import cv2
import numpy as np
from tqdm import tqdm
import shi_tomasi_corners as stc
from utils import debug_messages, error_metrics
from plot_utils import make_comparison_image, draw_corner_markers, draw_lines, group_corners_nearest_neighbors, make_bounding_boxes_from_groups, add_bounding_boxes_to_image
import video_utils as vid
from copy import copy
import argparse
from image_operations import gaussian_low_pass_cv2, gaussian_low_pass_numpy, sobel_operator_cv2, sobel_operator_numpy, gaussian_high_pass_filter, averaging_low_pass_filter, histogram_equalization
from tqdm import tqdm

gradient = { 'cv2': sobel_operator_cv2,
                    'numpy': sobel_operator_numpy
                  }
gaussian_low_pass   = { 'cv2': gaussian_low_pass_cv2,
                    'numpy': gaussian_low_pass_numpy
                  }

def calcOpticalFlowPyrLK(prev_img, curr_img, points_prev, lk_params=None, ksize=3, method='cv2'):

    winSize, maxLevel, criteria = lk_params.values()
    __, max_iter, epsilon = criteria
    points_curr = np.copy(points_prev)  # Initial flow starts as the initial points
    st = np.ones(len(points_prev), dtype=np.uint8)  # Status array
    err = np.zeros(len(points_prev), dtype=np.float32)  # Error array

    # Generate pyramids for prev_img and curr_img
    pyramid_prev = [prev_img]
    pyramid_curr = [curr_img]
    for i in range(1, maxLevel + 1):
        pyramid_prev.append(cv2.pyrDown(pyramid_prev[i - 1]))
        pyramid_curr.append(cv2.pyrDown(pyramid_curr[i - 1]))

    # Start from the top of the pyramid (smallest scale)
    scale_factor = 2 ** maxLevel
    for level in range(maxLevel, -1, -1):
        # Scale down points for current pyramid level
        points_prev_scaled = points_prev / scale_factor
        points_curr_scaled = points_curr / scale_factor

        half_w, half_h = winSize[0] // 2, winSize[1] // 2

        for i, point in enumerate(points_prev_scaled):
            x, y = int(point[0][0]), int(point[0][1])
            prev_patch = pyramid_prev[level][y - half_h:y + half_h + 1, x - half_w:x + half_w + 1]
            curr_patch = pyramid_curr[level][y - half_h:y + half_h + 1, x - half_w:x + half_w + 1]

            if prev_patch.shape == winSize and curr_patch.shape == winSize:
                Ix = gradient[method](prev_patch, ksize=ksize, xy=0)
                Iy = gradient[method](prev_patch, ksize=ksize, xy=1)
                It = curr_patch.astype(np.float32) - prev_patch.astype(np.float32)

                Ixx = Ix ** 2
                Iyy = Iy ** 2
                Ixy = Ix * Iy
                Ixt = Ix * It
                Iyt = Iy * It

                A = np.array([[Ixx.sum(), Ixy.sum()], [Ixy.sum(), Iyy.sum()]])
                b = -np.array([Ixt.sum(), Iyt.sum()])

                flow = np.zeros(2)
                for _ in range(max_iter):
                    if np.linalg.det(A) > 1e-5:
                        new_flow = np.linalg.solve(A, b)
                        if np.linalg.norm(new_flow - flow) < epsilon:
                            break
                        flow = new_flow
                    else:
                        st[i] = 0  # Tracking failed
                        flow = np.zeros(2)
                        break

                # Apply incremental flow to scaled point location
                points_curr_scaled[i][0][0] += flow[0]
                points_curr_scaled[i][0][1] += flow[1]
                err[i] = np.abs(It).mean()

        # Scale `points_curr_scaled` back up and accumulate into `points_curr`
        points_curr += (points_curr_scaled * scale_factor - points_curr) / scale_factor
        scale_factor //= 2

    points_curr = np.array(points_curr, dtype=np.float32)
    st = st.reshape(-1, 1)
    err = err.reshape(-1, 1)
    
    points_curr_cv2, st_cv2, err_cv2 = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, points_prev, None, **lk_params)

    return points_curr, st, err, points_curr_cv2, st_cv2, err_cv2


def lucas_kanade_optical_flow(input_video_path, output_video_path, frame_rate=30, lk_params=None, feature_params=None, reinit_threshold=50, err_thresh=0.7, skip_frames=0, method='cv2', pyrdown_level=1):
    
    print(f"Kanade-Lucas-Tomasi Feature Tracker")
    print(f"Processing video: {input_video_path}\n")

    ## Load video, get grep info, setup output video
    cap, frame_width, frame_height, total_frames = vid.open_video(input_video_path)
    crop_size, crop_x, crop_y = vid.get_crop_info(frame_width, frame_height)
    out_video = vid.prepare_output_video(output_video_path, frame_rate, crop_size, pyrdown_level)
    
    ## Read first frame, crop, pyrdown, change to grayscale
    first_frame = vid.read_frame(cap, True, crop_size, crop_x, crop_y, pyrdown_level)
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    ## Detect initial corners
    corners_0, corners_0_cv2 = stc.shi_tomasi_corners(gray_first_frame, **feature_params)
    points_prev = vid.reshape_points(corners_0)
    
    ## Write first frame to output video
    first_frame = draw_corner_markers(first_frame, corners_0, vid.green)  # Green markers
    out_video.write(first_frame)

    ## Process each frame
    for i in tqdm(range(total_frames - 1), desc="Processing Frames", unit="frame"):
        
        try:
            if i % skip_frames == 1:
                continue
        except ZeroDivisionError as e:
            if i == 0:
                print("No frames skipped")
        frame = vid.read_frame(cap, True, crop_size, crop_x, crop_y, pyrdown_level)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ## Check if there are points to track
        if points_prev is not None and len(points_prev) > reinit_threshold:
            points_curr, st, err, points_curr_cv2, st_cv2, err_cv2 = calcOpticalFlowPyrLK(gray_first_frame, gray_frame, points_prev, method=method, lk_params=lk_params)
            
            if points_curr is not None and st is not None:                
                valid_mask = st.flatten() == 1
                
                # Filter points based on the valid mask
                good_new = points_curr[valid_mask]
                good_old = points_prev[valid_mask]
                err_valid = err[valid_mask]

                # Further filter based on error threshold using the same mask
                err_mask = err_valid.flatten() < err_thresh

                # Final filtering
                good_new = good_new[err_mask]
                good_new = good_new.reshape(len(good_new), 2)
                good_old = good_old[err_mask]
                good_old = good_old.reshape(len(good_old), 2)
                
                ## Draw the tracks on the color frame
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = map(int, new.ravel())
                    c, d = map(int, old.ravel())
                    frame = draw_corner_markers(frame, [(a, b)], vid.green)  # Green markers
                    frame = draw_lines(frame, (b, a), (d, c), vid.blue)    # Blue lines
                    
                ## Bounding box around clusters of points
                if good_new.size > 0:
                    corner_groups_nn = group_corners_nearest_neighbors(good_new, 50)
                    bboxes_nn = make_bounding_boxes_from_groups(corner_groups_nn, frame)
                    frame = add_bounding_boxes_to_image(frame, bboxes_nn, vid.red)

                ## Update points for the next frame
                points_prev = good_new.reshape(-1, 1, 2)
            else:
                ## Reinitialize points if tracking fails
                corners_0, corners_0_cv2 = stc.shi_tomasi_corners(gray_frame, **feature_params)
                points_prev = vid.reshape_points(corners_0)

        else:
            ## Reinitialize points if few points remain or were lost
            corners_0, corners_0_cv2 = stc.shi_tomasi_corners(gray_frame, **feature_params)
            points_prev = vid.reshape_points(corners_0)

        # Update the reference frame for the next iteration, and write out frame
        gray_first_frame = gray_frame.copy()
        out_video.write(frame)

    # Release resources
    cap.release()
    out_video.release()
    print(f"Video saved successfully as {output_video_path}.")


# Example usage
if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='Run Lucas-Kanade CV2 Method.')

    # Add arguments
    parser.add_argument('-in_vid', '--input_video', type=str, required=True, help='Path to the input image file')
    parser.add_argument('-mc', '--max_corners', type=int, default=2000, required=False, help='Maximum number of corners to detect (default: 25)')
    parser.add_argument('-ks', '--kernel_size', type=int, choices=[3, 5, 7], default=3, required=False, help='Size of the Sobel kernel (3, 5, or 7, default: 3)')
    parser.add_argument('-m', '--method', type=str, choices=['cv2', 'numpy'], default='numpy', required=False, help='Sobel operator to use (default: numpy)')
    parser.add_argument('-gs', '--gaussian_sigma', type=int, default=0, required=False, help='Sigma value for gaussian blur kernel')
    parser.add_argument('-s', '--sensitivity', type=float, default=0.001, required=False, help='Sensitivity for corner detection (default: 0.04)')
    parser.add_argument('-md', '--minimum_distance', type=int, default=5, required=False, help='Minumum distance between detected corners (used for removing oversample corners)')
    parser.add_argument('-debug', '--debug', action='store_true', required=False, default=False, help='Enable debugging print statements')
    parser.add_argument('-debug_images', '--debug_images', action='store_true', required=False, default=False, help='Enable debugging image showing')
    parser.add_argument('-fr', '--frame_rate', default=30, type=int, required=False, help='Frame rate for output video writer')
    parser.add_argument('-rt', '--reinit_threshold', default=50, type=int, required=False, help='Threshold for reinitialization of corner detections.')
    parser.add_argument('-ws', '--window_size', default=5, type=int, required=False, help='Optical flow window size')
    parser.add_argument('-et', '--error_threshold', default=0.7, type=float, required=False, help='Error threshold for tracks being successful between frames')
    parser.add_argument('-sf', '--skip_frames', default=0, type=int, required=False, help='Num frames to skip in between video processing')
    parser.add_argument('-pl', '--pyrdown_level', default=1, type=int, required=False, help='Image pyramid donw function iterations')


    # Parse the arguments
    args = parser.parse_args()
    
    # Parameters for Shi-Tomasi corner detection and KLT tracking
    lk_params = dict(winSize=(args.window_size, args.window_size), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
    feature_params = dict(max_corners=args.max_corners, ksize=args.kernel_size, sensitivity=args.sensitivity, min_dist=args.minimum_distance, method=args.method)
    
    # Example usage:
    video_file = 'data/videos/blue_angels_formation.mp4'
    # output_video = f"{args.input_video.split('.')[0]}_output_klt_20241002.mp4"
    output_video = f"output.mp4"
    lucas_kanade_optical_flow(args.input_video, output_video, frame_rate=args.frame_rate, lk_params=lk_params, feature_params=feature_params, reinit_threshold=args.reinit_threshold, err_thresh=args.error_threshold, skip_frames=args.skip_frames, method=args.method, pyrdown_level=args.pyrdown_level)

    #TODO
    # Evalute performance
    # Implement mean-shift algorithm
    # comment code
    # DOEs for kernel sizes, and additional image operations
    # write report detailing the algorithm design and code implementation