import cv2
import numpy as np
from tqdm import tqdm
import shi_tomasi_corners as stc
import utilities as utils
import video_utils as vid
from copy import copy
import argparse

def lucas_kanade_optical_flow(input_video_path, output_video_path, frame_rate=30, pyrdown=True, lk_params=None, feature_params=None, reinit_threshold=50, err_thresh=0.7):
    
    print(f"Kanade-Lucas-Tomasi Feature Tracker")
    print(f"Processing video: {input_video_path}\n")

    ## Load video, get grep info, setup output video
    cap, frame_width, frame_height, total_frames = vid.open_video(input_video_path)
    crop_size, crop_x, crop_y = vid.get_crop_info(frame_width, frame_height)
    out_video = vid.prepare_output_video(output_video_path, frame_rate, crop_size)
    
    ## Read first frame, crop, pyrdown, change to grayscale
    first_frame = vid.read_frame(cap, True, crop_size, crop_x, crop_y, pyrdown)
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    ## Detect initial corners
    corners_0, corners_0_cv2 = stc.shi_tomasi_corners(gray_first_frame, **feature_params)
    points_prev = vid.reshape_points(corners_0)
    
    ## Write first frame to output video
    first_frame = utils.draw_corner_markers(first_frame, corners_0, vid.green)  # Green markers
    out_video.write(first_frame)

    ## Process each frame
    for _ in tqdm(range(total_frames - 1), desc="Processing Frames", unit="frame"):
        
        frame = vid.read_frame(cap, True, crop_size, crop_x, crop_y, pyrdown)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ## Check if there are points to track
        if points_prev is not None and len(points_prev) > reinit_threshold:
            points_curr, st, err = cv2.calcOpticalFlowPyrLK(gray_first_frame, gray_frame, points_prev, None, **lk_params)
            
            if points_curr is not None and st is not None:
                # good_new = points_curr[st==1]
                # good_old = points_prev[st==1]
                
                valid_mask = st.flatten() == 1
                
                # Filter points based on the valid mask
                good_new = points_curr[valid_mask]
                good_old = points_prev[valid_mask]
                err_valid = err[valid_mask]

                # Further filter based on error threshold using the same mask
                err_mask = err_valid.flatten() < err_thresh

                # Final filtering
                good_new = good_new[err_mask].squeeze()
                good_old = good_old[err_mask].squeeze()
                
                ## Draw the tracks on the color frame
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = map(int, new.ravel())
                    c, d = map(int, old.ravel())
                    frame = utils.draw_corner_markers(frame, [(a, b)], vid.green)  # Green markers
                    frame = utils.draw_lines(frame, (b, a), (d, c), vid.blue)    # Blue lines
                    
                ## Bounding box around clusters of points
                if good_new.size > 0:
                    corner_groups_nn = utils.group_corners_nearest_neighbors(good_new, 50)
                    bboxes_nn = utils.make_bounding_boxes_from_groups(corner_groups_nn, frame)
                    frame = utils.add_bounding_boxes_to_image(frame, bboxes_nn, vid.red)

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
    parser.add_argument('-m', '--method', type=str, choices=['cv2', 'numpy'], default='cv2', required=False, help='Sobel operator to use (default: cv2)')
    parser.add_argument('-gs', '--gaussian_sigma', type=int, default=0, required=False, help='Sigma value for gaussian blur kernel')
    parser.add_argument('-s', '--sensitivity', type=float, default=0.001, required=False, help='Sensitivity for corner detection (default: 0.04)')
    parser.add_argument('-md', '--minimum_distance', type=int, default=5, required=False, help='Minumum distance between detected corners (used for removing oversample corners)')
    parser.add_argument('-debug', '--debug', action='store_true', required=False, default=False, help='Enable debugging print statements')
    parser.add_argument('-debug_images', '--debug_images', action='store_true', required=False, default=False, help='Enable debugging image showing')
    parser.add_argument('-fr', '--frame_rate', default=30, type=int, required=False, help='Frame rate for output video writer')
    parser.add_argument('-rt', '--reinit_threshold', default=50, type=int, required=False, help='Threshold for reinitialization of corner detections.')
    parser.add_argument('-ws', '--window_size', default=5, type=int, required=False, help='Optical flow window size')
    parser.add_argument('-et', '--error_threshold', default=0.7, type=float, required=False, help='Error threshold for tracks being successful between frames')

    # Parse the arguments
    args = parser.parse_args()
    
    # Parameters for Shi-Tomasi corner detection and KLT tracking
    lk_params = dict(winSize=(args.window_size, args.window_size), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
    feature_params = dict(max_corners=args.max_corners, ksize=args.kernel_size, sensitivity=args.sensitivity, min_dist=args.minimum_distance)
    
    # Example usage:
    video_file = 'data/videos/blue_angels_formation.mp4'
    output_video = f"{args.input_video.split('.')[0]}_output_klt.mp4"
    lucas_kanade_optical_flow(args.input_video, output_video, frame_rate=args.frame_rate, lk_params=lk_params, feature_params=feature_params, reinit_threshold=args.reinit_threshold, err_thresh=args.error_threshold)
    
    # lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # feature_params = dict(max_corners=1000, ksize=3, sensitivity=0.001, min_dist=5)
    # reinit_threshold = 50