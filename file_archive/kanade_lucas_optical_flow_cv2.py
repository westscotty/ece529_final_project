import cv2
import numpy as np
from tqdm import tqdm
import shi_tomasi_corners as stc
import utilities as utils

def process_video(input_video_path, output_video_path, frame_rate=30):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file at {input_video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup the VideoWriter to save the output in color
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height), isColor=True)

    # Parameters for Shi-Tomasi corner detection and KLT
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(max_corners=800, ksize=3, sensitivity=0.005, min_dist=5)

    print(f"Processing video: {input_video_path}")
    
    # Read the first frame and find initial corners
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        return
    
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    points_prev, __ = np.array(stc.shi_tomasi_corners(gray_first_frame, gray_first_frame.shape, **feature_params), dtype=np.float32).reshape(-1, 1, 2)

    for _ in tqdm(range(total_frames - 1), desc="Processing video", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check if there are points to track
        if points_prev is not None and len(points_prev) > 0:
            points_curr, st, err = cv2.calcOpticalFlowPyrLK(gray_first_frame, gray_frame, points_prev, None, **lk_params)
            
            if points_curr is not None and st is not None:
                good_new = points_curr[st == 1]
                good_old = points_prev[st == 1]

                # Draw the tracks on the color frame
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = map(int, new.ravel())
                    c, d = map(int, old.ravel())
                    frame = utils.draw_corner_markers(frame, [(a, b)], (0, 255, 0))  # Green markers
                    frame = utils.draw_lines(frame, (a, b), (c, d), (255, 0, 0))    # Blue lines
                    
                if good_new.size > 0:
                    corner_groups_nn =  utils.group_corners_nearest_neighbors(good_new, 60)
                    bboxes_nn = utils.make_bounding_boxes_from_groups(corner_groups_nn, frame)
                    frame = utils.add_bounding_boxes_to_image(frame, bboxes_nn, color=(0, 0, 255))
                # else:
                #     print("No good new points found for clustering.")
                
                # Update points for the next frame
                points_prev = good_new.reshape(-1, 1, 2)
            else:
                print("Reinitializing points due to tracking failure.")
                points_prev = np.array(stc.shi_tomasi_corners(gray_frame, gray_frame.shape, **feature_params), dtype=np.float32).reshape(-1, 1, 2)
        else:
            # Reinitialize if no points found
            # print("Reinitializing points due to no points found.")
            points_prev = np.array(stc.shi_tomasi_corners(gray_frame, gray_frame.shape, **feature_params), dtype=np.float32).reshape(-1, 1, 2)

        # Update the reference frame for the next iteration
        gray_first_frame = gray_frame.copy()

        # Write the processed color frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Video saved successfully as {output_video_path}.")

# Example usage:
video_file = 'data/videos/blue_angels_formation.mp4'
process_video(video_file, f"{video_file.split('.')[0]}_output_klt.mp4")
