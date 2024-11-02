import cv2
import os
import shutil

def save_frames(input_video_path, output_image_dir="extracted_frames"):
    # Create the output directory if it doesn't exist
    if os.path.isdir(output_image_dir):
        shutil.rmtree(output_image_dir)  # Remove existing directory if it exists
    os.makedirs(output_image_dir, exist_ok=True)  # Create the output directory

    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file at {input_video_path}")
        return

    frame_id = 0  # Track the current frame number
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames

    print(f"Extracting frames from {input_video_path}... Total frames: {total_frames}")

    # Loop through the frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames

        # Save every 50th frame as a PNG
        # if frame_id % 50 == 0:
        image_path = os.path.join(output_image_dir, f"frame_{frame_id}.png")
        if cv2.imwrite(image_path, frame):
            print(f"Saved {image_path}")
        else:
            print(f"Error saving frame {frame_id} to {image_path}")

        frame_id += 1  # Increment the frame counter

    # Release resources
    cap.release()
    print(f"Finished extracting frames. Total frames processed: {frame_id}. Frames saved: {frame_id // 50}.")

# Example usage
video_file = '../data/videos/blue_angels_formation.mp4'
new_image_dir = f'../data/full_frame_video_images/{os.path.basename(video_file).rsplit(".", 1)[0]}'
save_frames(video_file, output_image_dir=new_image_dir)
