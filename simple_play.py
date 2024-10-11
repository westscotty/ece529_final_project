# import cv2

# def play_video(video_path):
#     # Capture the video from the specified file
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     while True:
#         # Read a frame from the video
#         ret, frame = cap.read()

#         # If the frame is not returned, we've reached the end of the video
#         if not ret:
#             break

#         # Display the frame
#         cv2.imshow('Video Playback', frame)

#         # Exit playback when 'q' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture object and close windows
#     cap.release()
#     cv2.destroyAllWindows()

# # Example usage
# # video_file = 'data/videos/drone_following_model_plane.mp4'  # Replace with your video file path
# video_file = 'output_video.mp4'
# play_video(video_file)

import cv2

def process_video(input_video_path, output_video_path):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file at {input_video_path}")
        return

    # Get original frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Set up the VideoWriter for the output file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec for AVI
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    print(f"Processing video: {input_video_path}")
    print(f"Writing output video: {output_video_path}")

    # Read and write frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no more frames

        # Write the current frame to the output file
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Video saved successfully as {output_video_path}.")

# Example usage
video_file = 'data/videos/drone_following_model_plane.mp4'
output_video = 'output_video.avi'  # Output video in AVI format
process_video(video_file, output_video)