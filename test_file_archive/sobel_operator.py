import numpy as np
import cv2
from tqdm import tqdm
from numba import jit


@jit(nopython=True)
def apply_convolution(image, kernel):
    """Applies a 2D convolution to an image with the given kernel using Numba."""
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Create a padded image
    padded_image_height = image_height + 2 * pad_height
    padded_image_width = image_width + 2 * pad_width
    padded_image = np.zeros((padded_image_height, padded_image_width), dtype=np.float32)

    # Fill the padded image with the original image
    for i in range(image_height):
        for j in range(image_width):
            padded_image[i + pad_height, j + pad_width] = image[i, j]

    # Prepare output image
    convolved_image = np.zeros_like(image, dtype=np.float32)

    # Perform convolution using nested loops
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Apply the kernel to the region and sum the result
            convolved_image[i, j] = np.sum(region * kernel)
    
    return convolved_image

def sobel_operator(image):
    """Applies Sobel operator to compute image gradients Ix and Iy."""
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    # Apply convolution using the Sobel X and Y kernels
    Ix = apply_convolution(image, sobel_x)
    Iy = apply_convolution(image, sobel_y)
    
    return Ix, Iy

def process_video(input_video_path, output_video_path, frame_rate=30, window_size_inches=(10, 7), dpi=100):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file at {input_video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the desired output video dimensions in pixels based on window size in inches and DPI
    window_width_pixels = int(window_size_inches[0] * dpi)
    window_height_pixels = int(window_size_inches[1] * dpi)

    # Calculate scaling factors to fit the window size while maintaining aspect ratio
    scale_factor = min(window_width_pixels / (2 * frame_width), window_height_pixels / frame_height)
    scaled_width = int(frame_width * scale_factor)
    scaled_height = int(frame_height * scale_factor)

    # Setup the VideoWriter to save the side-by-side video (grayscale output)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (scaled_width * 2, scaled_height), isColor=False)

    # Process frames with a progress bar
    print(f"Processing video: {input_video_path}")
    for _ in tqdm(range(total_frames), desc="Processing video", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break  # No more frames to process

        # Convert the original frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Sobel operator to grayscale frame (example processing)
        sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = cv2.convertScaleAbs(sobel)

        # Concatenate the original grayscale and processed frames side by side (no conversion to BGR)
        concatenated_frame = cv2.hconcat([gray_frame, sobel])

        # Resize the concatenated frame to fit within the specified window size
        resized_frame = cv2.resize(concatenated_frame, (scaled_width * 2, scaled_height))

        # Write the resized concatenated grayscale frame to the output video
        out.write(resized_frame)

    # Release the resources
    cap.release()
    out.release()
    print(f"Video saved successfully as {output_video_path}.")


# video_file = 'data/videos/drone_following_model_plane.mp4'
video_file = 'data/videos/helicopter2.mp4'
process_video(video_file, f"{video_file.split('.')[0]}_output.mp4")