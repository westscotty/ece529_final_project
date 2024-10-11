import numpy as np
import cv2
from tqdm import tqdm
from numba import jit


@jit(nopython=True)
def manual_threshold(gradient_magnitude, threshold):
    """Applies thresholding from scratch to the gradient magnitude."""
    height, width = gradient_magnitude.shape
    thresholded = np.zeros_like(gradient_magnitude, dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            if gradient_magnitude[i, j] >= threshold:
                thresholded[i, j] = 255  # Set pixel to maximum value
            else:
                thresholded[i, j] = 0    # Set pixel to 0
    
    return thresholded

@jit(nopython=True)
def flood_fill(image, x, y, label, labeled_image):
    """Performs flood fill from pixel (x, y) and labels the connected component."""
    stack = [(x, y)]
    labeled_image[y, x] = label
    height, width = image.shape
    min_x, min_y = x, y
    max_x, max_y = x, y

    while stack:
        cx, cy = stack.pop()
        
        # Update bounding box coordinates
        min_x = min(min_x, cx)
        max_x = max(max_x, cx)
        min_y = min(min_y, cy)
        max_y = max(max_y, cy)
        
        # Check neighbors (4-connectivity: up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height and image[ny, nx] == 255 and labeled_image[ny, nx] == 0:
                stack.append((nx, ny))
                labeled_image[ny, nx] = label
    
    return min_x, min_y, max_x, max_y

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

def compute_structure_tensor(Ix, Iy, window_size):
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    kernel = np.ones((window_size, window_size), dtype=np.float32)
    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)
    
    return Sxx, Syy, Sxy

def shi_tomasi_response(Sxx, Syy, Sxy):
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy * Sxy
    eigenvalue_min = 0.5 * (trace - np.sqrt(trace ** 2 - 4 * det))
    return eigenvalue_min

def non_maximum_suppression(corners, radius=10):
    suppressed_corners = []
    corners = sorted(corners, key=lambda x: -x[2])  # Sort by response strength
    mask = np.zeros(len(corners), dtype=bool)
    
    for i, (x, y, response) in enumerate(corners):
        if not mask[i]:
            suppressed_corners.append((x, y))
            for j, (x2, y2, _) in enumerate(corners):
                if (x - x2)**2 + (y - y2)**2 < radius**2:
                    mask[j] = True
    
    return np.array(suppressed_corners)

def find_connected_components(image):
    """Finds connected components in a binary image and returns bounding boxes."""
    height, width = image.shape
    labeled_image = np.zeros_like(image, dtype=np.int32)
    current_label = 1
    bounding_boxes = []

    for y in range(height):
        for x in range(width):
            if image[y, x] == 255 and labeled_image[y, x] == 0:
                # Perform flood fill for each new connected component
                min_x, min_y, max_x, max_y = flood_fill(image, x, y, current_label, labeled_image)
                bounding_boxes.append((min_x, min_y, max_x, max_y))
                current_label += 1

    return bounding_boxes

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

def detect_corners_sobel(image, threshold=50):
    """Detects edges using Sobel and applies manual thresholding."""
    Ix, Iy = sobel_operator(image)
    
    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    
    # Apply manual thresholding
    thresholded = manual_threshold(gradient_magnitude, threshold)
    
    # Find connected components and bounding boxes
    bounding_boxes = find_connected_components(thresholded)
    
    return bounding_boxes

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
    # window_width_pixels = int(window_size_inches[0] * dpi)
    # window_height_pixels = int(window_size_inches[1] * dpi)

    # Setup the VideoWriter to save the grayscale video with bounding boxes
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height), isColor=False)

    # Process frames with a progress bar
    print(f"Processing video: {input_video_path}")
    
    frame_id = 0
    for _ in tqdm(range(total_frames), desc="Processing video", unit="frame"):

        ret, frame = cap.read()
        
        # if not frame_id%4==0:
        #     frame_id += 1
        #     continue
        
        if not ret:
            break  # No more frames to process

        # Convert the original frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect edges and draw bounding boxes
        bounding_boxes = detect_corners_sobel(gray_frame)

        # Draw bounding boxes around the detected contours
        for (min_x, min_y, max_x, max_y) in bounding_boxes:
            cv2.rectangle(gray_frame, (min_x, min_y), (max_x, max_y), (255, 255, 255), 2)

        # Write the processed grayscale frame with bounding boxes to the output video
        out.write(gray_frame)
        # Write the processed grayscale frame with bounding boxes to the output video
        out.write(gray_frame)
        
        if frame_id == 20:
            break
            
        frame_id += 1
        
    # Release the resources
    cap.release()
    out.release()
    print(f"Video saved successfully as {output_video_path}.")


# Example usage:
video_file = 'data/videos/blue_angels_formation.mp4'
process_video(video_file, f"{video_file.split('.')[0]}_output.mp4")
