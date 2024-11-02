import numpy as np
import cv2
from tqdm import tqdm
from numba import jit

@jit(nopython=True)
def manual_threshold(gradient_magnitude, threshold):
    height, width = gradient_magnitude.shape
    thresholded = np.zeros_like(gradient_magnitude, dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            if gradient_magnitude[i, j] >= threshold:
                thresholded[i, j] = 255
            else:
                thresholded[i, j] = 0
    
    return thresholded

@jit(nopython=True)
def flood_fill(image, x, y, label, labeled_image):
    stack = [(x, y)]
    labeled_image[y, x] = label
    height, width = image.shape
    min_x, min_y, max_x, max_y = x, y, x, y

    while stack:
        cx, cy = stack.pop()
        min_x = min(min_x, cx)
        max_x = max(max_x, cx)
        min_y = min(min_y, cy)
        max_y = max(max_y, cy)
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height and image[ny, nx] == 255 and labeled_image[ny, nx] == 0:
                stack.append((nx, ny))
                labeled_image[ny, nx] = label
    
    return min_x, min_y, max_x, max_y

@jit(nopython=True)
def apply_convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.zeros((image_height + 2 * pad_height, image_width + 2 * pad_width), dtype=np.float32)

    for i in range(image_height):
        for j in range(image_width):
            padded_image[i + pad_height, j + pad_width] = image[i, j]

    convolved_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
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
        if not mask[i]:  # Only consider corners that haven't been suppressed
            suppressed_corners.append((x, y))
            for j in range(i + 1, len(corners)):
                x2, y2, _ = corners[j]
                if (x - x2) ** 2 + (y - y2) ** 2 < radius ** 2:
                    mask[j] = True  # Suppress this corner
                # Exit early if the rest of the corners are too far away
                if (x - x2) ** 2 >= radius ** 2:
                    break  # Since the list is sorted, we can break early

    return np.array(suppressed_corners)

# def detect_corners_sobel(image, threshold=50):
#     Ix, Iy = sobel_operator(image)
#     gradient_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
#     thresholded = manual_threshold(gradient_magnitude, threshold)
#     bounding_boxes = find_connected_components(thresholded)
#     return bounding_boxes

def detect_corners(image, threshold=50):
    Ix, Iy = sobel_operator(image)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    # Apply manual thresholding
    thresholded = manual_threshold(gradient_magnitude, threshold)

    # Compute the structure tensor and the Shi-Tomasi response
    window_size = 5  # Adjust as needed
    Sxx, Syy, Sxy = compute_structure_tensor(Ix, Iy, window_size)
    response = shi_tomasi_response(Sxx, Syy, Sxy)

    # Prepare corners as tuples (x, y, response)
    corners = [(x, y, response[y, x]) for y in range(response.shape[0]) for x in range(response.shape[1]) if response[y, x] > threshold]

    # Apply non-maximum suppression
    suppressed_corners = non_maximum_suppression(corners)

    # Generate bounding boxes based on suppressed corners and the thresholded image
    bounding_boxes = []
    for (x, y) in suppressed_corners:
        # Ensure the corner is within the thresholded image
        if thresholded[y, x] == 255:  # Check if the corner is part of the thresholded response
            # Define a small box around each corner
            box_size = 5  # Adjust size as needed
            min_x = max(0, x - box_size)
            min_y = max(0, y - box_size)
            max_x = min(image.shape[1], x + box_size)
            max_y = min(image.shape[0], y + box_size)
            bounding_boxes.append((min_x, min_y, max_x, max_y))

    return bounding_boxes


def sobel_operator(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    Ix = apply_convolution(image, sobel_x)
    Iy = apply_convolution(image, sobel_y)
    
    return Ix, Iy

def find_connected_components(image):
    height, width = image.shape
    labeled_image = np.zeros_like(image, dtype=np.int32)
    current_label = 1
    bounding_boxes = []

    for y in range(height):
        for x in range(width):
            if image[y, x] == 255 and labeled_image[y, x] == 0:
                min_x, min_y, max_x, max_y = flood_fill(image, x, y, current_label, labeled_image)
                bounding_boxes.append((min_x, min_y, max_x, max_y))
                current_label += 1

    return bounding_boxes

def process_video(input_video_path, output_video_path, frame_rate=30):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file at {input_video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # window_width_pixels = int(window_size_inches[0] * dpi)
    # window_height_pixels = int(window_size_inches[1] * dpi)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height), isColor=False)

    print(f"Processing video: {input_video_path}")
    
    frame_id = 0
    for _ in tqdm(range(total_frames), desc="Processing video", unit="frame"):
        ret, frame = cap.read()
        
        
        if not ret:
            break  
        # downscaled_frame = cv2.pyrDown(frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bounding_boxes = detect_corners(gray_frame)

        for (min_x, min_y, max_x, max_y) in bounding_boxes:
            cv2.rectangle(gray_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

        out.write(gray_frame)
        frame_id += 1
        if frame_id == 5:
            break
    cap.release()
    out.release()
    print(f"Video saved successfully as {output_video_path}.")
    

# Example usage:
video_file = 'data/videos/blue_angels_formation.mp4'
process_video(video_file, f"{video_file.split('.')[0]}_output.mp4")
