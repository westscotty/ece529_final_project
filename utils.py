import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2


def debug_messages(message):
    print(f"\n<< DEBUG >>")
    print(message)
    print("<< DEBUG >>\n")
    
    
def coordinate_density_filter(coords, min_dist, max_corners, image_height, image_width, image_edge_threshold=5):
    # Enforce minimum distance constraint
    filtered_coords = []
    for coord in coords:
        if all(np.linalg.norm(coord - np.array(fc)) >= min_dist for fc in filtered_coords):
            filtered_coords.append(coord)
            if len(filtered_coords) >= max_corners:
                break
    
    # filtered_coords = coords
    final_corners = []    
    for coord in filtered_coords:
        y, x = coord
        # Check if the point is not within the threshold of the edges
        if (x > image_edge_threshold and x < (image_width - image_edge_threshold) and
            y > image_edge_threshold and y < (image_height - image_edge_threshold)):
            final_corners.append(coord)
    # final_corners = filtered_coords
    return np.array(final_corners)


def error_metrics(image1, image2):
    """
    Calculate the percent absolute error and PSNR between two images.
    
    Parameters:
        image1 (np.ndarray): First image (e.g., Sobel output from NumPy).
        image2 (np.ndarray): Second image (e.g., Sobel output from OpenCV).
    
    Returns:
        float: The percent absolute error between the two images.
        float: The PSNR value between the two images.
    """
    # Ensure the images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for error calculation.")

    # Convert both images to float32 to ensure consistent data type
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # Calculate the absolute error
    absolute_error = np.abs(image1 - image2)

    # Calculate percent absolute error
    percent_error = (absolute_error / (np.abs(image2) + 1e-8)) * 100  # Add a small value to avoid division by zero
    
    # Calculate PSNR value
    psnr_value = psnr(image1, image2, data_range=image2.max() - image2.min())
    
    # # Normalize Cross-correlation (NCC)
    # Ix_cv2_flat = Ix_cv2.flatten()
    # Ix_numpy_flat = Ix_numpy.flatten()
    # cosine_similarity = np.dot(Ix_cv2_flat, Ix_numpy_flat) / (
    # np.linalg.norm(Ix_cv2_flat) * np.linalg.norm(Ix_numpy_flat) + 1e-8)

    # # Cosine similarity
    # numerator = np.sum((Ix_cv2 - Ix_cv2.mean()) * (Ix_numpy - Ix_numpy.mean()))
    # denominator = np.sqrt(np.sum((Ix_cv2 - Ix_cv2.mean())**2) * np.sum((Ix_numpy - Ix_numpy.mean())**2))
    # ncc = numerator / (denominator + 1e-8)  # To avoid division by zero

    # # Gradient direction similarity (GDS)
    # angle_diff = np.arctan2(Iy_cv2, Ix_cv2) - np.arctan2(Iy_numpy, Ix_numpy)
    # angle_diff = np.abs((angle_diff + np.pi) % (2 * np.pi) - np.pi)  # Normalize to [0, pi]
    # gds = np.mean(angle_diff)

    # Return the mean percent absolute error and PSNR value
    return np.mean(percent_error), psnr_value