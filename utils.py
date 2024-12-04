import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import cv2
from skimage.metrics import structural_similarity
from video_utils import reshape_points


def debug_messages(message):
    print(f"\n\n<< DEBUG >>")
    print(message)
    print("<< DEBUG >>")
    
    
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


def psnr(image1, image2):
   
    # Ensure the images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for error calculation.")

    # Convert both images to float32 to ensure consistent data type
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Calculate PSNR value
    psnr_value = peak_signal_noise_ratio(image1, image2, data_range=image1.max() - image1.min())

    return psnr_value

def ssim(image1, image2):
    
    # Ensure the images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for error calculation.")

    # Convert both images to float32 to ensure consistent data type
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    ssim_value = structural_similarity(image1, image2, data_range=image1.max() - image1.min())

    return ssim_value

def mae(image1, image2):
    
    # Ensure the images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for error calculation.")

    # Convert both images to float32 to ensure consistent data type
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute error
    absolute_error = np.abs(image1 - image2)

    # Calculate percent absolute error
    percent_error = (absolute_error / (np.abs(image2) + 1e-8)) * 100  # Add a small value to avoid division by zero
    
    return np.mean(percent_error)

def precision(corners1, corners2, threshold):
    
    corners1 = np.squeeze(corners1)
    corners2 = np.squeeze(corners2) 
    
    distances = np.linalg.norm(corners1[:, None] - corners2, axis=2)
    min_distances = np.min(distances, axis=1)
    
    # True Positives: corners in corners_1 that are within the threshold distance to corners_2
    TP = np.sum(min_distances <= threshold)
    
    # False Positives: corners in corners_2 that do not match any corner in corners_1 within threshold
    FP = len(corners2) - TP
    
    # False Negatives: corners in corners_1 that do not match any corner in corners_2 within threshold
    FN = len(corners1) - TP
    
    # Calculate Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    return precision

def recall(corners1, corners2, threshold):
    
    corners1 = np.squeeze(corners1)
    corners2 = np.squeeze(corners2) 
    
    distances = np.linalg.norm(corners1[:, None] - corners2, axis=2)
    min_distances = np.min(distances, axis=1)
    
    # True Positives: corners in corners_1 that are within the threshold distance to corners_2
    TP = np.sum(min_distances <= threshold)
    
    # False Positives: corners in corners_2 that do not match any corner in corners_1 within threshold
    FP = len(corners2) - TP
    
    # False Negatives: corners in corners_1 that do not match any corner in corners_2 within threshold
    FN = len(corners1) - TP
    
    # Calculate Precision and Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return recall
    

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