import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from numba import jit

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

def draw_corner_markers(img, corners, color):
    
    for (y, x) in corners:
        cv2.circle(img, (x, y), 3, color, -1)
    return img

def draw_lines(img, corner1, corner2, color):
    
    cv2.line(img, corner1, corner2, color, 2)
    return img

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


def make_comparison_image(images, titles, suptitle):
    
    # Display and save the image with corners
    n = len(images)
    plt.figure(figsize=(12, 8))
    plt.suptitle(suptitle)
    for i, image in enumerate(images):
        plt.axis('off')
        plt.subplot(1, n, i+1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray')
        
        
def make_bounding_boxes(corners, image):

    # Generate bounding boxes based on suppressed corners and the thresholded image
    bounding_boxes = []
    for (y, x) in corners:
        # Define a small box around each corner
        box_size = 10  # Adjust size as needed
        min_x = max(0, x - box_size)
        min_y = max(0, y - box_size)
        max_x = min(image.shape[1], x + box_size)
        max_y = min(image.shape[0], y + box_size)
        bounding_boxes.append((min_x, min_y, max_x, max_y))

    return bounding_boxes

def add_bounding_boxes_to_image(image, bounding_boxes, color=(0, 255, 0)):
    
    for box in bounding_boxes:
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        cv2.rectangle(image, pt1, pt2, color, 2)
        
    return image

def group_nearby_corners(corners, distance_threshold=15):
    """
    Groups corners that are within the specified distance threshold.
    
    Args:
        corners (list): A list of (x, y) tuples representing corner coordinates.
        distance_threshold (float): The maximum distance for corners to be considered part of the same group.
        
    Returns:
        list: A list of lists, where each inner list contains grouped corner coordinates.
    """
    groups = []  # List to hold the groups of corners
    visited = set()  # Set to keep track of visited corners
    
    for i, corner in enumerate(corners):
        if i in visited:
            continue  # Skip if this corner has already been grouped
        
        # Start a new group with the current corner
        group = [corner]
        visited.add(i)
        
        # Check against all other corners
        for j, other_corner in enumerate(corners):
            if j in visited:
                continue  # Skip already visited corners
            # Calculate the distance between corners
            distance = np.linalg.norm(np.array(corner) - np.array(other_corner))
            if distance <= distance_threshold:
                group.append(other_corner)  # Add to current group
                visited.add(j)  # Mark this corner as visited
        
        groups.append(group)  # Add the completed group to the list of groups
    
    return groups

def group_corners_nearest_neighbors(corners, max_distance):
    """
    Group corners using nearest neighbor search.
    
    Args:
        corners (list): A list of corner coordinates.
        max_distance (float): The maximum distance for grouping corners.
        
    Returns:
        list: A list of groups of corner coordinates.
    """
    # Convert the list of corners to a NumPy array for KDTree
    corner_array = np.array(corners)

    # Build a KDTree for efficient nearest neighbor search
    tree = KDTree(corner_array)
    
    # To keep track of which corners have been grouped
    groups = []
    visited = set()
    
    for i, corner in enumerate(corners):
        if i in visited:
            continue
        
        # Find all neighbors within max_distance
        indices = tree.query_ball_point(corner, max_distance)
        group = [corner_array[j] for j in indices]
        
        # Mark these corners as visited
        visited.update(indices)
        
        # Append the group of corners
        groups.append(group)
        
    return groups

def cluster_corners(corners, n_clusters):
    """
    Cluster corners into a specified number of groups using K-Means.
    
    Args:
        corners (list): A list of corner coordinates.
        n_clusters (int): The number of clusters to form.
        
    Returns:
        list: A list of groups of corner coordinates.
    """
    # Convert the list of corners to a NumPy array
    corner_array = np.array(corners)
    breakpoint()
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(corner_array)

    # Get the labels (cluster assignments) and cluster centers
    labels = kmeans.labels_

    # Group corners based on cluster labels
    grouped = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        grouped[label].append(corner_array[idx])

    return grouped

# Visualization Function
def draw_clusters_on_image(image, corner_groups):
    """
    Draw clusters of corners and their bounding boxes on the image.
    
    Args:
        image (numpy array): The image to draw on.
        corner_groups (list): List of groups of corner coordinates.
        
    Returns:
        numpy array: The image with clusters drawn.
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # Add more colors as needed
    
    for i, group in enumerate(corner_groups):
        # Get the coordinates for the bounding box
        if len(group) > 0:
            x_coords = [corner[1] for corner in group]
            y_coords = [corner[0] for corner in group]
            min_x, min_y = int(min(x_coords)), int(min(y_coords))
            max_x, max_y = int(max(x_coords)), int(max(y_coords))
            # Draw the bounding box with a unique color for each cluster
            color = colors[i % len(colors)]
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color, 2)

    return image

def make_bounding_boxes_from_groups(corner_groups, image):
    """
    Generate bounding boxes for groups of corners.
    
    Args:
        corner_groups (list): A list of lists of grouped corner coordinates.
        image (numpy array): The image to which the bounding boxes will be applied.
        
    Returns:
        list: A list of bounding boxes defined by (min_x, min_y, max_x, max_y).
    """
    bounding_boxes = []
    
    for group in corner_groups:
        # Check if the group is not empty
        if len(group) == 0:
            continue  # Skip empty groups

        # Extract x and y coordinates from the group of corners
        x_coords = [corner[1] for corner in group]
        y_coords = [corner[0] for corner in group]
        
        # Calculate the bounding box coordinates
        min_x = max(0, min(x_coords))
        min_y = max(0, min(y_coords))
        max_x = min(image.shape[1], max(x_coords))
        max_y = min(image.shape[0], max(y_coords))
        
        # Add the bounding box to the list
        bounding_boxes.append((min_x, min_y, max_x, max_y))

        # Debug output
        # print(f"Group: {group}, Bounding Box: ({min_x}, {min_y}, {max_x}, {max_y})")

    return bounding_boxes
