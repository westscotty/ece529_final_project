import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import KMeans

green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)

def draw_corner_markers(img, corners, color=red, size=2):
    # if not type(corners) == type([]):
    #     breakpoint()
    for (y, x) in corners:
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, -1)
    return img

def draw_lines(img, corner1, corner2, color):
    cv2.line(img, corner1, corner2, color, 2)
    return img

def make_comparison_image(images, titles, suptitle, output_file):
    
    # Display and save the image with corners
    n = len(images)
    plt.figure(figsize=(12, 8))
    # plt.suptitle(suptitle)
    for i, image in enumerate(images):
        # plt.axis('off')
        plt.subplot(1, n, i+1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close() 
        
def write_image(image, suptitle, output_file):
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Display and save the image with corners
    plt.figure(figsize=(12, 8))
    plt.suptitle(suptitle)
    plt.axis('off')
    plt.imshow(image_rgb)
    # plt.imsave(output_file, image, cmap='gray')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close() 
        
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
    # Groups corners that are within the specified distance threshold.

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
    # Group corners using nearest neighbor search.

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
    # Cluster corners into a specified number of groups using K-Means.

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
    # Draw clusters of corners and their bounding boxes on the image.
    
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
    # Generate bounding boxes for groups of corners.
    
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

def plot_stats(frames, stat1, stat2, yaxis="Y Axis", title="", output_file=""):
    
    plt.figure()
    plt.plot(frames, stat1, color='red', label='Numpy')
    plt.plot(frames, stat2, color='green', label='OpenCV')
    plt.xlabel('Frame ID')
    plt.ylabel(yaxis)
    # plt.yticks([0,1])
    plt.legend()
    # plt.tight_layout()
    plt.grid(visible=True)
    if title:
        plt.suptitle(title)
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()

def plot_error(frames, error, yaxis="Percent Error", title="", output_file=""):
    
    plt.figure()
    plt.plot(frames, error, color='red', label='Numpy')
    plt.xlabel('Frame ID')
    plt.ylabel(yaxis)
    # plt.yticks([0,1])

    plt.legend()
    # plt.tight_layout()
    plt.grid(visible=True)
    if title:
        plt.suptitle(title)
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()
    
def create_histogram(data_dict, output_file):
    
    # Create a grid of histograms
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    # Plot histograms for each parameter
    for i, (var_name, values) in enumerate(data_dict.items()):
        ax = axes[i]
        ax.hist(values, bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax.set_title(f"Histogram of {var_name}")
        ax.set_xlabel(var_name)
        ax.set_ylabel("Frequency")

    # Remove unused subplots if any
    for j in range(len(data_dict), len(axes)):
        fig.delaxes(axes[j])

    plt.title(f"Histogram for MC Variables")
    # plt.tight_layout()
    plt.savefig(output_file)
    plt.close()