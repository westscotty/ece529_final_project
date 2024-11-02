import cv2
import numpy as np
import sys
from numba import jit

normalize_image = lambda x : cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

def create_sobel_operator(ksize):
    """Creates the needed sobel convolutional kernel based in input ksize."""
    
    if ksize == 3:
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=np.float32)

    elif ksize == 5:
        sobel_x = np.array([[-2, -1, 0, 1, 2],
                            [-4, -2, 0, 2, 4],
                            [-8, -4, 0, 4, 8],
                            [-4, -2, 0, 2, 4],
                            [-2, -1, 0, 1, 2]], dtype=np.float32)

        sobel_y = np.array([[2, 4, 8, 4, 2],
                            [1, 2, 4, 2, 1],
                            [0, 0, 0, 0, 0],
                            [-1, -2, -4, -2, -1],
                            [-2, -4, -8, -4, -2]], dtype=np.float32)

    elif ksize == 7:
        sobel_x = np.array([[-3, -2, -1, 0, 1, 2, 3],
                            [-6, -5, -4, 0, 4, 5, 6],
                            [-12, -10, -8, 0, 8, 10, 12],
                            [-6, -5, -4, 0, 4, 5, 6],
                            [-3, -2, -1, 0, 1, 2, 3]], dtype=np.float32)

        sobel_y = np.array([[3, 6, 12, 6, 3, 0, 0],
                            [2, 5, 10, 5, 2, 0, 0],
                            [1, 4, 8, 4, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [-1, -4, -8, -4, -1, 0, 0],
                            [-2, -5, -10, -5, -2, 0, 0],
                            [-3, -6, -12, -6, -3, 0, 0]], dtype=np.float32)

    else:
        sys.exit("Only sizes 3, 5, and 7 are supported.")

    return sobel_x, sobel_y

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

def normalize_image2(image):
    """Normalize an image to the range [0, 255]."""
    # Ensure the input is a float32 array
    image = image.astype(np.float32)

    # Find the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)

    # Scale the image to the range [0, 255]
    normalized_image = (image - min_val) / (max_val - min_val) * 255.0

    # Clip to ensure no values exceed the range
    normalized_image = np.clip(normalized_image, 0, 255)

    return normalized_image.astype(np.uint8)  # Convert back to uint8

def sobel_operator_numpy(array, ksize):
    """Applies Sobel operator to compute image gradients Ix and Iy using numpy routine."""
    
    # Sobel kernels
    sobel_x, sobel_y = create_sobel_operator(ksize)
    
    image_norm = normalize_image(array)

    # Apply convolution using the Sobel X and Y kernels
    Ix = apply_convolution(image_norm, sobel_x)
    Iy = apply_convolution(image_norm, sobel_y)

    return Ix, Iy

def sobel_operator_cv2(array, ksize):
    """Applies Sobel operator to compute image gradients Ix and Iy using cv2 routine."""
    
    image_norm = normalize_image(array)
    
    Ix = cv2.Sobel(image_norm, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(image_norm, cv2.CV_64F, 0, 1, ksize=ksize)
    
    return Ix, Iy