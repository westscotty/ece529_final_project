import numpy as np
import cv2
from numba import jit
from sobel_operator import apply_convolution

normalize_image = lambda x : cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

def gaussian_blur_cv2(Ixx, Iyy, Ixy, ksize, sigma):
    """Applies Gaussian blur to the provided matrices Ixx, Iyy, and Ixy using cv2 routine."""
    
    if sigma == 0:
        sigma = calculate_sigma(normalize_image(Ixx), ksize)
    
    Ixx = cv2.GaussianBlur(Ixx, (ksize, ksize), sigma)
    Iyy = cv2.GaussianBlur(Iyy, (ksize, ksize), sigma)
    Ixy = cv2.GaussianBlur(Ixy, (ksize, ksize), sigma)

    return Ixx, Iyy, Ixy, sigma

def calculate_sigma(image, ksize):
    """
    Calculate an adaptive sigma value based on image statistics.
    A common approach is to use the average of the pixel intensities.
    """
    
    # Convert to float and normalize
    image_float = np.float32(image) / 255.0
    mean_intensity = np.mean(image_float)

    # Define a range for sigma based on the kernel size and mean intensity
    # The higher the mean intensity, the smaller the sigma (less blur)
    # The smaller the mean intensity, the larger the sigma (more blur)
    sigma = (ksize / 2) * (1 - mean_intensity)

    # Ensure sigma is within a reasonable range
    sigma = max(0.1, sigma)  # Avoid zero sigma
    return sigma

def gaussian_kernel(size, sigma):
    """Generates a Gaussian kernel."""
            
    ax = np.linspace(-(size // 2), size // 2, size)
    kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()  # Normalize the kernel
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def gaussian_blur_numpy(Ixx, Iyy, Ixy, ksize, sigma):
    """Applies Gaussian blur to the provided matrices Ixx, Iyy, and Ixy using numpy routine."""
    
    if sigma == 0:
        sigma = calculate_sigma(normalize_image(Ixx), ksize)
        
    kernel = gaussian_kernel(ksize, sigma)
    Ixx = apply_convolution(Ixx, kernel)
    Iyy = apply_convolution(Iyy, kernel)
    Ixy = apply_convolution(Ixy, kernel)
    
    return Ixx, Iyy, Ixy, sigma