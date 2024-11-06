import cv2
import numpy as np

def histogram_equalization(image):
    """
    Perform histogram equalization on a grayscale image.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Histogram equalized image.
    """
    # Check if the image is grayscale
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")

    # Step 1: Calculate the histogram
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Step 2: Compute the cumulative distribution function (CDF)
    cdf = histogram.cumsum()

    # Step 3: Normalize the CDF to the range [0, 255]
    cdf_normalized = cdf * 255 / cdf[-1]  # Scale to 0-255

    # Step 4: Map the pixel values in the original image to equalized values
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)

    # Step 5: Reshape to original image shape
    equalized_image = equalized_image.reshape(image.shape).astype(np.uint8)

    return equalized_image



def averaging_low_pass_filter(image, kernel_size=5):
    """
    Apply a simple averaging low-pass filter to an image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the averaging kernel. Must be odd.

    Returns:
        numpy.ndarray: Filtered image.
    """
    # Create an averaging kernel
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)

    # Apply the filter using cv2.filter2D
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

def gaussian_low_pass_filter(image, kernel_size=5, sigma=1.0):
    """
    Apply a Gaussian low-pass filter to an image.

    Parameters:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the Gaussian kernel. Must be odd.
        sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
        numpy.ndarray: Filtered image.
    """
    # Create a Gaussian kernel
    kernel = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return kernel

def high_pass_filter(image, kernel_size=5, sigma=1):
    """
    Apply a high-pass filter to an image.
    
    Parameters:
        image (numpy.ndarray): The input image (BGR format).
        kernel_size (int): Size of the Gaussian kernel. Must be odd.
        sigma (float): Standard deviation for the Gaussian kernel.
        
    Returns:
        numpy.ndarray: The high-pass filtered image.
    """
    # Ensure the kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Step 1: Apply Gaussian low-pass filter to the image
    low_pass = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Step 2: Subtract the low-pass filtered image from the original image
    high_pass = cv2.subtract(image, low_pass)
    
    return high_pass