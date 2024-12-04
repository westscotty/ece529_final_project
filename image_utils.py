import cv2
import numpy as np
import sys
from numba import jit
from scipy.signal import convolve2d as conv2d

## minmax normalization on image ##
normalize_image = lambda x : cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

##########################################################################################
### Soble operator creation for calculating gradients in x and y directions of image ###

def create_sobel_operator(ksize, xy):
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

    # breakpoint()
    if xy == 0:
        return sobel_x
    else:
        return sobel_y

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

def sobel_operator_numpy(img, ksize, xy=0, scipy=False):
    """Applies Sobel operator to compute image gradients Ix and Iy using numpy routine."""
    
    # Sobel kernels
    sobel = create_sobel_operator(ksize, xy)
    image_norm = normalize_image(img)

    if not scipy:
        #Apply convolution using the Sobel kernels
        gradient_img = apply_convolution(image_norm, sobel)
        # gradient_img = cv2.filter2D(src=image_norm, ddepth=-1, kernel=sobel)
    else:
        gradient_img = conv2d(image_norm, sobel, mode='same', boundary='fill', fillvalue=0)
        
    return gradient_img

def sobel_operator_cv2(img, ksize, xy = 0):
    """Applies Sobel operator to compute image gradients Ix and Iy using cv2 routine."""
    
    image_norm = normalize_image(img)
    x = 1
    y = 0
    if xy == 1:
        x = 0
        y = 1
    
    gradient_img = cv2.Sobel(image_norm, cv2.CV_32F, x, y, ksize=ksize)
    
    return gradient_img


##########################################################################################
### Low pass filter for image sharpening ###

def gaussian_low_pass_cv2(img, ksize, sigma):
    """Applies Gaussian blur to the provided matrices Ixx, Iyy, and Ixy using cv2 routine."""
    
    if sigma == 0:
        sigma = calculate_sigma(normalize_image(img), ksize)
    
    conv_img = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    return conv_img, sigma

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

def gaussian_low_pass_numpy(img, ksize, sigma, scipy=False):
    """Applies Gaussian blur to the provided matrices Ixx, Iyy, and Ixy using numpy routine."""
    
    if sigma == 0:
        sigma = calculate_sigma(normalize_image(img), ksize)
        
    kernel = gaussian_kernel(ksize, sigma)
    if not scipy:
        conv_img = apply_convolution(img, kernel)
    else:
        conv_img = conv2d(img, kernel, mode='same', boundary='fill', fillvalue=0)
    
    return conv_img, sigma


##########################################################################################
### Other image operations ###

def histogram_equalization(image):
    # Perform histogram equalization on a grayscale image.

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
    # Apply a simple averaging low-pass filter to an image.
    # Create an averaging kernel
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)

    # Apply the filter using cv2.filter2D
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image


def gaussian_high_pass_filter(image, kernel_size=5, sigma=1):
    # Apply a high-pass filter to an image.
    # Ensure the kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Step 1: Apply Gaussian low-pass filter to the image
    low_pass = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Step 2: Subtract the low-pass filtered image from the original image
    high_pass = cv2.subtract(image, low_pass)
    
    return high_pass