import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from sobel_operator import sobel_operator_cv2, sobel_operator_numpy
from gaussian_blur import gaussian_blur_cv2, gaussian_blur_numpy
from error_checking import error_metrics
from tqdm import tqdm

sobel_operators = { 'cv2': sobel_operator_cv2,
                    'numpy': sobel_operator_numpy
                  }
gaussian_blur   = { 'cv2': gaussian_blur_cv2,
                    'numpy': gaussian_blur_numpy
                  }
def debug_messages(message):
    print(f"\n<< DEBUG >>\n")
    print(message)
    print("\n<< DEBUG >>\n")

def shi_tomasi_corners(img, max_corners=25, ksize=3, method='cv2', sensitivity=0.04, sigma0=0, debug=False):
    
    # Ensure the input is already in float format for processing
    gray = np.float32(img)
        
    # Ix: Gradient in the x-direction (horizontal changes).
    # Iy: Gradient in the y-direction (vertical changes).
    # The gradients highlight areas of rapid intensity change, which are candidates for corners.
    # Done by alculating the derivative of the intensity function
    # Compute image gradients (Ix, Iy) using Sobel filters
    Ix, Iy = sobel_operators[method](gray, ksize)
    
    if debug:
        test_Ix1, test_Iy1 = sobel_operators['cv2'](gray, 3)
        test_Ix2, test_Iy2 = sobel_operators['numpy'](gray, 3)
        mae_Ix, pnsr_Ix = error_metrics(test_Ix1, test_Ix2)
        mae_Iy, pnsr_Iy = error_metrics(test_Iy1, test_Iy2)
        debug_messages(f"""\
            MAE Ix: {mae_Ix:.5f}
            MAE Iy: {mae_Iy:.5f}
            PNSR Ix: {pnsr_Ix:.5f}
            PNSR Iy: {pnsr_Iy:.5f}
            """)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Sobel Ix Delta')
        plt.imshow(test_Ix1-test_Ix2, cmap='gray')
        
        plt.subplot(1, 2, 2)
        plt.title('Sobel Iy Delta')
        plt.imshow(test_Iy1-test_Iy2, cmap='gray')
        plt.show()
        
    # Compute elements of the covariance matrix
    # Ixx: Represents the gradient squared in the x-direction, emphasizing how the intensity changes horizontally.
    # Iyy: Represents the gradient squared in the y-direction, emphasizing how the intensity changes vertically.
    # Ixy: Represents the product of gradients in both directions, capturing how they interact, which is crucial for corner detection.
    Ixx0 = Ix * Ix
    Iyy0 = Iy * Iy
    Ixy0 = Ix * Iy

    # Apply Gaussian filter to the matrix elements
    # The Gaussian blur is applied to smooth the covariance matrix elements. 
    # This step reduces noise and minor variations in the gradient images, allowing for more reliable corner detection. 
    # The kernel size for the Gaussian blur is determined by ksize, which is the same as that used for the Sobel operator. 
    # The smoothing ensures that the corner response is more robust by averaging local intensity variations.
    Ixx, Iyy, Ixy, sigma = gaussian_blur[method](Ixx0, Iyy0, Ixy0, ksize, sigma0)
    if sigma != sigma0:
        print(f"Calculated new sigma value: {sigma}")
    if debug:
        test_Ixx1, test_Iyy1, test_Ixy1, sigma1 = gaussian_blur['cv2'](Ixx0, Iyy0, Ixy0, 3, 0)
        test_Ixx2, test_Iyy2, test_Ixy2, sigma2 = gaussian_blur['numpy'](Ixx0, Iyy0, Ixy0, 3, 0)
        mae_Ixx, pnsr_Ixx = error_metrics(test_Ixx1, test_Ixx2)
        mae_Iyy, pnsr_Iyy = error_metrics(test_Iyy1, test_Iyy2)
        mae_Ixy, pnsr_Ixy = error_metrics(test_Ixy1, test_Ixy2)
        debug_messages(f"""
            MAE Ixx: {mae_Ixx:.5f}
            MAE Iyy: {mae_Iyy:.5f}
            MAE Ixy: {mae_Ixy:.5f}
            PSNR Ixx: {pnsr_Ixx:.5f}
            PSNR Iyy: {pnsr_Iyy:.5f}
            PSNR Ixy: {pnsr_Ixy:.5f}
            Sigmas: {sigma1}, {sigma2}
            """)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title('Sobel Ix Delta')
        plt.imshow(test_Ixx1-test_Ixx2, cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title('Sobel Iy Delta')
        plt.imshow(test_Iyy1-test_Iyy2, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title('Sobel Iy Delta')
        plt.imshow(test_Ixy1-test_Ixy2, cmap='gray')
        plt.show()
        
    # Compute the minimum eigenvalue (Shi-Tomasi score) for each pixel
    # det_M: This computes the determinant of the covariance matrix 𝑀
    # M = ([Ixx, Ixy], [Ixy, Iyy])
    # The determinant helps to identify the strength of the corners.
    # trace_M: This computes the trace of the covariance matrix 𝑀 which is the sum of its diagonal elements.
    # response: The final response is calculated using a combination of the determinant and trace.
    # The subtraction of 0.04×trace_M^2 helps adjust the sensitivity of corner detection, ensuring that the corners detected are of high quality.
    det_M = Ixx * Iyy - (Ixy ** 2)
    trace_M = Ixx + Iyy
    response = det_M - sensitivity * (trace_M ** 2)

    # Flatten the response matrix and get top N corners
    # The indices of the highest values are sorted to determine the locations of the strongest corners
    flat_response = response.flatten()
    top_indices = np.argsort(flat_response)[-max_corners:]

    # Convert flat indices back to 2D coordinates
    coords = np.array(np.unravel_index(top_indices, response.shape)).T

    # Copy Image then draw corners on the BGR converted
    result_img = img.copy()
    result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization

    for (y, x) in tqdm(coords):
        cv2.circle(result_img_bgr, (x, y), 10, (75, 180, 0), -1)  # Draw red circles

    return result_img_bgr, coords  # Return the colored image and coordinates


# Example usage
if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='Detect corners in an image using the Shi-Tomasi method.')

    # Add arguments
    parser.add_argument('-img_in', '--input_image', type=str, required=True, help='Path to the input image file')
    parser.add_argument('-img_out', '--output_image', type=str, required=False, default=None, help='Path to the output image file')
    parser.add_argument('-mc', '--max_corners', type=int, default=25, required=False, help='Maximum number of corners to detect (default: 25)')
    parser.add_argument('-ks', '--kernel_size', type=int, choices=[3, 5, 7], default=3, required=False, help='Size of the Sobel kernel (3, 5, or 7, default: 3)')
    parser.add_argument('-m', '--method', type=str, choices=['cv2', 'numpy'], default='cv2', required=False, help='Sobel operator to use (default: cv2)')
    parser.add_argument('-gs', '--gaussian_sigma', type=int, default=0, required=False, help='Sigma value for gaussian blur kernel')
    parser.add_argument('-s', '--shi_tomasi_sensitivty', type=float, default=0.04, required=False, help='Sensitivity for corner detection (default: 0.04)')
    parser.add_argument('-debug', '--debug', action='store_true', required=False, default=False, help='Enable debugging print statements')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the input file exists
    if not os.path.isfile(args.input_image):
        sys.exit(f"File not found: {args.input_image}")
    
    # Ensure the output directory exists
    if args.output_image:
        if not os.path.isdir(os.path.dirname(args.output_file)):
            os.makedirs(os.path.dirname(args.output_file))

    # Load image as grayscale
    img = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)

    # Call the Shi-Tomasi corner detection function
    result, corners = shi_tomasi_corners(
        img=img,
        ksize=args.kernel_size,
        max_corners=args.max_corners,
        method=args.method,
        sensitivity=args.shi_tomasi_sensitivty,
        sigma0=args.gaussian_sigma,
        debug=args.debug
    )

    # Print corners if the flag is set
    if args.debug:
        debug_messages(f"Detected corners: \n{corners}")

    # Display and save the image with corners
    plt.figure(figsize=(12, 8))
    plt.title(f"Shi-Tomasi Corners")
    plt.axis('off')
    if args.output_image:
        plt.imsave(args.output_file, result)
        
    if not args.debug:
        # Display image to screen
        plt.imshow(result)
        plt.show()
