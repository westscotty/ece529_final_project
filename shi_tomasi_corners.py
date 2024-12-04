import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from copy import copy
import argparse
from utils import debug_messages, coordinate_density_filter, mae, psnr, ssim
from plot_utils import make_comparison_image, draw_corner_markers
from image_utils import gaussian_low_pass_cv2, gaussian_low_pass_numpy, sobel_operator_cv2, sobel_operator_numpy, gaussian_high_pass_filter, averaging_low_pass_filter, histogram_equalization
from tqdm import tqdm
from video_utils import reshape_points

gradient = { 'cv2': sobel_operator_cv2,
                    'numpy': sobel_operator_numpy
                  }
gaussian_low_pass   = { 'cv2': gaussian_low_pass_cv2,
                    'numpy': gaussian_low_pass_numpy
                  }

def shi_tomasi_corners(img, max_corners=200, ksize=3, method='numpy', sensitivity=0.04, sigma0=0, min_dist=10, debug=False, show_image=False, plots_dir=None):
    
    # Calculate image dimensions
    height, width = img.shape
    
    # Ensure the input is already in float format for processing
    gray = np.float32(img)
        
    # Ix: Gradient in the x-direction (horizontal changes).
    # Iy: Gradient in the y-direction (vertical changes).
    # The gradients highlight areas of rapid intensity change, which are candidates for corners.
    # Done by alculating the derivative of the intensity function
    # Compute image gradients (Ix, Iy) using Sobel filters
    Ix = gradient[method](gray.copy(), ksize, 0)
    Iy = gradient[method](gray.copy(), ksize, 1)
    
    if debug:
        test_Ix1 = gradient['cv2'](gray.copy(), 3, 0)
        test_Iy1 = gradient['cv2'](gray.copy(), 3, 1)
        test_Ix2 = gradient['numpy'](gray.copy(), 3, 0)
        test_Iy2 = gradient['numpy'](gray.copy(), 3, 1)
        mae_Ix, pnsr_Ix = mae(test_Ix1, test_Ix2), psnr(test_Ix1, test_Ix2)
        mae_Iy, pnsr_Iy = mae(test_Iy1, test_Iy2), psnr(test_Iy1, test_Iy2)
        debug_messages(f"""
            Sobel Operators:
            MAE Ix: {mae_Ix:.5f}
            MAE Iy: {mae_Iy:.5f}
            PNSR Ix: {pnsr_Ix:.5f}
            PNSR Iy: {pnsr_Iy:.5f}
            """)
        
        out_file = None
        if plots_dir:
            out_file = f"{plots_dir}/gradients_sample.png"
        make_comparison_image([test_Ix2, test_Iy2], ['Sobel Ix', 'Sobel Iy'], "Sobel Kernels", out_file)
        
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
    Ixx, sigma_xx = gaussian_low_pass[method](Ixx0, ksize, sigma0)
    Iyy, sigma_yy = gaussian_low_pass[method](Iyy0, ksize, sigma0)
    Ixy, sigma_xy = gaussian_low_pass[method](Ixy0, ksize, sigma0)
    if debug:
        debug_messages(f"""
            Sigmas: 
            Iyy: {sigma_xx:.5f}
            Iy: y{sigma_yy:.5f}
            Ixy: {sigma_xy:.5f}
                    """)
 
        test_Ixx1, sigma_xx1 = gaussian_low_pass['cv2'](Ixx0, 3, 0)
        test_Iyy1, sigma_yy1 = gaussian_low_pass['cv2'](Iyy0, 3, 0)
        test_Ixy1, sigma_xy1 = gaussian_low_pass['cv2'](Ixy0, 3, 0)
        test_Ixx2, sigma_xx2 = gaussian_low_pass['numpy'](Ixx0, 3, 0)
        test_Iyy2, sigma_yy2 = gaussian_low_pass['numpy'](Iyy0, 3, 0)
        test_Ixy2, sigma_xy2 = gaussian_low_pass['numpy'](Ixy0, 3, 0)
        mae_Ixx, pnsr_Ixx = mae(test_Ixx1, test_Ixx2), psnr(test_Ixx1, test_Ixx2)
        mae_Iyy, pnsr_Iyy = mae(test_Iyy1, test_Iyy2), psnr(test_Iyy1, test_Iyy2)
        mae_Ixy, pnsr_Ixy = mae(test_Ixy1, test_Ixy2), psnr(test_Ixy1, test_Ixy2)
        debug_messages(f"""
            Guassian Blur Kernels:
            MAE Ixx: {mae_Ixx:.5f}
            MAE Iyy: {mae_Iyy:.5f}
            MAE Ixy: {mae_Ixy:.5f}
            PSNR Ixx: {pnsr_Ixx:.5f}
            PSNR Iyy: {pnsr_Iyy:.5f}
            PSNR Ixy: {pnsr_Ixy:.5f}
            Sigmas XX: {sigma_xx1}, {sigma_xx2}
            Sigmas YY: {sigma_yy1}, {sigma_yy2}
            Sigmas XY: {sigma_xy1}, {sigma_xy2}
            """)

        out_file = None
        if plots_dir:
            out_file = f"{plots_dir}/gaussian_low_pass_filters.png"
        make_comparison_image([test_Ixx2, test_Iyy2, test_Ixy2], ['Gaussian Ixx', 'Gaussian Iyy', 'Gaussian Ixy'], "Gaussian Low Pass Filters", out_file)

        
    # Compute the minimum eigenvalue (Shi-Tomasi score) for each pixel
    # det_M: This computes the determinant of the covariance matrix 𝑀
    # M = ([Ixx, Ixy], [Ixy, Iyy])
    # The determinant helps to identify the strength of the corners.
    # trace_M: This computes the trace of the covariance matrix 𝑀 which is the sum of its diagonal elements.
    # response: The final response is calculated using a combination of the determinant and trace.
    # The subtraction of <sensitivity>×trace_M^2 helps adjust the sensitivity of corner detection, ensuring that the corners detected are of high quality.
    det_M = Ixx * Iyy - (Ixy ** 2)
    trace_M = Ixx + Iyy
    response = det_M - (sensitivity * (trace_M ** 2))

    # Flatten the response matrix and get top N corners
    # The indices of the highest values are sorted to determine the locations of the strongest corners
    flat_response = response.flatten()
    top_indices = np.argsort(flat_response)[-max_corners:]

    # Convert flat indices back to 2D coordinates
    coords = np.array(np.unravel_index(top_indices, response.shape)).T
    filtered_coords = coordinate_density_filter(coords, min_dist, max_corners, height, width)

    # Use CV2 implementation for comparison  
    coords_cv2 = cv2.goodFeaturesToTrack(image=img, maxCorners=max_corners, qualityLevel=sensitivity, minDistance=min_dist)
    coords_cv2 = np.array([(int(corner[0][1]), int(corner[0][0])) for corner in coords_cv2])
    # filtered_coords_cv2 = coordinate_density_filter(coords_cv2, min_dist, max_corners, height, width)
    filtered_coords_cv2 = coords_cv2
    
    return reshape_points(filtered_coords), reshape_points(filtered_coords_cv2)

# Example usage
if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='Detect corners in an image using the Shi-Tomasi method.')

    # Add arguments
    parser.add_argument('-img_in', '--input_image', type=str, required=True, help='Path to the input image file')
    parser.add_argument('-img_out', '--output_image', action='store_true', required=False, help='Save output files')
    parser.add_argument('-mc', '--max_corners', type=int, default=200, required=False, help='Maximum number of corners to detect (default: 25)')
    parser.add_argument('-ks', '--kernel_size', type=int, choices=[3, 5, 7], default=3, required=False, help='Size of the Sobel kernel (3, 5, or 7, default: 3)')
    parser.add_argument('-m', '--method', type=str, choices=['cv2', 'numpy'], default='numpy', required=False, help='Sobel operator to use (default: numpy)')
    parser.add_argument('-gs', '--gaussian_sigma', type=int, default=0, required=False, help='Sigma value for gaussian blur kernel')
    parser.add_argument('-s', '--shi_tomasi_sensitivity', type=float, default=0.04, required=False, help='Sensitivity for corner detection (default: 0.04)')
    parser.add_argument('-md', '--minimum_distance', type=int, default=10, required=False, help='Minumum distance between detected corners (used for removing oversample corners)')
    parser.add_argument('-debug', '--debug', action='store_true', required=False, default=False, help='Enable debugging print statements')
    parser.add_argument('-debug_images', '--debug_images', action='store_true', required=False, default=False, help='Enable debugging image showing')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the input file exists
    if not os.path.isfile(args.input_image):
        sys.exit(f"File not found: {args.input_image}")
    
    # Ensure the output directory exists
    if args.output_image:
        output_image = copy(args.input_image)
        output_image = output_image.split('/')[1:]
        output_image = os.path.join('test_results', *output_image)
        if not os.path.isdir(os.path.dirname(output_image)):
            os.makedirs(os.path.dirname(output_image))

    # Load image as grayscale
    img = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
    img = cv2.pyrDown(img)
    # img = histogram_equalization(img)
    height, width = img.shape
    # img = cv2.pyrDown(img)


    # Call the Shi-Tomasi corner detection function
    corners, corners_cv2 = shi_tomasi_corners(
        img=img,
        ksize=args.kernel_size,
        max_corners=args.max_corners,
        method=args.method,
        sensitivity=args.shi_tomasi_sensitivity,
        sigma0=args.gaussian_sigma,
        min_dist=args.minimum_distance,
        debug=args.debug,
        show_image=args.debug_images
    )

    # Copy Image then draw corners on the BGR converted
    result_img = img.copy()
    result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization
    result_img_bgr = draw_corner_markers(result_img_bgr, np.squeeze(corners), (0, 255, 0))
    
    # Copy Image then draw corners on the BGR converted
    result_img2 = img.copy()
    result_img_bgr2 = cv2.cvtColor(result_img2, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization
    result_img_bgr2 = draw_corner_markers(result_img_bgr2, np.squeeze(corners_cv2), (255, 0, 0))

    # Print corners if the flag is set
    if args.debug:
        debug_messages(f"Detected corners: \n{list(corners)}")
        debug_messages(f"Detected corners2: \n{list(corners_cv2)}")
       
    # Add bounding boxes in various methods
    # corner_groups =  group_nearby_corners(corners, 50)
    # bboxes = make_bounding_boxes_from_groups(corner_groups, result_img_bgr)
    # result_img_bgr = add_bounding_boxes_to_image(result_img_bgr, bboxes, color=(0, 255, 0))
    
    # corner_groups_nn =  group_corners_nearest_neighbors(corners, 60)
    # bboxes_nn = make_bounding_boxes_from_groups(corner_groups_nn, result_img_bgr)
    # result_img_bgr = add_bounding_boxes_to_image(result_img_bgr, bboxes_nn, color=(0, 0, 255))
    
    # corner_groups =  cluster_corners(corners, n_clusters=5)
    # result_img_bgr = draw_clusters_on_image(result_img_bgr, corner_groups)

    # Display and save the image with corners
    make_comparison_image([result_img_bgr, result_img_bgr2], ['Shi-tomasi Corners from Scratch', 'Shi-tomasi Corners from OpenCV'], "Shi-Tomasi Corners")
    
    if args.output_image:
        plt.savefig(output_image)
        