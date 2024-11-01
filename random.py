import cv2
import numpy as np

def shi_tomasi_corners(img, max_corners=50, ksize=3):
    # Ensure the input is already in float format for processing
    gray = np.float32(img)

    # Compute image gradients (Ix, Iy) using Sobel filters
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # Compute elements of the covariance matrix
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Apply Gaussian filter to the matrix elements
    Ixx = cv2.GaussianBlur(Ixx, (ksize, ksize), 0)
    Iyy = cv2.GaussianBlur(Iyy, (ksize, ksize), 0)
    Ixy = cv2.GaussianBlur(Ixy, (ksize, ksize), 0)

    # Compute the minimum eigenvalue (Shi-Tomasi score) for each pixel
    det_M = Ixx * Iyy - Ixy ** 2
    trace_M = Ixx + Iyy
    response = det_M - 0.04 * trace_M ** 2

    # Flatten the response matrix and get top N corners
    flat_response = response.flatten()
    top_indices = np.argsort(flat_response)[-max_corners:]

    # Convert flat indices back to 2D coordinates
    coords = np.array(np.unravel_index(top_indices, response.shape)).T

    # Draw corners on the original grayscale image
    result_img = img.copy()
    for (y, x) in coords:
        cv2.circle(result_img, (x, y), 3, 255, -1)  # Draw in white

    return result_img, coords

# Example usage
img = cv2.imread("data/extracted_frames/frame_150.png", cv2.IMREAD_GRAYSCALE)  # Load as grayscale
result, corners = shi_tomasi_corners(img)

print("Detected corners:", corners)

# Display the image with corners
cv2.imshow("Shi-Tomasi Corners", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
