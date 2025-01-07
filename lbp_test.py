import cv2
import numpy as np
from funcs import get_lbp_3x3, add_to_histogram
import matplotlib.pyplot as plt

def segment_lbp_image(lbp_image: np.ndarray, m: int = 4, n: int = 4):
    '''
    Segments the LBP image into m x n blocks.
    '''
    segment_height = lbp_image.shape[0] // m
    segment_width = lbp_image.shape[1] // n
    segments = []
    
    for i in range(m):
        for j in range(n):
            segment = lbp_image[
                i * segment_height : (i + 1) * segment_height,
                j * segment_width : (j + 1) * segment_width
            ]
            segments.append(segment)
    
    return segments
def compute_histogram_difference(hist1: np.ndarray, hist2: np.ndarray, metric='euclidean') -> float:
    '''
    Computes the difference between two histograms using a given metric.
    Supported metrics: 'euclidean', 'manhattan'.
    '''
    if metric == 'euclidean':
        return np.sqrt(np.sum((hist1 - hist2) ** 2))
    elif metric == 'manhattan':
        return np.sum(np.abs(hist1 - hist2))
    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'manhattan'.")


image = cv2.imread('road_1.jpg')
image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
rows, cols, dim = image.shape
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobel_kern = 19
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kern)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kern)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))

# Create an empty array for the LBP image
rows, cols = image.shape
lbp_image = np.zeros((rows - 2, cols - 2), dtype=np.uint8)  # Output will be smaller by 1 pixel on all sides

# Calculate the LBP for each pixel (excluding the borders)
for i in range(1, rows - 1):  # Avoid borders
    for j in range(1, cols - 1):
        lbp_image[i - 1, j - 1] = get_lbp_3x3(sobel_combined_normalized, (i, j))

m = 8
n = 8
segments = segment_lbp_image(lbp_image, m=m, n=n)

# Display and save the LBP image
cv2.imshow('LBP Image', lbp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

