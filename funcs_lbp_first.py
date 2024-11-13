import numpy as np
from collections import deque
import cv2
import matplotlib.pyplot as plt

def get_lbp_image(sobel_array: np.ndarray) -> np.ndarray:
    """
    Calculate the LBP image using get_lbp_3x3 for every pixel with a full 3x3 neighborhood.
    """
    lbp_array = np.zeros(sobel_array.shape, dtype=np.uint8)
    
    # Calculate LBP for each pixel, excluding the border
    for y in range(1, sobel_array.shape[0] - 1):
        for x in range(1, sobel_array.shape[1] - 1):
            lbp_array[y, x] = get_lbp_3x3(sobel_array, (y, x))
            
    return lbp_array

def get_bfs_kernel(lbp_array: np.ndarray, kernel_size: int, kernel_center: tuple, bfs_stride: np.uint8):
    """
    Perform BFS to compute histograms of LBP values in specified kernels using a precomputed LBP array.
    """
    neighbors = [(0, -1 * bfs_stride), (-1 * bfs_stride, 0), (0, 1 * bfs_stride), (1 * bfs_stride, 0)]
    processing_queue = deque([kernel_center])
    checked_array = np.zeros((lbp_array.shape[0], lbp_array.shape[1]), dtype='bool')
    
    offset = kernel_size // 2
    lbp_hist_initial = compute_lbp_histogram(lbp_array, kernel_center, kernel_size)
    
    distances = np.full(lbp_array.shape, -1.0)

    while processing_queue:
        currently_checking_array = processing_queue.popleft()
        
        # Use the precomputed LBP array to obtain the histogram for the current kernel center
        lbp_hist_current = compute_lbp_histogram(lbp_array, currently_checking_array, kernel_size)
        top_modes = get_top_n_modes(lbp_hist_current, 5)

        # Calculate Chi-square distance between the initial and current histograms
        chi_square_dist = chi_square_distance(lbp_hist_initial, lbp_hist_current)
        
        # Store the calculated distance in the distances array
        distances[currently_checking_array[0], currently_checking_array[1]] = chi_square_dist
        checked_array[currently_checking_array] = True

        # Process each neighbor for BFS traversal
        for dy, dx in neighbors:
            ny, nx = currently_checking_array[0] + dy, currently_checking_array[1] + dx
            if 0 <= ny < lbp_array.shape[0] and 0 <= nx < lbp_array.shape[1] and not checked_array[ny, nx]:
                processing_queue.append((ny, nx))
                checked_array[ny, nx] = True

    return distances

def compute_lbp_histogram(lbp_array: np.ndarray, kernel_center: tuple, kernel_size: int):
    """
    Compute an LBP histogram for a specific kernel centered at kernel_center in the precomputed LBP array.
    """
    offset = kernel_size // 2
    lbp_hist = np.zeros(256)
    
    for dy in range(-offset, offset + 1):
        for dx in range(-offset, offset + 1):
            ny = kernel_center[0] + dy
            nx = kernel_center[1] + dx
            if 0 <= ny < lbp_array.shape[0] and 0 <= nx < lbp_array.shape[1]:
                lbp = lbp_array[ny, nx]  # Use the LBP value from the precomputed array
                add_to_histogram(lbp_hist, lbp)
    
    return lbp_hist

def get_lbp_3x3(sobel_array: np.ndarray, kernel_center: tuple) -> np.uint8:
    """
    Calculate the LBP value for a 3x3 neighborhood centered at kernel_center.
    """
    neighboring_lbp_k_dist = ((-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0))
    center = sobel_array[kernel_center]
    lbp_val = 0
    for idx, (dy, dx) in enumerate(neighboring_lbp_k_dist):
        to_bitshift = 1 if center < sobel_array[kernel_center[0] - dy, kernel_center[1] - dx] else 0
        lbp_val |= (to_bitshift << idx)
    return lbp_val

def add_to_histogram(lbp_hist: np.ndarray, lbp_val: np.uint8):
    lbp_hist[lbp_val] += 1

def chi_square_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    distance = 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-10))
    return distance

def get_top_n_modes(lbp_hist: np.ndarray, N: int) -> np.ndarray:
    """
    Get the top N most frequent values (modes) from the lbp_hist, ignoring intensity.
    
    Parameters:
    lbp_hist (np.ndarray): The histogram array with frequency counts for each LBP value.
    N (int): The number of top modes to return.
    
    Returns:
    np.ndarray: An array of the top N mode bin indices.
    """
    # Get the indices of the top N highest frequencies in the histogram
    top_n_indices = np.argsort(lbp_hist)[-N:][::-1]  # Sort in descending order of frequency
    
    return top_n_indices
# Example usage
image = cv2.imread('road_1.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = image.shape
if rows > 1280 and cols > 720:
    image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
    rows, cols = image.shape

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))

# Compute the LBP array for the entire image
lbp_array = get_lbp_image(sobel_combined_normalized)

# Run BFS with the precomputed LBP array
# start_index = (int(rows / 2), int(cols / 2))
start_index = (rows - 1, int(cols / 2))

distances = get_bfs_kernel(lbp_array, 39, start_index, 5)

# limited_distances = np.clip(distances, -1, 500)
limited_distances = distances

# Convert -1 values to 0 temporarily for dilation
dilated_distances = np.where(limited_distances == -1, 0, limited_distances)

# Define a kernel for dilation (e.g., 3x3 square kernel)
kernel = np.ones((5, 5), np.uint8)

# Apply dilation
dilated_distances = cv2.dilate(dilated_distances, kernel, iterations=1)
# dilated_distances = dilated_distances

# # Convert back 0 values to np.nan for plotting
dilated_distances = np.where(dilated_distances == 0, np.nan, dilated_distances)

# Display the distances matrix as an image (optional)
plt.imshow(dilated_distances, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.title("Chi-Square Distance Map")
plt.tight_layout()
plt.show()
