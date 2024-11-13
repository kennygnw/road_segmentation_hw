import numpy as np
from collections import deque
import cv2
import matplotlib.pyplot as plt

def get_top_n_modes(lbp_hist: np.ndarray, N: int) -> np.ndarray:
    """
    Get the top N most frequent values (modes) from the lbp_hist, ignoring intensity.
    
    Parameters:
    lbp_hist (np.ndarray): The histogram array with frequency counts for each LBP value.
    N (int): The number of top modes to return.
    
    Returns:
    np.ndarray: An array of the top N mode bin indices.
    """
    top_n_indices = np.argsort(lbp_hist)[-N:][::-1]
    return top_n_indices

def euclidean_distance_top_modes_weighted(hist1: np.ndarray, hist2: np.ndarray, N: int) -> float:
    """
    Calculate the weighted Euclidean distance based on the top N modes of two histograms.
    
    Parameters:
    hist1 (np.ndarray): The first histogram.
    hist2 (np.ndarray): The second histogram.
    N (int): The number of top modes to consider.
    
    Returns:
    float: The weighted Euclidean distance between the top N modes of hist1 and hist2.
    """
    # Get the top N modes for each histogram
    top_modes_1 = get_top_n_modes(hist1, N)
    
    # Calculate the total frequency of the top N modes in hist1 to use for weighting
    total_frequency = sum(hist1[mode] for mode in top_modes_1)
    
    # Calculate Euclidean distance with weights based on hist1 frequencies
    distance = 0
    for mode in top_modes_1:
        # Get the counts from both histograms for this mode (bin)
        count1 = hist1[mode]
        count2 = hist2[mode]
        
        # Calculate the weight for this mode based on its frequency in hist1
        weight = count1 / total_frequency if total_frequency != 0 else 0  # Avoid division by zero
        
        # Accumulate the weighted squared differences
        distance += weight * (count1 - count2) ** 2
    
    # Return the square root of the weighted sum of squared differences
    return np.sqrt(distance)

def get_bfs_kernel(lbp_array: np.ndarray, kernel_size: int, kernel_center: tuple, bfs_stride: np.uint8, top_n_modes: int):
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
        
        # Calculate Euclidean distance between the top N modes of the initial and current histograms
        euclidean_dist = euclidean_distance_top_modes_weighted(lbp_hist_initial, lbp_hist_current, top_n_modes)
        
        # Store the calculated distance in the distances array
        distances[currently_checking_array[0], currently_checking_array[1]] = euclidean_dist
        checked_array[currently_checking_array] = True

        # Process each neighbor for BFS traversal
        for dy, dx in neighbors:
            ny, nx = currently_checking_array[0] + dy, currently_checking_array[1] + dx
            if 0 <= ny < lbp_array.shape[0] and 0 <= nx < lbp_array.shape[1] and not checked_array[ny, nx]:
                processing_queue.append((ny, nx))
                checked_array[ny, nx] = True

    return distances

def compute_lbp_histogram(lbp_array: np.ndarray, lbp_kernel_center: tuple, lbp_kernel_size: int):
    offset = lbp_kernel_size // 2
    lbp_hist = np.zeros(256)
    
    for dy in range(-offset, offset + 1):
        for dx in range(-offset, offset + 1):
            ny = lbp_kernel_center[0] + dy
            nx = lbp_kernel_center[1] + dx
            if 0 <= ny < lbp_array.shape[0] and 0 <= nx < lbp_array.shape[1]:
                lbp = lbp_array[ny, nx]
                add_to_histogram(lbp_hist, lbp)
    
    return lbp_hist

def get_lbp_3x3(sobel_array: np.ndarray, kernel_center: tuple) -> np.uint8:
    neighboring_lbp_k_dist = ((-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0))
    center = sobel_array[kernel_center]
    lbp_val = 0
    for idx, (dy, dx) in enumerate(neighboring_lbp_k_dist):
        to_bitshift = 1 if center < sobel_array[kernel_center[0] - dy, kernel_center[1] - dx] else 0
        lbp_val |= (to_bitshift << idx)
    return lbp_val

def add_to_histogram(lbp_hist: np.ndarray, lbp_val: np.uint8):
    lbp_hist[lbp_val] += 1

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

# Set the start point at the bottom center of the image
start_index = (rows - 1, int(cols / 2))

# Set the number of top modes to use
top_n_modes = 5
kernel_size = 39
stride_size = 5
# Run BFS with the precomputed LBP array and Euclidean distance
distances = get_bfs_kernel(lbp_array, kernel_size, start_index, stride_size, top_n_modes)

limited_distances = np.clip(distances, -1, 400)
# limited_distances = distances

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
plt.title(f"Euclidean Distance Map with {kernel_size} KernelSize and {stride_size} Stride")
plt.tight_layout()
plt.show()
