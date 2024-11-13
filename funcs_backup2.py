import numpy as np
from collections import deque
import cv2
import matplotlib.pyplot as plt
def get_bfs_kernel(sobel_array: np.ndarray, kernel_size: int, kernel_center: tuple, bfs_stride:np.uint8):
    neighbors = [(0, -1*bfs_stride), (-1*bfs_stride, 0), (0, 1*bfs_stride), (1*bfs_stride, 0)]
    processing_queue = deque([kernel_center])
    checked_array = np.zeros((sobel_array.shape[0], sobel_array.shape[1]), dtype='bool')
    
    # Define the offset range for the kernel around the center
    offset = kernel_size // 2
    lbp_hist_initial = np.zeros(256)
    distances = []  # Store distances from the initial kernel center

    # Compute the initial LBP histogram at the starting kernel_center
    lbp_hist_initial = compute_lbp_histogram(sobel_array, kernel_center, kernel_size)
    
    # Continue processing until all points in the queue are processed
    while processing_queue:
        # Pop the current point to process
        currently_checking_array = processing_queue.popleft()
        
        # Calculate the LBP histogram for the current kernel_center
        lbp_hist_current = compute_lbp_histogram(sobel_array, currently_checking_array, kernel_size)
        
        # Compute the Chi-square distance between the initial and current histograms
        chi_square_dist = chi_square_distance(lbp_hist_initial, lbp_hist_current)
        distances.append((currently_checking_array, chi_square_dist))
        # print(currently_checking_array, chi_square_dist)
        # Mark the center location as checked
        checked_array[currently_checking_array] = True
        
        # Process each neighbor for BFS traversal
        for dy, dx in neighbors:
            ny, nx = currently_checking_array[0] + dy, currently_checking_array[1] + dx
            if 0 <= ny < sobel_array.shape[0] and 0 <= nx < sobel_array.shape[1] and not checked_array[ny, nx]:
                # Add unprocessed neighbor to the queue
                processing_queue.append((ny, nx))
                checked_array[ny, nx] = True

    return distances

def compute_lbp_histogram(sobel_array: np.ndarray, kernel_center: tuple, kernel_size: int):
    offset = kernel_size // 2
    lbp_hist = np.zeros(256)
    # Iterate over the kernel area centered around the current point
    for dy in range(-offset, offset + 1):
        for dx in range(-offset, offset + 1):
            # Calculate the coordinates within the sobel_array
            ny = kernel_center[0] + dy
            nx = kernel_center[1] + dx
            # Ensure the coordinates are within bounds
            if offset <= ny < sobel_array.shape[0]-offset and offset <= nx < sobel_array.shape[1]-offset:
                # Calculate the LBP value for this position
                lbp = get_lbp_3x3(sobel_array, (ny, nx))
            else:
                lbp = 0
            add_to_histogram(lbp_hist, lbp)
    return lbp_hist

def chi_square_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    # Compute the Chi-square distance
    distance = 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-10))  # Add a small value to avoid division by zero
    return distance

def get_lbp_3x3(sobel_array:np.ndarray, kernel_center: tuple) -> np.uint8:
    '''
    counts 3x3 lbp

    starts from top left going clockwise
    '''
    neighboring_lbp_k_dist = ((-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0))
    center = sobel_array[kernel_center]
    lbp_val = np.uint8
    lbp_val = 0
    for idx , (dy, dx) in enumerate(neighboring_lbp_k_dist):
        to_bitshift = 1 if center < sobel_array[kernel_center[0]-dy,kernel_center[1]-dx] else 0
        lbp_val = lbp_val | (to_bitshift << idx)
    return lbp_val
def add_to_histogram(lbp_hist:np.ndarray, lbp_val: np.uint8):
    lbp_hist[lbp_val] += 1
def get_top_n_lbp_indexes(lbp_hist: np.ndarray, N: int)-> np.ndarray:
    # Get the indices of the top N values in lbp_hist
    top_n_indexes = np.argsort(lbp_hist)[-N:][::-1]
    # Retrieve the corresponding values for these top indexes (optional)
    top_n_values = lbp_hist[top_n_indexes]
    
    return np.array((top_n_indexes, top_n_values))

image = cv2.imread('road_1.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = image.shape
if rows > 1280 and cols > 720:
    image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
    rows, cols = image.shape
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Normalize sobel_combined to the range 0-255 (necessary for LBP)
sobel_combined_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))

start_index = (int(rows/2), int(cols/2))

get_bfs_kernel(sobel_combined_normalized, 79, start_index, 15)