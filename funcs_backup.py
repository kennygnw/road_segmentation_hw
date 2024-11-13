import numpy as np
from collections import deque
import cv2
def get_bfs_kernel(sobel_array: np.ndarray, kernel_size: int, kernel_center: tuple):
    neighbors = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    processing_queue = deque([kernel_center])
    checked_array = np.zeros((sobel_array.shape[0], sobel_array.shape[1]), dtype='bool')
    
    # Initialize the LBP buffer
    lbp_buffer = np.zeros((kernel_size, kernel_size))
    # Define the offset range for the kernel around the center
    offset = kernel_size // 2
    
    lbp_hist = np.zeros(256)
    
    # Continue processing until all points in the queue are processed
    while processing_queue:
        # Pop the current point to process
        currently_checking_array = processing_queue.popleft()
        lbp_hist[:] = 0
        # Iterate over the kernel area centered around the current point
        for dy in range(-offset, offset + 1):
            for dx in range(-offset, offset + 1):
                # Calculate the coordinates within the sobel_array
                ny = currently_checking_array[0] + dy
                nx = currently_checking_array[1] + dx
                
                # Ensure the coordinates are within bounds
                if 0 <= ny < sobel_array.shape[0] and 0 <= nx < sobel_array.shape[1]:
                    # Calculate the LBP value for this position and assign it to lbp_buffer
                    lbp = get_lbp_3x3(sobel_array, (ny, nx))
                    lbp_buffer[dy + offset, dx + offset] = lbp
                    add_to_histogram(lbp_hist, lbp)
                    top_hist = get_top_n_lbp_indexes(lbp_hist,5)

        # Mark the center location as checked
        checked_array[currently_checking_array] = True
        
        # Process each neighbor for BFS traversal
        for dy, dx in neighbors:
            ny, nx = currently_checking_array[0] + dy, currently_checking_array[1] + dx
            if 0 <= ny < sobel_array.shape[0] and 0 <= nx < sobel_array.shape[1] and not checked_array[ny, nx]:
                # Add unprocessed neighbor to the queue
                processing_queue.append((ny, nx))
                checked_array[ny, nx] = True
    # Retrieve the top N values from the histogram
    top_hist = get_top_n_lbp_indexes(lbp_hist, 5)
    return lbp_buffer, top_hist

# def get_bfs_kernel(sobel_array:np.ndarray, kernel_size: np.uint8, kernel_center: tuple):
#     neighbors = [(0, -1), (-1, 0), (0, 1), (1, 0)]
#     processing_queue = deque()
#     checked_array = np.zeros((sobel_array.shape[0], sobel_array.shape[1]), dtype='bool')
#     processing_queue.append(kernel_center)
#     # Initialize the LBP buffer
#     lbp_buffer = np.zeros((kernel_size, kernel_size))
#     # Define the offset range for the kernel around the center
#     offset = kernel_size // 2
#     # Process the central point and fill the lbp_buffer
#     currently_checking_array = processing_queue.popleft()
#     # Iterate over the kernel area centered around the current point
#     lbp_hist = np.zeros(256)
#     for dy in range(-offset, offset + 1):
#         for dx in range(-offset, offset + 1):
#             # Calculate the coordinates within the sobel_array
#             ny = currently_checking_array[0] + dy
#             nx = currently_checking_array[1] + dx
#             # Ensure the coordinates are within bounds
#             if 0 <= ny < sobel_array.shape[0] and 0 <= nx < sobel_array.shape[1]:
#                 # Calculate the LBP value for this position and assign it to lbp_buffer
#                 lbp = get_lbp_3x3(sobel_array, (ny, nx))
#                 lbp_buffer[dy + offset, dx + offset] = lbp
#                 add_to_histogram(lbp_hist,lbp)
#     # Mark the center location as checked
#     checked_array[currently_checking_array] = True
#     top_hist = get_top_n_lbp_indexes(lbp_hist,5)
#     # Process each neighbor for BFS traversal
#     for dy, dx in neighbors:
#         ny, nx = currently_checking_array[0] + dy, currently_checking_array[1] + dx
#         if 0 <= ny < sobel_array.shape[0] and 0 <= nx < sobel_array.shape[1] and not checked_array[ny, nx]:
#             processing_queue.append((ny, nx))
#             checked_array[ny, nx] = True

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
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
# Normalize sobel_combined to the range 0-255 (necessary for LBP)
sobel_combined_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))
rows, cols = image.shape
start_index = (int(rows/2), int(cols/2))

get_bfs_kernel(sobel_combined_normalized, 19, start_index)