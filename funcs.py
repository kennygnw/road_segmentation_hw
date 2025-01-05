import numpy as np
from collections import deque
import cv2
import matplotlib.pyplot as plt

def get_dfs_kernel(sobel_array: np.ndarray, kernel_size: int, kernel_center: tuple, dfs_stride: np.uint8):
    # Initialize distances array with the same shape as sobel_array, filled with -1 to indicate unprocessed pixels
    distances = np.full(sobel_array.shape, -1.0)
    previous_lbp_hist = np.zeros(256)
    for sobel_row in range(kernel_size, sobel_array.shape[0], dfs_stride):
        for sobel_col in range(kernel_size,sobel_array.shape[1],dfs_stride):
            lbp_hist = compute_lbp_histogram(sobel_array, (sobel_row, sobel_col), kernel_size)
            # Compute the Chi-square distance using cv2.compareHist between the last and current histograms
            chi_square_dist = cv2.compareHist(
                previous_lbp_hist.astype('float32'),
                lbp_hist.astype('float32'),
                cv2.HISTCMP_CHISQR
            )
            # Store the calculated distance directly in the distances array at the current location
            distances[sobel_row, sobel_col] = chi_square_dist

            print(sobel_row, sobel_col, chi_square_dist)
            previous_lbp_hist = lbp_hist
        # Compute the initial LBP histogram at the starting kernel_center
        # lbp_hist = compute_lbp_histogram(sobel_array, kernel_center, kernel_size)
    return distances

    # neighbors = [(0, -1 * dfs_stride), (-1 * dfs_stride, 0), (0, 1 * dfs_stride), (1 * dfs_stride, 0)]
    # working_kernel_center = kernel_center
    # processing_stack = [working_kernel_center]  # Stack for DFS
    # checked_array = np.zeros((sobel_array.shape[0], sobel_array.shape[1]), dtype='bool')

    # # Define the offset range for the kernel around the center
    # offset = kernel_size // 2


    # # Compute the initial LBP histogram at the starting kernel_center
    # last_lbp_hist = compute_lbp_histogram(sobel_array, kernel_center, kernel_size)

    # # Continue processing until all points in the stack are processed
    # while processing_stack:
    #     # Pop the current point from the stack
    #     currently_checking_array = processing_stack.pop()

    #     # Calculate the LBP histogram for the current kernel_center
    #     current_lbp_hist = compute_lbp_histogram(sobel_array, currently_checking_array, kernel_size)

    #     # Compute the Chi-square distance using cv2.compareHist between the last and current histograms
    #     chi_square_dist = cv2.compareHist(
    #         last_lbp_hist.astype('float32'),
    #         current_lbp_hist.astype('float32'),
    #         cv2.HISTCMP_CHISQR
    #     )

    #     # Store the calculated distance directly in the distances array at the current location
    #     distances[currently_checking_array[0], currently_checking_array[1]] = chi_square_dist

    #     # Update the last histogram to the current histogram
    #     last_lbp_hist = current_lbp_hist

    #     # Mark the current location as checked
    #     checked_array[currently_checking_array] = True

    #     # Process each neighbor for DFS traversal
    #     for dy, dx in neighbors:
    #         ny, nx = currently_checking_array[0] + dy, currently_checking_array[1] + dx
    #         if 0 <= ny < sobel_array.shape[0] and 0 <= nx < sobel_array.shape[1] and not checked_array[ny, nx]:
    #             # Add unprocessed neighbor to the stack
    #             processing_stack.append((ny, nx))
    #             checked_array[ny, nx] = True

    # return distances

def get_bfs_kernel(sobel_array: np.ndarray, kernel_size: int, kernel_center: tuple, bfs_stride: np.uint8):
    neighbors = [(0, -1 * bfs_stride), (-1 * bfs_stride, 0), (0, 1 * bfs_stride), (1 * bfs_stride, 0)]
    processing_queue = deque([kernel_center])

    # previous_hist_queue = deque()
    
    checked_array = np.zeros((sobel_array.shape[0], sobel_array.shape[1]), dtype='bool')
    
    # Define the offset range for the kernel around the center
    offset = kernel_size // 2
    lbp_hist_initial = np.zeros(256)
    
    # Initialize distances array with the same shape as sobel_array, filled with -1 to indicate unprocessed pixels
    distances = np.full(sobel_array.shape,-1.0)

    # Compute the initial LBP histogram at the starting kernel_center
    lbp_hist_initial = compute_lbp_histogram(sobel_array, kernel_center, kernel_size)

    # previous_hist_queue.append(lbp_hist_initial)

    # # Compute the initial LBP histogram at the starting kernel_center
    # last_lbp_hist = compute_lbp_histogram(sobel_array, kernel_center, kernel_size)

    # Continue processing until all points in the queue are processed
    while processing_queue:
        # Pop the current point to process
        currently_checking_array = processing_queue.popleft()
        
        # Calculate the LBP histogram for the current kernel_center
        lbp_hist_current = compute_lbp_histogram(sobel_array, currently_checking_array, kernel_size)
        # previous_lbp_hist = previous_hist_queue.popleft()

        # Compute the Chi-square distance between the initial and current histograms
        # chi_square_dist = chi_square_distance(previous_lbp_hist, lbp_hist_current)
        chi_square_dist = chi_square_distance(lbp_hist_initial, lbp_hist_current)
        
        # # Compute the Chi-square distance using cv2.compareHist between the last and current histograms
        # chi_square_dist = cv2.compareHist(
        #     last_lbp_hist.astype('float32'), 
        #     lbp_hist_current.astype('float32'), 
        #     cv2.HISTCMP_CORREL
        # )

        # Store the calculated distance directly in the distances array at the current location
        distances[currently_checking_array[0], currently_checking_array[1]] = chi_square_dist
        
        # # Update the last histogram to the current histogram
        # last_lbp_hist = lbp_hist_current
        
        # Mark the center location as checked
        checked_array[currently_checking_array] = True
        
        # Process each neighbor for BFS traversal
        for dy, dx in neighbors:
            ny, nx = currently_checking_array[0] + dy, currently_checking_array[1] + dx
            if 0 <= ny < sobel_array.shape[0] and 0 <= nx < sobel_array.shape[1] and not checked_array[ny, nx]:
                # Add unprocessed neighbor to the queue
                processing_queue.append((ny, nx))
                # previous_hist_queue.append(lbp_hist_current)
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
    lbp_val = 0
    for idx, (dy, dx) in enumerate(neighboring_lbp_k_dist):
        to_bitshift = 1 if center < sobel_array[kernel_center[0] - dy, kernel_center[1] - dx] else 0
        lbp_val = lbp_val | (to_bitshift << idx)
    return lbp_val

def add_to_histogram(lbp_hist: np.ndarray, lbp_val: np.uint8):
    lbp_hist[lbp_val] += 1
    
# # Example code to run the BFS kernel and store the distance matrix
# image = cv2.imread('road_1.jpg', cv2.IMREAD_GRAYSCALE)
# rows, cols = image.shape
# if rows > 1280 and cols > 720:
#     image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
#     rows, cols = image.shape
# sobel_kern = 19
# sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kern)
# sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kern)
# sobel_combined = cv2.magnitude(sobel_x, sobel_y)
# sobel_combined_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))
# start_index = (int(rows / 2), int(cols / 2))

# # Run BFS and get the distances matrix
# kernel_size = 23
# stride_size = 5
# distances = get_bfs_kernel(sobel_combined_normalized, kernel_size, start_index, stride_size)

# # Save the distances array as a .npy file
# np.save(f'dist_kern{kernel_size}_strd{stride_size}_sbl{sobel_kern}.npy', distances)


# # limited_distances = np.clip(distances, -1, 500)
# limited_distances = distances

# # Convert -1 values to 0 temporarily for dilation
# dilated_distances = np.where(limited_distances == -1, 0, limited_distances)

# # Define a kernel for dilation (e.g., 3x3 square kernel)
# kernel = np.ones((3, 3), np.uint8)

# # Apply dilation
# dilated_distances = cv2.dilate(dilated_distances, kernel, iterations=1)

# # Convert back 0 values to np.nan for plotting
# dilated_distances = np.where(dilated_distances == 0, np.nan, dilated_distances)
# # Display the distances matrix as an image (optional)
# plt.imshow(dilated_distances, cmap='viridis', interpolation='nearest')
# plt.colorbar()
# plt.title("Chi-Square Distance Map")
# plt.show()
