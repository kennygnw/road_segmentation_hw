import cv2
import numpy as np
import matplotlib.pyplot as plt
import funcs
# Example code to run the BFS kernel and store the distance matrix
image = cv2.imread('road_1.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = image.shape
if rows > 1280 and cols > 720:
    image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
    rows, cols = image.shape
sobel_kern = 19
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kern)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kern)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))
start_index = (int(rows / 2), int(cols / 2))

# Run BFS and get the distances matrix
kernel_size = 23
stride_size = 5
distances =funcs.get_bfs_kernel(sobel_combined_normalized, kernel_size, start_index, stride_size)

# Save the distances array as a .npy file
np.save(f'dist_kern{kernel_size}_strd{stride_size}_sbl{sobel_kern}.npy', distances)


# limited_distances = np.clip(distances, -1, 500)
limited_distances = distances

# Convert -1 values to 0 temporarily for dilation
dilated_distances = np.where(limited_distances == -1, 0, limited_distances)

# Define a kernel for dilation (e.g., 3x3 square kernel)
kernel = np.ones((3, 3), np.uint8)

# Apply dilation
dilated_distances = cv2.dilate(dilated_distances, kernel, iterations=1)

# Convert back 0 values to np.nan for plotting
dilated_distances = np.where(dilated_distances == 0, np.nan, dilated_distances)
# Display the distances matrix as an image (optional)
plt.imshow(dilated_distances, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("Chi-Square Distance Map")
plt.show()
