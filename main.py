'''
Copyright 2025 吳永保

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import funcs
import datetime
# Example code to run the BFS kernel and store the distance matrix
filename = 'road_1'
image_color = cv2.imread(f'{filename}.jpg')
# rows, cols, dim = image_color.shape
# if rows > 1280 and cols > 720:
image_color = cv2.resize(image_color, (1280, 720), interpolation=cv2.INTER_AREA)
rows, cols, dim = image_color.shape
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
sobel_kern = 19
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kern)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kern)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))
# start on the 80% of row and 66% of length
start_index = (int(rows / 2*1), int(cols / 2*1))

# Run BFS and get the distances matrix
kernel_size = 23
stride_size = 5
distances =funcs.get_bfs_kernel(sobel_combined_normalized, kernel_size, start_index, stride_size)

# # Save the distances array as a .npy file
# np.save(f'dist_kern{kernel_size}_strd{stride_size}_sbl{sobel_kern}.npy', distances)
current_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
np.save(f'{filename}_{current_time}.npy', distances)

# limited_distances = np.clip(distances, -1, 500)
limited_distances = distances

# Convert -1 values to 0 temporarily for dilation
dilated_distances = np.where(limited_distances == -1, 0, limited_distances)

# Define a kernel for dilation (e.g., 3x3 square kernel)
kernel = np.ones((5, 5), np.uint8)

# Apply dilation
dilated_distances = cv2.dilate(dilated_distances, kernel, iterations=1)

# Convert back 0 values to np.nan for plotting
dilated_distances = np.where(dilated_distances == 0, np.nan, dilated_distances)

# Rescale the array to 0-255
min_val = np.nanmin(dilated_distances)
max_val = np.nanmax(dilated_distances)

# Avoid division by zero if max_val equals min_val
rescaled_array = ((dilated_distances - min_val) / (max_val - min_val) * 255).astype(np.uint8)
(center_y, center_x) = (int(dilated_distances.shape[0]*0.80), int(dilated_distances.shape[1]*0.66))
average_threshold = np.average(rescaled_array[center_y-10:center_y+10,center_x-10:center_x+10])
max_threshold = np.max(rescaled_array[center_y-10:center_y+10,center_x-10:center_x+10])

mask = np.zeros((dilated_distances.shape[0],dilated_distances.shape[1]),dtype=np.uint8)
mask[(rescaled_array <= int(max_threshold))] = 255
second_kernel = np.ones((15, 15), np.uint8)
eroded_mask = cv2.erode(mask, second_kernel,iterations=1)
dilated_mask = cv2.dilate(eroded_mask, second_kernel, iterations=1)

# Create a red overlay image (same size as the original)
red_overlay = np.zeros_like(image_color)
red_overlay[:, :, 2] = 255  # Set the red channel to maximum

# Apply the binary mask to the red overlay
# red_overlay_masked = cv2.bitwise_and(red_overlay, red_overlay, mask=mask)
red_overlay_masked = cv2.bitwise_and(red_overlay, red_overlay, mask=dilated_mask)

# Blend the original image with the red overlay
alpha = 0.5  # Transparency factor (0.0: only original, 1.0: only overlay)
blended_image = cv2.addWeighted(image_color, 1 - alpha, red_overlay_masked, alpha, 0)

cv2.imshow('Translucent Red Mask', blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the distances matrix as an image (optional)
# plt.imshow(limited_distances, cmap='viridis', interpolation='nearest')
plt.imshow(dilated_distances, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("Chi-Square Distance Map")
plt.show()
