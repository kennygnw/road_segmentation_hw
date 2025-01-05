import matplotlib.pyplot as plt
import numpy as np
import cv2
image = cv2.imread('road_1.jpg')
rows, cols, dim = image.shape
if rows > 1280 and cols > 720:
    image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
    rows, cols, dim = image.shape

distance = np.load('66_80_percent_center_dist_kern23_strd5_sbl19.npy')

dilated_distances = np.where(distance == -1, 0, distance)

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
red_overlay = np.zeros_like(image)
red_overlay[:, :, 2] = 255  # Set the red channel to maximum

# Apply the binary mask to the red overlay
# red_overlay_masked = cv2.bitwise_and(red_overlay, red_overlay, mask=mask)
red_overlay_masked = cv2.bitwise_and(red_overlay, red_overlay, mask=dilated_mask)

# Blend the original image with the red overlay
alpha = 0.5  # Transparency factor (0.0: only original, 1.0: only overlay)
blended_image = cv2.addWeighted(image, 1 - alpha, red_overlay_masked, alpha, 0)

cv2.imshow('Translucent Red Mask', blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.figure()
# # plt.imshow(rescaled_array, cmap='viridis', interpolation='nearest')
# plt.imshow(dilated_mask,cmap='gray')
# # plt.colorbar()
# plt.title("Chi-Square Distance Map")
# plt.show()