import matplotlib.pyplot as plt
import numpy as np
import cv2
distance = np.load('dist_kern23_strd5.npy')

dilated_distances = np.where(distance == -1, 0, distance)

# Define a kernel for dilation (e.g., 3x3 square kernel)
kernel = np.ones((3, 3), np.uint8)

# Apply dilation
dilated_distances = cv2.dilate(dilated_distances, kernel, iterations=1)

# Convert back 0 values to np.nan for plotting
dilated_distances = np.where(dilated_distances == 0, np.nan, dilated_distances)

plt.imshow(distance, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("Chi-Square Distance Map")
plt.show()