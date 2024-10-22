import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
# Load the image
image = cv2.imread('road_1.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the Sobel operator in the x and y direction
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=15)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=15)

# Compute the gradient magnitude
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Normalize sobel_combined to the range 0-255 (necessary for LBP)
sobel_combined_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))

# Apply LBP on the Sobel-filtered image
radius = 1  # Radius of the LBP
n_points = 8 * radius  # Number of points for the LBP
lbp = local_binary_pattern(sobel_combined_normalized, n_points, radius, method='uniform')

# Display the Sobel and LBP results using Matplotlib
plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
plt.title('Sobel Combined')
plt.imshow(sobel_combined_normalized, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('LBP of Sobel Combined')
plt.imshow(lbp, cmap='gray')

plt.show()
