import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
# Load the image
image = cv2.imread('road_1.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)

# Apply the Sobel operator in the x and y direction
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=25)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=25)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))

# Apply LBP on the Sobel-filtered image
radius = 1  # Radius of the LBP
n_points = 8 * radius  # Number of points for the LBP
lbp = local_binary_pattern(sobel_combined_normalized, n_points, radius, method='uniform')

# Display the Sobel and LBP results using Matplotlib

from funcs import get_lbp_3x3
def get_lbp_image(sobel_array: np.ndarray) -> np.ndarray:
    """
    Calculate the LBP image using get_lbp_3x3 for every pixel with a full 3x3 neighborhood.
    """
    # Initialize the LBP array with the same shape as the input image, filled with zeros
    lbp_array = np.zeros(sobel_array.shape, dtype=np.uint8)
    
    # Loop over each pixel in the image, excluding the border pixels
    for y in range(1, sobel_array.shape[0] - 1):
        for x in range(1, sobel_array.shape[1] - 1):
            # Calculate the LBP value for the current pixel using the get_lbp_3x3 function
            lbp_value = get_lbp_3x3(sobel_array, (y, x))
            # Store the calculated LBP value in the lbp_array
            lbp_array[y, x] = lbp_value
            
    return lbp_array

lbp_image = get_lbp_image(sobel_combined_normalized)

plt.figure(figsize=(16, 9))

plt.title('Sobel Combined')
plt.imshow(sobel_combined_normalized, cmap='gray')
plt.tight_layout()

plt.figure(figsize=(16, 9))
plt.subplot(1, 2, 1)
plt.title('LBP of Sobel Combined')
plt.imshow(lbp, cmap='gray')

plt.subplot(1,2,2)
plt.title('Self')
plt.imshow(lbp_image, cmap='gray')
plt.tight_layout()
plt.show()
input()
