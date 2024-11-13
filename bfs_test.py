import cv2
import numpy as np
from collections import deque

def bfs_segmentation(image, start, visited, label):
    rows, cols = image.shape
    queue = deque([start])
    visited[start] = True
    segmented_image[start] = label

    while queue:
        x, y = queue.popleft()

        # Check the four possible neighbors (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols:  # Check within bounds
                if not visited[nx, ny] and image[nx, ny] == image[start]:
                    visited[nx, ny] = True
                    segmented_image[nx, ny] = label
                    queue.append((nx, ny))

# Load and preprocess the image
image_path = 'road_1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Threshold the image to binary (if not already binary)
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Initialize variables
rows, cols = binary_image.shape
visited = np.zeros((rows, cols), dtype=bool)
segmented_image = np.zeros_like(binary_image)
label = 1  # Label for connected components

# Perform BFS on each pixel
for i in range(rows):
    for j in range(cols):
        if binary_image[i, j] == 255 and not visited[i, j]:  # Unvisited foreground pixel
            bfs_segmentation(binary_image, (i, j), visited, label)
            label += 1

# Display segmented result
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()