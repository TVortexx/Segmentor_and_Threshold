########### Steps to choose the optimal number of clusters K:-(Elbow Method)

#  1. Compute K-Means clustering for different values of K by varying K from 1 to 10 clusters.

#  2. For each K, calculate the total within-cluster sum of square (WCSS).

 # 3. Plot the curve of WCSS vs the number of clusters K.

 # 4. The location of a bend (knee) in the plot is generally considered as an indicator of the appropriate number of clusters.

import numpy as np
from PyQt5.QtGui import QImage

# select k initial centroids for rgb image segmentation
def create_random_centroids(pixels, k):
    # m is number of pixels (rows), and n would be 3 (r,g,b)
    m, n = pixels.shape
    # Initialize k centroids random from the pixels with 3 channels
    centroids = np.zeros((k, n))
    for i in range(k):
        # Randomly select a pixel from the image
        rand_index = np.random.randint(0, m)
        # Assign the pixel value to the centroid
        centroids[i] = pixels[rand_index]
    return centroids


def kmeans(image, k=3, max_iters=100):
    if len(image.shape) == 3:
        # Reshape the image to a 2D array of pixels
        pixels = image.reshape(-1, 3)
    else:
        # If the image is grayscale, we need to reshape it to 2D
        pixels = image.reshape(-1, 1)

    # m is number of pixels (rows), and n would be 3 (r,g,b)
    m, n = pixels.shape

    # Initialize centroids
    centroids = create_random_centroids(pixels, k)

    # Initialize variables
    prev_centroids = np.zeros(centroids.shape)
    idx = np.zeros(m, dtype=int)  # integer indexing

    for _ in range(max_iters):
        print(f"Iteration {_ + 1}/{max_iters}")
        # Assign clusters
        for i in range(m):
            distances = np.linalg.norm(pixels[i] - centroids, axis=1)
            # Find the index of the closest centroid
            idx[i] = np.argmin(distances)

        # Update centroids
        prev_centroids = centroids.copy()
        for i in range(k):
            # Get all points assigned to the i-th centroid
            points_in_cluster = pixels[idx == i]
            # If there are points in the cluster, update the centroid
            if len(points_in_cluster) > 0:
                centroids[i] = np.mean(points_in_cluster, axis=0)

        # Check for convergence
        # because they might be close enough but not exactly equal
        if np.allclose(centroids, prev_centroids):
            print(f"Converged at {_} iterations")
            break

    # Build segmented image
    # Assign each pixel to the nearest centroid
    segmented_image = centroids[idx.astype(int)]
    # Reshape back to original image shape
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image


def kmeans_result_to_qimage(segmented_image):
    """Convert the K-means segmentation result to a QImage"""
    # Make sure the data is in the correct format for QImage
    if segmented_image.dtype != np.uint8:
        # Scale values to 0-255 range
        if segmented_image.max() > 1.0:
            # Already in a higher range
            segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)
        else:
            # Scale from [0,1] to [0,255]
            segmented_image = (segmented_image * 255).astype(np.uint8)

    # Ensure memory is contiguous
    segmented_image = np.ascontiguousarray(segmented_image)

    # Convert to QImage
    height, width = segmented_image.shape[:2]

    if len(segmented_image.shape) == 3:
        # RGB image
        bytes_per_line = 3 * width
        return QImage(segmented_image.data, width, height,
                      bytes_per_line, QImage.Format_RGB888).copy()
    else:
        # Grayscale image
        return QImage(segmented_image.data, width, height,
                      width, QImage.Format_Grayscale8).copy()
