# Computer Vision: Thresholding and Segmentation

## Authors

- Ibrahim Fateen  
- Bassant Zaki  
- Mohamed Hisham  
- Yasmine Mahmoud  

## Overview

This task explores several classical and modern image segmentation techniques. The main objective is to separate regions of interest in grayscale and color images using thresholding and clustering-based methods.

Covered techniques include:
1. Otsu Thresholding (Global & Local)
2. Optimal Thresholding (Global & Local)
3. Spectral Thresholding (Binary & Multi-level)
4. K-Means Segmentation
5. Region Growing Segmentation
6. Mean-Shift Segmentation
7. Agglomerative Segmentation

## 1. Otsu Thresholding

Otsu's method finds a threshold value that minimizes intra-class variance and maximizes inter-class variance.

- Applies to histogram-based analysis.
- Efficient for bi-modal distributions.
- Can be extended to local thresholding using windowed regions.

## 2. Optimal Thresholding

An iterative inter-means method that updates the threshold based on mean values of pixel groups until convergence.

- Works on pixel values, not histograms.
- Applied globally or locally using small windows.
- Tolerant to intensity distributions not suited to Otsu.

## 3. Spectral Thresholding

Spectral thresholding segments images based on intensity using between-class variance analysis.

- Binary Mode: Separates pixels into foreground/background.
- Multi-Level Mode: Uses two thresholds for three-class segmentation.
- Fallback to binary if three-class separation is ineffective.

Challenges include high computational cost and increased histogram sensitivity for multi-class segmentation.

## 4. K-Means Segmentation

K-Means clusters pixels based on color similarity.

- Input space: RGB or LUV (optional for better perceptual clustering)
- Each pixel is a feature vector
- Random centroids, distance-based assignment, and iterative update
- Output: clustered image by dominant colors

LUV space sometimes oversegments or missegments due to uniformity sensitivity.

## 5. Region Growing Segmentation

An interactive method that starts from user-defined seed points and expands regions based on intensity similarity.

- Supports multiple seeds
- Uses 8-connected neighborhood
- Threshold-based inclusion and queue for processing
- GUI allows seed selection and visual feedback

Challenges include finding optimal similarity threshold and visualizing binary masks.

## 6. Mean-Shift Segmentation

A density-based clustering algorithm that shifts feature vectors toward the mode.

- No need to preset number of clusters
- Feature space: pixel values (+ optional spatial info)
- Bandwidth controls convergence and granularity
- Low bandwidth → many fine clusters, High bandwidth → fewer coarse clusters

Performance is highly dependent on bandwidth selection.

## 7. Agglomerative Segmentation

A bottom-up hierarchical clustering method where each pixel starts as a cluster.

- Distance-based merging of clusters using a priority queue (heap)
- Two modes: Vanilla (pixel-based) and Superpixel-initialized (faster but less accurate)
- Builds a binary tree structure representing cluster hierarchy

Vanilla mode yields better results but is computationally expensive, requiring image downscaling for feasibility.

## Dependencies

- Python 3.8+
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib
- PyQt (for region growing GUI)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Conclusion

This task presents a comprehensive comparison of segmentation methods. From statistical thresholding to clustering and region-based approaches, each method offers distinct strengths depending on image content and computational constraints. Combined, they form a powerful toolkit for various segmentation applications.
