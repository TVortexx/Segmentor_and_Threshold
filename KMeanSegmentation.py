########### Steps to choose the optimal number of clusters K:-(Elbow Method)

#  1. Compute K-Means clustering for different values of K by varying K from 1 to 10 clusters.

#  2. For each K, calculate the total within-cluster sum of square (WCSS).

 # 3. Plot the curve of WCSS vs the number of clusters K.

 # 4. The location of a bend (knee) in the plot is generally considered as an indicator of the appropriate number of clusters.

import numpy as np

# QImage se importa solo dentro de kmeans_result_to_qimage: así kmeans() se
# puede usar sin tener PyQt5 instalado (benchmark, pruebas, app web).

# select k initial centroids for rgb image segmentation
def create_random_centroids(pixels, k, rng=None):
    """Elige k píxeles al azar como centroides iniciales.
    Cambio: acepta un generador (rng) para que el resultado sea reproducible
    y elige píxeles DISTINTOS (el original podía repetir el mismo píxel y
    dejar un grupo vacío)."""
    rng = np.random.default_rng() if rng is None else rng
    m = pixels.shape[0]
    indices = rng.choice(m, size=min(k, m), replace=False)
    return pixels[indices].astype(np.float64)


def kmeans(image, k=3, max_iters=100, seed=None, verbose=False, return_labels=False):
    """K-means sobre el color de cada píxel.

    Cambio principal: la asignación de cada píxel al centroide más cercano
    se calcula para TODOS los píxeles a la vez (matriz de distancias m×k con
    NumPy) en lugar de un bucle `for i in range(m)` en Python. Mismo
    algoritmo y mismo resultado, del orden de 100 veces más rápido.

    seed:          semilla para que la segmentación sea reproducible.
    verbose:       el original imprimía cada iteración; ahora es opcional.
    return_labels: devuelve también el índice de grupo de cada píxel (h, w),
                   necesario para construir una máscara objeto/fondo.
    """
    pixels = image.reshape(-1, 3) if len(image.shape) == 3 else image.reshape(-1, 1)
    pixels = pixels.astype(np.float64)
    rng = np.random.default_rng(seed)
    centroids = create_random_centroids(pixels, k, rng)
    idx = np.zeros(pixels.shape[0], dtype=int)

    for it in range(max_iters):
        if verbose:
            print(f"Iteration {it + 1}/{max_iters}")
        # ||p − c||² = ||p||² − 2 p·c + ||c||²  (sin bucles por píxel)
        distancias = ((pixels ** 2).sum(1)[:, None] - 2 * pixels @ centroids.T
                      + (centroids ** 2).sum(1)[None, :])
        idx = np.argmin(distancias, axis=1)

        prev_centroids = centroids.copy()
        for i in range(len(centroids)):
            puntos = pixels[idx == i]
            if len(puntos) > 0:
                centroids[i] = puntos.mean(axis=0)

        if np.allclose(centroids, prev_centroids):
            if verbose:
                print(f"Converged at {it} iterations")
            break

    segmented_image = centroids[idx].reshape(image.shape)
    if return_labels:
        return segmented_image, idx.reshape(image.shape[:2])
    return segmented_image


def kmeans_result_to_qimage(segmented_image):
    """Convert the K-means segmentation result to a QImage"""
    from PyQt5.QtGui import QImage
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
