"""Crecimiento de regiones SIN dependencias de interfaz (antes estaba mezclado
con PyQt5 dentro de RegionGrowingSegmentation.py, por lo que no se podía usar
ni medir fuera de la ventana).

Cambios respecto a original/RegionGrowingSegmentation.py:

1. Cola con collections.deque: `list.pop(0)` es O(n) en cada paso; con
   regiones grandes el algoritmo se volvía cuadrático.
2. Etiquetas en int32: el original multiplicaba una máscara uint8 por el
   número de semilla, así que con 256 semillas o más fallaba con
   "OverflowError: Python integer 256 out of bounds for uint8"
   (en NumPy < 2 el número se desbordaba en silencio y mezclaba regiones).
3. Criterio de pertenencia configurable:
     criterio="semilla"  (original) |f(p) − f(semilla)| ≤ umbral
     criterio="media"    (Gonzalez & Woods) |f(p) − media(región)| ≤ umbral,
                         con la media actualizada a medida que la región crece.
   Con "semilla" el resultado depende de qué píxel exacto se eligió: si la
   semilla cae sobre ruido, la región entera queda corrida.
4. Los píxeles rechazados no se marcan como visitados para siempre: con
   criterio="media" un píxel rechazado al principio puede aceptarse más tarde.
"""

from __future__ import annotations

from collections import deque

import numpy as np

VECINOS_8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def rgb_to_luv_vectorized(image):
    """Conversión RGB (uint8) -> CIELUV. Sin cambios respecto al original."""
    rgb = image.astype(float) / 255.0
    mask = rgb <= 0.04045
    rgb[mask] /= 12.92
    rgb[~mask] = ((rgb[~mask] + 0.055) / 1.055) ** 2.4

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    X = r * 0.412453 + g * 0.357580 + b * 0.180423
    Y = r * 0.212671 + g * 0.715160 + b * 0.072169
    Z = r * 0.019334 + g * 0.119193 + b * 0.950227

    Yn = 1.0
    xn, yn = 0.312713, 0.329016
    un = 4 * xn / (xn + 15 * yn + 3 * (1 - xn - yn))
    vn = 9 * yn / (xn + 15 * yn + 3 * (1 - xn - yn))

    denominator = np.maximum(X + 15 * Y + 3 * Z, 1e-10)
    u = 4 * X / denominator
    v = 9 * Y / denominator

    Y_ratio = Y / Yn
    mask = Y_ratio > 0.008856
    L = np.zeros_like(Y_ratio)
    L[mask] = 116 * (Y_ratio[mask] ** (1 / 3)) - 16
    L[~mask] = 903.3 * Y_ratio[~mask]

    U = 13 * L * (u - un)
    V = 13 * L * (v - vn)
    return np.stack([L, U, V], axis=2)


def region_growing(image, seeds, threshold=50, use_luv=True, criterio="semilla"):
    """Crecimiento de regiones desde semillas (x, y), 8-conectividad.

    Devuelve un mapa de etiquetas int32: 0 = sin región, k = región de la
    semilla k (1..N). Si dos semillas alcanzan el mismo píxel gana la de
    número mayor, igual que en el original (np.maximum).
    """
    img = np.asarray(image)
    if img.ndim == 3 and use_luv:
        datos = rgb_to_luv_vectorized(img)
    else:
        datos = img.astype(np.float64)
    color = datos.ndim == 3
    alto, ancho = datos.shape[:2]
    etiquetas = np.zeros((alto, ancho), dtype=np.int32)

    for idx, (sx, sy) in enumerate(seeds, start=1):
        if not (0 <= sx < ancho and 0 <= sy < alto):
            continue
        en_region = np.zeros((alto, ancho), dtype=bool)
        semilla = datos[sy, sx].astype(np.float64)
        suma = semilla.copy()
        n = 1
        en_region[sy, sx] = True
        cola = deque([(sx, sy)])

        while cola:
            x, y = cola.popleft()
            referencia = semilla if criterio == "semilla" else suma / n
            for dx, dy in VECINOS_8:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < ancho and 0 <= ny < alto) or en_region[ny, nx]:
                    continue
                valor = datos[ny, nx]
                if color:
                    diferencia = float(np.sqrt(np.sum((valor - referencia) ** 2)))
                else:
                    diferencia = abs(float(valor) - float(referencia))
                if diferencia <= threshold:
                    en_region[ny, nx] = True
                    suma = suma + valor
                    n += 1
                    cola.append((nx, ny))

        etiquetas = np.maximum(etiquetas, en_region.astype(np.int32) * idx)

    return etiquetas
