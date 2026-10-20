"""Tema 3e · Umbrales basados en varias variables.

En vez de un escalar por píxel (intensidad) cada píxel es un vector, por
ejemplo (L*, a*, b*). Se implementan dos variantes vistas en Gonzalez & Woods:

    kmeans_color       Agrupa los vectores de color en K grupos (k-means) y
                       declara FONDO al grupo que domina el borde de la imagen.
    distancia_color    'Umbral' sobre la distancia de cada píxel a un color de
                       referencia (Euclídea o de Mahalanobis). Sirve cuando se
                       conoce el color del fondo o del objeto.

Se trabaja en CIELAB porque allí la distancia Euclídea es aproximadamente
proporcional a la diferencia de color percibida, y la luminancia (L*) queda
separada de la cromaticidad (a*, b*).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import color

from ..utils import a_rgb, limpiar_mascara


def _caracteristicas(img, espacio: str, suavizado: float, peso_luminancia: float) -> np.ndarray:
    rgb = a_rgb(img)
    if suavizado > 0:
        rgb = np.dstack([ndi.gaussian_filter(rgb[..., k], suavizado) for k in range(3)])
    if espacio == "lab":
        x = color.rgb2lab(rgb)
        x[..., 0] *= peso_luminancia     # <1 da menos importancia al brillo (robustez a iluminación)
    elif espacio == "hsv":
        hsv = color.rgb2hsv(rgb)
        # El tono es un ángulo: se representa con (cos, sin) para que 0° y 360° estén cerca.
        ang = 2 * np.pi * hsv[..., 0]
        x = np.dstack([hsv[..., 1] * np.cos(ang), hsv[..., 1] * np.sin(ang), hsv[..., 2] * peso_luminancia])
    else:
        x = rgb
    return x.reshape(-1, x.shape[-1]).astype(np.float32)


def _grupo_fondo(etiquetas: np.ndarray, margen: int = 3) -> int:
    """El fondo es el grupo más frecuente en un marco de `margen` píxeles."""
    borde = np.concatenate([etiquetas[:margen].ravel(), etiquetas[-margen:].ravel(),
                            etiquetas[:, :margen].ravel(), etiquetas[:, -margen:].ravel()])
    return int(np.bincount(borde).argmax())


def kmeans_color(img, k: int = 4, espacio: str = "lab", suavizado: float = 1.5,
                 peso_luminancia: float = 0.5, area_min: int = 30, semilla: int = 0):
    h, w = a_rgb(img).shape[:2]
    x = _caracteristicas(img, espacio, suavizado, peso_luminancia)
    cv2.setRNGSeed(semilla)  # resultados reproducibles
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    compacidad, etiquetas, centros = cv2.kmeans(x, k, None, criterio, 3, cv2.KMEANS_PP_CENTERS)
    etiquetas = etiquetas.reshape(h, w)
    fondo = _grupo_fondo(etiquetas)
    mascara = limpiar_mascara(etiquetas != fondo, area_min)
    return mascara, {"k": k, "espacio": espacio, "grupo_fondo": fondo, "etiquetas": etiquetas,
                     "centros": centros, "compacidad": float(compacidad)}


def distancia_color(img, referencia=None, umbral: float | None = None, mahalanobis: bool = True,
                    suavizado: float = 1.5, area_min: int = 30):
    """Si no se da `referencia`, se estima el color del FONDO con el marco de la
    imagen; el objeto es todo lo que esté 'lejos' del fondo. Con Mahalanobis la
    distancia se mide en unidades de la dispersión propia del fondo (elipsoide
    en lugar de esfera), lo que la hace insensible a la escala de cada canal.
    Si `umbral` es None se usa 3 (≈ 3 desviaciones estándar)."""
    rgb = a_rgb(img)
    h, w = rgb.shape[:2]
    x = _caracteristicas(rgb, "lab", suavizado, 1.0).astype(np.float64)
    if referencia is None:
        marco = np.zeros((h, w), bool)
        marco[:4], marco[-4:], marco[:, :4], marco[:, -4:] = True, True, True, True
        muestra = x[marco.ravel()]
    else:
        muestra = np.atleast_2d(np.asarray(referencia, float))
    media = muestra.mean(axis=0)
    dif = x - media
    if mahalanobis and muestra.shape[0] > 3:
        cov = np.cov(muestra, rowvar=False) + 1e-6 * np.eye(3)
        d = np.sqrt(np.einsum("ij,jk,ik->i", dif, np.linalg.inv(cov), dif))
        t = 3.0 if umbral is None else umbral
    else:
        d = np.linalg.norm(dif, axis=1)
        t = 10.0 if umbral is None else umbral   # ΔE ≈ 10: diferencia de color claramente visible
    d = d.reshape(h, w)
    return limpiar_mascara(d > t, area_min), {"umbral": t, "distancia": d, "media_fondo": media}
