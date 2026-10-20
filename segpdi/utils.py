"""Utilidades comunes: conversión de tipos, limpieza morfológica y superposición."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import color, morphology


def a_flotante(img: np.ndarray) -> np.ndarray:
    """Devuelve la imagen como float64 en [0, 1]."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float64) / 65535.0
    img = img.astype(np.float64)
    if img.max() > 1.0 + 1e-9:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def a_gris(img: np.ndarray) -> np.ndarray:
    """Convierte RGB(A) o gris a gris float en [0, 1] (luminancia ITU-R 709)."""
    img = a_flotante(img)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        return color.rgb2gray(img)
    return img


def a_rgb(img: np.ndarray) -> np.ndarray:
    """Convierte gris a RGB float en [0, 1]."""
    img = a_flotante(img)
    if img.ndim == 2:
        return np.dstack([img] * 3)
    return img[..., :3]


def a_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(a_flotante(img), 0, 1) * 255).round().astype(np.uint8)


def limpiar_mascara(mascara: np.ndarray, area_min: int = 30, radio: int = 1) -> np.ndarray:
    """Post-procesamiento morfológico estándar aplicado a todas las técnicas
    por igual (para que la comparación sea justa): apertura, relleno de
    huecos y eliminación de componentes pequeñas."""
    m = mascara.astype(bool)
    if radio > 0:
        m = ndi.binary_opening(m, structure=morphology.disk(radio))
    m = ndi.binary_fill_holes(m)
    if area_min > 0:
        m = quitar_objetos_pequenos(m, area_min)
    return m


def quitar_objetos_pequenos(mascara: np.ndarray, area_min: int) -> np.ndarray:
    """Elimina componentes conexas con área < area_min (independiente de la
    versión de scikit-image, cuya API cambió en 0.26)."""
    etiquetas, n = ndi.label(mascara)
    if n == 0:
        return mascara.astype(bool)
    areas = np.bincount(etiquetas.ravel())
    conservar = areas >= area_min
    conservar[0] = False
    return conservar[etiquetas]


def superponer(img: np.ndarray, mascara: np.ndarray, color_rgb=(1.0, 0.2, 0.2), alfa=0.45):
    """Superpone la máscara coloreada sobre la imagen para visualización."""
    base = a_rgb(img).copy()
    m = mascara.astype(bool)
    capa = np.array(color_rgb, dtype=np.float64)
    base[m] = (1 - alfa) * base[m] + alfa * capa
    return base


def comparar_con_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Imagen RGB de errores: verde = acierto (VP), rojo = falso positivo,
    azul = falso negativo, negro = verdadero negativo."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    out = np.zeros(pred.shape + (3,), dtype=np.float64)
    out[pred & gt] = (0.15, 0.8, 0.3)
    out[pred & ~gt] = (0.9, 0.2, 0.2)
    out[~pred & gt] = (0.2, 0.4, 0.95)
    return out


def normalizar(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)
