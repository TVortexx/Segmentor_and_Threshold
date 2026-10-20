"""Tema 3 · Umbralización.

    global_iterativo   Umbral global simple (algoritmo iterativo de Gonzalez & Woods).
    otsu               Umbral óptimo de Otsu, implementado desde el histograma.
    adaptativo         Umbral local: media gaussiana del vecindario - C.
    tophat_otsu        Corrige la iluminación con un top-hat morfológico y luego Otsu.
                       (une "procesamiento morfológico" con "papel de la iluminación").

Todas trabajan sobre la luminancia y devuelven (máscara, info) donde info
documenta el umbral calculado para poder explicarlo en la defensa.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

from ..utils import a_gris, limpiar_mascara


def umbral_global_iterativo(g: np.ndarray, eps: float = 1e-4, max_iter: int = 100):
    """1. T0 = media global.  2. Separar G1 (>T) y G2 (<=T).
    3. T = (m1 + m2) / 2.     4. Repetir hasta que |ΔT| < eps.
    Devuelve (T, historial de T) para mostrar la convergencia."""
    t = float(g.mean())
    historial = [t]
    for _ in range(max_iter):
        g1, g2 = g[g > t], g[g <= t]
        if g1.size == 0 or g2.size == 0:
            break
        nuevo = 0.5 * (g1.mean() + g2.mean())
        historial.append(float(nuevo))
        if abs(nuevo - t) < eps:
            t = nuevo
            break
        t = nuevo
    return float(t), historial


def umbral_otsu(g: np.ndarray, niveles: int = 256):
    """Otsu: elige k que maximiza la varianza entre clases
        σB²(k) = [mG·P1(k) − m(k)]² / [P1(k)·(1 − P1(k))]
    calculada para todos los k a la vez con sumas acumuladas del histograma.
    Devuelve (T, separabilidad η = σB²/σG², curva σB²)."""
    hist, bordes = np.histogram(g, bins=niveles, range=(0.0, 1.0))
    p = hist.astype(float) / hist.sum()
    centros = 0.5 * (bordes[:-1] + bordes[1:])
    p1 = np.cumsum(p)                    # probabilidad acumulada de la clase 1
    m = np.cumsum(p * centros)           # media acumulada
    mg = m[-1]                           # media global
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mg * p1 - m) ** 2 / (p1 * (1.0 - p1))
    sigma_b = np.nan_to_num(sigma_b, nan=0.0, posinf=0.0)
    k_max = np.flatnonzero(sigma_b == sigma_b.max())
    k = int(round(k_max.mean()))         # si hay empate se promedia (como en el libro)
    sigma_g = float(np.sum(p * (centros - mg) ** 2))
    eta = float(sigma_b[k] / sigma_g) if sigma_g > 0 else 0.0
    # El umbral es el borde superior del bin k: g > T pertenece a la clase 2.
    return float(bordes[k + 1]), eta, sigma_b


def global_iterativo(img, suavizado: float = 1.0, area_min: int = 30):
    g = a_gris(img)
    if suavizado > 0:
        g = ndi.gaussian_filter(g, suavizado)
    t, hist = umbral_global_iterativo(g)
    return limpiar_mascara(g > t, area_min), {"umbral": t, "iteraciones": len(hist) - 1,
                                               "historial": hist}


def otsu(img, suavizado: float = 1.0, area_min: int = 30):
    g = a_gris(img)
    if suavizado > 0:
        g = ndi.gaussian_filter(g, suavizado)
    t, eta, curva = umbral_otsu(g)
    return limpiar_mascara(g > t, area_min), {"umbral": t, "separabilidad": eta, "curva": curva}


def adaptativo(img, sigma_ventana: float = 15.0, c: float = 0.02, suavizado: float = 1.0,
               area_min: int = 30):
    """Umbral variable T(x,y) = media gaussiana local(x,y) + c.
    sigma_ventana debe ser mayor que los objetos (si no, el interior de un
    objeto grande y uniforme se confunde con 'fondo local')."""
    g = a_gris(img)
    if suavizado > 0:
        g = ndi.gaussian_filter(g, suavizado)
    local = ndi.gaussian_filter(g, sigma_ventana)
    return limpiar_mascara(g > local + c, area_min), {"umbral": "variable", "sigma_ventana": sigma_ventana,
                                                       "c": c, "mapa_umbral": local + c}


def tophat_otsu(img, radio: int = 45, suavizado: float = 1.0, area_min: int = 30, escala: int = 4,
                modo: str = "division"):
    """Top-hat blanco = imagen − apertura(imagen). La apertura con un disco más
    grande que los objetos estima el FONDO (iluminación); al quitarla queda una
    imagen con fondo ~plano y se aplica Otsu.

    modo="resta"    top-hat clásico: sirve si la iluminación es ADITIVA.
    modo="division" corrección de campo plano g/fondo: sirve si la iluminación
                    es MULTIPLICATIVA (f = reflectancia × iluminación), que es el
                    modelo físico real y el del escenario E2.
    Para que sea rápido la apertura se calcula en una versión reducida
    `escala` veces y se reinterpola (el fondo es suave, no se pierde nada)."""
    g = a_gris(img)
    if suavizado > 0:
        g = ndi.gaussian_filter(g, suavizado)
    if escala > 1:
        chica = g[::escala, ::escala]
        r = max(1, radio // escala)
        fondo_chico = ndi.grey_opening(chica, footprint=morphology.disk(r))
        fondo = ndi.zoom(fondo_chico, (g.shape[0] / chica.shape[0], g.shape[1] / chica.shape[1]), order=1)
        fondo = fondo[: g.shape[0], : g.shape[1]]
        fondo = ndi.gaussian_filter(fondo, escala)
    else:
        fondo = ndi.grey_opening(g, footprint=morphology.disk(radio))
    if modo == "division":
        corregida = np.clip(g / np.maximum(fondo, 1e-3) - 1.0, 0, None)
    else:
        corregida = np.clip(g - fondo, 0, None)
    corregida = corregida / max(corregida.max(), 1e-12)
    t, eta, curva = umbral_otsu(corregida)
    return limpiar_mascara(corregida > t, area_min), {"umbral": t, "separabilidad": eta,
                                                       "fondo_estimado": fondo, "corregida": corregida,
                                                       "modo": modo}
