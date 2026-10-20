"""Tema 2 · Detección de discontinuidades: puntos, líneas, bordes y combinada.

Las máscaras de convolución siguen a Gonzalez & Woods (Digital Image Processing,
cap. 10). Además se incluyen dos segmentadores basados en bordes (Canny y
Marr-Hildreth) que convierten contornos en regiones cerradas para poder
compararlos con las demás técnicas usando las mismas métricas.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, morphology

from ..utils import a_gris, limpiar_mascara, normalizar

# ---------------------------------------------------------------------------
# Máscaras clásicas
# ---------------------------------------------------------------------------
MASCARA_PUNTO = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=float)

MASCARAS_LINEA = {
    "horizontal": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], float),
    "+45°":       np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], float),
    "vertical":   np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], float),
    "-45°":       np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], float),
}

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
SOBEL_Y = SOBEL_X.T


def detectar_puntos(img, umbral_rel: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Puntos aislados: |R| >= T con R = Laplaciano de 8 vecinos.
    El umbral es relativo al máximo de la respuesta. Devuelve (respuesta, máscara)."""
    g = a_gris(img)
    r = np.abs(ndi.convolve(g, MASCARA_PUNTO, mode="reflect"))
    return r, r >= umbral_rel * r.max()


def detectar_lineas(img, direccion: str | None = None, umbral_rel: float = 0.5):
    """Líneas de 1 px con las 4 máscaras direccionales.
    Si direccion es None se toma, en cada píxel, la máscara de mayor respuesta.
    Devuelve (respuesta, máscara, mapa_de_direcciones)."""
    g = a_gris(img)
    nombres = list(MASCARAS_LINEA)
    respuestas = np.stack([ndi.convolve(g, MASCARAS_LINEA[n], mode="reflect") for n in nombres])
    if direccion is not None:
        r = np.clip(respuestas[nombres.index(direccion)], 0, None)
        idx = np.full(g.shape, nombres.index(direccion))
    else:
        r = np.clip(respuestas.max(axis=0), 0, None)
        idx = respuestas.argmax(axis=0)
    return r, r >= umbral_rel * max(r.max(), 1e-12), idx


def gradiente_sobel(img, suavizado: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Magnitud y ángulo del gradiente (Sobel) tras suavizado gaussiano."""
    g = a_gris(img)
    if suavizado > 0:
        g = ndi.gaussian_filter(g, suavizado)
    gx = ndi.convolve(g, SOBEL_X, mode="reflect")
    gy = ndi.convolve(g, SOBEL_Y, mode="reflect")
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def laplaciano_gaussiano(img, sigma: float = 2.0) -> np.ndarray:
    return ndi.gaussian_laplace(a_gris(img), sigma)


def cruces_por_cero(log: np.ndarray, umbral_rel: float = 0.04) -> np.ndarray:
    """Cruces por cero del LoG cuya diferencia entre vecinos opuestos supera
    un umbral (evita los cruces producidos por el ruido)."""
    t = umbral_rel * np.abs(log).max()
    z = np.zeros(log.shape, dtype=bool)
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a = log
        b = np.roll(np.roll(log, -dy, axis=0), -dx, axis=1)
        z |= (np.sign(a) != np.sign(b)) & (np.abs(a - b) >= t)
    z[[0, -1], :] = False
    z[:, [0, -1]] = False
    return z


def deteccion_combinada(img, suavizado: float = 1.0) -> dict[str, np.ndarray]:
    """Combina las respuestas de punto, línea y borde (normalizadas a [0,1]).
    Cada píxel toma el detector con mayor respuesta relativa: así se ve en un
    solo mapa qué tipo de discontinuidad domina en cada zona."""
    g = a_gris(img)
    rp, _ = detectar_puntos(g)
    rl, _, _ = detectar_lineas(g)
    rb, _ = gradiente_sobel(g, suavizado)
    pila = np.stack([normalizar(rp), normalizar(rl), normalizar(rb)])
    return {"punto": pila[0], "linea": pila[1], "borde": pila[2],
            "combinada": pila.max(axis=0), "tipo": pila.argmax(axis=0)}


# ---------------------------------------------------------------------------
# Segmentación basada en bordes (contorno -> región)
# ---------------------------------------------------------------------------
def _contornos_a_region(bordes: np.ndarray, cierre: int, area_min: int) -> np.ndarray:
    """Cierra huecos de los contornos, rellena el interior y elimina restos."""
    cerrado = ndi.binary_closing(bordes, structure=morphology.disk(cierre), iterations=1)
    cerrado = ndi.binary_dilation(cerrado, structure=morphology.disk(1))
    relleno = ndi.binary_fill_holes(cerrado)
    # Las regiones que eran SOLO contorno (no encerraban nada) se descartan:
    # se quita el borde engrosado y se vuelve a dilatar lo que queda.
    interior = relleno & ~cerrado
    interior = ndi.binary_dilation(interior, structure=morphology.disk(2)) & relleno
    return limpiar_mascara(interior, area_min=area_min, radio=1)


def segmentar_canny(img, sigma: float = 2.0, bajo: float | None = None, alto: float | None = None,
                    cierre: int = 2, area_min: int = 60) -> np.ndarray:
    """Canny (suavizado + gradiente + supresión de no máximos + histéresis) y
    luego relleno de contornos cerrados. Umbrales por defecto: cuantiles del
    gradiente (automáticos, no dependen del contraste absoluto)."""
    g = a_gris(img)
    if bajo is None or alto is None:
        bordes = feature.canny(g, sigma=sigma, low_threshold=0.80, high_threshold=0.92,
                               use_quantiles=True)
    else:
        bordes = feature.canny(g, sigma=sigma, low_threshold=bajo, high_threshold=alto)
    return _contornos_a_region(bordes, cierre, area_min)


def segmentar_marr_hildreth(img, sigma: float = 2.5, umbral_rel: float = 0.06,
                            cierre: int = 2, area_min: int = 60) -> np.ndarray:
    """Marr-Hildreth: cruces por cero del Laplaciano del Gaussiano.
    Los cruces por cero forman contornos cerrados por naturaleza; además el
    signo del LoG indica de qué lado está el objeto claro (LoG < 0 dentro)."""
    log = laplaciano_gaussiano(img, sigma)
    z = cruces_por_cero(log, umbral_rel)
    return _contornos_a_region(z, cierre, area_min)
