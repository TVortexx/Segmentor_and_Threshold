"""Tema 5 · Utilización del movimiento en la segmentación.

Técnicas espaciales (método básico, Gonzalez & Woods):
    diferencia_consecutiva   d(x,y) = 1 si |f(t) − f(t−1)| > T.
    diferencia_referencia    d(x,y) = 1 si |f(t) − R| > T, con R = fondo estimado
                             por la MEDIANA TEMPORAL de los frames (imagen de
                             referencia sin objetos, como pide el libro).
    imagenes_diferencia_acumulada (ADI)  cuenta cuántas veces cambió cada píxel
                             respecto al primer frame (absoluta, positiva y negativa).
    mog2                     Mezcla de gaussianas adaptativa por píxel (OpenCV),
                             estado del arte "clásico" para cámaras fijas.

Técnicas en el dominio de la frecuencia:
    fft_temporal             Para cada píxel se toma su serie temporal en una
                             ventana de W frames y se calcula su FFT. Un píxel
                             estático solo tiene energía en la componente DC;
                             uno atravesado por un objeto tiene energía en las
                             frecuencias temporales ≠ 0.
    velocidad_correlacion_fase  Estima el desplazamiento (dx, dy) de cada objeto
                             entre frames con la correlación de fase:
                             F1·F2* / |F1·F2*| → su IFFT tiene un pico en el
                             desplazamiento (teorema del desplazamiento de Fourier).

Todas reciben `frames` (T, H, W) gris float y devuelven máscaras (T, H, W).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..utils import a_uint8, limpiar_mascara
from .umbralizacion import umbral_otsu


def compensar_brillo(frames: np.ndarray) -> np.ndarray:
    """Cancela el parpadeo GLOBAL de iluminación (aporte propio): escala cada
    frame para que su mediana sea igual a la mediana de todo el video. La
    mediana casi no se ve afectada por los objetos (ocupan poca área)."""
    medianas = np.median(frames, axis=(1, 2))
    ganancia = np.median(medianas) / np.maximum(medianas, 1e-6)
    return frames * ganancia[:, None, None]


def _preparar(frames, suavizado, compensar):
    f = np.stack([ndi.gaussian_filter(x, suavizado) for x in frames]) if suavizado > 0 else frames.copy()
    return compensar_brillo(f) if compensar else f


def _limpiar_secuencia(mascaras: np.ndarray, area_min: int = 40, radio: int = 1) -> np.ndarray:
    return np.stack([limpiar_mascara(m, area_min=area_min, radio=radio) for m in mascaras])


def _umbral_auto(dif: np.ndarray, k: float = 4.0, metodo: str = "ruido") -> float:
    """Umbral automático para mapas de diferencia/energía.

    metodo="ruido" (por defecto): la mayoría de los píxeles NO se mueven, así
        que la mediana de |d| describe al ruido. Para ruido gaussiano
        mediana(|d|) = 0.6745·σ  ->  T = k·σ.  (estimador MAD, robusto)
    metodo="otsu": Otsu sobre el histograma de diferencias. Falla cuando los
        bordes de objetos uniformes producen diferencias muy grandes: el
        umbral queda entre 'borde' y 'resto' y se pierde el interior."""
    if metodo == "otsu":
        escala = max(float(np.percentile(dif, 99.9)), 1e-6)
        t, _, _ = umbral_otsu(np.clip(dif / escala, 0, 1))
        return t * escala
    sigma = float(np.median(np.abs(dif))) / 0.6745
    return k * max(sigma, 1e-4)


def diferencia_consecutiva(frames: np.ndarray, umbral: float | None = None, suavizado: float = 1.0,
                           compensar: bool = True, metodo_umbral: str = "ruido", k: float = 3.0,
                           area_min: int = 40):
    """Limitación clave: si el objeto es UNIFORME y se desplaza menos que su
    tamaño, su interior es igual en t y t−1 y la diferencia se anula; solo se
    detectan los bordes delantero y trasero."""
    f = _preparar(frames, suavizado, compensar)
    dif = np.zeros_like(f)
    dif[1:] = np.abs(f[1:] - f[:-1])
    dif[0] = dif[1]
    t = _umbral_auto(dif, k, metodo_umbral) if umbral is None else umbral
    return _limpiar_secuencia(dif > t, area_min), {"umbral": t, "diferencias": dif}


def fondo_mediana(frames: np.ndarray) -> np.ndarray:
    """Imagen de referencia: mediana temporal. Si cada píxel está libre de
    objetos más de la mitad del tiempo, la mediana es el fondo exacto."""
    return np.median(frames, axis=0)


def diferencia_referencia(frames: np.ndarray, umbral: float | None = None, suavizado: float = 1.0,
                          compensar: bool = True, metodo_umbral: str = "otsu", k: float = 4.0,
                          area_min: int = 40):
    """Con una referencia sin objetos, TODO el objeto aparece en la diferencia:
    el histograma es bimodal (fondo ≈ 0, objeto ≫ 0) y Otsu es la elección
    natural del umbral."""
    f = _preparar(frames, suavizado, compensar)
    ref = fondo_mediana(f)
    dif = np.abs(f - ref[None])
    t = _umbral_auto(dif, k, metodo_umbral) if umbral is None else umbral
    return _limpiar_secuencia(dif > t, area_min), {"umbral": t, "referencia": ref, "diferencias": dif}


def imagenes_diferencia_acumulada(frames: np.ndarray, umbral: float = 0.1, suavizado: float = 1.0):
    """ADI absoluta, positiva y negativa respecto al primer frame (referencia).
    La positiva (objeto más claro que el fondo) deja de crecer cuando el objeto
    abandona su posición inicial: su soporte indica la posición de partida y,
    junto con la negativa, la dirección y velocidad del movimiento."""
    f = np.stack([ndi.gaussian_filter(x, suavizado) for x in frames]) if suavizado > 0 else frames
    ref = f[0]
    d = f - ref[None]
    return {"absoluta": (np.abs(d) > umbral).sum(0), "positiva": (d > umbral).sum(0),
            "negativa": (d < -umbral).sum(0)}


def mog2(frames: np.ndarray, historia: int = 50, var_umbral: float = 16.0, aprendizaje: float = 0.005,
         inicializar_con_mediana: bool = False, area_min: int = 40):
    """Background subtractor MOG2 de OpenCV (Zivkovic 2004). Cada píxel se
    modela con una mezcla de K gaussianas que se actualiza en línea.

    Trade-off de la tasa de aprendizaje α:
      α alta  -> el objeto lento se 'absorbe' en el fondo (cae el recall).
      α baja  -> los objetos del primer frame dejan 'fantasmas' (cae la precisión).
    inicializar_con_mediana (aporte propio): antes del video se entrena el
    modelo con la mediana temporal (fondo sin objetos) más ruido sintético
    del nivel estimado; así se puede usar α baja sin fantasmas."""
    sub = cv2.createBackgroundSubtractorMOG2(history=historia, varThreshold=var_umbral,
                                             detectShadows=False)
    if inicializar_con_mediana:
        ref = fondo_mediana(frames)
        ruido = float(np.median(np.abs(frames - ref[None]))) / 0.6745
        rng = np.random.default_rng(0)
        for _ in range(historia):
            sub.apply(a_uint8(ref + rng.normal(0, ruido, ref.shape)), learningRate=1.0 / historia)
    salidas = []
    for f in frames:
        m = sub.apply(a_uint8(f), learningRate=aprendizaje)
        salidas.append(m > 127)
    return _limpiar_secuencia(np.stack(salidas), area_min), {"historia": historia, "var_umbral": var_umbral,
                                                            "aprendizaje": aprendizaje,
                                                            "inicializado": inicializar_con_mediana}


def fft_temporal(frames: np.ndarray, ventana: int = 5, umbral: float | None = None,
                 suavizado: float = 1.0, compensar: bool = True, metodo_umbral: str = "ruido",
                 k: float = 4.0, area_min: int = 40):
    """Energía AC normalizada por píxel en una ventana temporal CENTRADA en t:

        E_ac(x,y,t) = Σ_{k≠0} |F_k(x,y)|²  /  W²

    donde F_k es la DFT de la serie f(x,y,t−W/2 .. t+W/2) multiplicada por una
    ventana de Hann (reduce el 'leakage' de los extremos). Por Parseval,
    E_ac = varianza temporal; la FFT además permite filtrar por banda: el
    parpadeo de iluminación es lento (baja frecuencia) y se puede descartar
    ignorando k=1 si la ventana es larga.

    Limitación: la máscara de t marca la TRAYECTORIA del objeto dentro de la
    ventana, no solo su posición actual (resolución temporal ↔ ventana)."""
    f = _preparar(frames, suavizado, compensar)
    n = len(f)
    mitad = ventana // 2
    hann = np.hanning(ventana + 2)[1:-1]
    energia = np.zeros_like(f)
    for t in range(n):
        a = min(max(t - mitad, 0), n - ventana)
        seg = f[a:a + ventana]
        seg = (seg - seg.mean(axis=0)) * hann[:, None, None]
        espectro = np.fft.rfft(seg, axis=0)
        energia[t] = np.sum(np.abs(espectro[1:]) ** 2, axis=0) / ventana ** 2
    raiz = np.sqrt(energia)
    t_ = _umbral_auto(raiz, k, metodo_umbral) if umbral is None else umbral
    return _limpiar_secuencia(raiz > t_, area_min), {"umbral": t_, "energia": raiz, "ventana": ventana}


def velocidad_correlacion_fase(frames: np.ndarray, mascaras: np.ndarray, t: int,
                               margen: int = 12, area_min: int = 80):
    """Para cada objeto (componente conexa de mascaras[t]) recorta una ventana
    alrededor en t y t+1, les resta el fondo (mediana temporal) y aplica la
    correlación de fase de OpenCV con ventana de Hann.
    Devuelve una lista de dicts {centro, dx, dy, respuesta}."""
    if t + 1 >= len(frames):
        t = len(frames) - 2
    ref = fondo_mediana(frames)
    a = (frames[t] - ref).astype(np.float32)
    b = (frames[t + 1] - ref).astype(np.float32)
    etiquetas, n = ndi.label(mascaras[t])
    resultados = []
    for sl, idx in zip(ndi.find_objects(etiquetas), range(1, n + 1)):
        if sl is None or (etiquetas[sl] == idx).sum() < area_min:
            continue
        y0 = max(sl[0].start - margen, 0); y1 = min(sl[0].stop + margen, a.shape[0])
        x0 = max(sl[1].start - margen, 0); x1 = min(sl[1].stop + margen, a.shape[1])
        ra, rb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
        if min(ra.shape) < 8:
            continue
        hann = cv2.createHanningWindow((ra.shape[1], ra.shape[0]), cv2.CV_32F)
        (dx, dy), respuesta = cv2.phaseCorrelate(ra, rb, hann)
        cy, cx = ndi.center_of_mass(etiquetas == idx)
        resultados.append({"centro": (float(cx), float(cy)), "dx": float(dx), "dy": float(dy),
                           "respuesta": float(respuesta), "caja": (x0, y0, x1, y1)})
    return resultados
