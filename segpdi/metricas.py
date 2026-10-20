"""Métricas objetivas de comparación, definidas ANTES de correr los experimentos.

Todas comparan una máscara binaria predicha contra el ground truth (GT):

    VP = píxeles objeto bien detectados      FP = fondo marcado como objeto
    FN = objeto que se perdió                VN = fondo bien descartado

    IoU (Jaccard) = VP / (VP + FP + FN)        -> métrica principal
    Dice (F1)     = 2VP / (2VP + FP + FN)
    Precisión     = VP / (VP + FP)             -> ¿cuánto de lo detectado es objeto?
    Recall        = VP / (VP + FN)             -> ¿cuánto del objeto se detectó?
    Exactitud     = (VP + VN) / total          -> engañosa si el fondo domina

Además se mide el tiempo de ejecución (ms) y, en el estudio de robustez,
cómo cae el IoU cuando aumenta el ruido o la no uniformidad de iluminación.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Resultado:
    iou: float
    dice: float
    precision: float
    recall: float
    exactitud: float

    def como_dict(self) -> dict:
        return asdict(self)


def matriz_confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Formas distintas: pred {pred.shape} vs gt {gt.shape}")
    vp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    vn = int(np.count_nonzero(~pred & ~gt))
    return vp, fp, fn, vn


def _div(a: float, b: float, vacio: float = 1.0) -> float:
    # Si el denominador es 0 (p. ej. GT vacío y predicción vacía) el resultado es perfecto.
    return float(a / b) if b > 0 else vacio


def evaluar(pred: np.ndarray, gt: np.ndarray) -> Resultado:
    vp, fp, fn, vn = matriz_confusion(pred, gt)
    return Resultado(
        iou=_div(vp, vp + fp + fn),
        dice=_div(2 * vp, 2 * vp + fp + fn),
        precision=_div(vp, vp + fp, vacio=1.0 if fn == 0 else 0.0),
        recall=_div(vp, vp + fn),
        exactitud=_div(vp + vn, vp + fp + fn + vn),
    )


def iou(pred, gt) -> float:
    return evaluar(pred, gt).iou


def dice(pred, gt) -> float:
    return evaluar(pred, gt).dice


def cronometrar(funcion, *args, repeticiones: int = 1, **kwargs):
    """Ejecuta la función y devuelve (resultado, tiempo_ms). Con varias
    repeticiones devuelve la mediana, más estable frente a ruido del SO."""
    tiempos = []
    salida = None
    for _ in range(max(1, repeticiones)):
        t0 = time.perf_counter()
        salida = funcion(*args, **kwargs)
        tiempos.append((time.perf_counter() - t0) * 1000.0)
    return salida, float(np.median(tiempos))


def error_velocidad(estimada, real) -> float:
    """Error de punto final (EPE) en píxeles/frame entre dos vectores (dx, dy)."""
    e = np.asarray(estimada, dtype=float)
    r = np.asarray(real, dtype=float)
    return float(np.hypot(*(e - r)))
