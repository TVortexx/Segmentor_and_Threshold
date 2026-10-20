"""Catálogo único de técnicas. El benchmark y la app leen de aquí, así que
agregar una técnica nueva = agregar una entrada (sin tocar nada más).

Cada técnica estática recibe una imagen RGB float y devuelve (máscara, info).
Cada técnica de movimiento recibe frames (T, H, W) y devuelve (máscaras, info).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .tecnicas import discontinuidades as disc
from .tecnicas import movimiento as mov
from .tecnicas import multivariable as mv
from .tecnicas import regiones as reg
from .tecnicas import umbralizacion as umb


@dataclass
class Tecnica:
    clave: str
    nombre: str
    tema: str                 # tema de la guía de la asignatura
    funcion: Callable
    descripcion: str
    aporte_propio: bool = False
    parametros: dict = field(default_factory=dict)   # para los controles de la app


def _sin_info(f):
    """Adapta funciones que devuelven solo la máscara al formato (máscara, info)."""
    def envoltura(img, **kw):
        return f(img, **kw), {}
    envoltura.__name__ = f.__name__
    return envoltura


TECNICAS_ESTATICAS: list[Tecnica] = [
    Tecnica("canny", "Bordes · Canny", "2. Discontinuidades", _sin_info(disc.segmentar_canny),
            "Canny con umbrales por cuantiles + cierre morfológico + relleno de contornos.",
            parametros={"sigma": (0.5, 5.0, 2.0, 0.25)}),
    Tecnica("marr_hildreth", "Bordes · Marr-Hildreth (LoG)", "2. Discontinuidades",
            _sin_info(disc.segmentar_marr_hildreth),
            "Cruces por cero del Laplaciano del Gaussiano + relleno.",
            parametros={"sigma": (1.0, 6.0, 2.5, 0.25), "umbral_rel": (0.0, 0.3, 0.06, 0.01)}),
    Tecnica("global_iterativo", "Umbral global iterativo", "3. Umbralización", umb.global_iterativo,
            "Algoritmo básico: T = (m1 + m2)/2 hasta converger."),
    Tecnica("otsu", "Umbral óptimo de Otsu", "3. Umbralización", umb.otsu,
            "Maximiza la varianza entre clases (implementación propia desde el histograma)."),
    Tecnica("adaptativo", "Umbral adaptativo (local)", "3. Umbralización", umb.adaptativo,
            "T(x,y) = media gaussiana local + c.",
            parametros={"sigma_ventana": (3.0, 60.0, 15.0, 1.0), "c": (-0.1, 0.2, 0.02, 0.005)}),
    Tecnica("tophat_otsu", "Corrección de fondo morfológica + Otsu", "3. Umbralización", umb.tophat_otsu,
            "Apertura morfológica estima la iluminación; se divide (campo plano) y luego Otsu.",
            aporte_propio=True, parametros={"radio": (10, 120, 45, 5)}),
    Tecnica("kmeans_color", "K-means en color (Lab)", "3e. Varias variables", mv.kmeans_color,
            "Agrupa vectores (L*, a*, b*); fondo = grupo que domina el borde.",
            parametros={"k": (2, 8, 4, 1), "peso_luminancia": (0.0, 1.0, 0.5, 0.05)}),
    Tecnica("distancia_color", "Distancia de Mahalanobis al fondo", "3e. Varias variables",
            mv.distancia_color,
            "Distancia de cada píxel al color del fondo, en unidades de su dispersión.",
            parametros={"umbral": (1.0, 15.0, 3.0, 0.5)}),
    Tecnica("crecimiento", "Crecimiento de regiones", "4b. Regiones", reg.crecimiento_regiones,
            "Semillas por contraste local; agrega vecinos con |g − μ| ≤ tolerancia.",
            parametros={"tolerancia": (0.01, 0.4, 0.10, 0.01)}),
    Tecnica("division_fusion_libro", "División y fusión (versión libro)", "4c. Regiones",
            reg.division_fusion_libro,
            "Quadtree + fusión por diferencia de medias, sin refinamiento (referencia)."),
    Tecnica("division_fusion", "División y fusión mejorada", "4c. Regiones", reg.division_fusion,
            "Quadtree + fusión por gradiente de frontera + refinamiento píxel. APORTE PROPIO.",
            aporte_propio=True, parametros={"bloque_min": (2, 16, 4, 1)}),
]

TECNICAS_MOVIMIENTO: list[Tecnica] = [
    Tecnica("dif_consecutiva", "Diferencia de frames consecutivos", "5a. Espacial",
            mov.diferencia_consecutiva, "|f(t) − f(t−1)| > T (umbral por ruido, MAD)."),
    Tecnica("dif_referencia", "Diferencia con referencia (mediana)", "5a. Espacial",
            mov.diferencia_referencia, "|f(t) − mediana temporal| > T (Otsu)."),
    Tecnica("mog2", "MOG2 (inicio en frío)", "5a. Espacial", mov.mog2,
            "Mezcla de gaussianas adaptativa por píxel."),
    Tecnica("mog2_mediana", "MOG2 inicializado con mediana", "5a. Espacial",
            lambda fr, **kw: mov.mog2(fr, inicializar_con_mediana=True, **kw),
            "MOG2 pre-entrenado con el fondo sin objetos: sin fantasmas. APORTE PROPIO.",
            aporte_propio=True),
    Tecnica("fft_temporal", "FFT temporal (energía AC)", "5b. Frecuencia", mov.fft_temporal,
            "Energía en frecuencias temporales ≠ 0 en una ventana de W frames.",
            parametros={"ventana": (3, 15, 5, 2)}),
]


def por_clave(clave: str) -> Tecnica:
    for t in TECNICAS_ESTATICAS + TECNICAS_MOVIMIENTO:
        if t.clave == clave:
            return t
    raise KeyError(clave)
