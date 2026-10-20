"""Comparación 1:1 entre el código ORIGINAL (carpeta original/, sin modificar)
y su versión MEJORADA (archivos de la raíz del repositorio).

Cada par recibe la misma imagen RGB float [0,1] y devuelve (máscara, info),
igual que el resto de las técnicas, para medirlos con las mismas métricas.

Los algoritmos originales devuelven imágenes (0/255, grupos de color o
etiquetas), no máscaras objeto/fondo. La conversión a máscara se hace aquí
con una regla única para original y mejorado, así la comparación es justa:
    - umbrales:      objeto = píxeles en 255 (o > 0 en el multinivel)
    - k-means:       fondo = grupo que domina el borde de la imagen
    - crecimiento:   objeto = píxeles alcanzados por alguna semilla
Ninguno aplica limpieza morfológica: se compara el algoritmo tal cual.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .tecnicas.regiones import parametros_crecimiento
from .utils import a_gris, a_rgb, a_uint8

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_ORIGINAL = RAIZ / "original"
_lock = threading.Lock()
_cache: dict[str, types.ModuleType] = {}


# ---------------------------------------------------------------------------
# Carga de módulos
# ---------------------------------------------------------------------------
def _asegurar_pyqt():
    """Los módulos originales importan PyQt5 al inicio aunque el algoritmo no
    lo use. Si PyQt5 no está instalado se registra un sustituto mínimo, solo
    para poder importar el archivo; los algoritmos no lo necesitan."""
    try:
        importlib.import_module("PyQt5.QtGui")
        return
    except Exception:
        pass
    raiz = types.ModuleType("PyQt5")
    for sub in ("QtGui", "QtCore", "QtWidgets", "uic"):
        m = types.ModuleType(f"PyQt5.{sub}")
        m.__getattr__ = lambda nombre: type(nombre, (object,), {"__init__": lambda s, *a, **k: None})
        sys.modules[f"PyQt5.{sub}"] = m
        setattr(raiz, sub, m)
    sys.modules["PyQt5"] = raiz


def _cargar(nombre: str, version: str) -> types.ModuleType:
    """version = 'original' (carpeta original/) o 'mejorado' (raíz)."""
    clave = f"{version}:{nombre}"
    if clave in _cache:
        return _cache[clave]
    _asegurar_pyqt()
    carpeta = CARPETA_ORIGINAL if version == "original" else RAIZ
    spec = importlib.util.spec_from_file_location(f"_{version}_{nombre}", carpeta / f"{nombre}.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(carpeta))          # por si importa módulos hermanos
    try:
        spec.loader.exec_module(modulo)
    finally:
        sys.path.remove(str(carpeta))
    _cache[clave] = modulo
    return modulo


@contextlib.contextmanager
def _directorio_temporal():
    """El Otsu local original escribe temp_window.png en el directorio actual.
    Se ejecuta dentro de una carpeta temporal para no ensuciar el proyecto."""
    with _lock, tempfile.TemporaryDirectory() as tmp:
        anterior = os.getcwd()
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(anterior)


def _gris_uint8(img) -> np.ndarray:
    return a_uint8(a_gris(img))


def _bgr_uint8(img) -> np.ndarray:
    return cv2.cvtColor(a_uint8(a_rgb(img)), cv2.COLOR_RGB2BGR)


def _fondo_por_borde(etiquetas: np.ndarray, margen: int = 3) -> int:
    borde = np.concatenate([etiquetas[:margen].ravel(), etiquetas[-margen:].ravel(),
                            etiquetas[:, :margen].ravel(), etiquetas[:, -margen:].ravel()])
    return int(np.bincount(borde).argmax())


# ---------------------------------------------------------------------------
# Umbralización (OtsuAndOptimal.py)
# ---------------------------------------------------------------------------
def _otsu_and_optimal(img, version: str, tmp: Path | None):
    clase = _cargar("OtsuAndOptimal", version).OtsuAndOptimal
    if version == "original":
        # El original solo acepta una ruta de archivo.
        ruta = tmp / "entrada.png"
        cv2.imwrite(str(ruta), _gris_uint8(img))
        return clase(str(ruta))
    return clase(_gris_uint8(img))


def otsu_global(img, version="mejorado"):
    with _directorio_temporal() as tmp:
        m = _otsu_and_optimal(img, version, tmp).otsu_thresholding() > 127
    return m, {}


def optimo_iterativo(img, version="mejorado"):
    with _directorio_temporal() as tmp:
        m = _otsu_and_optimal(img, version, tmp).optimal_thresholding() > 127
    return m, {}


def otsu_local(img, version="mejorado", ventana: int = 13):
    """ventana=13 es el valor por defecto de la interfaz original."""
    with _directorio_temporal() as tmp:
        m = _otsu_and_optimal(img, version, tmp).local_otsu(ventana) > 127
        archivos = sorted(p.name for p in tmp.iterdir() if p.name != "entrada.png")
    return m, {"ventana": ventana, "archivos_temporales": archivos}


def espectral_multinivel(img, version="mejorado"):
    proc = _cargar("spectralThresholding", version).ThresholdingProcessor()
    seg = proc.spectral_second_mode(_gris_uint8(img))
    return seg > 0, {"niveles": sorted(int(v) for v in np.unique(seg))}


# ---------------------------------------------------------------------------
# K-means (KMeanSegmentation.py)
# ---------------------------------------------------------------------------
def kmeans_rgb(img, version="mejorado", k: int = 4, semilla: int = 0):
    modulo = _cargar("KMeanSegmentation", version)
    rgb = a_uint8(a_rgb(img))
    if version == "original":
        np.random.seed(semilla)               # el original usa np.random global
        with contextlib.redirect_stdout(io.StringIO()):   # imprime cada iteración
            seg = modulo.kmeans(rgb, k=k)
        _, etiquetas = np.unique(seg.reshape(-1, seg.shape[-1]), axis=0, return_inverse=True)
        etiquetas = etiquetas.reshape(rgb.shape[:2])
    else:
        _, etiquetas = modulo.kmeans(rgb, k=k, seed=semilla, return_labels=True)
    fondo = _fondo_por_borde(etiquetas)
    return etiquetas != fondo, {"k": k}


# ---------------------------------------------------------------------------
# Crecimiento de regiones (RegionGrowingSegmentation.py / region_growing_core.py)
# ---------------------------------------------------------------------------
UMBRAL_CRECIMIENTO = 70   # valor por defecto del deslizador en la interfaz original


def crecimiento(img, version="mejorado", umbral: int = UMBRAL_CRECIMIENTO, semillas: str = "auto",
                criterio: str = "semilla"):
    """Ambas versiones reciben las MISMAS semillas (en la interfaz original
    las elige el usuario con el mouse).

    semillas="auto"    semillas por contraste local (pocas, sobre los objetos).
    semillas="grilla"  grilla de 17×17 = 289 semillas: con más de 255 el
                       original falla por desbordamiento de uint8.
    criterio           solo para la versión mejorada: "semilla" (igual al
                       original) o "media" (Gonzalez & Woods).
    """
    if semillas == "grilla":
        h, w = a_gris(img).shape
        ys = np.linspace(4, h - 5, 17).astype(int)
        xs = np.linspace(4, w - 5, 17).astype(int)
        puntos = [(int(x), int(y)) for y in ys for x in xs]
    else:
        _, sem, _, _ = parametros_crecimiento(img)
        puntos = [(int(c), int(f)) for f, c in sem]           # (x, y) como espera el original
    gris = _gris_uint8(img)
    if version == "original":
        funcion = _cargar("RegionGrowingSegmentation", "original").region_growing
        etiquetas = funcion(gris, puntos, threshold=umbral, use_luv=False)
    else:
        funcion = _cargar("region_growing_core", "mejorado").region_growing
        etiquetas = funcion(gris, puntos, threshold=umbral, use_luv=False, criterio=criterio)
    return etiquetas > 0, {"semillas": len(puntos), "umbral": umbral}


# ---------------------------------------------------------------------------
# Catálogo de pares
# ---------------------------------------------------------------------------
@dataclass
class Par:
    clave: str
    nombre: str
    archivo: str               # archivo original / mejorado que se compara
    original: Callable
    mejorado: Callable
    cambio: str                # qué cambió, en una línea


def _v(funcion, version):
    def envoltura(img, **kw):
        return funcion(img, version=version, **kw)
    envoltura.__name__ = f"{funcion.__name__}_{version}"
    return envoltura


PARES: list[Par] = [
    Par("otsu", "Otsu global", "OtsuAndOptimal.py",
        _v(otsu_global, "original"), _v(otsu_global, "mejorado"),
        "Acepta arreglos (sin leer de disco) y corrige g >= T por g > T."),
    Par("optimo", "Umbral óptimo iterativo", "OtsuAndOptimal.py",
        _v(optimo_iterativo, "original"), _v(optimo_iterativo, "mejorado"),
        "Mismo algoritmo; ya no requiere guardar la imagen en un archivo."),
    Par("otsu_local", "Otsu local por bloques", "OtsuAndOptimal.py",
        _v(otsu_local, "original"), _v(otsu_local, "mejorado"),
        "Sin temp_window.png: cada ventana se procesa en memoria."),
    Par("espectral", "Umbral espectral multinivel", "spectralThresholding.py",
        _v(espectral_multinivel, "original"), _v(espectral_multinivel, "mejorado"),
        "Búsqueda de (t1, t2) vectorizada con sumas acumuladas."),
    Par("kmeans", "K-means RGB", "KMeanSegmentation.py",
        _v(kmeans_rgb, "original"), _v(kmeans_rgb, "mejorado"),
        "Asignación de píxeles vectorizada; semilla reproducible."),
    Par("crecimiento", "Crecimiento de regiones", "RegionGrowingSegmentation.py",
        _v(crecimiento, "original"), _v(crecimiento, "mejorado"),
        "Separado de la interfaz, cola deque y sin desborde con más de 255 semillas."),
]

# Para cada par, la técnica NUEVA del paquete segpdi que lo reemplaza o lo
# supera (clave en registro.TECNICAS_ESTATICAS).
ALTERNATIVAS = {
    "otsu": "tophat_otsu",
    "optimo": "adaptativo",
    "otsu_local": "adaptativo",
    "espectral": "tophat_otsu",
    "kmeans": "kmeans_color",
    "crecimiento": "crecimiento",
}


def par_por_clave(clave: str) -> Par:
    for p in PARES:
        if p.clave == clave:
            return p
    raise KeyError(clave)
