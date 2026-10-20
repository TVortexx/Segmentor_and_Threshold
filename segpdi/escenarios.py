"""Escenarios de prueba con ground truth (GT) exacto.

Se usan imágenes sintéticas porque permiten conocer la máscara correcta píxel
por píxel; así las métricas son objetivas y reproducibles (semilla fija).
Cada escenario ataca una debilidad distinta de las técnicas:

  E1 "uniforme"            Iluminación pareja + ruido leve. Caso fácil / línea base.
  E2 "iluminacion"         Gradiente de iluminación + viñeteado + más ruido.
                           Rompe a los umbrales GLOBALES (cap. "papel de la iluminación").
  E3 "color_isoluminante"  Objetos con el MISMO brillo que el fondo, solo cambia el color.
                           Rompe a todo lo que trabaja en escala de grises;
                           motiva los umbrales en VARIAS VARIABLES.
  E4 "movimiento"          Secuencia de video con objetos que se desplazan sobre un fondo
                           texturado, con parpadeo de iluminación. Para segmentación
                           por movimiento (técnicas espaciales y en frecuencia).

También se exponen imágenes reales de scikit-image (sin GT) para la demo visual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage import draw

from .utils import a_rgb

TAM = (256, 256)
SEMILLA = 2026
FONDO = 0.25
LUM = np.array([0.2125, 0.7154, 0.0721])  # pesos de luminancia de skimage.rgb2gray


@dataclass
class Escenario:
    clave: str
    nombre: str
    descripcion: str
    desafio: str
    imagen: np.ndarray          # RGB float [0,1], (H, W, 3)
    gt: np.ndarray | None       # bool (H, W) o None si no hay GT


@dataclass
class EscenarioVideo:
    clave: str
    nombre: str
    descripcion: str
    desafio: str
    frames: np.ndarray                  # gris float [0,1], (T, H, W)
    gt: np.ndarray                      # bool (T, H, W): unión de objetos
    gt_objetos: list = field(default_factory=list)   # por objeto: bool (T, H, W)
    velocidades: list = field(default_factory=list)  # por objeto: (dx, dy) px/frame
    calentamiento: int = 5              # frames iniciales excluidos de la evaluación


# --------------------------------------------------------------------------
# Objetos de la escena estática
# --------------------------------------------------------------------------
def _mascaras_objetos(forma=TAM) -> list[np.ndarray]:
    """Cinco objetos de formas y tamaños distintos (incluye uno pequeño y
    uno con esquinas agudas, que castigan a los métodos por bloques)."""
    mascaras = []

    def nueva():
        return np.zeros(forma, dtype=bool)

    m = nueva(); rr, cc = draw.disk((68, 68), 36, shape=forma); m[rr, cc] = True; mascaras.append(m)
    m = nueva(); rr, cc = draw.rectangle((150, 30), extent=(62, 84), shape=forma); m[rr, cc] = True; mascaras.append(m)
    m = nueva(); rr, cc = draw.ellipse(78, 182, 30, 52, shape=forma, rotation=0.5); m[rr, cc] = True; mascaras.append(m)
    m = nueva(); rr, cc = draw.polygon([150, 232, 232], [196, 150, 236], shape=forma); m[rr, cc] = True; mascaras.append(m)
    m = nueva(); rr, cc = draw.disk((126, 128), 11, shape=forma); m[rr, cc] = True; mascaras.append(m)
    return mascaras


_INTENSIDADES = [0.85, 0.68, 0.92, 0.62, 0.78]


def campo_iluminacion(forma=TAM, intensidad: float = 1.0) -> np.ndarray:
    """Campo multiplicativo de iluminación: gradiente diagonal + viñeteado.
    intensidad=0 -> iluminación perfecta (todo 1); intensidad=1 -> rango ~[0.18, 1.7]."""
    h, w = forma
    y, x = np.mgrid[0:h, 0:w]
    diagonal = (x / (w - 1) + y / (h - 1)) / 2.0                       # 0 arriba-izq, 1 abajo-der
    r = np.hypot((x - w * 0.8) / w, (y - h * 0.8) / h)                   # foco abajo-derecha
    vineta = 1.0 - np.clip(r / r.max(), 0, 1)
    g = 0.75 * diagonal + 0.25 * vineta
    g = ((g - g.min()) / (g.max() - g.min())) ** 1.5                     # [0, 1], no lineal
    objetivo = 0.18 + 1.52 * g                                           # [0.18, 1.7]
    return 1.0 + intensidad * (objetivo - 1.0)


def escena_gris(ruido: float = 0.03, iluminacion: float = 0.0, semilla: int = SEMILLA):
    """Escena base en gris con GT. Parámetros usados por E1, E2 y el estudio de robustez."""
    rng = np.random.default_rng(semilla)
    objetos = _mascaras_objetos()
    img = np.full(TAM, FONDO)
    gt = np.zeros(TAM, dtype=bool)
    for m, v in zip(objetos, _INTENSIDADES):
        img[m] = v
        gt |= m
    img = ndi.gaussian_filter(img, 0.8)          # bordes realistas (no escalón ideal)
    img = img * campo_iluminacion(TAM, iluminacion)
    img = img + rng.normal(0, ruido, TAM)
    return np.clip(img, 0, 1), gt


def escenario_uniforme() -> Escenario:
    img, gt = escena_gris(ruido=0.04, iluminacion=0.0)
    return Escenario(
        "uniforme", "E1 · Iluminación uniforme",
        "Cinco objetos más claros que el fondo, iluminación pareja y ruido gaussiano σ=0.04.",
        "Línea base: casi todas las técnicas deberían funcionar bien.",
        a_rgb(img), gt,
    )


def escenario_iluminacion() -> Escenario:
    img, gt = escena_gris(ruido=0.06, iluminacion=1.0)
    return Escenario(
        "iluminacion", "E2 · Iluminación no uniforme",
        "Misma escena multiplicada por un gradiente diagonal + viñeteado (0.18× a 1.7×) y ruido σ=0.06.",
        "Los objetos de la zona oscura son más oscuros que el fondo de la zona clara: "
        "ningún umbral global único puede separarlos.",
        a_rgb(img), gt,
    )


def _color_isoluminante(base: np.ndarray, direccion: np.ndarray, magnitud: float) -> np.ndarray:
    """Color con la misma luminancia que `base`: se suma un vector ortogonal
    a los pesos de luminancia (w·delta = 0)."""
    d = direccion - (LUM @ direccion) / (LUM @ LUM) * LUM
    d = d / np.linalg.norm(d) * magnitud
    return base + d


def escenario_color_isoluminante(ruido: float = 0.03) -> Escenario:
    rng = np.random.default_rng(SEMILLA + 1)
    fondo = np.array([0.45, 0.45, 0.45])
    direcciones = [np.array(v, float) for v in
                   ([1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0.0], [0.2, 0, 1])]
    img = np.ones(TAM + (3,)) * fondo
    gt = np.zeros(TAM, dtype=bool)
    for m, d in zip(_mascaras_objetos(), direcciones):
        c = np.clip(_color_isoluminante(fondo, d, 0.30), 0, 1)
        img[m] = c
        gt |= m
    for k in range(3):
        img[..., k] = ndi.gaussian_filter(img[..., k], 0.8)
    img = np.clip(img + rng.normal(0, ruido, img.shape), 0, 1)
    return Escenario(
        "color_isoluminante", "E3 · Color isoluminante",
        "Objetos de distinto color pero con la misma luminancia que el fondo gris.",
        "En escala de grises la imagen es casi plana: solo un método que use las "
        "tres variables de color puede segmentarla.",
        img, gt,
    )


# --------------------------------------------------------------------------
# Escenario de video
# --------------------------------------------------------------------------
def escenario_movimiento(n_frames: int = 32, ruido: float = 0.03, parpadeo: float = 0.04,
                         semilla: int = SEMILLA) -> EscenarioVideo:
    rng = np.random.default_rng(semilla + 2)
    h, w = TAM
    textura = ndi.gaussian_filter(rng.normal(0, 1, TAM), 4)
    fondo = 0.42 + 0.12 * textura / np.abs(textura).max()

    # Objeto 1: disco claro y uniforme -> la diferencia consecutiva solo ve sus bordes.
    # Objeto 2: rectángulo oscuro con textura a cuadros.
    v1, v2 = (4.0, 1.0), (-3.0, -2.0)
    p1, p2 = np.array([40.0, 70.0]), np.array([220.0, 200.0])  # (x, y)
    yy, xx = np.mgrid[0:h, 0:w]

    frames = np.zeros((n_frames, h, w))
    gt1 = np.zeros((n_frames, h, w), bool)
    gt2 = np.zeros((n_frames, h, w), bool)
    for t in range(n_frames):
        f = fondo.copy()
        c1 = p1 + t * np.array(v1)
        c2 = p2 + t * np.array(v2)
        m1 = (xx - c1[0]) ** 2 + (yy - c1[1]) ** 2 <= 20 ** 2
        m2 = (np.abs(xx - c2[0]) <= 22) & (np.abs(yy - c2[1]) <= 15)
        # La textura viaja con el objeto (coordenadas relativas a su centro).
        tablero = ((np.floor((xx - c2[0]) / 6) + np.floor((yy - c2[1]) / 6)) % 2).astype(float)
        f[m1] = 0.85
        f[m2] = 0.12 + 0.08 * tablero[m2]
        gain = 1.0 + parpadeo * np.sin(2 * np.pi * t / 9.0)
        f = f * gain + rng.normal(0, ruido, TAM)
        frames[t] = np.clip(f, 0, 1)
        gt1[t], gt2[t] = m1, m2

    return EscenarioVideo(
        "movimiento", "E4 · Objetos en movimiento",
        f"{n_frames} frames 256×256: disco claro (4,1) px/frame y rectángulo texturado "
        f"oscuro (−3,−2) px/frame sobre fondo texturado, con parpadeo de iluminación ±{parpadeo:.0%}.",
        "El disco es uniforme: al restar frames consecutivos su interior se cancela. "
        "El parpadeo cambia todo el fondo y genera falsos positivos.",
        frames, gt1 | gt2, [gt1, gt2], [v1, v2],
    )


# --------------------------------------------------------------------------
# Catálogo
# --------------------------------------------------------------------------
def escenarios_estaticos() -> list[Escenario]:
    return [escenario_uniforme(), escenario_iluminacion(), escenario_color_isoluminante()]


def imagenes_reales() -> dict[str, np.ndarray]:
    """Imágenes incluidas en scikit-image (no requieren internet), sin GT."""
    from skimage import data
    salida = {}
    for nombre, f in [("Monedas (coins)", data.coins), ("Página escaneada (page)", data.page),
                      ("Café (coffee)", data.coffee), ("Luna (moon)", data.moon),
                      ("Fotógrafo (camera)", data.camera)]:
        try:
            salida[nombre] = a_rgb(f())
        except Exception:  # pragma: no cover - depende de la instalación
            pass
    return salida
