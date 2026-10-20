"""Pruebas automáticas: validan las implementaciones propias contra
referencias conocidas (scikit-image / OpenCV / valores analíticos) y
garantizan que ninguna técnica se rompa antes de la demo.

    python -m pytest -q
"""

import numpy as np
import pytest
from skimage import filters

from segpdi import escenarios as esc
from segpdi.metricas import evaluar, error_velocidad
from segpdi.registro import TECNICAS_ESTATICAS, TECNICAS_MOVIMIENTO
from segpdi.tecnicas import discontinuidades as disc
from segpdi.tecnicas import movimiento as mov
from segpdi.tecnicas import regiones as reg
from segpdi.tecnicas import umbralizacion as umb
from segpdi.utils import a_gris


# ---------------------------------------------------------------- métricas
def test_metricas_perfectas_y_vacias():
    gt = np.zeros((10, 10), bool); gt[2:5, 2:5] = True
    assert evaluar(gt, gt).iou == 1.0
    assert evaluar(np.zeros_like(gt), gt).iou == 0.0
    vacio = np.zeros_like(gt)
    assert evaluar(vacio, vacio).iou == 1.0


def test_metricas_valores_conocidos():
    gt = np.zeros((1, 4), bool); gt[0, :2] = True       # [1 1 0 0]
    pred = np.zeros((1, 4), bool); pred[0, 1:3] = True  # [0 1 1 0]
    r = evaluar(pred, gt)
    assert r.iou == pytest.approx(1 / 3)
    assert r.dice == pytest.approx(0.5)
    assert r.precision == pytest.approx(0.5) and r.recall == pytest.approx(0.5)
    assert error_velocidad((3, 4), (0, 0)) == pytest.approx(5.0)


# ------------------------------------------------------------ umbralización
def test_otsu_propio_coincide_con_skimage():
    g = a_gris(esc.escenario_uniforme().imagen)
    t, eta, _ = umb.umbral_otsu(g)
    assert abs(t - filters.threshold_otsu(g, nbins=256)) < 2 / 256
    assert 0 < eta <= 1


def test_global_iterativo_coincide_con_isodata():
    g = a_gris(esc.escenario_uniforme().imagen)
    t, hist = umb.umbral_global_iterativo(g)
    assert abs(t - filters.threshold_isodata(g)) < 0.02
    assert len(hist) >= 2


def test_iluminacion_no_uniforme_rompe_umbral_global():
    """Propiedad del escenario E2: ni el MEJOR umbral global posible supera IoU 0.8."""
    e = esc.escenario_iluminacion()
    g = a_gris(e.imagen)
    mejor = max(evaluar(g > t, e.gt).iou for t in np.linspace(0.05, 0.95, 91))
    assert mejor < 0.8


# ----------------------------------------------------------------- regiones
def test_ruido_immerkaer():
    rng = np.random.default_rng(1)
    for s in (0.02, 0.08):
        assert reg.estimar_ruido(0.5 + rng.normal(0, s, (200, 200))) == pytest.approx(s, rel=0.1)


def test_quadtree_cubre_toda_la_imagen_sin_solaparse():
    g = a_gris(esc.escenario_uniforme().imagen)[:250, :190]   # tamaño no potencia de 2
    hojas = reg.dividir_quadtree(g, 0.02, 4)
    cobertura = np.zeros(g.shape, int)
    for y, x, h, w in hojas:
        cobertura[y:y + h, x:x + w] += 1
    assert (cobertura == 1).all()


def test_division_fusion_mejora_a_la_version_libro_en_E2():
    e = esc.escenario_iluminacion()
    libro = evaluar(reg.division_fusion_libro(e.imagen)[0], e.gt).iou
    mejorada = evaluar(reg.division_fusion(e.imagen)[0], e.gt).iou
    assert mejorada > 0.9 and mejorada - libro > 0.3


def test_crecimiento_respeta_semillas_manuales():
    e = esc.escenario_uniforme()
    m, info = reg.crecimiento_regiones(e.imagen, semillas=[(68, 68)])   # centro del disco
    assert m[68, 68] and m.sum() < e.gt.sum()   # solo crece el disco, no los demás objetos


# ---------------------------------------------------------- discontinuidades
def test_punto_aislado_detectado():
    img = np.zeros((21, 21)); img[10, 10] = 1.0
    _, m = disc.detectar_puntos(img)
    assert m[10, 10] and m.sum() == 1


def test_linea_horizontal_detectada():
    img = np.zeros((21, 21)); img[10, :] = 1.0
    _, _, idx = disc.detectar_lineas(img)
    assert list(disc.MASCARAS_LINEA)[idx[10, 10]] == "horizontal"


# ----------------------------------------------------------------- movimiento
def test_correlacion_de_fase_recupera_desplazamiento():
    v = esc.escenario_movimiento()
    m, _ = mov.diferencia_referencia(v.frames)
    est = mov.velocidad_correlacion_fase(v.frames, m, 12)
    assert any(error_velocidad((r["dx"], r["dy"]), v.velocidades[0]) < 0.5 for r in est)


def test_mog2_inicializado_elimina_fantasmas():
    v = esc.escenario_movimiento()
    frio = evaluar(mov.mog2(v.frames)[0][5:], v.gt[5:]).precision
    inic = evaluar(mov.mog2(v.frames, inicializar_con_mediana=True)[0][5:], v.gt[5:]).precision
    assert inic > frio + 0.2


# -------------------------------------------------------------- catálogo
@pytest.mark.parametrize("tec", TECNICAS_ESTATICAS, ids=lambda t: t.clave)
def test_todas_las_tecnicas_estaticas_corren(tec):
    e = esc.escenario_uniforme()
    m, info = tec.funcion(e.imagen)
    assert m.shape == e.gt.shape and m.dtype == bool and isinstance(info, dict)


@pytest.mark.parametrize("tec", TECNICAS_MOVIMIENTO, ids=lambda t: t.clave)
def test_todas_las_tecnicas_de_movimiento_corren(tec):
    v = esc.escenario_movimiento(n_frames=12)
    m, _ = tec.funcion(v.frames)
    assert m.shape == v.gt.shape
