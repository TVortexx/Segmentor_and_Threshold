"""Pruebas de las correcciones al código original.

Cada prueba ejecuta la versión de original/ (sin modificar) y la de la raíz
y verifica: (a) que el resultado se conserva cuando el algoritmo no cambió y
(b) que los errores encontrados en el original están corregidos.

    python -m pytest -q
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from skimage import filters

from segpdi import escenarios as esc
from segpdi.metricas import evaluar
from segpdi.originales import PARES, _asegurar_pyqt, crecimiento

RAIZ = Path(__file__).resolve().parent.parent


def cargar(ruta: str, nombre: str):
    _asegurar_pyqt()
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / ruta)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def imagen_bimodal():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(60, 10, 5000), rng.normal(170, 15, 3000)])
    return np.clip(v, 0, 255).astype(np.uint8).reshape(80, 100)


# ------------------------------------------------------------ OtsuAndOptimal
def test_otsu_mejorado_coincide_con_skimage(imagen_bimodal):
    clase = cargar("OtsuAndOptimal.py", "oo_mej").OtsuAndOptimal
    o = clase(imagen_bimodal)                      # ahora acepta arreglos
    assert o.otsu_threshold_value() == filters.threshold_otsu(imagen_bimodal)
    assert ((o.otsu_thresholding() > 0) == (imagen_bimodal > o.otsu_threshold_value())).all()


def test_otsu_local_ya_no_escribe_archivos(imagen_bimodal, tmp_path):
    clase = cargar("OtsuAndOptimal.py", "oo_mej2").OtsuAndOptimal
    anterior = os.getcwd()
    os.chdir(tmp_path)
    try:
        clase(imagen_bimodal).local_otsu(16)
        clase(imagen_bimodal).local_optimal(16)
        assert list(tmp_path.iterdir()) == []
    finally:
        os.chdir(anterior)


# ----------------------------------------------------- crecimiento de regiones
def test_crecimiento_identico_al_original():
    orig = cargar("original/RegionGrowingSegmentation.py", "rg_orig").region_growing
    mej = cargar("region_growing_core.py", "rg_mej").region_growing
    rng = np.random.default_rng(1)
    img = (rng.random((60, 70)) * 80).astype(np.uint8)
    img[30:] += 120
    semillas = [(int(x), int(y)) for x, y in zip(rng.integers(0, 70, 25), rng.integers(0, 60, 25))]
    for umbral in (10, 40, 90):
        assert np.array_equal(orig(img, semillas, threshold=umbral, use_luv=False).astype(np.int32),
                              mej(img, semillas, threshold=umbral, use_luv=False))


def test_crecimiento_original_falla_con_mas_de_255_semillas():
    e = esc.escenario_uniforme()
    with pytest.raises(OverflowError):
        crecimiento(e.imagen, version="original", semillas="grilla")
    m, info = crecimiento(e.imagen, version="mejorado", semillas="grilla")
    assert info["semillas"] == 289 and m.shape == e.gt.shape


def test_crecimiento_semilla_fuera_de_rango():
    orig = cargar("original/RegionGrowingSegmentation.py", "rg_orig2").region_growing
    mej = cargar("region_growing_core.py", "rg_mej2").region_growing
    img = np.zeros((20, 20), np.uint8)
    with pytest.raises(UnboundLocalError):
        orig(img, [(50, 50), (5, 5)], threshold=10, use_luv=False)
    assert set(np.unique(mej(img, [(50, 50), (5, 5)], threshold=10, use_luv=False))) == {2}


def test_criterio_media_disponible():
    mej = cargar("region_growing_core.py", "rg_mej3").region_growing
    img = np.full((30, 30), 50, np.uint8)
    img[10:20, 10:20] = 200
    r = mej(img, [(15, 15)], threshold=30, use_luv=False, criterio="media")
    assert r[15, 15] == 1 and r.sum() == 100


# ------------------------------------------------------------------- k-means
def test_kmeans_vectorizado_da_el_mismo_resultado():
    import contextlib
    import io
    orig = cargar("original/KMeanSegmentation.py", "km_orig")
    mej = cargar("KMeanSegmentation.py", "km_mej")
    img = (np.random.default_rng(2).random((40, 50, 3)) * 255).astype(np.uint8)
    inicio = img.reshape(-1, 3)[[0, 500, 1000, 1999]].astype(float)
    orig.create_random_centroids = lambda p, k: inicio.copy()
    mej.create_random_centroids = lambda p, k, rng=None: inicio.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        a = orig.kmeans(img, k=4)
    assert np.allclose(a, mej.kmeans(img, k=4))


# ------------------------------------------------------- umbral espectral
def test_espectral_multinivel_mismos_umbrales():
    orig = cargar("original/spectralThresholding.py", "sp_orig").ThresholdingProcessor()
    mej = cargar("spectralThresholding.py", "sp_mej").ThresholdingProcessor()
    rng = np.random.default_rng(3)
    v = np.concatenate([rng.normal(40, 8, 3000), rng.normal(120, 10, 3000), rng.normal(210, 8, 2000)])
    img = np.clip(v, 0, 255).astype(np.uint8).reshape(80, 100)
    assert np.array_equal(orig.spectral_second_mode(img), mej.spectral_second_mode(img))


# ----------------------------------------------------- todos los pares corren
@pytest.mark.parametrize("par", PARES, ids=lambda p: p.clave)
def test_pares_original_y_mejorado_corren(par):
    e = esc.escenario_uniforme()
    for funcion in (par.original, par.mejorado):
        m, _ = funcion(e.imagen)
        assert m.shape == e.gt.shape
        assert 0.0 <= evaluar(m, e.gt).iou <= 1.0
