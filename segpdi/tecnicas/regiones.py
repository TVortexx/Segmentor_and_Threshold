"""Tema 4 · Segmentación orientada a regiones.

Formulación básica (Gonzalez & Woods): una segmentación es una partición
R1..Rn de la imagen tal que (a) la unión es toda la imagen, (b) cada Ri es
conexa, (c) las regiones son disjuntas, (d) Q(Ri) = VERDADERO para cada región
y (e) Q(Ri ∪ Rj) = FALSO para regiones adyacentes. Q es el PREDICADO de
homogeneidad.

    crecimiento_regiones   Agregación de píxeles a partir de semillas.
    division_fusion        División en quadtree + fusión de regiones adyacentes.
                           >>> APORTE PROPIO: scikit-image y OpenCV no traen esta
                           técnica. Ver el docstring de `division_fusion` para las
                           decisiones de diseño.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from ..utils import a_gris, limpiar_mascara

OCHO_VECINOS = np.ones((3, 3), dtype=bool)


_MASCARA_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)


def estimar_ruido(g: np.ndarray, suavizado: float = 0.0) -> float:
    """σ del ruido gaussiano por el método de Immerkær (1996): la máscara es
    la diferencia de dos Laplacianos, anula bordes y rampas suaves y responde
    casi solo al ruido. σ = sqrt(π/2) · mean|g * M| / 6.

    Si luego la imagen se suaviza con un gaussiano de desvío `suavizado`, el
    ruido blanco se reduce en un factor 1 / (2·sqrt(π)·suavizado); se devuelve
    el σ del ruido que queda en la imagen YA suavizada."""
    r = np.abs(ndi.convolve(g, _MASCARA_IMMERKAER, mode="reflect"))[1:-1, 1:-1]
    sigma = float(np.sqrt(np.pi / 2.0) * r.mean() / 6.0)
    if suavizado >= 0.5:
        sigma = sigma / (2.0 * np.sqrt(np.pi) * suavizado)
    return sigma


def _region_fondo(etiquetas: np.ndarray, margen: int = 2) -> int:
    borde = np.concatenate([etiquetas[:margen].ravel(), etiquetas[-margen:].ravel(),
                            etiquetas[:, :margen].ravel(), etiquetas[:, -margen:].ravel()])
    return int(np.bincount(borde).argmax())


# ===========================================================================
# Crecimiento de regiones por agregación de píxeles
# ===========================================================================
def semillas_automaticas(g: np.ndarray, sigma_fondo: float = 20.0, k: float = 4.0,
                         ruido: float | None = None) -> np.ndarray:
    """Semillas = un punto por cada zona de alto CONTRASTE LOCAL.

    Contraste local c = g − fondo_local, con fondo_local = desenfoque gaussiano
    amplio. Se marcan los píxeles con c > k·σ_ruido + percentil 90 de c y cada componente
    conexa se reduce a su píxel de máximo contraste (como en el ejemplo del
    libro, donde las semillas se erosionan a un punto). Usar contraste local y no
    intensidad absoluta hace que las semillas no dependan de la iluminación."""
    if ruido is None:
        ruido = estimar_ruido(g)
    c = g - ndi.gaussian_filter(g, sigma_fondo)
    candidatos = c > max(k * ruido, np.percentile(c, 90))
    etiquetas, n = ndi.label(candidatos, structure=OCHO_VECINOS)
    if n == 0:
        return np.empty((0, 2), dtype=int)
    idx = ndi.maximum_position(c, etiquetas, index=np.arange(1, n + 1))
    return np.array(idx, dtype=int).reshape(-1, 2)


def parametros_crecimiento(img, semillas=None, suavizado: float = 1.0):
    """Prepara la imagen y calcula semillas y tolerancia automáticas.
    Se expone aparte para que la comparación con el crecimiento ORIGINAL use
    exactamente las mismas semillas. Devuelve (g_suavizada, semillas (fila, col),
    tolerancia, σ_ruido)."""
    g = a_gris(img)
    ruido = estimar_ruido(g, suavizado)
    if suavizado > 0:
        g = ndi.gaussian_filter(g, suavizado)
    if semillas is None:
        semillas = semillas_automaticas(g, ruido=ruido)
    semillas = np.asarray(semillas, dtype=int).reshape(-1, 2)
    if len(semillas):
        c = g - ndi.gaussian_filter(g, 20.0)
        contraste = float(np.median(c[semillas[:, 0], semillas[:, 1]]))
    else:
        contraste = 0.0
    tolerancia = max(3 * ruido, 0.5 * contraste, 0.02)
    return g, semillas, tolerancia, ruido


def crecimiento_regiones(img, semillas=None, tolerancia: float | None = None, suavizado: float = 1.0,
                         max_iter: int = 400, area_min: int = 30):
    """Agrega a cada región los píxeles vecinos (8-conectividad) que cumplen
    el predicado Q: |g(p) − media(región)| ≤ tolerancia.

    Implementación vectorizada por CAPAS: en cada iteración se dilata el mapa
    de etiquetas un píxel, se evalúa Q para toda la frontera a la vez con la
    media actual de cada región y se aceptan los que cumplen. Es equivalente a
    un BFS que actualiza la media por frente de onda en lugar de por píxel,
    pero ~100× más rápido en Python.

    tolerancia=None -> 0.5·(contraste típico objeto/fondo local), acotada para que nunca sea menor
    a 3·σ_ruido (si fuera menor, el propio ruido cortaría el crecimiento).
    """
    g, semillas, tol_auto, ruido = parametros_crecimiento(img, semillas, suavizado)
    if tolerancia is None:
        tolerancia = tol_auto

    etiquetas = np.zeros(g.shape, dtype=np.int32)
    for i, (r, c_) in enumerate(semillas, start=1):
        etiquetas[r, c_] = i
    n = len(semillas)
    plano = g.ravel()
    iteraciones = 0
    for iteraciones in range(1, max_iter + 1):
        cuenta = np.bincount(etiquetas.ravel(), minlength=n + 1)
        suma = np.bincount(etiquetas.ravel(), weights=plano, minlength=n + 1)
        media = suma / np.maximum(cuenta, 1)
        candidato = ndi.grey_dilation(etiquetas, footprint=OCHO_VECINOS)
        frontera = (etiquetas == 0) & (candidato > 0)
        if not frontera.any():
            break
        acepta = frontera & (np.abs(g - media[candidato]) <= tolerancia)
        if not acepta.any():
            break
        etiquetas[acepta] = candidato[acepta]

    mascara = limpiar_mascara(etiquetas > 0, area_min)
    return mascara, {"semillas": semillas, "tolerancia": tolerancia, "ruido": ruido,
                     "iteraciones": iteraciones, "etiquetas": etiquetas}


# ===========================================================================
# División y fusión (quadtree split & merge)  ---  APORTE PROPIO
# ===========================================================================
@dataclass
class _Integral:
    """Imágenes integrales de g y g²: la suma de cualquier rectángulo en O(1).
    Así la media y la desviación de cada bloque del quadtree no requieren
    recorrer sus píxeles (la división pasa de O(N log N) a O(#bloques))."""
    s: np.ndarray
    s2: np.ndarray

    @classmethod
    def de(cls, g):
        s = np.pad(g, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        s2 = np.pad(g * g, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return cls(s, s2)

    def stats(self, y, x, h, w):
        def rect(t):
            return t[y + h, x + w] - t[y, x + w] - t[y + h, x] + t[y, x]
        n = h * w
        m = rect(self.s) / n
        var = max(rect(self.s2) / n - m * m, 0.0)
        return n, m, np.sqrt(var)


def dividir_quadtree(g: np.ndarray, umbral_std: float, bloque_min: int = 4):
    """Divide recursivamente en 4 cuadrantes todo bloque cuyo predicado
    Q(R) = [σ(R) ≤ umbral_std] sea FALSO. Admite tamaños que no son potencia
    de 2 (cuadrantes desiguales). Devuelve la lista de hojas (y, x, h, w)."""
    integral = _Integral.de(g)
    hojas = []
    pila = [(0, 0, g.shape[0], g.shape[1])]
    while pila:
        y, x, h, w = pila.pop()
        _, _, std = integral.stats(y, x, h, w)
        if std <= umbral_std or h <= bloque_min or w <= bloque_min:
            hojas.append((y, x, h, w))
            continue
        h2, w2 = h // 2, w // 2
        pila.extend([(y, x, h2, w2), (y, x + w2, h2, w - w2),
                     (y + h2, x, h - h2, w2), (y + h2, x + w2, h - h2, w - w2)])
    return hojas


def _adyacencias(etiquetas: np.ndarray) -> np.ndarray:
    """Pares (a, b) de etiquetas que se tocan (4-conectividad), sin repetir."""
    pares = []
    for a, b in ((etiquetas[:, :-1], etiquetas[:, 1:]), (etiquetas[:-1, :], etiquetas[1:, :])):
        dif = a != b
        p = np.stack([a[dif], b[dif]], axis=1)
        pares.append(np.sort(p, axis=1))
    pares = np.concatenate(pares)
    return np.unique(pares, axis=0) if len(pares) else pares


def _pares_frontera(etiquetas: np.ndarray, valores: np.ndarray):
    """Para cada par de regiones adyacentes (4-conectividad) devuelve la SUMA
    de `valores` sobre los pares de píxeles de la frontera y cuántos pares hay.
    Es aditivo: al fusionar dos regiones basta sumar sus aristas."""
    n = int(etiquetas.max()) + 1
    claves, sumas = [], []
    for (la, lb), (va, vb) in (
        ((etiquetas[:, :-1], etiquetas[:, 1:]), (valores[:, :-1], valores[:, 1:])),
        ((etiquetas[:-1, :], etiquetas[1:, :]), (valores[:-1, :], valores[1:, :])),
    ):
        dif = la != lb
        a, b = la[dif].astype(np.int64), lb[dif].astype(np.int64)
        claves.append(np.minimum(a, b) * n + np.maximum(a, b))
        sumas.append(0.5 * (va[dif] + vb[dif]))
    claves = np.concatenate(claves)
    sumas = np.concatenate(sumas)
    unicas, inv = np.unique(claves, return_inverse=True)
    total = np.bincount(inv, weights=sumas)
    cuenta = np.bincount(inv)
    return {(int(k // n), int(k % n)): [float(t), float(c)] for k, t, c in zip(unicas, total, cuenta)}


def fusionar_regiones(g: np.ndarray, etiquetas: np.ndarray, umbral: float, criterio: str = "frontera",
                      gradiente: np.ndarray | None = None):
    """Fusión jerárquica sobre el grafo de adyacencia de regiones (RAG).

    Se procesan las aristas de MENOR a MAYOR peso con una cola de prioridad y
    se unen las regiones mientras el peso sea ≤ umbral. Tras cada unión se
    recalculan los pesos de las aristas afectadas; las entradas viejas de la
    cola se invalidan con un contador de versión.

    criterio="media"     (libro) peso = |μa − μb|. Problema: con un gradiente de
                         iluminación la media de una región grande se aleja de
                         la de los bloques lejanos y el fondo queda partido.
    criterio="frontera"  (mejora) peso = gradiente medio A LO LARGO de la frontera
                         común. Un cambio suave de iluminación tiene gradiente
                         bajo en todas partes, así que el fondo se fusiona
                         completo; el borde real de un objeto tiene gradiente
                         alto aunque esté en la zona oscura.
    """
    n = int(etiquetas.max()) + 1
    plano = etiquetas.ravel()
    cuenta = np.bincount(plano, minlength=n).astype(float)
    suma = np.bincount(plano, weights=g.ravel(), minlength=n)
    padre = np.arange(n)
    version = np.zeros(n, dtype=int)

    if criterio == "frontera":
        if gradiente is None:
            raise ValueError("criterio='frontera' requiere el mapa de gradiente")
        aristas = _pares_frontera(etiquetas, gradiente)
    else:
        aristas = {(int(a), int(b)): [0.0, 1.0] for a, b in _adyacencias(etiquetas)}
    vecinos = [set() for _ in range(n)]
    for a, b in aristas:
        vecinos[a].add(b)
        vecinos[b].add(a)

    def clave(a, b):
        return (a, b) if a < b else (b, a)

    def peso(a, b):
        if criterio == "frontera":
            t, c = aristas[clave(a, b)]
            return t / c
        return abs(suma[a] / cuenta[a] - suma[b] / cuenta[b])

    cola = [(peso(a, b), a, b, 0, 0) for a, b in aristas]
    heapq.heapify(cola)

    fusiones = 0
    while cola:
        d, a, b, va, vb = heapq.heappop(cola)
        if d > umbral:
            break                                    # el resto de aristas pesa aún más
        if padre[a] != a or padre[b] != b or version[a] != va or version[b] != vb:
            continue                                 # arista obsoleta
        if cuenta[b] > cuenta[a]:                    # la región grande absorbe a la chica
            a, b = b, a
        padre[b] = a
        cuenta[a] += cuenta[b]
        suma[a] += suma[b]
        version[a] += 1
        version[b] += 1
        aristas.pop(clave(a, b), None)
        for v in vecinos[b]:
            if v == a:
                continue
            datos = aristas.pop(clave(b, v))
            existente = aristas.get(clave(a, v))
            if existente is None:
                aristas[clave(a, v)] = datos
            else:
                existente[0] += datos[0]
                existente[1] += datos[1]
            vecinos[v].discard(b)
            vecinos[v].add(a)
            vecinos[a].add(v)
        vecinos[a].discard(b)
        vecinos[b] = set()
        fusiones += 1
        for v in vecinos[a]:
            x, y = clave(a, v)
            heapq.heappush(cola, (peso(a, v), x, y, version[x], version[y]))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    raices = np.array([raiz(i) for i in range(n)])
    _, compacto = np.unique(raices, return_inverse=True)
    return compacto[etiquetas].astype(np.int32), fusiones


def disolver_regiones_pequenas(g: np.ndarray, etiquetas: np.ndarray, area_min: int) -> np.ndarray:
    """Refinamiento a nivel PÍXEL. Las regiones chicas que quedan tras la fusión
    son casi siempre bloques mínimos que 'montan' sobre un borde (mitad objeto,
    mitad fondo). Sus píxeles se reasignan, capa por capa desde afuera hacia
    adentro, a la región grande vecina cuya media se parezca más al valor de
    cada píxel. Esto devuelve la precisión de borde que el quadtree pierde
    por trabajar con bloques."""
    etiquetas = etiquetas.copy()
    areas = np.bincount(etiquetas.ravel())
    grande = areas >= area_min
    if grande.sum() == 0:
        return etiquetas
    medias = np.bincount(etiquetas.ravel(), weights=g.ravel()) / np.maximum(areas, 1)
    pendiente = ~grande[etiquetas]
    h, w = g.shape
    desplazamientos = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while pendiente.any():
        mejor_costo = np.full(g.shape, np.inf)
        mejor_etq = np.zeros_like(etiquetas)
        relleno = np.pad(etiquetas, 1)
        rel_pend = np.pad(pendiente, 1, constant_values=True)
        for dy, dx in desplazamientos:
            vecina = relleno[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            vec_pend = rel_pend[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            costo = np.where(vec_pend, np.inf, np.abs(g - medias[vecina]))
            mejora = pendiente & (costo < mejor_costo)
            mejor_costo[mejora] = costo[mejora]
            mejor_etq[mejora] = vecina[mejora]
        resueltos = pendiente & np.isfinite(mejor_costo)
        if not resueltos.any():
            break
        etiquetas[resueltos] = mejor_etq[resueltos]
        pendiente &= ~resueltos
    return etiquetas


def division_fusion(img, umbral_std: float | None = None, umbral_fusion: float | None = None,
                    criterio: str = "frontera", refinar: bool = True, bloque_min: int = 4,
                    suavizado: float | str = "auto", area_min: int = 30):
    """Segmentación por división y fusión de regiones (quadtree split & merge).

    Pasos: (1) dividir en quadtree mientras el bloque no sea homogéneo,
    (2) fusionar regiones adyacentes parecidas, (3) refinar a nivel píxel las
    regiones chicas, (4) decidir qué regiones son objeto.

    Decisiones de diseño (APORTE PROPIO respecto al algoritmo del libro):
      1. Umbrales ADAPTATIVOS AL RUIDO (σ estimado con Immerkær): un bloque es
         homogéneo si su desviación es compatible con ruido puro (≤ 2.5σ).
      2. Imágenes integrales: cada bloque se evalúa en O(1).
      3. Fusión con cola de prioridad (lo más parecido primero).
      4. Criterio de fusión por FRONTERA (gradiente a lo largo del borde común)
         en lugar de diferencia de medias: robusto a iluminación no uniforme.
         Con criterio="media", refinar=False se obtiene la versión del libro
         para compararlas con las mismas métricas.
      5. Refinamiento de bordes a nivel píxel (disolver_regiones_pequenas).
      6. Objeto/fondo: el FONDO es la región que más toca el borde de la
         imagen; todo lo demás es objeto (no supone objetos claros u oscuros).
      7. Suavizado ADAPTATIVO (suavizado="auto"): σ_suav = max(1, σ_ruido/0.06).
         Con mucho ruido el gradiente de fondo sube, el umbral de fusión también
         y los objetos tenues terminan fusionados al fondo; suavizar más
         devuelve el contraste borde/ruido (ver estudio de robustez).
    """
    g = a_gris(img)
    if suavizado == "auto":
        suavizado = max(1.0, estimar_ruido(g) / 0.06)
    ruido = estimar_ruido(g, suavizado)
    if suavizado > 0:
        g = ndi.gaussian_filter(g, suavizado)
    if umbral_std is None:
        umbral_std = 2.5 * ruido + 1e-3

    gradiente = None
    if criterio == "frontera":
        gradiente = np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1)) / 8.0   # derivada por píxel
        if umbral_fusion is None:
            # Nivel de gradiente que produce el ruido solo (la mediana: la mayoría
            # de los píxeles no son borde) multiplicado por un margen.
            umbral_fusion = 3.0 * float(np.median(gradiente)) + 0.004
    elif umbral_fusion is None:
        umbral_fusion = 3.0 * ruido + 0.02

    hojas = dividir_quadtree(g, umbral_std, bloque_min)
    etiquetas_hojas = np.zeros(g.shape, dtype=np.int32)
    for i, (y, x, h, w) in enumerate(hojas):
        etiquetas_hojas[y:y + h, x:x + w] = i
    etiquetas, fusiones = fusionar_regiones(g, etiquetas_hojas, umbral_fusion, criterio, gradiente)
    if refinar:
        etiquetas = disolver_regiones_pequenas(g, etiquetas, area_min=max(area_min, 4 * bloque_min ** 2))
        _, etiquetas = np.unique(etiquetas, return_inverse=True)
        etiquetas = etiquetas.reshape(g.shape).astype(np.int32)
    fondo = _region_fondo(etiquetas)
    mascara = limpiar_mascara(etiquetas != fondo, area_min)
    return mascara, {"hojas": hojas, "n_hojas": len(hojas), "n_regiones": int(etiquetas.max()) + 1,
                     "fusiones": fusiones, "umbral_std": umbral_std, "umbral_fusion": umbral_fusion,
                     "criterio": criterio, "ruido": ruido, "suavizado": suavizado, "etiquetas": etiquetas,
                     "etiquetas_hojas": etiquetas_hojas}


def division_fusion_libro(img, **kw):
    """Versión 'de libro' para comparar: fusión por diferencia de medias y sin
    refinamiento a nivel píxel."""
    kw.setdefault("criterio", "media")
    kw.setdefault("refinar", False)
    kw.setdefault("suavizado", 1.0)
    return division_fusion(img, **kw)
