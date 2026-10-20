"""Demo en vivo · Segmentación de imágenes (PDI - FPUNE).

Extensión de Segmentor_and_Threshold (Computer-Vision-Toolkit). La interfaz
original de escritorio sigue disponible: python original/main.py

    streamlit run app.py
"""

from __future__ import annotations

import difflib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.segmentation import mark_boundaries

from segpdi import escenarios as esc
from segpdi.metricas import cronometrar, error_velocidad, evaluar
from segpdi.originales import ALTERNATIVAS, CARPETA_ORIGINAL, PARES, RAIZ, crecimiento, par_por_clave, precargar
from segpdi.registro import TECNICAS_ESTATICAS, TECNICAS_MOVIMIENTO, por_clave
from segpdi.tecnicas import discontinuidades as disc
from segpdi.tecnicas import movimiento as mov
from segpdi.tecnicas import multivariable as mv
from segpdi.tecnicas import regiones as reg
from segpdi.tecnicas import umbralizacion as umb
from segpdi.utils import a_gris, a_rgb, comparar_con_gt, normalizar, superponer

st.set_page_config(page_title="Segmentación de imágenes · PDI", layout="wide")


# ---------------------------------------------------------------- caché
@st.cache_data(show_spinner=False)
def cargar_estaticos():
    return {e.nombre: e for e in esc.escenarios_estaticos()}


@st.cache_data(show_spinner=False)
def cargar_reales():
    return esc.imagenes_reales()


@st.cache_data(show_spinner=False)
def cargar_video(n_frames: int, ruido: float, parpadeo: float):
    return esc.escenario_movimiento(n_frames=n_frames, ruido=ruido, parpadeo=parpadeo)


@st.cache_data(show_spinner=False, max_entries=256)
def correr(clave: str, imagen: np.ndarray, parametros: tuple):
    tec = por_clave(clave)
    (m, info), ms = cronometrar(tec.funcion, imagen, **dict(parametros))
    return m, info, ms


@st.cache_data(show_spinner=False, max_entries=32)
def correr_video(clave: str, frames: np.ndarray, parametros: tuple):
    tec = por_clave(clave)
    (m, info), ms = cronometrar(tec.funcion, frames, **dict(parametros))
    return m, ms


def mostrar(img, titulo=None, ancho=None):
    st.image(np.clip(img, 0, 1), caption=titulo, width=ancho or "stretch", clamp=True)


# ---------------------------------------------------------------- barra lateral
st.sidebar.title("Segmentación de imágenes")
st.sidebar.caption("Procesamiento Digital de Imágenes · FPUNE  \nBasado en Segmentor_and_Threshold")

fuente = st.sidebar.radio("Imagen de entrada", ["Escenario con ground truth", "Imagen real (sin GT)",
                                                "Subir imagen"], index=0)
imagen, gt, nombre_img, escenario = None, None, "", None
if fuente == "Escenario con ground truth":
    estaticos = cargar_estaticos()
    nombre_img = st.sidebar.selectbox("Escenario", list(estaticos))
    escenario = estaticos[nombre_img]
    imagen, gt = escenario.imagen, escenario.gt
    with st.sidebar.expander("Modificar el escenario en vivo"):
        st.caption("Regenera la escena E1/E2 con otro nivel de ruido o de iluminación.")
        if escenario.clave in ("uniforme", "iluminacion"):
            r = st.slider("Ruido σ", 0.0, 0.25, 0.04 if escenario.clave == "uniforme" else 0.06, 0.01)
            il = st.slider("No uniformidad de iluminación", 0.0, 1.0,
                           0.0 if escenario.clave == "uniforme" else 1.0, 0.1)
            g, gt = esc.escena_gris(ruido=r, iluminacion=il)
            imagen = a_rgb(g)
        else:
            st.info("Disponible para E1 y E2.")
elif fuente == "Imagen real (sin GT)":
    reales = cargar_reales()
    nombre_img = st.sidebar.selectbox("Imagen", list(reales))
    imagen = reales[nombre_img]
else:
    archivo = st.sidebar.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if archivo is not None:
        pil = Image.open(io.BytesIO(archivo.read())).convert("RGB")
        if max(pil.size) > 512:
            pil.thumbnail((512, 512))
        imagen = np.asarray(pil, dtype=np.float64) / 255.0
        nombre_img = archivo.name
        mascara_gt = st.sidebar.file_uploader("Máscara GT opcional (blanco = objeto)", type=["png", "bmp"])
        if mascara_gt is not None:
            mg = Image.open(io.BytesIO(mascara_gt.read())).convert("L").resize(pil.size, Image.NEAREST)
            gt = np.asarray(mg) > 127

st.sidebar.divider()
st.sidebar.markdown("**Leyenda del mapa de errores**  \n🟩 acierto · 🟥 falso positivo · 🟦 falso negativo")

pestanas = st.tabs(["Original vs. Mejorado", "Comparar técnicas", "Paso a paso", "Discontinuidades", "Movimiento",
                    "Resultados del benchmark"])


@st.cache_data(show_spinner=False, max_entries=64)
def correr_par(clave: str, version: str, imagen: np.ndarray, extra: tuple):
    """version: 'original' | 'mejorado' | 'nueva'. Devuelve (máscara, ms, error)."""
    par = par_por_clave(clave)
    try:
        if clave == "crecimiento" and version != "nueva":
            (m, info), ms = cronometrar(crecimiento, imagen, version=version, **dict(extra))
        elif version == "nueva":
            (m, info), ms = cronometrar(por_clave(ALTERNATIVAS[clave]).funcion, imagen)
        else:
            funcion = par.original if version == "original" else par.mejorado
            (m, info), ms = cronometrar(funcion, imagen)
        return m, ms, "", {k: v for k, v in info.items() if isinstance(v, (int, float, str, list))}
    except Exception as ex:   # el original puede fallar: se muestra el error
        return None, float("nan"), f"{type(ex).__name__}: {ex}", {}


@st.cache_resource(show_spinner=False)
def _precarga():
    precargar()     # importa los módulos antes de medir: el tiempo no incluye la carga
    return True


_precarga()


def reducir(img: np.ndarray, gt: np.ndarray | None, lado: int = 256):
    h, w = img.shape[:2]
    if max(h, w) <= lado:
        return img, gt
    f = lado / max(h, w)
    nuevo = (max(1, int(round(w * f))), max(1, int(round(h * f))))
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(nuevo, Image.BILINEAR)
    img2 = np.asarray(pil, dtype=np.float64) / 255.0
    gt2 = None
    if gt is not None:
        gt2 = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize(nuevo, Image.NEAREST)) > 127
    return img2, gt2


# =====================================================================
# 0. Original vs. Mejorado (comparación 1:1 con el repositorio base)
# =====================================================================
with pestanas[0]:
    st.markdown("Cada algoritmo del repositorio base **tal como lo escribieron sus autores** (carpeta `original/`) "
                "frente a **la misma función corregida** (raíz del repo) y la **técnica nueva** que agregamos para "
                "resolver su limitación.")
    if imagen is None:
        st.info("Subí una imagen en la barra lateral.")
    else:
        nombres_par = {f"{p.nombre}  ·  {p.archivo}": p for p in PARES}
        elegido = st.selectbox("Algoritmo del repositorio original", list(nombres_par))
        par = nombres_par[elegido]
        st.caption(f"**Qué cambió:** {par.cambio}")
        extra = ()
        if par.clave == "crecimiento":
            c1, c2 = st.columns(2)
            modo_sem = c1.radio("Semillas (las mismas para ambas versiones)", ["auto", "grilla"], horizontal=True,
                                format_func=lambda x: {"auto": "Automáticas", "grilla": "Grilla de 289 (>255)"}[x])
            umbral_rg = c2.slider("Umbral (0–255, por defecto 70 como en la interfaz original)", 5, 150, 70, 5)
            extra = (("semillas", modo_sem), ("umbral", umbral_rg))
        img_par, gt_par = reducir(imagen, gt)
        if img_par.shape[:2] != imagen.shape[:2]:
            st.caption(f"La imagen se redujo a {img_par.shape[1]}×{img_par.shape[0]} px para ambas versiones "
                       "(el k-means original tarda varios segundos en imágenes grandes).")
        tecnica_nueva = por_clave(ALTERNATIVAS[par.clave])
        cols = st.columns(4)
        with cols[0]:
            mostrar(img_par, "Entrada")
            if gt_par is not None:
                mostrar(gt_par.astype(float), "Ground truth")
        filas_par = []
        for col, (version, titulo) in zip(cols[1:], (("original", "Original (sin modificar)"),
                                                    ("mejorado", "Mejorado (mismo algoritmo)"),
                                                    ("nueva", f"Técnica nueva: {tecnica_nueva.nombre}"))):
            with col:
                st.markdown(f"**{titulo}**")
                with st.spinner("Ejecutando…"):
                    m, ms, error, info = correr_par(par.clave, version, img_par, extra if version != "nueva" else ())
                if error:
                    st.error(f"El código falla:\n\n`{error}`")
                    filas_par.append({"Versión": titulo, "IoU": np.nan, "Tiempo (ms)": np.nan, "Resultado": error})
                    continue
                vis = comparar_con_gt(m, gt_par) if gt_par is not None else superponer(img_par, m)
                iou_txt = f" · IoU {evaluar(m, gt_par).iou:.3f}" if gt_par is not None else ""
                mostrar(vis, f"{ms:.1f} ms{iou_txt}")
                if info.get("archivos_temporales"):
                    st.warning(f"Escribió en disco: {', '.join(info['archivos_temporales'])}")
                filas_par.append({"Versión": titulo, "IoU": evaluar(m, gt_par).iou if gt_par is not None else np.nan,
                                  "Tiempo (ms)": ms, "Resultado": "ok"})
        df_par = pd.DataFrame(filas_par)
        t_o, t_m = df_par["Tiempo (ms)"].iloc[0], df_par["Tiempo (ms)"].iloc[1] if len(df_par) > 1 else np.nan
        if np.isfinite(t_o) and np.isfinite(t_m) and t_m > 0:
            st.metric("Aceleración del mismo algoritmo", f"{t_o / t_m:.1f}×",
                      help="Tiempo del original dividido por el tiempo de la versión mejorada")
        st.dataframe(df_par.style.format({"IoU": "{:.3f}", "Tiempo (ms)": "{:.1f}"}, na_rep="—"),
                     hide_index=True, width="stretch")

        with st.expander(f"Ver el código: diferencias en {par.archivo}"):
            archivo_mejorado = "region_growing_core.py" if par.clave == "crecimiento" else par.archivo
            texto_o = (CARPETA_ORIGINAL / par.archivo).read_text(encoding="utf-8").splitlines()
            texto_m = (RAIZ / archivo_mejorado).read_text(encoding="utf-8").splitlines()
            diff = "\n".join(difflib.unified_diff(texto_o, texto_m, fromfile=f"original/{par.archivo}",
                                                  tofile=archivo_mejorado, lineterm="", n=2))
            st.code(diff or "Sin diferencias", language="diff")

        tabla = RAIZ / "resultados" / "original_vs_mejorado.csv"
        if tabla.exists():
            with st.expander("Tabla completa del benchmark (todos los algoritmos y escenarios)"):
                d = pd.read_csv(tabla)
                d = d[["par", "escenario", "iou_original", "iou_mejorado", "iou_nueva", "tecnica_nueva",
                       "ms_original", "ms_mejorado", "aceleracion", "error_original"]]
                st.dataframe(d, hide_index=True, width="stretch")

# =====================================================================
# 1. Comparar técnicas
# =====================================================================
with pestanas[1]:
    if imagen is None:
        st.info("Subí una imagen en la barra lateral.")
    else:
        if escenario is not None:
            st.markdown(f"**{escenario.nombre}** — {escenario.descripcion}  \n*Desafío:* {escenario.desafio}")
        nombres = {t.nombre: t for t in TECNICAS_ESTATICAS}
        elegidas = st.multiselect("Técnicas", list(nombres), default=list(nombres))
        parametros = {}
        with st.expander("Parámetros (dejar por defecto = automáticos)"):
            cols = st.columns(3)
            i = 0
            for nom in elegidas:
                t = nombres[nom]
                if not t.parametros:
                    continue
                with cols[i % 3]:
                    st.markdown(f"**{t.nombre}**")
                    usar = st.checkbox("Ajustar manualmente", key=f"usar_{t.clave}")
                    p = {}
                    for par, (lo, hi, defecto, paso) in t.parametros.items():
                        if isinstance(defecto, int) and isinstance(paso, int):
                            v = st.slider(par, int(lo), int(hi), int(defecto), int(paso), key=f"{t.clave}_{par}",
                                          disabled=not usar)
                        else:
                            v = st.slider(par, float(lo), float(hi), float(defecto), float(paso),
                                          key=f"{t.clave}_{par}", disabled=not usar)
                        p[par] = v
                    if usar:
                        parametros[t.clave] = p
                i += 1

        c1, c2 = st.columns(2)
        with c1:
            mostrar(imagen, "Entrada")
        with c2:
            if gt is not None:
                mostrar(gt.astype(float), "Ground truth")

        filas, resultados = [], {}
        progreso = st.progress(0.0, text="Segmentando…")
        for k, nom in enumerate(elegidas):
            t = nombres[nom]
            m, info, ms = correr(t.clave, imagen, tuple(sorted(parametros.get(t.clave, {}).items())))
            resultados[nom] = m
            fila = {"Técnica": nom, "Tema": t.tema, "Aporte propio": "✔" if t.aporte_propio else "",
                    "Tiempo (ms)": round(ms, 1)}
            if gt is not None:
                r = evaluar(m, gt)
                fila.update({"IoU": r.iou, "Dice": r.dice, "Precisión": r.precision, "Recall": r.recall})
            filas.append(fila)
            progreso.progress((k + 1) / max(len(elegidas), 1), text=f"Segmentando… {nom}")
        progreso.empty()

        if filas:
            df = pd.DataFrame(filas)
            if gt is not None:
                df = df.sort_values("IoU", ascending=False)
                st.subheader("Métricas")
                st.dataframe(df.style.format({"IoU": "{:.3f}", "Dice": "{:.3f}", "Precisión": "{:.3f}",
                                              "Recall": "{:.3f}"})
                             .background_gradient(subset=["IoU"], cmap="Greens", vmin=0, vmax=1),
                             hide_index=True, width="stretch")
                st.bar_chart(df.set_index("Técnica")["IoU"], horizontal=True, height=36 * len(df) + 60)
            else:
                st.dataframe(df, hide_index=True, width="stretch")

            st.subheader("Resultados")
            vista = st.radio("Vista", ["Mapa de errores", "Superposición", "Máscara"], horizontal=True,
                             index=0 if gt is not None else 1)
            cols = st.columns(4)
            for k, (nom, m) in enumerate(resultados.items()):
                with cols[k % 4]:
                    if vista == "Mapa de errores" and gt is not None:
                        vis = comparar_con_gt(m, gt)
                    elif vista == "Máscara":
                        vis = m.astype(float)
                    else:
                        vis = superponer(imagen, m)
                    extra = f" · IoU {evaluar(m, gt).iou:.3f}" if gt is not None else ""
                    mostrar(vis, nom + extra)

# =====================================================================
# 2. Paso a paso (internos de cada algoritmo)
# =====================================================================
with pestanas[2]:
    if imagen is None:
        st.info("Subí una imagen en la barra lateral.")
    else:
        opcion = st.selectbox("Algoritmo", ["Otsu y umbral global iterativo", "Umbral adaptativo",
                                            "Corrección de fondo morfológica", "K-means en color",
                                            "Crecimiento de regiones", "División y fusión (quadtree)"])
        g = a_gris(imagen)
        if opcion == "Otsu y umbral global iterativo":
            from scipy import ndimage as ndi
            gs = ndi.gaussian_filter(g, 1.0)
            t_otsu, eta, curva = umb.umbral_otsu(gs)
            t_it, hist = umb.umbral_global_iterativo(gs)
            h, bordes = np.histogram(gs, bins=256, range=(0, 1))
            dfh = pd.DataFrame({"intensidad": bordes[:-1], "histograma": h / h.max(),
                                "σB² (normalizada)": curva / max(curva.max(), 1e-12)}).set_index("intensidad")
            st.line_chart(dfh, height=300)
            c1, c2, c3 = st.columns(3)
            c1.metric("Umbral de Otsu", f"{t_otsu:.3f}")
            c2.metric("Separabilidad η", f"{eta:.3f}", help="σB²/σG²: 1 = clases perfectamente separables")
            c3.metric("Umbral iterativo", f"{t_it:.3f}", help=f"Convergió en {len(hist) - 1} iteraciones")
            st.caption("Historial del algoritmo iterativo: " + " → ".join(f"{x:.3f}" for x in hist))
            c1, c2 = st.columns(2)
            with c1:
                mostrar((gs > t_otsu).astype(float), "g > T (Otsu)")
            with c2:
                mostrar(gs, "Imagen suavizada")
            st.info("Si la iluminación no es uniforme el histograma deja de ser bimodal: η baja y ningún T "
                    "global separa bien. Probalo con E2.")
        elif opcion == "Umbral adaptativo":
            s = st.slider("σ de la ventana", 3.0, 60.0, 15.0, 1.0)
            c = st.slider("c", -0.1, 0.2, 0.02, 0.005)
            m, info = umb.adaptativo(imagen, sigma_ventana=s, c=c)
            c1, c2, c3 = st.columns(3)
            with c1: mostrar(g, "Gris")
            with c2: mostrar(normalizar(info["mapa_umbral"]), "Umbral T(x,y) (normalizado)")
            with c3: mostrar(superponer(imagen, m), "Resultado")
            st.info("Con σ chico (menor que los objetos) el interior de un objeto grande se vuelve 'fondo local' "
                    "y queda hueco: ese es el trade-off del tamaño de ventana.")
        elif opcion == "Corrección de fondo morfológica":
            radio = st.slider("Radio del disco (debe superar al objeto más grande)", 10, 120, 45, 5)
            modo = st.radio("Modelo de iluminación", ["division", "resta"], horizontal=True,
                            format_func=lambda x: {"division": "Multiplicativo (g / fondo)",
                                                   "resta": "Aditivo (top-hat clásico)"}[x])
            m, info = umb.tophat_otsu(imagen, radio=radio, modo=modo)
            c1, c2, c3 = st.columns(3)
            with c1: mostrar(normalizar(info["fondo_estimado"]), "Fondo estimado (apertura)")
            with c2: mostrar(info["corregida"], "Imagen corregida")
            with c3: mostrar(superponer(imagen, m), "Otsu sobre la corregida")
        elif opcion == "K-means en color":
            k = st.slider("K (grupos)", 2, 8, 4)
            peso = st.slider("Peso de la luminancia L*", 0.0, 1.0, 0.5, 0.05)
            m, info = mv.kmeans_color(imagen, k=k, peso_luminancia=peso)
            c1, c2 = st.columns(2)
            with c1: mostrar(normalizar(info["etiquetas"].astype(float)), f"Grupos (fondo = {info['grupo_fondo']})")
            with c2: mostrar(superponer(imagen, m), "No-fondo")
            st.info("Bajar el peso de L* hace al método más robusto a la iluminación, pero si los objetos solo "
                    "se distinguen por brillo (E1/E2) se pierde la información útil.")
        elif opcion == "Crecimiento de regiones":
            tol_auto = st.checkbox("Tolerancia automática", True)
            tol = None if tol_auto else st.slider("Tolerancia", 0.01, 0.4, 0.1, 0.01)
            m, info = reg.crecimiento_regiones(imagen, tolerancia=tol)
            vis = superponer(imagen, m)
            for (r_, c_) in info["semillas"]:
                vis[max(r_ - 2, 0):r_ + 3, max(c_ - 2, 0):c_ + 3] = (1.0, 0.85, 0.0)
            c1, c2 = st.columns(2)
            with c1: mostrar(vis, f"{len(info['semillas'])} semillas (amarillo) y regiones crecidas")
            with c2: mostrar(normalizar(info["etiquetas"].astype(float)), "Etiquetas por semilla")
            st.caption(f"Tolerancia usada: {info['tolerancia']:.3f} · ruido estimado σ = {info['ruido']:.4f} · "
                       f"{info['iteraciones']} capas de crecimiento")
        else:
            c1, c2, c3 = st.columns(3)
            criterio = c1.radio("Criterio de fusión", ["frontera", "media"],
                                format_func=lambda x: {"frontera": "Gradiente en la frontera (mejora)",
                                                       "media": "Diferencia de medias (libro)"}[x])
            refinar = c2.checkbox("Refinamiento a nivel píxel", True)
            bloque = c3.slider("Bloque mínimo", 2, 16, 4)
            (m, info), ms = cronometrar(reg.division_fusion, imagen, criterio=criterio, refinar=refinar,
                                        bloque_min=bloque)
            c1, c2, c3 = st.columns(3)
            with c1: mostrar(mark_boundaries(a_rgb(imagen), info["etiquetas_hojas"], color=(1, 0.8, 0)),
                             f"1 · División: {info['n_hojas']} hojas")
            with c2: mostrar(mark_boundaries(a_rgb(imagen), info["etiquetas"], color=(1, 0.2, 0.2)),
                             f"2 · Fusión: {info['n_regiones']} regiones")
            with c3:
                titulo = "3 · Objeto = todo menos la región de fondo"
                mostrar(comparar_con_gt(m, gt) if gt is not None else superponer(imagen, m),
                        titulo + (f" · IoU {evaluar(m, gt).iou:.3f}" if gt is not None else ""))
            st.caption(f"σ ruido = {info['ruido']:.4f} · umbral de homogeneidad = {info['umbral_std']:.4f} · "
                       f"umbral de fusión = {info['umbral_fusion']:.4f} · {info['fusiones']} fusiones · {ms:.0f} ms")
            st.info("Probá E2 con criterio 'media': el fondo queda partido en varias regiones porque la media de "
                    "una región grande se aleja de la de los bloques del otro extremo del gradiente.")

# =====================================================================
# 3. Discontinuidades
# =====================================================================
with pestanas[3]:
    st.markdown("Detección de **puntos**, **líneas**, **bordes** y **detección combinada**.")
    patron = st.radio("Imagen", ["Patrón de prueba", "Imagen seleccionada"], horizontal=True)
    if patron == "Patrón de prueba" or imagen is None:
        rng = np.random.default_rng(3)
        prueba = np.full((200, 200), 0.3) + rng.normal(0, 0.01, (200, 200))
        prueba[40, 40] = prueba[160, 30] = prueba[100, 170] = 1.0     # puntos aislados
        prueba[20, 70:190] = 0.9                                      # línea horizontal
        prueba[60:190, 120] = 0.9                                     # línea vertical
        for i in range(90):
            prueba[105 + i, 10 + i] = 0.9                             # línea diagonal
        prueba[130:180, 140:190] = 0.8                                # región (bordes)
        fuente_disc = np.clip(prueba, 0, 1)
    else:
        fuente_disc = a_gris(imagen)
    c1, c2, c3 = st.columns(3)
    up = c1.slider("Umbral relativo de puntos", 0.1, 1.0, 0.6, 0.05)
    ul = c2.slider("Umbral relativo de líneas", 0.1, 1.0, 0.5, 0.05)
    sg = c3.slider("σ (bordes / LoG)", 0.5, 5.0, 1.5, 0.25)
    _, mp = disc.detectar_puntos(fuente_disc, up)
    rl, ml, _ = disc.detectar_lineas(fuente_disc, umbral_rel=ul)
    mag, _ = disc.gradiente_sobel(fuente_disc, sg)
    log = disc.laplaciano_gaussiano(fuente_disc, sg)
    comb = disc.deteccion_combinada(fuente_disc, sg)
    from skimage import feature
    canny = feature.canny(fuente_disc, sigma=sg)
    fila1 = st.columns(4)
    with fila1[0]: mostrar(fuente_disc, "Entrada")
    with fila1[1]: mostrar(mp.astype(float), f"Puntos ({int(mp.sum())} px)")
    with fila1[2]: mostrar(ml.astype(float), "Líneas (4 máscaras)")
    with fila1[3]: mostrar(normalizar(mag), "Magnitud del gradiente (Sobel)")
    fila2 = st.columns(4)
    with fila2[0]: mostrar(disc.cruces_por_cero(log).astype(float), "Cruces por cero del LoG")
    with fila2[1]: mostrar(canny.astype(float), "Canny")
    with fila2[2]: mostrar(comb["combinada"], "Combinada (máx. de respuestas)")
    colores = np.array([[0.95, 0.3, 0.3], [0.2, 0.8, 0.3], [0.25, 0.5, 0.95]])
    tipo_rgb = colores[comb["tipo"]] * comb["combinada"][..., None]
    with fila2[3]: mostrar(normalizar(tipo_rgb), "Tipo dominante: rojo punto, verde línea, azul borde")

# =====================================================================
# 4. Movimiento
# =====================================================================
with pestanas[4]:
    c1, c2, c3 = st.columns(3)
    n_frames = c1.slider("Frames", 12, 48, 32, 4)
    ruido_v = c2.slider("Ruido σ", 0.0, 0.1, 0.03, 0.01)
    parpadeo = c3.slider("Parpadeo de iluminación", 0.0, 0.15, 0.04, 0.01)
    video = cargar_video(n_frames, ruido_v, parpadeo)
    st.caption(video.descripcion + " " + video.desafio)
    nombres_m = {t.nombre: t for t in TECNICAS_MOVIMIENTO}
    elegidas_m = st.multiselect("Técnicas", list(nombres_m), default=list(nombres_m), key="mov_sel")
    t_frame = st.slider("Frame a visualizar", 0, n_frames - 1, min(20, n_frames - 1))

    resultados_m, filas_m, por_frame = {}, [], {}
    w = video.calentamiento
    for nom in elegidas_m:
        m, ms = correr_video(nombres_m[nom].clave, video.frames, ())
        resultados_m[nom] = m
        ious = [evaluar(m[i], video.gt[i]).iou for i in range(len(m))]
        por_frame[nom] = ious
        r = evaluar(m[w:], video.gt[w:])
        filas_m.append({"Técnica": nom, "Tema": nombres_m[nom].tema, "IoU medio": float(np.mean(ious[w:])),
                        "Precisión": r.precision, "Recall": r.recall, "ms/frame": ms / len(video.frames)})
    cols = st.columns(len(resultados_m) + 1)
    with cols[0]:
        mostrar(video.frames[t_frame], f"Frame {t_frame}")
    for k, (nom, m) in enumerate(resultados_m.items(), start=1):
        with cols[k]:
            mostrar(comparar_con_gt(m[t_frame], video.gt[t_frame]), f"{nom} · IoU {por_frame[nom][t_frame]:.2f}")
    if filas_m:
        st.dataframe(pd.DataFrame(filas_m).sort_values("IoU medio", ascending=False)
                     .style.format({"IoU medio": "{:.3f}", "Precisión": "{:.3f}", "Recall": "{:.3f}",
                                    "ms/frame": "{:.1f}"}), hide_index=True, width="stretch")
        st.line_chart(pd.DataFrame(por_frame), height=260, x_label="frame", y_label="IoU")
        st.caption(f"Los primeros {w} frames son de calentamiento y no entran en el promedio.")

    st.subheader("Dominio de la frecuencia · velocidad por correlación de fase")
    m_ref, _ = mov.diferencia_referencia(video.frames)
    est = mov.velocidad_correlacion_fase(video.frames, m_ref, t_frame)
    filas_v = []
    for e_ in est:
        cx, cy = int(round(e_["centro"][0])), int(round(e_["centro"][1]))
        real = None
        for k, (g_obj, vel) in enumerate(zip(video.gt_objetos, video.velocidades)):
            if g_obj[min(t_frame, len(g_obj) - 1)][min(cy, 255), min(cx, 255)]:
                real = vel
        filas_v.append({"centro (x, y)": f"({cx}, {cy})", "dx estimado": e_["dx"], "dy estimado": e_["dy"],
                        "real": str(real), "EPE (px/frame)": error_velocidad((e_["dx"], e_["dy"]), real)
                        if real else np.nan, "pico de correlación": e_["respuesta"]})
    if filas_v:
        st.dataframe(pd.DataFrame(filas_v), hide_index=True, width="stretch")
    adi = mov.imagenes_diferencia_acumulada(video.frames)
    c1, c2, c3 = st.columns(3)
    with c1: mostrar(normalizar(adi["absoluta"].astype(float)), "ADI absoluta")
    with c2: mostrar(normalizar(adi["positiva"].astype(float)), "ADI positiva")
    with c3: mostrar(normalizar(adi["negativa"].astype(float)), "ADI negativa")

# =====================================================================
# 5. Resultados del benchmark
# =====================================================================
with pestanas[5]:
    carpeta = Path(__file__).parent / "resultados"
    resumen = carpeta / "RESUMEN.md"
    if not resumen.exists():
        st.warning("Todavía no hay resultados. Ejecutá `python -m segpdi.benchmark`.")
    else:
        for png, titulo in [("iou_por_escenario.png", None), ("robustez_ruido.png", None),
                            ("robustez_iluminacion.png", None), ("iou_por_frame.png", None)]:
            if (carpeta / png).exists():
                st.image(str(carpeta / png), width=900)
        with st.expander("Resumen completo (RESUMEN.md)"):
            st.markdown(resumen.read_text(encoding="utf-8"))
