"""Banco de pruebas: corre TODAS las técnicas sobre TODOS los escenarios con
las MISMAS métricas y guarda tablas (CSV), figuras (PNG) y un resumen (Markdown).

Uso:
    python -m segpdi.benchmark                 # todo (≈1-2 min)
    python -m segpdi.benchmark --rapido        # sin estudio de robustez
    python -m segpdi.benchmark --salida otra_carpeta
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import escenarios as esc
from . import graficos
from .metricas import cronometrar, error_velocidad, evaluar
from .originales import ALTERNATIVAS, PARES, crecimiento
from .registro import TECNICAS_ESTATICAS, TECNICAS_MOVIMIENTO, por_clave
from .tecnicas import movimiento as mov
from .tecnicas import regiones as reg
from .utils import a_rgb


def _fmt(df: pd.DataFrame, cols=None) -> str:
    """Tabla Markdown sin depender de 'tabulate'."""
    cols = cols or list(df.columns)
    filas = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for fila in df[cols].itertuples(index=False):
        celdas = []
        for v in fila:
            celdas.append(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v))
        filas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(filas)


# ---------------------------------------------------------------------------
def experimento_estatico(salida: Path, repeticiones: int = 3) -> pd.DataFrame:
    filas = []
    for e in esc.escenarios_estaticos():
        mascaras, ious = {}, {}
        for t in TECNICAS_ESTATICAS:
            (m, _), ms = cronometrar(t.funcion, e.imagen, repeticiones=repeticiones)
            r = evaluar(m, e.gt)
            mascaras[t.nombre], ious[t.nombre] = m, r.iou
            filas.append({"escenario": e.nombre, "tecnica": t.nombre, "tema": t.tema,
                          "aporte_propio": t.aporte_propio, **r.como_dict(), "tiempo_ms": ms})
            print(f"  {e.clave:<20} {t.nombre:<42} IoU={r.iou:.3f}  {ms:7.1f} ms")
        graficos.mosaico_escenario(e, mascaras, ious, salida / f"mosaico_{e.clave}.png")
    df = pd.DataFrame(filas)
    df.to_csv(salida / "metricas_estaticas.csv", index=False)
    graficos.barras_iou(df, salida / "iou_por_escenario.png")
    return df


def original_vs_mejorado(salida: Path, repeticiones: int = 3) -> pd.DataFrame:
    """Compara 1:1 cada algoritmo ORIGINAL (carpeta original/) con su versión
    corregida (raíz) y con la técnica nueva de segpdi que lo reemplaza."""
    filas = []
    escenarios = esc.escenarios_estaticos()
    for par in PARES:
        par.original(escenarios[0].imagen)      # calentamiento: carga de módulos
        par.mejorado(escenarios[0].imagen)
        alternativa = por_clave(ALTERNATIVAS[par.clave])
        for e in escenarios:
            fila = {"par": par.nombre, "archivo": par.archivo, "escenario": e.nombre}
            for etiqueta, funcion, reps in (("original", par.original, 1 if par.clave == "kmeans" else repeticiones),
                                            ("mejorado", par.mejorado, repeticiones),
                                            ("nueva", alternativa.funcion, repeticiones)):
                try:
                    (m, _), ms = cronometrar(funcion, e.imagen, repeticiones=reps)
                    fila[f"iou_{etiqueta}"], fila[f"ms_{etiqueta}"] = evaluar(m, e.gt).iou, ms
                    fila[f"error_{etiqueta}"] = ""
                except Exception as ex:  # el original puede fallar: se registra
                    fila[f"iou_{etiqueta}"], fila[f"ms_{etiqueta}"] = np.nan, np.nan
                    fila[f"error_{etiqueta}"] = type(ex).__name__
            fila["tecnica_nueva"] = alternativa.nombre
            filas.append(fila)
            print(f"  {par.nombre:<30} {e.clave:<20} original {fila['iou_original']:.3f} / "
                  f"{fila['ms_original']:7.1f} ms   mejorado {fila['iou_mejorado']:.3f} / "
                  f"{fila['ms_mejorado']:6.1f} ms   nueva {fila['iou_nueva']:.3f}")
    # Caso de falla documentado: crecimiento con más de 255 semillas
    e = escenarios[0]
    fila = {"par": "Crecimiento de regiones (289 semillas)", "archivo": "RegionGrowingSegmentation.py",
            "escenario": e.nombre, "tecnica_nueva": ""}
    for etiqueta, version in (("original", "original"), ("mejorado", "mejorado")):
        try:
            (m, _), ms = cronometrar(crecimiento, e.imagen, version=version, semillas="grilla")
            fila[f"iou_{etiqueta}"], fila[f"ms_{etiqueta}"], fila[f"error_{etiqueta}"] = evaluar(m, e.gt).iou, ms, ""
        except Exception as ex:
            fila[f"iou_{etiqueta}"], fila[f"ms_{etiqueta}"], fila[f"error_{etiqueta}"] = np.nan, np.nan, type(ex).__name__
    fila.update({"iou_nueva": np.nan, "ms_nueva": np.nan, "error_nueva": ""})
    filas.append(fila)
    df = pd.DataFrame(filas)
    df["aceleracion"] = df["ms_original"] / df["ms_mejorado"]
    df.to_csv(salida / "original_vs_mejorado.csv", index=False)
    graficos.tiempos_original_mejorado(df, salida / "original_vs_mejorado_tiempo.png")
    return df


def experimento_movimiento(salida: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = esc.escenario_movimiento()
    w = v.calentamiento
    filas, por_frame, mascaras, medios = [], [], {}, {}
    for t in TECNICAS_MOVIMIENTO:
        (m, _), ms = cronometrar(t.funcion, v.frames)
        ious = [evaluar(m[i], v.gt[i]).iou for i in range(len(m))]
        r = evaluar(m[w:], v.gt[w:])
        medio = float(np.mean(ious[w:]))
        mascaras[t.nombre], medios[t.nombre] = m, medio
        por_frame += [{"tecnica": t.nombre, "frame": i, "iou": x} for i, x in enumerate(ious)]
        filas.append({"escenario": v.nombre, "tecnica": t.nombre, "tema": t.tema,
                      "aporte_propio": t.aporte_propio, "iou_medio": medio,
                      "iou_min": float(np.min(ious[w:])), "precision": r.precision, "recall": r.recall,
                      "dice": r.dice, "ms_por_frame": ms / len(v.frames)})
        print(f"  {'movimiento':<20} {t.nombre:<42} IoU medio={medio:.3f}  {ms / len(v.frames):6.1f} ms/frame")
    df = pd.DataFrame(filas)
    df.to_csv(salida / "metricas_movimiento.csv", index=False)
    dpf = pd.DataFrame(por_frame)
    dpf.to_csv(salida / "iou_por_frame.csv", index=False)
    graficos.mosaico_video(v, mascaras, medios, t=20, ruta=salida / "mosaico_movimiento.png")
    graficos.iou_por_frame(dpf, w, salida / "iou_por_frame.png")
    return df, dpf


def experimento_velocidad(salida: Path) -> pd.DataFrame:
    """Dominio de la frecuencia: velocidad estimada por correlación de fase vs real."""
    v = esc.escenario_movimiento()
    m, _ = mov.diferencia_referencia(v.frames)
    filas = []
    for t in range(v.calentamiento, len(v.frames) - 1):
        for est in mov.velocidad_correlacion_fase(v.frames, m, t):
            cx, cy = (int(round(c)) for c in est["centro"])
            for k, (gt_obj, vel) in enumerate(zip(v.gt_objetos, v.velocidades)):
                if gt_obj[t][min(cy, gt_obj.shape[1] - 1), min(cx, gt_obj.shape[2] - 1)]:
                    filas.append({"frame": t, "objeto": k + 1, "dx": est["dx"], "dy": est["dy"],
                                  "dx_real": vel[0], "dy_real": vel[1], "respuesta": est["respuesta"],
                                  "epe": error_velocidad((est["dx"], est["dy"]), vel)})
    df = pd.DataFrame(filas)
    df.to_csv(salida / "velocidad_correlacion_fase.csv", index=False)
    return df


def estudio_robustez(salida: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Degrada la escena de forma controlada y mide cómo cae el IoU."""
    filas_r, filas_i = [], []
    for sigma in [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.20]:
        img, gt = esc.escena_gris(ruido=sigma, iluminacion=0.0)
        img = a_rgb(img)
        for t in TECNICAS_ESTATICAS:
            m, _ = t.funcion(img)
            filas_r.append({"tecnica": t.nombre, "ruido": sigma, "iou": evaluar(m, gt).iou})
        print(f"  robustez ruido σ={sigma:.2f} listo")
    for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        img, gt = esc.escena_gris(ruido=0.04, iluminacion=s)
        img = a_rgb(img)
        for t in TECNICAS_ESTATICAS:
            m, _ = t.funcion(img)
            filas_i.append({"tecnica": t.nombre, "iluminacion": s, "iou": evaluar(m, gt).iou})
        print(f"  robustez iluminación={s:.1f} listo")
    dr, di = pd.DataFrame(filas_r), pd.DataFrame(filas_i)
    dr.to_csv(salida / "robustez_ruido.csv", index=False)
    di.to_csv(salida / "robustez_iluminacion.csv", index=False)
    destacar = ["Umbral óptimo de Otsu", "Umbral adaptativo (local)", "División y fusión mejorada",
                "Crecimiento de regiones"]
    graficos.curvas_robustez(dr, "ruido", "Desvío del ruido gaussiano σ", destacar,
                             salida / "robustez_ruido.png", "Robustez al ruido (escena E1)")
    graficos.curvas_robustez(di, "iluminacion", "Intensidad de la no uniformidad de iluminación (0 = pareja)",
                             destacar, salida / "robustez_iluminacion.png",
                             "Robustez a la iluminación no uniforme")
    return dr, di


def ablaciones(salida: Path) -> pd.DataFrame:
    """Mide el efecto de cada aporte propio por separado (encendido/apagado)."""
    filas = []
    e1, e2 = esc.escenario_uniforme(), esc.escenario_iluminacion()
    variantes = [
        ("División-fusión · libro (media, sin refinar)", dict(criterio="media", refinar=False)),
        ("+ refinamiento a nivel píxel", dict(criterio="media", refinar=True)),
        ("+ criterio de frontera (sin refinar)", dict(criterio="frontera", refinar=False)),
        ("División-fusión · completa (frontera + refinamiento)", dict(criterio="frontera", refinar=True)),
    ]
    for nombre, kw in variantes:
        for e in (e1, e2):
            (m, info), ms = cronometrar(reg.division_fusion, e.imagen, repeticiones=3, **kw)
            filas.append({"aporte": "División y fusión", "variante": nombre, "escenario": e.nombre,
                          "iou": evaluar(m, e.gt).iou, "regiones": info["n_regiones"], "tiempo_ms": ms})
    ruidosa, gt_r = esc.escena_gris(ruido=0.15)
    for nombre, s in (("suavizado fijo σ=1", 1.0), ("suavizado adaptativo al ruido", "auto")):
        m, _ = reg.division_fusion(a_rgb(ruidosa), suavizado=s)
        filas.append({"aporte": "División y fusión", "variante": nombre, "escenario": "E1 con ruido σ=0.15",
                      "iou": evaluar(m, gt_r).iou, "regiones": np.nan, "tiempo_ms": np.nan})
    from .tecnicas import umbralizacion as umb
    for modo in ("resta", "division"):
        m, _ = umb.tophat_otsu(e2.imagen, modo=modo)
        filas.append({"aporte": "Corrección de fondo", "variante": f"modo = {modo}", "escenario": e2.nombre,
                      "iou": evaluar(m, e2.gt).iou, "regiones": np.nan, "tiempo_ms": np.nan})
    v = esc.escenario_movimiento()
    w = v.calentamiento
    for nombre, f in [("MOG2 · inicio en frío", lambda fr: mov.mog2(fr)),
                      ("MOG2 · inicializado con mediana", lambda fr: mov.mog2(fr, inicializar_con_mediana=True)),
                      ("Dif. consecutiva · sin compensar brillo", lambda fr: mov.diferencia_consecutiva(fr, compensar=False)),
                      ("Dif. consecutiva · compensando brillo", lambda fr: mov.diferencia_consecutiva(fr)),
                      ("FFT temporal · sin compensar brillo", lambda fr: mov.fft_temporal(fr, compensar=False)),
                      ("FFT temporal · compensando brillo", lambda fr: mov.fft_temporal(fr))]:
        m, _ = f(v.frames)
        filas.append({"aporte": "Movimiento", "variante": nombre, "escenario": v.nombre,
                      "iou": float(np.mean([evaluar(m[i], v.gt[i]).iou for i in range(w, len(m))])),
                      "regiones": np.nan, "tiempo_ms": np.nan})
    df = pd.DataFrame(filas)
    df.to_csv(salida / "ablaciones.csv", index=False)
    return df


def _tabla_original(do: pd.DataFrame) -> str:
    filas = ["| Algoritmo | Escenario | IoU original | IoU mejorado | IoU técnica nueva | ms original | ms mejorado | Aceleración |",
             "|---|---|---|---|---|---|---|---|"]
    for r in do.itertuples(index=False):
        def iou(v, err):
            return f"falla ({err})" if err else f"{v:.3f}"
        nueva = "—" if not r.tecnica_nueva else f"{r.iou_nueva:.3f} ({r.tecnica_nueva})"
        acel = "—" if not np.isfinite(r.aceleracion) else f"{r.aceleracion:.0f}×"
        ms_o = "—" if not np.isfinite(r.ms_original) else f"{r.ms_original:.1f}"
        ms_m = "—" if not np.isfinite(r.ms_mejorado) else f"{r.ms_mejorado:.1f}"
        filas.append(f"| {r.par} | {r.escenario} | {iou(r.iou_original, r.error_original)} | "
                     f"{iou(r.iou_mejorado, r.error_mejorado)} | {nueva} | {ms_o} | {ms_m} | {acel} |")
    return "\n".join(filas)


def escribir_resumen(salida: Path, de, dm, dv, da, dr=None, di=None, do=None):
    lineas = ["# Resultados del banco de pruebas", "",
              f"Generado: {time.strftime('%Y-%m-%d %H:%M')}. Todos los números se reproducen con "
              "`python -m segpdi.benchmark` (semilla fija).", "",
              "Métricas: IoU (principal), Dice, precisión, recall y tiempo. Definiciones en `segpdi/metricas.py`.", ""]
    if do is not None:
        lineas += ["## 0. Código original vs. mejorado (comparación 1:1)", "",
                   "*Original* = carpeta `original/` sin modificar. *Mejorado* = mismo algoritmo corregido "
                   "(raíz del repo). *Técnica nueva* = método de `segpdi` que lo reemplaza. Sin limpieza "
                   "morfológica en original ni mejorado.", "", _tabla_original(do), ""]
    lineas += ["## 1. Escenarios estáticos (E1–E3) · IoU", ""]
    piv = de.pivot(index="tecnica", columns="escenario", values="iou").loc[de.drop_duplicates("tecnica")["tecnica"]]
    piv["promedio"] = piv.mean(axis=1)
    piv = piv.reset_index()
    lineas += [_fmt(piv), ""]
    lineas += ["## 2. Tiempo de ejecución (ms, mediana de 3 corridas, escenario E1)", ""]
    t = de[de.escenario == de.escenario.iloc[0]][["tecnica", "tiempo_ms", "precision", "recall"]]
    lineas += [_fmt(t), ""]
    lineas += ["## 3. Movimiento (E4)", ""]
    lineas += [_fmt(dm, ["tecnica", "iou_medio", "iou_min", "precision", "recall", "ms_por_frame"]), ""]
    if len(dv):
        resumen_v = dv.groupby("objeto").agg(dx_medio=("dx", "mean"), dy_medio=("dy", "mean"),
                                             dx_real=("dx_real", "first"), dy_real=("dy_real", "first"),
                                             epe_medio=("epe", "mean"), epe_mediano=("epe", "median")).reset_index()
        lineas += ["### Velocidad por correlación de fase (dominio de la frecuencia)", "",
                   "EPE = error de punto final en píxeles/frame.", "", _fmt(resumen_v), ""]
    lineas += ["## 4. Ablaciones (efecto de cada aporte propio)", "", _fmt(da, ["aporte", "variante", "escenario", "iou"]), ""]
    if dr is not None:
        lineas += ["## 5. Robustez", "", "Ver `robustez_ruido.png` y `robustez_iluminacion.png`.", ""]
        for nombre, d, var in (("ruido σ", dr, "ruido"), ("iluminación", di, "iluminacion")):
            p = d.pivot(index="tecnica", columns=var, values="iou").loc[d.drop_duplicates("tecnica")["tecnica"]]
            p.columns = [f"{nombre}={c}" for c in p.columns]
            lineas += [_fmt(p.reset_index()), ""]
    (salida / "RESUMEN.md").write_text("\n".join(lineas), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Banco de pruebas de segmentación")
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--rapido", action="store_true", help="omite el estudio de robustez")
    args = ap.parse_args(argv)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("[1/6] Original vs. mejorado"); do = original_vs_mejorado(salida)
    print("[2/6] Escenarios estáticos"); de = experimento_estatico(salida)
    print("[3/6] Movimiento"); dm, _ = experimento_movimiento(salida)
    print("[4/6] Velocidad (correlación de fase)"); dv = experimento_velocidad(salida)
    print("[5/6] Ablaciones"); da = ablaciones(salida)
    dr = di = None
    if not args.rapido:
        print("[6/6] Robustez"); dr, di = estudio_robustez(salida)
    escribir_resumen(salida, de, dm, dv, da, dr, di, do)
    print(f"Listo en {time.perf_counter() - t0:.1f} s → {salida.resolve()}")


if __name__ == "__main__":
    main()
