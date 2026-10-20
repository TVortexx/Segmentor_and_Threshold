"""Figuras estáticas (PNG) para el informe y la presentación."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .utils import comparar_con_gt  # noqa: E402

# Paleta categórica validada (orden fijo, nunca se recicla).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRIS = "#b9b8b3"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
SUPERFICIE = "#fcfcfb"


def _estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": "#d6d5d0", "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2,
        "ytick.color": TEXTO_2, "text.color": TEXTO, "axes.grid": True, "grid.color": "#ebeae6",
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    })


def mosaico_escenario(escenario, mascaras: dict[str, np.ndarray], metricas: dict[str, float], ruta):
    """Imagen, GT y el mapa de errores de cada técnica con su IoU."""
    _estilo()
    n = len(mascaras) + 2
    cols = 5
    filas = int(np.ceil(n / cols))
    fig, ejes = plt.subplots(filas, cols, figsize=(cols * 2.6, filas * 3.25))
    ejes = np.atleast_1d(ejes).ravel()
    ejes[0].imshow(escenario.imagen)
    ejes[0].set_title("Entrada", fontsize=10)
    ejes[1].imshow(escenario.gt, cmap="gray")
    ejes[1].set_title("Ground truth", fontsize=10)
    for ax, (nombre, m) in zip(ejes[2:], mascaras.items()):
        ax.imshow(comparar_con_gt(m, escenario.gt))
        ax.set_title(f"{nombre}\nIoU = {metricas[nombre]:.3f}", fontsize=8.5)
    for ax in ejes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in ejes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{escenario.nombre}   ·   verde = acierto, rojo = falso positivo, azul = falso negativo",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(h_pad=2.2)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def barras_iou(df: pd.DataFrame, ruta, titulo="IoU por técnica y escenario"):
    """Barras horizontales agrupadas: técnicas en el eje, escenarios como series."""
    _estilo()
    tabla = df.pivot(index="tecnica", columns="escenario", values="iou")
    orden = df.drop_duplicates("tecnica")["tecnica"].tolist()
    tabla = tabla.loc[orden[::-1]]
    escenarios = df.drop_duplicates("escenario")["escenario"].tolist()
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(tabla) + 1.6))
    alto = 0.8 / len(escenarios)
    y = np.arange(len(tabla))
    for i, esc in enumerate(escenarios):
        ax.barh(y - (i - (len(escenarios) - 1) / 2) * alto, tabla[esc], height=alto * 0.86,
                color=SERIES[i], label=esc)
    ax.set_yticks(y, tabla.index)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("IoU (más alto es mejor)")
    ax.grid(axis="y", visible=False)
    ax.set_title(titulo, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=len(escenarios), fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def curvas_robustez(df: pd.DataFrame, variable: str, etiqueta_x: str, destacar: list[str], ruta, titulo):
    """IoU vs. variable de degradación. Las técnicas en `destacar` llevan color
    y etiqueta directa; el resto queda en gris como contexto."""
    _estilo()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for tec, sub in df.groupby("tecnica", sort=False):
        if tec in destacar:
            continue
        ax.plot(sub[variable], sub["iou"], color=GRIS, lw=1.2, alpha=0.8)
    for i, tec in enumerate(destacar):
        sub = df[df.tecnica == tec]
        if sub.empty:
            continue
        ax.plot(sub[variable], sub["iou"], color=SERIES[i], lw=2.2, marker="o", ms=4, label=tec)
    ax.plot([], [], color=GRIS, lw=1.2, label="otras técnicas")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel("IoU")
    ax.set_title(titulo, loc="left")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def mosaico_video(video, mascaras: dict[str, np.ndarray], iou_medio: dict[str, float], t: int, ruta):
    _estilo()
    n = len(mascaras) + 2
    fig, ejes = plt.subplots(1, n, figsize=(n * 2.4, 2.9))
    ejes[0].imshow(video.frames[t], cmap="gray", vmin=0, vmax=1)
    ejes[0].set_title(f"Frame {t}", fontsize=9)
    ejes[1].imshow(video.gt[t], cmap="gray")
    ejes[1].set_title("Ground truth", fontsize=9)
    for ax, (nombre, m) in zip(ejes[2:], mascaras.items()):
        ax.imshow(comparar_con_gt(m[t], video.gt[t]))
        ax.set_title(f"{nombre}\nIoU medio = {iou_medio[nombre]:.3f}", fontsize=8)
    for ax in ejes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(video.nombre, fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def iou_por_frame(df: pd.DataFrame, calentamiento: int, ruta):
    _estilo()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, (tec, sub) in enumerate(df.groupby("tecnica", sort=False)):
        ax.plot(sub["frame"], sub["iou"], color=SERIES[i % len(SERIES)], lw=2, label=tec)
    ax.axvspan(-0.5, calentamiento - 0.5, color="#ebeae6", zorder=0)
    ax.text(calentamiento / 2 - 0.5, 0.03, "calentamiento\n(no se evalúa)", ha="center", fontsize=8,
            color=TEXTO_2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Frame")
    ax.set_ylabel("IoU")
    ax.set_title("IoU por frame · segmentación por movimiento", loc="left")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def tiempos_original_mejorado(df: pd.DataFrame, ruta):
    """Tiempo (escala log) del código original vs. su versión corregida,
    promedio sobre los escenarios, por algoritmo."""
    _estilo()
    d = df.dropna(subset=["ms_original", "ms_mejorado"]).groupby("par", sort=False)[["ms_original", "ms_mejorado"]].mean()
    d = d.iloc[::-1]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(d) + 1.6))
    ax.barh(y + 0.2, d["ms_original"], height=0.36, color=SERIES[1], label="Original (sin modificar)")
    ax.barh(y - 0.2, d["ms_mejorado"], height=0.36, color=SERIES[0], label="Mejorado")
    for yi, (o, m) in zip(y, d.itertuples(index=False)):
        if m > 0:
            ax.text(max(o, m) * 1.15, yi, f"{o / m:.0f}× más rápido" if o / m >= 1.5 else "≈ igual",
                    va="center", fontsize=8.5, color=TEXTO_2)
    ax.set_xscale("log")
    ax.set_yticks(y, d.index)
    ax.set_xlabel("Tiempo por imagen 256×256 (ms, escala logarítmica)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=ax.get_xlim()[1] * 12)
    ax.set_title("Código original vs. mejorado · tiempo de ejecución", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
