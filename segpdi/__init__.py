"""segpdi: banco de pruebas de segmentación de imágenes.

Extensión del proyecto Segmentor_and_Threshold (Computer-Vision-Toolkit)
para Procesamiento Digital de Imágenes - Facultad Politécnica, UNE.

Módulos:
    tecnicas.discontinuidades  Detección de puntos, líneas, bordes y combinada.
    tecnicas.umbralizacion     Umbral global iterativo, Otsu, adaptativo, top-hat + Otsu.
    tecnicas.multivariable     Umbralización en varias variables (k-means en color Lab).
    tecnicas.regiones          Crecimiento de regiones y división-fusión (quadtree).
    tecnicas.movimiento        Diferencia de frames, MOG2, FFT temporal, correlación de fase.
    escenarios                 Escenarios sintéticos con ground truth exacto.
    metricas                   IoU, Dice, precisión, recall, tiempo.
    registro                   Catálogo único de técnicas usado por el benchmark y la app.
    originales                 Ejecuta los algoritmos ORIGINALES (carpeta original/) y sus
                               versiones mejoradas (raíz) con la misma interfaz, para
                               compararlos 1:1.
"""

__version__ = "1.0.0"
