# Resultados del banco de pruebas

Generado: 2026-10-19 23:06. Todos los números se reproducen con `python -m segpdi.benchmark` (semilla fija).

Métricas: IoU (principal), Dice, precisión, recall y tiempo. Definiciones en `segpdi/metricas.py`.

## 0. Código original vs. mejorado (comparación 1:1)

*Original* = carpeta `original/` sin modificar. *Mejorado* = mismo algoritmo corregido (raíz del repo). *Técnica nueva* = método de `segpdi` que lo reemplaza. Sin limpieza morfológica en original ni mejorado.

| Algoritmo | Escenario | IoU original | IoU mejorado | IoU técnica nueva | ms original | ms mejorado | Aceleración |
|---|---|---|---|---|---|---|---|
| Otsu global | E1 · Iluminación uniforme | 0.990 | 0.990 | 0.982 (Corrección de fondo morfológica + Otsu) | 5.4 | 2.4 | 2× |
| Otsu global | E2 · Iluminación no uniforme | 0.708 | 0.706 | 0.978 (Corrección de fondo morfológica + Otsu) | 4.9 | 3.0 | 2× |
| Otsu global | E3 · Color isoluminante | 0.233 | 0.221 | 0.237 (Corrección de fondo morfológica + Otsu) | 4.5 | 2.2 | 2× |
| Umbral óptimo iterativo | E1 · Iluminación uniforme | 0.990 | 0.990 | 0.991 (Umbral adaptativo (local)) | 4.4 | 2.2 | 2× |
| Umbral óptimo iterativo | E2 · Iluminación no uniforme | 0.701 | 0.701 | 0.986 (Umbral adaptativo (local)) | 5.4 | 3.5 | 2× |
| Umbral óptimo iterativo | E3 · Color isoluminante | 0.208 | 0.208 | 0.000 (Umbral adaptativo (local)) | 8.1 | 6.5 | 1× |
| Otsu local por bloques | E1 · Iluminación uniforme | 0.340 | 0.342 | 0.991 (Umbral adaptativo (local)) | 184.9 | 75.3 | 2× |
| Otsu local por bloques | E2 · Iluminación no uniforme | 0.333 | 0.333 | 0.986 (Umbral adaptativo (local)) | 218.3 | 61.9 | 4× |
| Otsu local por bloques | E3 · Color isoluminante | 0.229 | 0.217 | 0.000 (Umbral adaptativo (local)) | 168.4 | 44.4 | 4× |
| Umbral espectral multinivel | E1 · Iluminación uniforme | 0.990 | 0.990 | 0.982 (Corrección de fondo morfológica + Otsu) | 499.8 | 5.1 | 97× |
| Umbral espectral multinivel | E2 · Iluminación no uniforme | 0.526 | 0.526 | 0.978 (Corrección de fondo morfológica + Otsu) | 569.2 | 5.9 | 96× |
| Umbral espectral multinivel | E3 · Color isoluminante | 0.242 | 0.242 | 0.237 (Corrección de fondo morfológica + Otsu) | 207.1 | 6.3 | 33× |
| K-means RGB | E1 · Iluminación uniforme | 0.479 | 0.438 | 0.921 (K-means en color (Lab)) | 3000.2 | 116.3 | 26× |
| K-means RGB | E2 · Iluminación no uniforme | 0.438 | 0.415 | 0.439 (K-means en color (Lab)) | 6412.3 | 171.3 | 37× |
| K-means RGB | E3 · Color isoluminante | 0.996 | 0.996 | 0.995 (K-means en color (Lab)) | 4661.6 | 68.8 | 68× |
| Crecimiento de regiones | E1 · Iluminación uniforme | 0.986 | 0.986 | 0.965 (Crecimiento de regiones) | 144.5 | 26.6 | 5× |
| Crecimiento de regiones | E2 · Iluminación no uniforme | 0.397 | 0.397 | 0.943 (Crecimiento de regiones) | 470.1 | 39.8 | 12× |
| Crecimiento de regiones | E3 · Color isoluminante | 0.276 | 0.276 | 0.276 (Crecimiento de regiones) | 370.7 | 20.7 | 18× |
| Crecimiento de regiones (289 semillas) | E1 · Iluminación uniforme | falla (OverflowError) | 0.276 | — | — | 104.7 | — |

## 1. Escenarios estáticos (E1–E3) · IoU

| tecnica | E1 · Iluminación uniforme | E2 · Iluminación no uniforme | E3 · Color isoluminante | promedio |
|---|---|---|---|---|
| Bordes · Canny | 0.980 | 0.979 | 0.015 | 0.658 |
| Bordes · Marr-Hildreth (LoG) | 0.977 | 0.972 | 0.199 | 0.716 |
| Umbral global iterativo | 0.982 | 0.696 | 0.208 | 0.629 |
| Umbral óptimo de Otsu | 0.982 | 0.696 | 0.224 | 0.634 |
| Umbral adaptativo (local) | 0.991 | 0.986 | 0.000 | 0.659 |
| Corrección de fondo morfológica + Otsu | 0.982 | 0.978 | 0.237 | 0.732 |
| K-means en color (Lab) | 0.921 | 0.439 | 0.995 | 0.785 |
| Distancia de Mahalanobis al fondo | 0.851 | 0.522 | 0.856 | 0.743 |
| Crecimiento de regiones | 0.965 | 0.943 | 0.276 | 0.728 |
| División y fusión (versión libro) | 0.888 | 0.372 | 0.000 | 0.420 |
| División y fusión mejorada | 0.976 | 0.961 | 0.000 | 0.646 |

## 2. Tiempo de ejecución (ms, mediana de 3 corridas, escenario E1)

| tecnica | tiempo_ms | precision | recall |
|---|---|---|---|
| Bordes · Canny | 16.272 | 0.982 | 0.998 |
| Bordes · Marr-Hildreth (LoG) | 18.368 | 0.982 | 0.995 |
| Umbral global iterativo | 5.864 | 0.995 | 0.987 |
| Umbral óptimo de Otsu | 6.075 | 0.995 | 0.987 |
| Umbral adaptativo (local) | 14.244 | 0.991 | 1.000 |
| Corrección de fondo morfológica + Otsu | 18.281 | 0.994 | 0.988 |
| K-means en color (Lab) | 37.622 | 0.921 | 1.000 |
| Distancia de Mahalanobis al fondo | 20.255 | 0.851 | 1.000 |
| Crecimiento de regiones | 91.924 | 1.000 | 0.965 |
| División y fusión (versión libro) | 98.695 | 0.888 | 1.000 |
| División y fusión mejorada | 76.084 | 0.991 | 0.985 |

## 3. Movimiento (E4)

| tecnica | iou_medio | iou_min | precision | recall | ms_por_frame |
|---|---|---|---|---|---|
| Diferencia de frames consecutivos | 0.586 | 0.216 | 0.743 | 0.736 | 7.894 |
| Diferencia con referencia (mediana) | 0.967 | 0.953 | 0.991 | 0.976 | 9.339 |
| MOG2 (inicio en frío) | 0.596 | 0.376 | 0.572 | 0.966 | 6.975 |
| MOG2 inicializado con mediana | 0.991 | 0.967 | 0.999 | 0.991 | 13.938 |
| FFT temporal (energía AC) | 0.667 | 0.649 | 0.667 | 0.999 | 10.515 |

### Velocidad por correlación de fase (dominio de la frecuencia)

EPE = error de punto final en píxeles/frame.

| objeto | dx_medio | dy_medio | dx_real | dy_real | epe_medio | epe_mediano |
|---|---|---|---|---|---|---|
| 1 | 3.924 | 0.993 | 4.000 | 1.000 | 0.140 | 0.145 |
| 2 | -2.844 | -1.972 | -3.000 | -2.000 | 0.291 | 0.271 |

## 4. Ablaciones (efecto de cada aporte propio)

| aporte | variante | escenario | iou |
|---|---|---|---|
| División y fusión | División-fusión · libro (media, sin refinar) | E1 · Iluminación uniforme | 0.888 |
| División y fusión | División-fusión · libro (media, sin refinar) | E2 · Iluminación no uniforme | 0.372 |
| División y fusión | + refinamiento a nivel píxel | E1 · Iluminación uniforme | 0.935 |
| División y fusión | + refinamiento a nivel píxel | E2 · Iluminación no uniforme | 0.374 |
| División y fusión | + criterio de frontera (sin refinar) | E1 · Iluminación uniforme | 0.899 |
| División y fusión | + criterio de frontera (sin refinar) | E2 · Iluminación no uniforme | 0.921 |
| División y fusión | División-fusión · completa (frontera + refinamiento) | E1 · Iluminación uniforme | 0.976 |
| División y fusión | División-fusión · completa (frontera + refinamiento) | E2 · Iluminación no uniforme | 0.961 |
| División y fusión | suavizado fijo σ=1 | E1 con ruido σ=0.15 | 0.501 |
| División y fusión | suavizado adaptativo al ruido | E1 con ruido σ=0.15 | 0.963 |
| Corrección de fondo | modo = resta | E2 · Iluminación no uniforme | 0.903 |
| Corrección de fondo | modo = division | E2 · Iluminación no uniforme | 0.978 |
| Movimiento | MOG2 · inicio en frío | E4 · Objetos en movimiento | 0.596 |
| Movimiento | MOG2 · inicializado con mediana | E4 · Objetos en movimiento | 0.991 |
| Movimiento | Dif. consecutiva · sin compensar brillo | E4 · Objetos en movimiento | 0.384 |
| Movimiento | Dif. consecutiva · compensando brillo | E4 · Objetos en movimiento | 0.586 |
| Movimiento | FFT temporal · sin compensar brillo | E4 · Objetos en movimiento | 0.585 |
| Movimiento | FFT temporal · compensando brillo | E4 · Objetos en movimiento | 0.667 |

## 5. Robustez

Ver `robustez_ruido.png` y `robustez_iluminacion.png`.

| tecnica | ruido σ=0.0 | ruido σ=0.03 | ruido σ=0.06 | ruido σ=0.09 | ruido σ=0.12 | ruido σ=0.15 | ruido σ=0.2 |
|---|---|---|---|---|---|---|---|
| Bordes · Canny | 0.982 | 0.980 | 0.980 | 0.979 | 0.978 | 0.978 | 0.976 |
| Bordes · Marr-Hildreth (LoG) | 0.977 | 0.977 | 0.972 | 0.502 | 0.375 | 0.417 | 0.417 |
| Umbral global iterativo | 0.983 | 0.982 | 0.980 | 0.978 | 0.976 | 0.974 | 0.969 |
| Umbral óptimo de Otsu | 0.983 | 0.982 | 0.980 | 0.978 | 0.977 | 0.974 | 0.969 |
| Umbral adaptativo (local) | 0.991 | 0.991 | 0.989 | 0.979 | 0.964 | 0.948 | 0.918 |
| Corrección de fondo morfológica + Otsu | 0.983 | 0.982 | 0.981 | 0.980 | 0.981 | 0.978 | 0.971 |
| K-means en color (Lab) | 0.922 | 0.921 | 0.919 | 0.917 | 0.915 | 0.910 | 0.543 |
| Distancia de Mahalanobis al fondo | 0.701 | 0.840 | 0.869 | 0.889 | 0.906 | 0.919 | 0.940 |
| Crecimiento de regiones | 0.972 | 0.967 | 0.950 | 0.952 | 0.953 | 0.955 | 0.276 |
| División y fusión (versión libro) | 0.851 | 0.870 | 0.903 | 0.913 | 0.921 | 0.918 | 0.894 |
| División y fusión mejorada | 0.964 | 0.975 | 0.977 | 0.974 | 0.964 | 0.963 | 0.963 |

| tecnica | iluminación=0.0 | iluminación=0.2 | iluminación=0.4 | iluminación=0.6 | iluminación=0.8 | iluminación=1.0 |
|---|---|---|---|---|---|---|
| Bordes · Canny | 0.980 | 0.980 | 0.979 | 0.980 | 0.980 | 0.980 |
| Bordes · Marr-Hildreth (LoG) | 0.977 | 0.977 | 0.977 | 0.977 | 0.977 | 0.977 |
| Umbral global iterativo | 0.982 | 0.990 | 0.983 | 0.969 | 0.878 | 0.699 |
| Umbral óptimo de Otsu | 0.982 | 0.989 | 0.983 | 0.969 | 0.881 | 0.699 |
| Umbral adaptativo (local) | 0.991 | 0.992 | 0.992 | 0.992 | 0.991 | 0.990 |
| Corrección de fondo morfológica + Otsu | 0.982 | 0.983 | 0.983 | 0.984 | 0.984 | 0.983 |
| K-means en color (Lab) | 0.921 | 0.922 | 0.459 | 0.448 | 0.447 | 0.439 |
| Distancia de Mahalanobis al fondo | 0.851 | 0.900 | 0.951 | 0.959 | 0.693 | 0.521 |
| Crecimiento de regiones | 0.965 | 0.944 | 0.947 | 0.947 | 0.942 | 0.942 |
| División y fusión (versión libro) | 0.888 | 0.878 | 0.438 | 0.438 | 0.378 | 0.370 |
| División y fusión mejorada | 0.976 | 0.976 | 0.975 | 0.976 | 0.974 | 0.969 |
