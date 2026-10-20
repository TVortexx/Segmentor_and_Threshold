# Código original (sin modificar)

Esta carpeta es una copia **exacta** de los archivos `.py` y `.ui` del repositorio
[Computer-Vision-Toolkit/Segmentor_and_Threshold](https://github.com/Computer-Vision-Toolkit/Segmentor_and_Threshold)
en el commit `7ef53e7` (tag `base-original`), de los autores Ibrahim Fateen, Bassant Zaki,
Mohamed Hisham y Yasmine Mahmoud.

Se conserva para comparar 1:1 con las versiones mejoradas de la raíz del repositorio:

| Original (esta carpeta) | Mejorado (raíz) |
|---|---|
| `original/OtsuAndOptimal.py` | `OtsuAndOptimal.py` |
| `original/RegionGrowingSegmentation.py` | `RegionGrowingSegmentation.py` + `region_growing_core.py` |
| `original/KMeanSegmentation.py` | `KMeanSegmentation.py` |
| `original/spectralThresholding.py` | `spectralThresholding.py` |
| `original/main.py` | `main.py` |

Para abrir la interfaz original tal como era:

```bash
cd original
python main.py
```

Verificación de que no se modificó: `git diff base-original -- OtsuAndOptimal.py` compara
la raíz con el original, y `git show base-original:OtsuAndOptimal.py | diff - original/OtsuAndOptimal.py`
no debe mostrar diferencias.
