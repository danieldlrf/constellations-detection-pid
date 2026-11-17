import glob
import os
import re
import shutil
import sys
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocesado.preprocessing import preprocessing, EstrellasInsuficientesError

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

# Imágenes que quieres probar
IMAGENES = [
    "data/processed/testing/aquila/aquila_test_img9_ann30.jpg",
    "data/processed/testing/bootes/bootes_test_img9_ann28.jpg",
    "data/processed/testing/canis_major/canis_major_test_img13_ann48.jpg",
    "data/processed/testing/gemini/gemini_test_img13_ann47.jpg"
]

# Carpeta donde se guardan las imágenes de cada paso (una subcarpeta por foto)
CARPETA = "test/preprocessing"

# Parámetros de preprocessing
AREA_MIN = 2            # área mínima (px) para aceptar una componente como estrella
AREA_MAX = 150          # área máxima (px) permitida para una estrella
MAX_ITER = 50           # máximo de iteraciones al bajar el umbral
N_MIN = 3               # estrellas mínimas; si no se llega, lanza error
N_MAX = 15              # estrellas máximas devueltas (las más brillantes)
UMBRAL_INICIAL = 225    # umbral inicial (0-255) sobre la imagen del top-hat
PASO = 5                # cuánto baja el umbral en cada iteración
TAM_SE = None           # tamaño impar del elemento estructurante (None = automático desde AREA_MAX)
RATIO_MAX = 2.0         # relación máxima lado mayor / lado menor de la caja
RELLENO_MIN = 0.5       # relleno mínimo (área / área de la caja)
AREA_FORMA_MIN = 9      # área mínima a partir de la cual se aplican los filtros de forma
UMBRAL_MIN = 20         # umbral mínimo al que se puede llegar

COLUMNAS = ["Gris", "Mediana 3×3", "Top-hat (contraste ajustado)", "Binarización final"]

# ---------------------------------------------------------------------------


def numero(ruta): 
    return int(re.search(r"binaria(\d+)", ruta).group(1))


filas = []
for ruta in IMAGENES:
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    carpeta = os.path.join(CARPETA, nombre)
    shutil.rmtree(carpeta, ignore_errors=True) 
    try:
        preprocessing(ruta, carpeta,
                      area_min=AREA_MIN, area_max=AREA_MAX, max_iter=MAX_ITER,
                      n_min=N_MIN, n_max=N_MAX, umbral_inicial=UMBRAL_INICIAL,
                      paso=PASO, tam_se=TAM_SE, ratio_max=RATIO_MAX,
                      relleno_min=RELLENO_MIN, area_forma_min=AREA_FORMA_MIN,
                      umbral_min=UMBRAL_MIN, testing=True)
    except EstrellasInsuficientesError as e:
        print("FALLA:", e)
    binarias = sorted(glob.glob(os.path.join(carpeta, "binaria*.png")), key=numero)
    filas.append((nombre, [os.path.join(carpeta, "im_gris.png"),
                           os.path.join(carpeta, "im_median.png"),
                           os.path.join(carpeta, "im_tophat_visible.png"),
                           binarias[-1]]))

fig, ejes = plt.subplots(len(filas), len(COLUMNAS), figsize=(16, 3.5 * len(filas)), squeeze=False)
for f, (nombre, archivos) in enumerate(filas):
    for c, archivo in enumerate(archivos):
        eje = ejes[f][c]
        eje.imshow(cv.imread(archivo, 0), cmap="gray", vmin=0, vmax=255)
        eje.set_xticks([])
        eje.set_yticks([])
        if f == 0:
            eje.set_title(COLUMNAS[c])
    ejes[f][0].set_ylabel(nombre)

plt.tight_layout()
plt.savefig("test/tabla_comparativa.png", dpi=200)
plt.show()