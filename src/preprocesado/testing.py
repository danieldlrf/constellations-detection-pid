import glob
import os
import re
import shutil
import sys
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocesado.articulo_preprocessing import (
    EstrellasInsuficientesError as ArticuloEstrellasInsuficientesError,
    preprocessing_articulo,
)
from preprocesado.preprocessing import EstrellasInsuficientesError, preprocessing

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

# Imágenes que quieres probar
IMAGENES = [
    "data/processed/testing/aquila/aquila_test_img9_ann30.jpg",
    "data/processed/testing/bootes/bootes_test_img9_ann28.jpg",
    "data/processed/testing/canis_major/canis_major_test_img13_ann48.jpg",
    "data/processed/testing/gemini/gemini_test_img13_ann47.jpg",
]

# Carpeta donde se guardan las imágenes de cada paso (una subcarpeta por foto)
CARPETA = "test/preprocessing"

# Parámetros de nuestro preprocessing
AREA_MIN = 2            # área mínima (px) para aceptar una componente como estrella
AREA_MAX = 150          # área máxima (px) permitida para una estrella
MAX_ITER = 50           # máximo de iteraciones al bajar el umbral
N_MIN = 5               # estrellas mínimas; si no se llega, lanza error
N_MAX = 12              # estrellas máximas devueltas (las más brillantes)
UMBRAL_INICIAL = 35    # umbral inicial (0-255) sobre la imagen del top-hat
PASO = 5                # cuánto baja el umbral en cada iteración
TAM_SE = None           # tamaño impar del elemento estructurante (None = automático desde AREA_MAX)
RATIO_MAX = 2.0         # relación máxima lado mayor / lado menor de la caja
RELLENO_MIN = 0.5       # relleno mínimo (área / área de la caja)
AREA_FORMA_MIN = 9      # área mínima a partir de la cual se aplican los filtros de forma
UMBRAL_MIN = 20         # umbral mínimo al que se puede llegar

# Parámetros de preprocessing_articulo
ART_AREA_MIN = 1            # área mínima del contorno
ART_AREA_MAX = 150          # área máxima del contorno
ART_MAX_ITER = 100          # máximo de iteraciones ajustando umbral
ART_UMBRAL_INICIAL = None   # None = automático con Otsu
ART_PASO = 5                # incremento/decremento de umbral por iteración
ART_UMBRAL_MIN = 5          # límite inferior de umbral
ART_UMBRAL_MAX = 250        # límite superior de umbral

COLUMNAS = [
    "Gris",
    "Mediana 3×3",
    "Top-hat (contraste ajustado)",
    "Binaria (Nuestro)",
    "Binaria (Artículo)",
    "Detecciones (Nuestro vs Art.)",
]

# ---------------------------------------------------------------------------


def numero(ruta):
    return int(re.search(r"binaria(\d+)", os.path.basename(ruta)).group(1))


filas = []
for ruta in IMAGENES:
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    carpeta = os.path.join(CARPETA, nombre)
    shutil.rmtree(carpeta, ignore_errors=True)

    # 1. Nuestro preprocesado
    try:
        estrellas_nuestro = preprocessing(
            ruta,
            carpeta,
            area_min=AREA_MIN,
            area_max=AREA_MAX,
            max_iter=MAX_ITER,
            n_min=N_MIN,
            n_max=N_MAX,
            umbral_inicial=UMBRAL_INICIAL,
            paso=PASO,
            tam_se=TAM_SE,
            ratio_max=RATIO_MAX,
            relleno_min=RELLENO_MIN,
            area_forma_min=AREA_FORMA_MIN,
            umbral_min=UMBRAL_MIN,
            testing=True,
        )
    except EstrellasInsuficientesError as e:
        print("FALLA (Nuestro):", e)
        estrellas_nuestro = np.empty((0, 3))

    # 2. Preprocesado del artículo
    try:
        estrellas_art = preprocessing_articulo(
            ruta,
            carpeta,
            area_min=ART_AREA_MIN,
            area_max=ART_AREA_MAX,
            max_iter=ART_MAX_ITER,
            n_min=N_MIN,
            n_max=N_MAX,
            umbral_inicial=ART_UMBRAL_INICIAL,
            paso=ART_PASO,
            umbral_min=ART_UMBRAL_MIN,
            umbral_max=ART_UMBRAL_MAX,
            testing=True,
        )
    except ArticuloEstrellasInsuficientesError as e:
        print("FALLA (Artículo):", e)
        estrellas_art = np.empty((0, 3))

    binarias = sorted(glob.glob(os.path.join(carpeta, "binaria*.png")), key=numero)
    art_binarias = sorted(glob.glob(os.path.join(carpeta, "art_binaria*.png")), key=numero)

    filas.append(
        (
            nombre,
            [
                os.path.join(carpeta, "im_gris.png"),
                os.path.join(carpeta, "im_median.png"),
                os.path.join(carpeta, "im_tophat_visible.png"),
                binarias[-1],
                art_binarias[-1],
            ],
            estrellas_nuestro,
            estrellas_art,
        )
    )

fig, ejes = plt.subplots(
    len(filas),
    len(COLUMNAS),
    figsize=(22, 3.6 * len(filas)),
    squeeze=False,
)

for f, (nombre, archivos, est_nuestro, est_art) in enumerate(filas):
    for c, archivo in enumerate(archivos):
        eje = ejes[f][c]
        eje.imshow(cv.imread(archivo, 0), cmap="gray", vmin=0, vmax=255)
        eje.set_xticks([])
        eje.set_yticks([])
        if f == 0:
            eje.set_title(COLUMNAS[c])

    eje_det = ejes[f][-1]
    eje_det.imshow(cv.imread(archivos[0], 0), cmap="gray", vmin=0, vmax=255)
    if len(est_nuestro) > 0:
        eje_det.scatter(
            est_nuestro[:, 0],
            est_nuestro[:, 1],
            s=80,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.4,
            label=f"Nuestro ({len(est_nuestro)})",
        )
    if len(est_art) > 0:
        eje_det.scatter(
            est_art[:, 0],
            est_art[:, 1],
            s=140,
            facecolors="none",
            edgecolors="red",
            linewidths=1.2,
            linestyle="--",
            label=f"Artículo ({len(est_art)})",
        )
    eje_det.set_xticks([])
    eje_det.set_yticks([])
    eje_det.legend(loc="lower right", fontsize=8, framealpha=0.75)
    if f == 0:
        eje_det.set_title(COLUMNAS[-1])

    ejes[f][0].set_ylabel(nombre)

plt.tight_layout()
plt.savefig("test/tabla_comparativa.png", dpi=200)
plt.show()