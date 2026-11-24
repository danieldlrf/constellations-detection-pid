import cv2 as cv
import numpy as np
import os

class EstrellasInsuficientesError(ValueError):
    """Se detectaron menos estrellas de las necesarias."""

def preprocessing_articulo(img_path, path_testing, area_min=1, area_max=150, max_iter=100,
                           n_min=3, n_max=15, umbral_inicial=None, paso=5,
                           umbral_min=5, umbral_max=250, testing=False):
    #Conversion de la imagen a escala de grises
    im_gris = cv.imread(img_path, 0)
    if im_gris is None:
        raise FileNotFoundError(img_path)
    if testing: _guardar(path_testing, "art_im_gris.png", im_gris)

    #Umbral inicial: manual o automatico (Otsu)
    if umbral_inicial is None:
        umbral, _ = cv.threshold(im_gris, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        umbral = int(umbral)
    else:
        umbral = umbral_inicial

    args = (im_gris, area_min, area_max, testing, path_testing)
    estrellas = detectar_contornos(umbral, *args, i=0)

    #Si hay menos de n_min estrellas, bajamos el umbral hasta encontrar n_min
    i = 0
    while len(estrellas) < n_min and umbral > umbral_min and i < max_iter:
        i += 1
        umbral = max(umbral_min, umbral - paso)
        estrellas = detectar_contornos(umbral, *args, i=i)

    if len(estrellas) < n_min:
        raise EstrellasInsuficientesError(
            f"{img_path}: {len(estrellas)} estrellas (minimo {n_min}), umbral final {umbral}"
        )

    #Si hay mas de n_max estrellas, subimos el umbral hasta que haya menos de n_max
    while len(estrellas) > n_max and umbral < umbral_max and i < max_iter:
        i += 1
        umbral_nuevo = min(umbral_max, umbral + paso)
        candidatas = detectar_contornos(umbral_nuevo, *args, i=i)
        #Si al subir nos quedamos sin las n_min necesarias, nos quedamos con el resultado anterior
        if len(candidatas) < n_min:
            break
        umbral, estrellas = umbral_nuevo, candidatas

    return np.array(estrellas, dtype=float).reshape(-1, 3)

def detectar_contornos(umbral, im_gris, area_min, area_max, testing, path_testing, i):
    #Binarizacion con umbral global
    _, binaria = cv.threshold(im_gris, umbral, 255, cv.THRESH_BINARY)
    if testing: _guardar(path_testing, f"art_binaria{i}.png", binaria)

    #Busqueda de contornos
    contornos, _ = cv.findContours(binaria, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    res = []
    for c in contornos:
        #Solo contornos cuya area esta en el intervalo (area del contorno, no numero de pixeles)
        area = cv.contourArea(c)
        if not (area_min <= area <= area_max):
            continue
        #Centro del minimo circulo que encierra el contorno
        (x, y), _ = cv.minEnclosingCircle(c)
        #Brillo (no lo usa el articulo, se calcula para ser compatible con preprocessing())
        mask = np.zeros(im_gris.shape, dtype=np.uint8)
        cv.drawContours(mask, [c], -1, 255, -1)
        brillo = cv.minMaxLoc(im_gris, mask=mask)[1]
        res.append((x, y, brillo))
    return res

def _guardar(carpeta, nombre, imagen):
    os.makedirs(carpeta, exist_ok=True)
    cv.imwrite(os.path.join(carpeta, nombre), imagen)