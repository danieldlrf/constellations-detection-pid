import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np

class EstrellasInsuficientesError(ValueError):
    """Se detectaron menos estrellas de las necesarias."""
    
def tam_se_desde_area(area_max, margen=2):
    """Tamaño (impar) del elemento estructurante a partir del área máxima de una estrella."""
    diametro = int(np.ceil(2 * np.sqrt(area_max / np.pi)))  # diámetro de un círculo de esa área
    tam = diametro + margen
    return tam if tam % 2 == 1 else tam + 1

def preprocesing(img_path, area_min=2, area_max=150, max_iter=50,
                n_min=3, n_max=15, umbral_inicial=225, paso=5, tam_se=None, 
                ratio_max=2.0, relleno_min=0.5, area_forma_min=9, umbral_min=20):
    #Conversion de la imagen a escala de grises
    im_gris = cv.imread(img_path, 0)
    if im_gris is None:
        raise FileNotFoundError(img_path)
    
    #Calcular el tamaño del elemento estructural
    if tam_se is None:
        tam_se = tam_se_desde_area(area_max)
        
    #Limpieza de ruido usando la mediana 3x3
    im_median = cv.medianBlur(im_gris, 3)
    
    #Limpieza de ruido usnado operación morfologica Top-Hat 
    se_elipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (tam_se,tam_se))
    im_tophat = cv.morphologyEx(im_median,  cv.MORPH_TOPHAT, se_elipse, iterations=1)
    
    #Binarizacion usando umbrales adaptativos
    umbral = umbral_inicial
    args = (area_min, area_max, ratio_max, relleno_min, area_forma_min)
    estrellas = detectar_componentes(umbral, im_tophat, area_min, area_max, *args)
    
    for _ in range(max_iter):
        #Si hay mas de las estrellas minimas o el umbral es <= umbral_min para el bucle
        if len(estrellas) >= n_min or umbral <= umbral_min:
            break
        #Reducimos el umbral
        umbral = max(0, umbral - paso)
        estrellas = detectar_componentes(umbral, im_tophat, area_min, area_max, *args)
    
    if len(estrellas) < n_min:
        raise EstrellasInsuficientesError(
            f"{img_path}: {len(estrellas)} estrellas (mínimo {n_min}), umbral final {umbral}"
        )
    
    #Obtener las n_max estrellas de la lista con mas brillo
    estrellas.sort(key=lambda e: e[2], reverse=True)
    estrellas = estrellas[:n_max]
    return np.array(estrellas, dtype=float).reshape(-1, 3)
    
def detectar_componentes(umbral, tophat, area_min, area_max,
                         ratio_max, relleno_min, area_forma_min):
        _, binaria = cv.threshold(tophat, umbral, 255, cv.THRESH_BINARY)
        #Extracción de centroides
        n, labels, stats, cent = cv.connectedComponentsWithStats(binaria, connectivity=8)
        res = []
        for i in range(1, n):
            #Seleccionar estrellas que cumplan: min <= area <= max
            area = stats[i, cv.CC_STAT_AREA]
            if not (area_min <= area <= area_max):
                continue

            # Filtro de forma (solo si la mancha es lo bastante grande para que tenga sentido)
            if area >= area_forma_min:
                ancho = stats[i, cv.CC_STAT_WIDTH]
                alto = stats[i, cv.CC_STAT_HEIGHT]
                ratio = max(ancho, alto) / min(ancho, alto)   # 1 = cuadrada, grande = alargada
                relleno = area / (ancho * alto)                # círculo ≈ 0.78, línea diagonal ≈ 0
                if ratio > ratio_max or relleno < relleno_min:
                    continue
            #Aplicamos mascara de labels para seleccionar los pixeles de una misma estrella y sacar el mas brillanet
            brillo = tophat[labels == i].max()
            res.append((cent[i][0], cent[i][1], brillo))
        return res