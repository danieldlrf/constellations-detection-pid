from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from src.preprocesado.preprocessing import EstrellasInsuficientesError, preprocessing

RANGO_ESCALA = (0.35, 2.8)            # escala permitida respecto a la inicial s0
FACTORES_ESCALA = (0.7, 1.0, 1.4)     # escalas de partida de la búsqueda gruesa

N_MIN_ESTRELLAS = 3

def extraer_estrellas(ruta) -> np.ndarray:
    """Estrellas (x, y, brillo) con el preprocesado existente, admitiendo 3 o más."""
    return preprocessing(str(ruta), "", n_min=N_MIN_ESTRELLAS, testing=False)


def transformar_puntos(puntos, params, centro_ref, centro_obj) -> np.ndarray:
    """Rotación (alpha), escala uniforme (exp(log_s)) y traslación (tx, ty)."""
    tx, ty, alpha, log_s = params
    c, s = np.cos(alpha), np.sin(alpha)
    rot = np.array([[c, -s], [s, c]])
    return np.exp(log_s) * ((puntos - centro_ref) @ rot.T) + centro_obj + (tx, ty)


def _coste(params, ref_xy, obj_xy, centro_ref, centro_obj, escala_obj, tau) -> float:
    """Asignación 1 a 1 con distancias truncadas en tau. Las estrellas sin pareja
    (por diferencia de tamaño entre nubes) cuestan tau cada una."""
    pts = transformar_puntos(ref_xy, params, centro_ref, centro_obj)
    d = np.minimum(np.linalg.norm(pts[:, None, :] - obj_xy[None, :, :], axis=2), tau)
    filas, cols = linear_sum_assignment(d)
    sin_pareja = abs(len(ref_xy) - len(obj_xy))
    total = d[filas, cols].sum() + tau * sin_pareja
    return float(total / max(len(ref_xy), len(obj_xy)) / escala_obj)



def _preparar(ref_stars, obj_stars, n_angulos: int, tau_rel: float):
    """Búsqueda gruesa (rejilla de ángulos x escalas) sin optimizar."""
    ref_xy = np.asarray(ref_stars, dtype=float)[:, :2]
    obj_xy = np.asarray(obj_stars, dtype=float)[:, :2]
    if len(ref_xy) < 3 or len(obj_xy) < 3:
        return None

    centro_ref = np.median(ref_xy, axis=0)
    centro_obj = np.median(obj_xy, axis=0)
    escala_ref = float(np.median(np.linalg.norm(ref_xy - centro_ref, axis=1)))
    escala_obj = float(np.median(np.linalg.norm(obj_xy - centro_obj, axis=1)))
    if escala_ref < 1e-6 or escala_obj < 1e-6:
        return None

    s0 = escala_obj / escala_ref
    args = (ref_xy, obj_xy, centro_ref, centro_obj, escala_obj, tau_rel * escala_obj)
    arranques = [
        np.array([0.0, 0.0, a, np.log(s0 * f)])
        for a in np.linspace(0.0, 2.0 * np.pi, n_angulos, endpoint=False)
        for f in FACTORES_ESCALA
    ]
    costes = np.array([_coste(x0, *args) for x0 in arranques])
    limites = [
        (-escala_obj, escala_obj), (-escala_obj, escala_obj), (None, None),
        (np.log(s0 * RANGO_ESCALA[0]), np.log(s0 * RANGO_ESCALA[1])),
    ]
    return {"args": args, "arranques": arranques, "costes": costes, "limites": limites}


def _refinar(prep, n_refinar: int, maxiter: int) -> tuple[float, np.ndarray]:
    """Nelder-Mead desde los mejores arranques de la búsqueda gruesa."""
    mejor = None
    for i in np.argsort(prep["costes"])[:n_refinar]:
        res = minimize(
            _coste, prep["arranques"][i], args=prep["args"], method="Nelder-Mead",
            bounds=prep["limites"],
            options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4},
        )
        if mejor is None or res.fun < mejor.fun:
            mejor = res
    ref_xy, _, centro_ref, centro_obj, _, _ = prep["args"]
    return float(mejor.fun), transformar_puntos(ref_xy, mejor.x, centro_ref, centro_obj)


def emparejar_nubes(ref_stars, obj_stars, *, n_angulos=24, n_refinar=3,
                    maxiter=400, tau_rel=0.5) -> tuple[float, np.ndarray]:
    """Ajusta una plantilla sobre las estrellas detectadas (matrices Nx3).
    Devuelve (error_normalizado, puntos_ajustados_xy)."""
    prep = _preparar(ref_stars, obj_stars, n_angulos, tau_rel)
    if prep is None:
        return float("inf"), np.empty((0, 2))
    return _refinar(prep, n_refinar, maxiter)


def cargar_referencias(dir_referencias, verbose: bool = False) -> dict[str, list[np.ndarray]]:
    """Preprocesa reference/<constelacion>/*.jpg una sola vez."""
    base: dict[str, list[np.ndarray]] = {}
    for carpeta in sorted(Path(dir_referencias).iterdir()):
        if not carpeta.is_dir():
            continue
        for img in sorted(carpeta.glob("*.jpg")):
            try:
                base.setdefault(carpeta.name, []).append(extraer_estrellas(img))
            except EstrellasInsuficientesError:
                if verbose:
                    print(f"[AVISO] Referencia descartada (pocas estrellas): {img}")
    return base


def clasificar_estrellas(obj_stars, base_referencias, *, n_candidatas=12,
                         n_angulos=24, n_refinar=3, maxiter=400, tau_rel=0.5):
    """Devuelve el ranking [(constelacion, error, puntos)] ordenado por error."""
    preparadas = []
    for nombre, plantillas in base_referencias.items():
        for ref in plantillas:
            prep = _preparar(ref, obj_stars, n_angulos, tau_rel)
            if prep is not None:
                preparadas.append((nombre, prep))
    # Poda: solo se refinan las plantillas con mejor coste en la búsqueda gruesa
    preparadas.sort(key=lambda p: p[1]["costes"].min())

    mejor_por_clase: dict[str, tuple[float, np.ndarray]] = {}
    for nombre, prep in preparadas[:n_candidatas]:
        error, pts = _refinar(prep, n_refinar, maxiter)
        if nombre not in mejor_por_clase or error < mejor_por_clase[nombre][0]:
            mejor_por_clase[nombre] = (error, pts)
    return sorted(((n, e, p) for n, (e, p) in mejor_por_clase.items()), key=lambda r: r[1])


def clasificar_recorte(img_path, base_referencias, *, umbral_rechazo=0.25,
                       margen_min=0.0, **kwargs) -> dict[str, Any] | None:
    """Identifica la constelación más parecida; None si no hay estrellas
    suficientes, el error supera el umbral o el resultado es ambiguo."""
    try:
        obj_stars = preprocessing(str(img_path), "", testing=False)
    except EstrellasInsuficientesError:
        return None

    ranking = clasificar_estrellas(obj_stars, base_referencias, **kwargs)
    if not ranking:
        return None
    nombre, error, puntos = ranking[0]
    if error > umbral_rechazo:
        return None
    if margen_min > 0 and len(ranking) > 1 and ranking[1][1] < error * (1 + margen_min):
        return None
    return {
        "name": nombre,
        "error": error,
        "stars": [(int(round(x)), int(round(y))) for x, y in puntos],
        "ranking": [(n, round(e, 4)) for n, e, _ in ranking[:3]],
    }