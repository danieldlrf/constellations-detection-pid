from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from src.preprocesado.preprocessing import EstrellasInsuficientesError, preprocessing


def transformar_puntos(
    puntos: np.ndarray,
    params: np.ndarray,
    centro_ref: np.ndarray,
    centro_obj: np.ndarray,
) -> np.ndarray:
    """Aplica rotación (alpha), escala uniforme (s) y traslación (tx, ty)."""
    tx, ty, alpha, log_s = params
    s = float(np.exp(log_s))
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)
    return s * ((puntos - centro_ref) @ rot.T) + centro_obj + np.array([tx, ty], dtype=float)


def _coste_alineamiento(
    params: np.ndarray,
    ref_xy: np.ndarray,
    obj_xy: np.ndarray,
    centro_ref: np.ndarray,
    centro_obj: np.ndarray,
    escala_obj: float,
    s0: float,
) -> float:
    tx, ty, alpha, log_s = params
    s = float(np.exp(log_s))
    ratio_s = s / s0
    if ratio_s < 0.35 or ratio_s > 2.8:
        return 1e6 + (np.log(ratio_s) ** 2) * 1e4

    transformados = transformar_puntos(ref_xy, params, centro_ref, centro_obj)
    dists = np.linalg.norm(transformados[:, None, :] - obj_xy[None, :, :], axis=2)

    if len(ref_xy) <= len(obj_xy):
        filas, cols = linear_sum_assignment(dists)
        error_medio = float(np.mean(dists[filas, cols]))
    else:
        min_dists = np.min(dists, axis=1)
        unicos = len(np.unique(np.argmin(dists, axis=1)))
        penalizacion = 1.0 + 0.5 * (len(ref_xy) - unicos)
        error_medio = float(np.mean(min_dists)) * penalizacion

    return error_medio / max(escala_obj, 1e-6)


def emparejar_nubes(
    ref_stars: np.ndarray,
    obj_stars: np.ndarray,
    *,
    max_estrellas_ref: int = 8,
    n_angulos: int = 12,
    maxiter: int = 600,
) -> tuple[float, np.ndarray]:
    """Ajusta una plantilla de referencia sobre las estrellas detectadas.

    Recibe matrices Nx3 (x, y, brillo) como las que devuelve `preprocessing()`.
    Devuelve `(error_normalizado, puntos_ajustados_xy)`.
    """
    ref_xy = np.asarray(ref_stars[:max_estrellas_ref, :2], dtype=float)
    obj_xy = np.asarray(obj_stars[:, :2], dtype=float)

    if len(ref_xy) < 3 or len(obj_xy) < 3:
        return float("inf"), np.empty((0, 2), dtype=float)

    centro_ref = np.mean(ref_xy, axis=0)
    centro_obj = np.median(obj_xy, axis=0)

    escala_ref = float(np.median(np.linalg.norm(ref_xy - centro_ref, axis=1)))
    escala_obj = float(np.median(np.linalg.norm(obj_xy - centro_obj, axis=1)))
    s0 = max(escala_obj / max(escala_ref, 1e-6), 0.1)

    mejor_res = None
    for alpha0 in np.linspace(0.0, 2.0 * np.pi, n_angulos, endpoint=False):
        x0 = np.array([0.0, 0.0, alpha0, np.log(s0)], dtype=float)
        res = minimize(
            _coste_alineamiento,
            x0,
            args=(ref_xy, obj_xy, centro_ref, centro_obj, escala_obj, s0),
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4},
        )
        if mejor_res is None or res.fun < mejor_res.fun:
            mejor_res = res

    assert mejor_res is not None
    puntos_ajustados = transformar_puntos(ref_xy, mejor_res.x, centro_ref, centro_obj)
    return float(mejor_res.fun), puntos_ajustados


def clasificar_recorte(
    img_path: str | Path,
    base_referencias: dict[str, list[np.ndarray]],
    *,
    umbral_rechazo: float = 0.25,
) -> dict[str, Any] | None:
    """Identifica la constelación más parecida en `base_referencias`.

    `base_referencias` mapea `{nombre_constelacion: [array_Nx3_ref1, ...]}`.
    """
    try:
        obj_stars = preprocessing(str(img_path), path_testing="", testing=False)
    except EstrellasInsuficientesError:
        return None

    mejor_nombre: str | None = None
    mejor_error = float("inf")
    mejores_puntos = np.empty((0, 2), dtype=float)

    for nombre, plantillas in base_referencias.items():
        for ref_stars in plantillas:
            error, pts = emparejar_nubes(ref_stars, obj_stars)
            if error < mejor_error:
                mejor_error = error
                mejor_nombre = nombre
                mejores_puntos = pts

    if mejor_nombre is None or mejor_error > umbral_rechazo:
        return None

    return {
        "name": mejor_nombre,
        "error": mejor_error,
        "stars": [(int(round(x)), int(round(y))) for x, y in mejores_puntos],
    }

