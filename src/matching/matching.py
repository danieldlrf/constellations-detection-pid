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
    """Primero intenta obtener más estrellas (umbral bajo); si no, admite 3 o más."""
    try:
        return preprocessing(str(ruta), "", n_min=8, n_max=15, umbral_min=5, testing=False)
    except EstrellasInsuficientesError:
        return preprocessing(str(ruta), "", n_min=N_MIN_ESTRELLAS, n_max=15, testing=False)



def _simplex_inicial(x0, limites, escala_obj):
    """Símplex con pasos razonables: traslación 0,25·escala, giro 0,4 rad, log-escala 0,2."""
    pasos = np.array([0.25 * escala_obj, 0.25 * escala_obj, 0.4, 0.2])
    lo = np.array([-np.inf if b[0] is None else b[0] for b in limites])
    hi = np.array([np.inf if b[1] is None else b[1] for b in limites])
    sim = [np.asarray(x0, dtype=float)]
    for i in range(4):
        v = np.array(x0, dtype=float)
        v[i] += pasos[i]
        if v[i] > hi[i]:
            v[i] = x0[i] - pasos[i]
        sim.append(np.clip(v, lo, hi))
    return np.vstack(sim)

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
    ref_xy, _, centro_ref, centro_obj, escala_obj, _ = prep["args"]
    mejor = None
    for i in np.argsort(prep["costes"])[:n_refinar]:
        x0 = prep["arranques"][i]
        res = minimize(
            _coste, x0, args=prep["args"], method="Nelder-Mead",
            bounds=prep["limites"],
            options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4,
                     "initial_simplex": _simplex_inicial(x0, prep["limites"], escala_obj)},
        )
        if mejor is None or res.fun < mejor.fun:
            mejor = res
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


def clasificar_estrellas(obj_stars, base_referencias, *, n_candidatas=60, n_por_clase=3,
                         n_angulos=24, n_refinar=3, maxiter=400, tau_rel=0.5):
    preparadas = []
    for nombre, plantillas in base_referencias.items():
        for ref in plantillas:
            prep = _preparar(ref, obj_stars, n_angulos, tau_rel)
            if prep is not None:
                preparadas.append((nombre, prep))

    por_clase = {}
    for nombre, prep in preparadas:
        por_clase.setdefault(nombre, []).append(prep)
    candidatas = []
    for nombre, lista in por_clase.items():
        lista.sort(key=lambda p: p["costes"].min())
        candidatas += [(nombre, p) for p in lista[:n_por_clase]]
    candidatas.sort(key=lambda p: p[1]["costes"].min())

    errores = {}
    for nombre, prep in candidatas[:n_candidatas]:
        error, pts = _refinar(prep, n_refinar, maxiter)
        errores.setdefault(nombre, []).append((error, pts))
    resultado = []
    for nombre, lst in errores.items():
        lst.sort(key=lambda t: t[0])
        score = float(np.mean([e for e, _ in lst[:2]]))
        resultado.append((nombre, score, lst[0][1]))
    return sorted(resultado, key=lambda r: r[1])

def clasificar_recorte(img_path, base_referencias, *, umbral_rechazo=0.25,
                       margen_min=0.0, **kwargs) -> dict[str, Any] | None:
    """Identifica la constelación más parecida; None si no hay estrellas
    suficientes, el error supera el umbral o el resultado es ambiguo."""
    try:
        obj_stars = extraer_estrellas(img_path)
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

