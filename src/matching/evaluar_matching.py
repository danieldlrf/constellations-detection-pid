"""Evalúa el matching sobre data/processed/testing usando index.csv.

Fase 1: preprocesa UNA vez todas las referencias y los recortes de testing.
Fase 2: hace el matching sobre las estrellas ya extraídas.

Uso: py -3.13 src/matching/evaluar_matching.py --data data/processed [--limit 200]
"""
import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.matching.matching import clasificar_estrellas, extraer_estrellas
from src.preprocesado.preprocessing import EstrellasInsuficientesError

BASE = None
KW = {}


def _preprocesar(ruta):
    """Fase 1: estrellas (x, y, brillo) de una imagen, o None si hay pocas."""
    try:
        return extraer_estrellas(ruta)
    except EstrellasInsuficientesError:
        return None


def _init(base, kw):
    global BASE, KW
    BASE = base
    KW = kw


def _procesar(tarea):
    """Fase 2: matching de un recorte ya preprocesado."""
    real, estrellas = tarea
    if estrellas is None:
        return real, None, None, None
    ranking = clasificar_estrellas(estrellas, BASE, **KW)
    if not ranking:
        return real, None, None, None
    return real, ranking[0][0], ranking[0][1], ranking[1][1] if len(ranking) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--limit", type=int, default=None, help="nº máximo de recortes")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tau-rel", type=float, default=0.5)
    ap.add_argument("--n-candidatas", type=int, default=12)
    ap.add_argument("--n-refinar", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None,
                    help="si se indica, --limit toma una muestra aleatoria")
    args = ap.parse_args()

    with open(args.data / "index.csv", encoding="utf-8") as f:
        filas = [(str(args.data / r["crop"]), r["constellation"])
                 for r in csv.DictReader(f) if r["subset"] == "testing"]
    if args.limit:
        if args.seed is not None:
            import random
            random.Random(args.seed).shuffle(filas)
        filas = filas[:args.limit]

    rutas_ref = sorted((p.parent.name, str(p))
                       for p in (args.data / "reference").glob("*/*.jpg"))

    # ---------------- FASE 1: preprocesado ----------------
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        est_ref = list(ex.map(_preprocesar, [r for _, r in rutas_ref], chunksize=4))
        est_test = list(ex.map(_preprocesar, [r for r, _ in filas], chunksize=8))

    base = defaultdict(list)
    for (clase, ruta), est in zip(rutas_ref, est_ref):
        if est is None:
            print(f"[AVISO] Referencia descartada (pocas estrellas): {ruta}")
        else:
            base[clase].append(est)
    base = dict(base)

    n_con = [len(e) for e in est_test if e is not None]
    print(f"Fase 1 (preprocesado) en {time.perf_counter() - t0:.1f} s")
    print("Plantillas por constelación:", {k: len(v) for k, v in sorted(base.items())})
    print(f"Testing: {len(n_con)}/{len(est_test)} recortes con estrellas suficientes"
          f" | estrellas por recorte: media {np.mean(n_con):.1f}, mín {min(n_con)}, máx {max(n_con)}"
          if n_con else "Testing: ningún recorte con estrellas suficientes")

    # ---------------- FASE 2: matching ----------------
    t1 = time.perf_counter()
    kw = dict(tau_rel=args.tau_rel, n_candidatas=args.n_candidatas,
              n_refinar=args.n_refinar)
    tareas = [(real, est) for (_, real), est in zip(filas, est_test)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(base, kw)) as ex:
        resultados = list(ex.map(_procesar, tareas, chunksize=8))
    print(f"Fase 2 (matching) en {time.perf_counter() - t1:.1f} s")

    sin_estrellas = sum(1 for r in resultados if r[1] is None)
    validos = [r for r in resultados if r[1] is not None]
    aciertos = [r for r in validos if r[0] == r[1]]
    print(f"Recortes: {len(resultados)} | sin estrellas suficientes: {sin_estrellas}")
    print(f"Acierto top-1 (sobre válidos): {len(aciertos) / max(len(validos), 1):.1%}")
    print(f"Acierto top-1 (sobre todos):   {len(aciertos) / max(len(resultados), 1):.1%}")

    por_clase = defaultdict(lambda: [0, 0])
    for real, pred, *_ in validos:
        por_clase[real][1] += 1
        por_clase[real][0] += real == pred
    for c, (ok, n) in sorted(por_clase.items()):
        print(f"  {c:<14}{ok / n:6.1%} ({ok}/{n})")

    if aciertos and len(aciertos) < len(validos):
        e_ok = [r[2] for r in aciertos]
        e_ko = [r[2] for r in validos if r[0] != r[1]]
        print(f"Error mediano: aciertos {np.median(e_ok):.3f} | fallos {np.median(e_ko):.3f}")

    with open(args.data / "resultados_matching.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["real", "predicha", "error", "error_2a"])
        w.writerows(resultados)
    conf = Counter((r[0], r[1]) for r in validos if r[0] != r[1])
    print("Confusiones más frecuentes:", conf.most_common(5))


if __name__ == "__main__":
    main()