"""Evalúa el matching sobre data/processed/testing usando index.csv.

Uso: py -3.13 src/matching/evaluar_matching.py --data data/processed [--limit 200]
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.matching.matching import cargar_referencias, clasificar_estrellas, extraer_estrellas
from src.preprocesado.preprocessing import EstrellasInsuficientesError

BASE = None

KW = {}

def _init(dir_ref, kw):
    global BASE, KW
    BASE = cargar_referencias(dir_ref, verbose=False)
    KW = kw

def _procesar(fila):
    ruta, real = fila
    try:
        estrellas = extraer_estrellas(ruta)
    except EstrellasInsuficientesError:
        return real, None, None, None
    ranking = clasificar_estrellas(estrellas, BASE, **KW)
    if not ranking:
        return real, None, None, None
    nombres = [r[0] for r in ranking]
    return (real, ranking[0][0], ranking[0][1],
            ranking[1][1] if len(ranking) > 1 else None, nombres)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--limit", type=int, default=None, help="nº máximo de recortes")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tau-rel", type=float, default=0.5)
    ap.add_argument("--n-candidatas", type=int, default=100)
    ap.add_argument("--n-refinar", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None,
                    help="si se indica, --limit toma una muestra aleatoria")
    ap.add_argument("--n-por-clase", type=int, default=5)
    ap.add_argument("--n-finalistas", type=int, default=25)
    ap.add_argument("--maxiter-previo", type=int, default=40)
    ap.add_argument("--maxiter", type=int, default=150)
    args = ap.parse_args()

    with open(args.data / "index.csv", encoding="utf-8") as f:
        filas = [(str(args.data / r["crop"]), r["constellation"])
                 for r in csv.DictReader(f) if r["subset"] == "testing"]
    if args.limit:
        if args.seed is not None:
            import random
            random.Random(args.seed).shuffle(filas)
        filas = filas[:args.limit]

    kw = dict(tau_rel=args.tau_rel, n_candidatas=args.n_candidatas,
              n_refinar=args.n_refinar, n_por_clase=args.n_por_clase,
              maxiter=args.maxiter, n_finalistas=args.n_finalistas,
              maxiter_previo=args.maxiter_previo)
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.data / "reference", kw)) as ex:
        resultados = list(ex.map(_procesar, filas, chunksize=8))

    sin_estrellas = sum(1 for r in resultados if r[1] is None)
    validos = [r for r in resultados if r[1] is not None]
    aciertos = [r for r in validos if r[0] == r[1]]

    pos = [r[4].index(r[0]) + 1 for r in validos if r[0] in r[4]]
    print(f"Recall de la lista de candidatas: {len(pos) / max(len(validos), 1):.1%}")
    for k in (1, 3):
        print(f"Acierto top-{k}: {sum(p <= k for p in pos) / max(len(validos), 1):.1%}")
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