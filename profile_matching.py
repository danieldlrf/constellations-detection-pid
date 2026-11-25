import cProfile, csv, pstats, sys
from pathlib import Path
sys.path.insert(0, ".")
from src.matching.matching import cargar_referencias, clasificar_estrellas, extraer_estrellas
from src.preprocesado.preprocessing import EstrellasInsuficientesError

data = Path("data/processed")
base = cargar_referencias(data / "reference")
with open(data / "index.csv", encoding="utf-8") as f:
    filas = [r for r in csv.DictReader(f) if r["subset"] == "testing"][:12]

estrellas = []
for r in filas:
    try:
        estrellas.append(extraer_estrellas(data / r["crop"]))
    except EstrellasInsuficientesError:
        pass

def run():
    for e in estrellas[:5]:
        clasificar_estrellas(e, base, n_candidatas=100, n_por_clase=5, n_refinar=3, maxiter=150)

cProfile.run("run()", "prof.out")
pstats.Stats("prof.out").sort_stats("cumulative").print_stats(12)