"""Prueba del matching: original (rama main) frente a arreglado (rama actual).

Uso (desde la raíz del repo): python prueba/prueba_matching.py
Lo lanza PRUEBA_MATCHING.bat. Deja todo en resultados_prueba/.
"""
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data" / "processed"
SALIDA = RAIZ / "resultados_prueba"
ARGS_EVAL = ["--limit", "300", "--seed", "0"]
PY = sys.executable


def titulo(t):
    print("\n" + "=" * 70 + f"\n {t}\n" + "=" * 70, flush=True)


def ejecutar(cmd, log, cwd=RAIZ):
    """Ejecuta cmd mostrando la salida y guardándola en log. Devuelve (código, texto)."""
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    lineas = []
    for linea in p.stdout:
        print(linea, end="", flush=True)
        lineas.append(linea)
    p.wait()
    texto = "".join(lineas)
    log.write_text(texto, encoding="utf-8")
    print(f"[{time.time() - t0:.0f} s]", flush=True)
    return p.returncode, texto


def preparar_dataset():
    if (DATA / "index.csv").exists() and (DATA / "reference").exists():
        print(f"Ya existe {DATA}, no se regenera (bórralo para rehacerlo).")
        return
    titulo("1/4  Preparando el dataset (prepare_dataset --top-k 30)")
    cod, _ = ejecutar([PY, "-m", "src.prepare_dataset.prepare_dataset",
                       "--input", "data/dataset.zip", "--out", "data/processed",
                       "--top-k", "30"], SALIDA / "log_prepare_dataset.txt")
    if cod != 0:
        sys.exit("Falló prepare_dataset (mira resultados_prueba/log_prepare_dataset.txt)")


def copia_original():
    """Crea una copia del src con el matching de la rama main. None si no hay git."""
    git = shutil.which("git")
    if not git:
        print("No se encuentra git: se omite la comparación con el original.")
        return None
    dest = SALIDA / "_original"
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(RAIZ / "src", dest / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("src/matching/matching.py", "src/matching/evaluar_matching.py"):
        r = subprocess.run([git, "show", f"main:{f}"], cwd=RAIZ, capture_output=True)
        if r.returncode != 0:
            print(f"No se pudo leer main:{f}; se omite el original.")
            return None
        (dest / f).write_bytes(r.stdout)
    return dest


def resumen_de(texto):
    claves = ("Acierto top-1:", "Acierto top-3:", "Recortes:", "Recall")
    return [l for l in texto.splitlines() if l.startswith(claves)]


def main():
    SALIDA.mkdir(exist_ok=True)
    preparar_dataset()

    res = {}
    titulo("2/4  Matching ARREGLADO (rama actual), 300 recortes, seed 0")
    cod, txt = ejecutar([PY, "src/matching/evaluar_matching.py", "--data", str(DATA), *ARGS_EVAL],
                        SALIDA / "evaluacion_arreglado.txt")
    if cod != 0:
        sys.exit("Falló la evaluación del arreglado (mira resultados_prueba/evaluacion_arreglado.txt)")
    res["Arreglado (prueba-matching)"] = resumen_de(txt)
    shutil.copy(DATA / "resultados_matching.csv", SALIDA / "resultados_arreglado.csv")

    orig = copia_original()
    if orig:
        titulo("3/4  Matching ORIGINAL (rama main), mismos 300 recortes")
        cod, txt = ejecutar([PY, "src/matching/evaluar_matching.py", "--data", str(DATA), *ARGS_EVAL],
                            SALIDA / "evaluacion_original.txt", cwd=orig)
        if cod == 0:
            res["Original (main)"] = resumen_de(txt)
            shutil.copy(DATA / "resultados_matching.csv", SALIDA / "resultados_original.csv")
        shutil.rmtree(orig, ignore_errors=True)

    titulo("4/4  Imagen de ejemplos (4 aciertos y 2 fallos)")
    ejecutar([PY, "prueba/ver_ejemplos.py", str(SALIDA)], SALIDA / "log_ejemplos.txt")

    lineas = [f"Prueba del matching - {datetime.now():%d/%m/%Y %H:%M}", ""]
    for nombre, ls in res.items():
        lineas += [nombre] + [f"  {l}" for l in ls] + [""]
    lineas.append("Detalle por constelación en evaluacion_*.txt; imagen en ejemplos_matching.png")
    (SALIDA / "RESUMEN.txt").write_text("\n".join(lineas), encoding="utf-8")
    titulo("RESUMEN")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
