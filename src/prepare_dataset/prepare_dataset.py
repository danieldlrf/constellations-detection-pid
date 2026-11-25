import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocesado.preprocessing import EstrellasInsuficientesError, preprocessing

IGNORED_NAMES = {"star"}  # categoría 0: contenedor de Roboflow
ANN_FILE = "_annotations.coco.json"
CAND_DIR = "_candidates"  # carpeta temporal con todos los candidatos a referencia


def safe_extract(zip_path: Path, dest: Path):
    """Extrae el zip evitando rutas maliciosas (zip slip)."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if dest not in target.parents and target != dest:
                raise ValueError(f"Ruta insegura en el zip: {member.filename}")
        zf.extractall(dest)


def source_key(file_name: str) -> str:
    """Foto original, sin el sufijo '.rf.<hash>' de Roboflow."""
    stem = Path(file_name).stem
    return stem.split("_png_jpg.rf.")[0].split(".rf.")[0]


def assign_subset(key: str, ref_ratio: float) -> str:
    """Reparto determinista por foto original (evita fugas reference/testing)."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10_000 / 10_000
    return "reference" if h < ref_ratio else "testing"


def clamp_box(x, y, w, h, margin, W, H):
    mx, my = w * margin, h * margin
    return (max(0, int(round(x - mx))), max(0, int(round(y - my))),
            min(W, int(round(x + w + mx))), min(H, int(round(y + h + my))))


def contrast_score(crop) -> float:
    """Contraste RMS: desviación típica de la escala de grises (0-255)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray.astype(np.float32)))


def rank_norm(values) -> np.ndarray:
    """Percentil de cada valor dentro de su lista (0 = peor, 1 = mejor)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 1:
        return np.ones(1)
    return np.argsort(np.argsort(v)) / (len(v) - 1)


def process_split_dir(split_dir: Path, ann_path: Path, out: Path, ref_ratio,
                      margin, exclude, min_size, testing_rows, candidates, counts):
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    images = {im["id"]: im for im in data["images"]}
    cache = {}

    for ann in data["annotations"]:
        name = cats.get(ann["category_id"])
        if name is None or name in IGNORED_NAMES or name in exclude:
            continue

        im = images[ann["image_id"]]
        if im["id"] not in cache:
            cache = {im["id"]: cv2.imread(str(split_dir / im["file_name"]))}
        img = cache[im["id"]]
        if img is None:
            print(f"[AVISO] No se pudo leer {split_dir / im['file_name']}")
            continue

        H, W = img.shape[:2]
        x, y, w, h = ann["bbox"]
        x0, y0, x1, y1 = clamp_box(x, y, w, h, margin, W, H)
        cw, ch = x1 - x0, y1 - y0
        if cw < min_size or ch < min_size:
            continue

        key = source_key(im["file_name"])
        subset = assign_subset(key, ref_ratio)
        crop = img[y0:y1, x0:x1]
        crop_name = f"{name}_{split_dir.name}_img{im['id']}_ann{ann['id']}.jpg"

        if subset == "testing":
            dest_dir = out / "testing"  # todos juntos, sin subcarpetas
            dest_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest_dir / crop_name), crop)
            testing_rows.append([f"testing/{crop_name}", name, "testing",
                                 split_dir.name, im["file_name"], key,
                                 x0, y0, x1, y1, "", ""])
            counts[("testing", name)] = counts.get(("testing", name), 0) + 1
        else:
            cand_dir = out / CAND_DIR / name
            cand_dir.mkdir(parents=True, exist_ok=True)
            cand_path = cand_dir / crop_name
            cv2.imwrite(str(cand_path), crop)
            candidates.setdefault(name, []).append({
                "contrast": contrast_score(crop), "size": float(np.sqrt(cw * ch)),
                "key": key, "path": cand_path, "split": split_dir.name,
                "file": im["file_name"], "box": (x0, y0, x1, y1)})


def n_estrellas(ruta, minimo: int, cache: dict) -> int:
    """Estrellas que detecta el preprocesado existente (0 si no llega al mínimo)."""
    if ruta not in cache:
        try:
            cache[ruta] = len(preprocessing(str(ruta), "", n_min=minimo, testing=False))
        except EstrellasInsuficientesError:
            cache[ruta] = 0
    return cache[ruta]


def select_references(candidates, out: Path, top_k: int, w_contrast: float,
                      w_size: float, min_stars: int):
    """Elige las top_k con mayor puntuación = w_contrast*contraste + w_size*tamaño
    (ambos como percentil dentro de la constelación), exigiendo que el preprocesado
    detecte al menos min_stars estrellas. Máximo una por foto original mientras
    sea posible, porque las versiones aumentadas son casi idénticas."""
    rows, counts, cache = [], {}, {}
    for name, items in sorted(candidates.items()):
        rc = rank_norm([c["contrast"] for c in items])
        rs = rank_norm([c["size"] for c in items])
        for c, a, b in zip(items, rc, rs):
            c["score"] = w_contrast * a + w_size * b
        items.sort(key=lambda c: c["score"], reverse=True)

        def valida(c):
            return n_estrellas(c["path"], min_stars, cache) >= min_stars

        chosen, used_keys = [], set()
        for c in items:  # 1ª pasada: una por foto original
            if c["key"] in used_keys or not valida(c):
                continue
            chosen.append(c)
            used_keys.add(c["key"])
            if len(chosen) == top_k:
                break
        if len(chosen) < top_k:  # 2ª pasada: completa con otras versiones de la misma foto
            for c in items:
                if c not in chosen and valida(c):
                    chosen.append(c)
                if len(chosen) == top_k:
                    break
        if len(chosen) < top_k:
            print(f"[AVISO] {name}: solo {len(chosen)} referencias con "
                  f">={min_stars} estrellas detectables (de {len(items)} candidatos)")

        dest_dir = out / "reference" / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for rank, c in enumerate(chosen, 1):
            new_name = f"{name}_ref{rank}.jpg"
            shutil.copy2(c["path"], dest_dir / new_name)
            x0, y0, x1, y1 = c["box"]
            rows.append([f"reference/{name}/{new_name}", name, "reference",
                         c["split"], c["file"], c["key"], x0, y0, x1, y1,
                         f"{c['contrast']:.2f}", f"{c['size']:.0f}"])
        counts[("reference", name)] = len(chosen)
    shutil.rmtree(out / CAND_DIR, ignore_errors=True)
    return rows, counts


def run(root: Path, args):
    ann_files = sorted(root.rglob(ANN_FILE))
    if not ann_files:
        raise FileNotFoundError(f"No se encontró {ANN_FILE} dentro de {root}")

    args.out.mkdir(parents=True, exist_ok=True)
    counts, testing_rows, candidates = {}, [], {}
    for ann_path in ann_files:
        process_split_dir(ann_path.parent, ann_path, args.out, args.ref_ratio,
                          args.margin, set(args.exclude), args.min_size,
                          testing_rows, candidates, counts)

    ref_rows, ref_counts = select_references(candidates, args.out, args.top_k,
                                             args.w_contrast, args.w_size,
                                             args.ref_min_stars)
    counts.update(ref_counts)

    with open(args.out / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crop", "constellation", "subset", "orig_split",
                         "source_file", "source_key", "x0", "y0", "x1", "y1",
                         "contrast", "size_px"])
        writer.writerows(ref_rows + testing_rows)

    print(f"{'subset':<10}{'constelación':<15}{'recortes':>8}")
    for (subset, name), n in sorted(counts.items()):
        print(f"{subset:<10}{name:<15}{n:>8}")
    n_ref = sum(v for (s, _), v in counts.items() if s == "reference")
    n_test = sum(v for (s, _), v in counts.items() if s == "testing")
    print(f"Total: {n_ref} referencias y {n_test} recortes de testing en {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="ZIP o carpeta del dataset")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--ref-ratio", type=float, default=0.7)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--min-size", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=6,
                    help="nº de referencias por constelación (recomendado 5-7)")
    ap.add_argument("--w-contrast", type=float, default=0.5,
                    help="peso del contraste en la puntuación de referencia")
    ap.add_argument("--w-size", type=float, default=0.5,
                    help="peso del tamaño en la puntuación de referencia")
    ap.add_argument("--ref-min-stars", type=int, default=5,
                    help="estrellas mínimas que debe detectar el preprocesado en una referencia")
    ap.add_argument("--exclude", nargs="*", default=[], help="clases a ignorar, p.ej. moon")
    ap.add_argument("--zip-output", type=Path, default=None,
                    help="si se indica, empaqueta los recortes en este zip")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)  # regeneración limpia

    if args.input.is_file() and zipfile.is_zipfile(args.input):
        with tempfile.TemporaryDirectory() as tmp:
            print(f"Descomprimiendo {args.input} ...")
            safe_extract(args.input, Path(tmp))
            run(Path(tmp), args)
    elif args.input.is_dir():
        run(args.input, args)
    else:
        raise ValueError("--input debe ser un .zip válido o una carpeta")

    if args.zip_output:
        base = args.zip_output.with_suffix("")
        shutil.make_archive(str(base), "zip", args.out)
        print(f"Recortes empaquetados en {base}.zip")


if __name__ == "__main__":
    main()