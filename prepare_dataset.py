"""
Agrupa recortes de constelaciones por nombre a partir de un ZIP (o carpeta) con
el dataset COCO de Roboflow.

Uso:
  python src/prepare_dataset.py --input data/dataset.zip --out data/processed --exclude moon
  python src/prepare_dataset.py --input data/dataset.zip --out data/processed --zip-output data/recortes.zip
"""
import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2

IGNORED_NAMES = {"star"}  # categoría 0: contenedor de Roboflow
ANN_FILE = "_annotations.coco.json"


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


def process_split_dir(split_dir: Path, ann_path: Path, out: Path, ref_ratio,
                      margin, exclude, min_size, writer, counts):
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
        if (x1 - x0) < min_size or (y1 - y0) < min_size:
            continue

        key = source_key(im["file_name"])
        subset = assign_subset(key, ref_ratio)
        dest_dir = out / subset / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        crop_name = f"{name}_{split_dir.name}_img{im['id']}_ann{ann['id']}.jpg"
        cv2.imwrite(str(dest_dir / crop_name), img[y0:y1, x0:x1])

        writer.writerow([f"{subset}/{name}/{crop_name}", name, subset,
                         split_dir.name, im["file_name"], key, x0, y0, x1, y1])
        counts[(subset, name)] = counts.get((subset, name), 0) + 1


def run(root: Path, args):
    ann_files = sorted(root.rglob(ANN_FILE))
    if not ann_files:
        raise FileNotFoundError(f"No se encontró {ANN_FILE} dentro de {root}")

    args.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(args.out / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crop", "constellation", "subset", "orig_split",
                         "source_file", "source_key", "x0", "y0", "x1", "y1"])
        for ann_path in ann_files:
            process_split_dir(ann_path.parent, ann_path, args.out, args.ref_ratio,
                              args.margin, set(args.exclude), args.min_size,
                              writer, counts)

    print(f"{'subset':<10}{'constelación':<15}{'recortes':>8}")
    for (subset, name), n in sorted(counts.items()):
        print(f"{subset:<10}{name:<15}{n:>8}")
    print(f"Total: {sum(counts.values())} recortes en {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="ZIP o carpeta del dataset")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--ref-ratio", type=float, default=0.7)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--min-size", type=int, default=8)
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