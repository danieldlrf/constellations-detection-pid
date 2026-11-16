# Reconocimiento de constelaciones en fotografías nocturnas

Proyecto de la asignatura PID (Universidad de Sevilla). El sistema detecta las
estrellas visibles en una fotografía del cielo nocturno, las compara con imágenes
de referencia de constelaciones y dibuja sobre la foto la figura y el nombre de
las constelaciones reconocidas.

**Estado:** Fase 2 en curso (primera versión de la implementación).

## Contenido del repositorio

| Ruta | Descripción |
|------|-------------|
| `prepare_dataset.py` | Recorta y agrupa las constelaciones del dataset por nombre |
| `preprocesado/` | Preprocesado de imágenes (ruido, binarización, contornos) |
| `data/dataset.zip` | Dataset original de Roboflow en formato COCO |
| `data/processed/` | Recortes generados (no se versiona; se regenera con el script) |
| `requirements.txt` | Dependencias de Python |

## Instalación

Requiere Python 3.12 o 3.13.

```bash
pip install -r requirements.txt
```

Opcionalmente, con entorno virtual:

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparar el dataset

El dataset incluye varias constelaciones por imagen. El script lee las
anotaciones COCO (`bbox = [x, y, ancho, alto]`), recorta cada constelación y la
guarda en una carpeta con su nombre.

```bash
python prepare_dataset.py --input data/dataset.zip --out data/processed --exclude moon
```

Parámetros:

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `--input` | (obligatorio) | ZIP o carpeta del dataset |
| `--out` | `data/processed` | Carpeta de salida |
| `--ref-ratio` | `0.7` | Proporción de recortes para `reference` (el resto va a `testing`) |
| `--margin` | `0.05` | Margen relativo añadido alrededor de cada caja |
| `--min-size` | `8` | Lado mínimo del recorte, en píxeles |
| `--exclude` | ninguno | Clases a ignorar, por ejemplo `moon` |
| `--zip-output` | ninguno | Empaqueta los recortes en un zip |

Resultado:
