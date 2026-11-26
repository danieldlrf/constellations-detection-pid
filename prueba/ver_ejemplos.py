# Uso (desde la raíz del repo): python prueba/ver_ejemplos.py <carpeta_salida>
# Genera ejemplos_matching.png con 4 aciertos y 2 fallos de la muestra --seed 0.
import csv, random, sys
from pathlib import Path
import cv2 as cv, numpy as np
sys.path.insert(0, ".")
from src.matching.matching import cargar_referencias, clasificar_recorte, extraer_estrellas
from src.overlay.constellation_overlay import draw_constellation_overlay
from src.preprocesado.preprocessing import EstrellasInsuficientesError
data = Path("data/processed"); out = Path(sys.argv[1] if len(sys.argv) > 1 else "."); out.mkdir(parents=True, exist_ok=True)
base = cargar_referencias(data / "reference")
filas = [(r["crop"], r["constellation"]) for r in csv.DictReader(open(data/"index.csv")) if r["subset"]=="testing"]
random.Random(0).shuffle(filas)
kw = dict(n_candidatas=100, n_por_clase=5, n_finalistas=25, maxiter=150, umbral_rechazo=9e9)
bien, mal = [], []
S = 360
def nice(n): return n.replace("_"," ").title()
for crop, real in filas[:60]:
    if len(bien) >= 4 and len(mal) >= 2: break
    ruta = str(data/crop)
    try: det = extraer_estrellas(ruta)
    except EstrellasInsuficientesError: continue
    r = clasificar_recorte(ruta, base, **kw)
    if r is None: continue
    ok = r["name"] == real
    if (ok and len(bien) >= 4) or (not ok and len(mal) >= 2): continue
    img = cv.imread(ruta); h, w = img.shape[:2]; f = S / max(h, w)
    img = cv.resize(img, (round(w*f), round(h*f)), interpolation=cv.INTER_CUBIC)
    pad = np.zeros((S, S, 3), np.uint8); pad[:img.shape[0], :img.shape[1]] = img
    stars = [(int(x*f), int(y*f)) for x, y in r["stars"]]
    col = (80, 220, 80) if ok else (60, 60, 255)
    vis = draw_constellation_overlay(pad, [{"name": "", "bbox": [0,0,S,S], "stars": stars, "colour": col}], line_thickness=2)
    for x, y, _ in det:
        cv.circle(vis, (int(x*f), int(y*f)), 7, (255, 200, 0), 1, cv.LINE_AA)
    head = np.full((70, S, 3), 30, np.uint8)
    cv.putText(head, f"Real:     {nice(real)}", (8, 26), cv.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 1, cv.LINE_AA)
    cv.putText(head, f"Predicha: {nice(r['name'])}  (err {r['error']:.3f})", (8, 54), cv.FONT_HERSHEY_SIMPLEX, 0.62, col, 1, cv.LINE_AA)
    (bien if ok else mal).append(np.vstack([head, vis]))

panel = np.vstack([np.hstack(bien[:3]), np.hstack(bien[3:4] + mal[:2])])
ley = np.full((40, panel.shape[1], 3), 30, np.uint8)
cv.putText(ley, "Circulos azules: estrellas detectadas | Puntos y lineas: plantilla ajustada (verde = acierto, rojo = fallo)", (8, 26), cv.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1, cv.LINE_AA)
cv.imwrite(str(out/"ejemplos_matching.png"), np.vstack([panel, ley]))
print(len(bien), len(mal))
