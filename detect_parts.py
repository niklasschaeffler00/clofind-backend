# detect_parts.py
from ultralytics import YOLO
import cv2
from typing import Dict, Tuple

# kleines YOLOv8-Modell reicht fürs MVP
_model = YOLO("yolov8n.pt")

def detect_top_bottom(image_path: str) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Liefert Bounding Boxes für 'top' und 'bottom' als (x1, y1, x2, y2).
    Einfache Heuristik:
      - Betrachte alle erkannten Boxen
      - Obere Bildhälfte -> nimm die größte Box = 'top'
      - Untere Bildhälfte -> nimm die größte Box = 'bottom'
    Falls keine Boxen in einer Hälfte gefunden werden, fehlt der Part.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Bild nicht lesbar: {image_path}")
    h, w = img.shape[:2]

    res = _model(image_path)[0]
    boxes = []

    # res.boxes.xyxy: Tensor [N,4] mit x1,y1,x2,y2
    for b in res.boxes.xyxy.cpu().numpy().tolist():
        x1, y1, x2, y2 = map(int, b[:4])
        area = max(0, x2 - x1) * max(0, y2 - y1)
        cy = (y1 + y2) // 2
        boxes.append({"xyxy": (x1, y1, x2, y2), "area": area, "cy": cy})

    if not boxes:
        return {}

    upper = [b for b in boxes if b["cy"] < h // 2]
    lower = [b for b in boxes if b["cy"] >= h // 2]

    out: Dict[str, Tuple[int, int, int, int]] = {}

    if upper:
        top_box = max(upper, key=lambda b: b["area"])["xyxy"]
        out["top"] = top_box

    if lower:
        bottom_box = max(lower, key=lambda b: b["area"])["xyxy"]
        out["bottom"] = bottom_box

    return out

def crop(image_path: str, xyxy: Tuple[int, int, int, int], out_path: str) -> str:
    """Speichert den Zuschnitt an out_path und gibt den Pfad zurück."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Bild nicht lesbar: {image_path}")
    x1, y1, x2, y2 = xyxy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)  # simple Schutzkanten
    crop_img = img[y1:y2, x1:x2]
    ok = cv2.imwrite(out_path, crop_img)
    if not ok:
        raise RuntimeError(f"Konnte Crop nicht speichern: {out_path}")
    return out_path
