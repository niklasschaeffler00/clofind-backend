# app_search.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import os, io, urllib.request
import numpy as np
import faiss, torch, clip
from PIL import Image
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()

app = FastAPI(title="CloFind Visual Search API")

# --- CORS fürs Frontend (Domain später anpassen) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: im Prod einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globales State / Lazy Load ---
MODEL_NAME = "ViT-B/32"
INDEX_PATH = "faiss.index"
IDS_PATH = "ids.npy"
_device = "cpu"
_model = None
_preprocess = None
_index = None
_ids = None

def db_connect():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL fehlt")
    return psycopg2.connect(dsn)

def ensure_loaded():
    global _model, _preprocess, _index, _ids
    if _model is None or _preprocess is None:
        _model, _preprocess = clip.load(MODEL_NAME, device=_device)
        _model.eval()
    if _index is None or _ids is None:
        if not (os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH)):
            raise RuntimeError("Indexdateien fehlen. Bitte build_index.py ausführen.")
        _index = faiss.read_index(INDEX_PATH)
        _ids = np.load(IDS_PATH)

def _image_to_tensor(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return _preprocess(img).unsqueeze(0)

def _encode_image(tensor):
    with torch.no_grad():
        vec = _model.encode_image(tensor.to(_device)).cpu().numpy().astype("float32")
    faiss.normalize_L2(vec)
    return vec

def _label_for(score: float) -> str:
    # Cosine-Similarity (normiert). Identische Bilder ~0.98–1.0
    if score >= 0.85: 
        return "Exact"
    if score >= 0.60: 
        return "Sehr ähnlich"
    return "Alternative"


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search/by-url")
async def search_by_url(
    image_url: str = Form(...),
    topk: int = Form(5),
):
    ensure_loaded()

    # Query-Bild laden
    import urllib.request, io
    req = urllib.request.Request(
        image_url,
        headers={"User-Agent": "Mozilla/5.0 (CloFind bot)"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        data = r.read()

    q_tensor = _image_to_tensor(data)
    q_vec = _encode_image(q_tensor)

    # Suche
    topk = max(1, min(int(topk), 50))
    D, I = _index.search(q_vec, topk)
    sims = D[0].tolist()
    idxs = I[0].tolist()
    pid_hits = [int(_ids[i]) for i in idxs]

    # Produkte laden
    result = []
    with db_connect() as conn:
        from psycopg2.extras import RealDictCursor
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT p.id, p.title, p.price_cents, p.currency, p.deeplink, p.image_url,
                       m.name AS merchant
                FROM products p
                LEFT JOIN merchants m ON m.id = p.merchant_id
                WHERE p.id = ANY(%s)
            """, (pid_hits,))
            rows = {r["id"]: r for r in cur.fetchall()}

    for score, pid in zip(sims, pid_hits):
        r = rows.get(pid)
        if not r:
            continue
        result.append({
            "product_id": pid,
            "score": score,
            "label": _label_for(score),
            "title": r["title"],
            "price": (r["price_cents"] or 0) / 100.0,
            "currency": r["currency"],
            "merchant": r["merchant"],
            "deeplink": r["deeplink"],
            "image_url": r["image_url"],
        })

    return {"count": len(result), "results": result}


@app.post("/search/by-upload")
async def search_by_upload(
    file: UploadFile = File(...),
    topk: int = Form(5),
):
    ensure_loaded()

    data = await file.read()
    q_tensor = _image_to_tensor(data)
    q_vec = _encode_image(q_tensor)

    # Suche
    topk = max(1, min(int(topk), 50))
    D, I = _index.search(q_vec, topk)
    sims = D[0].tolist()
    idxs = I[0].tolist()
    pid_hits = [int(_ids[i]) for i in idxs]

    # Produkte laden
    result = []
    with db_connect() as conn:
        from psycopg2.extras import RealDictCursor
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT p.id, p.title, p.price_cents, p.currency, p.deeplink, p.image_url,
                       m.name AS merchant
                FROM products p
                LEFT JOIN merchants m ON m.id = p.merchant_id
                WHERE p.id = ANY(%s)
            """, (pid_hits,))
            rows = {r["id"]: r for r in cur.fetchall()}

    for score, pid in zip(sims, pid_hits):
        r = rows.get(pid)
        if not r:
            continue
        result.append({
            "product_id": pid,
            "score": score,
            "label": _label_for(score),
            "title": r["title"],
            "price": (r["price_cents"] or 0) / 100.0,
            "currency": r["currency"],
            "merchant": r["merchant"],
            "deeplink": r["deeplink"],
            "image_url": r["image_url"],
        })

    return {"count": len(result), "results": result}

