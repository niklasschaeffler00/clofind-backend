# app_search.py
import os
import io
import urllib.request
import urllib.error
from typing import List, Dict, Any

import numpy as np
import faiss
import torch
import clip
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# ---------------------------------------------------------------------
# ENV / Flags
# ---------------------------------------------------------------------
load_dotenv()
ENABLE_URL_SEARCH = os.getenv("ENABLE_URL_SEARCH", "0") == "1"

# Caches in /tmp, damit Railway nicht jammert
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
os.environ.setdefault("TORCH_HOME", "/tmp/.cache")

# ---------------------------------------------------------------------
# FastAPI-App
# ---------------------------------------------------------------------
app = FastAPI(title="CloFind Visual Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # TODO: im Prod einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Globale Ressourcen
# ---------------------------------------------------------------------
MODEL_NAME = "ViT-B/32"
INDEX_PATH = "faiss.index"
IDS_PATH   = "ids.npy"

_device: str = "cpu"
_model = None
_preprocess = None
_index = None
_ids: np.ndarray | None = None

def db_connect():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL fehlt")
    return psycopg2.connect(dsn)

def ensure_loaded():
    """Lädt CLIP + FAISS + IDs lazy einmalig."""
    global _model, _preprocess, _index, _ids
    if _model is None or _preprocess is None:
        # cache dir aus XDG_CACHE_HOME übernehmen
        cache_root = os.getenv("XDG_CACHE_HOME", "/tmp/.cache")
        _model, _preprocess = clip.load(MODEL_NAME, device=_device, download_root=cache_root)
        _model.eval()
    if _index is None or _ids is None:
        if not (os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH)):
            raise HTTPException(status_code=503, detail="Indexdateien fehlen (faiss.index / ids.npy).")
        _index = faiss.read_index(INDEX_PATH)
        _ids   = np.load(IDS_PATH)

def _image_to_tensor(data: bytes):
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Die Datei/URL liefert kein gültiges Bild.")
    return _preprocess(img).unsqueeze(0)

def _encode_image(tensor):
    with torch.no_grad():
        vec = _model.encode_image(tensor.to(_device)).cpu().numpy().astype("float32")
    faiss.normalize_L2(vec)
    return vec

def _label_for(score: float) -> str:
    if score >= 0.85:
        return "Exact"
    if score >= 0.60:
        return "Sehr ähnlich"
    return "Alternative"

def _fetch_image_bytes(url: str, timeout: int = 15) -> bytes:
    """Download mit User-Agent; mappt Netzwerkfehler auf 400 statt 500."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Image URL HTTP {e.code}")
    except urllib.error.URLError as e:
        raise HTTPException(status_code=400, detail=f"Image URL error: {e.reason}")

# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------- Upload-Suche (aktiv) --------
@app.post("/search/by-upload")
async def search_by_upload(
    file: UploadFile = File(...),
    topk: int = Form(5),
):
    ensure_loaded()

    data = await file.read()
    q_tensor = _image_to_tensor(data)
    q_vec = _encode_image(q_tensor)

    k = max(1, min(int(topk), 50))
    D, I = _index.search(q_vec, k)
    sims = D[0].tolist()
    idxs = I[0].tolist()
    pid_hits = [int(_ids[i]) for i in idxs]  # type: ignore[index]

    # Produkte laden
    result: List[Dict[str, Any]] = []
    with db_connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT p.id, p.title, p.price_cents, p.currency, p.deeplink, p.image_url,
                       m.name AS merchant
                FROM products p
                LEFT JOIN merchants m ON m.id = p.merchant_id
                WHERE p.id = ANY(%s)
                """,
                (pid_hits,),
            )
            rows = {r["id"]: r for r in cur.fetchall()}

    for score, pid in zip(sims, pid_hits):
        r = rows.get(pid)
        if not r:
            continue
        result.append(
            {
                "product_id": pid,
                "score": score,
                "label": _label_for(score),
                "title": r["title"],
                "price": (r["price_cents"] or 0) / 100.0,
                "currency": r["currency"],
                "merchant": r["merchant"],
                "deeplink": r["deeplink"],
                "image_url": r["image_url"],
            }
        )

    return {"count": len(result), "results": result}

# Optionaler Alias (falls dein Frontend /search/image nutzt)
@app.post("/search/image")
async def search_image_alias(
    file: UploadFile = File(...),
    topk: int = Form(5),
):
    return await search_by_upload(file=file, topk=topk)

# -------- URL-Suche (nur wenn aktiviert) --------
if ENABLE_URL_SEARCH:
    @app.post("/search/by-url")
    async def search_by_url(
        image_url: str = Form(...),
        topk: int = Form(5),
    ):
        ensure_loaded()

        data = _fetch_image_bytes(image_url)
        q_tensor = _image_to_tensor(data)
        q_vec = _encode_image(q_tensor)

        k = max(1, min(int(topk), 50))
        D, I = _index.search(q_vec, k)
        sims = D[0].tolist()
        idxs = I[0].tolist()
        pid_hits = [int(_ids[i]) for i in idxs]  # type: ignore[index]

        result: List[Dict[str, Any]] = []
        with db_connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT p.id, p.title, p.price_cents, p.currency, p.deeplink, p.image_url,
                           m.name AS merchant
                    FROM products p
                    LEFT JOIN merchants m ON m.id = p.merchant_id
                    WHERE p.id = ANY(%s)
                    """,
                    (pid_hits,),
                )
                rows = {r["id"]: r for r in cur.fetchall()}

        for score, pid in zip(sims, pid_hits):
            r = rows.get(pid)
            if not r:
                continue
            result.append(
                {
                    "product_id": pid,
                    "score": score,
                    "label": _label_for(score),
                    "title": r["title"],
                    "price": (r["price_cents"] or 0) / 100.0,
                    "currency": r["currency"],
                    "merchant": r["merchant"],
                    "deeplink": r["deeplink"],
                    "image_url": r["image_url"],
                }
            )

        return {"count": len(result), "results": result}
