# app_search.py
import os
import io
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# pgvector Adapter (für DB-Suche)
from pgvector.psycopg2 import register_vector

# ------------------------------------------------------------
# ENV / Flags
# ------------------------------------------------------------
load_dotenv()
ENABLE_URL_SEARCH = os.getenv("ENABLE_URL_SEARCH", "0") == "1"

# Caches (z.B. Railway)
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
os.environ.setdefault("TORCH_HOME", "/tmp/.cache")

# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
router = APIRouter(prefix="/search", tags=["search"])

# ------------------------------------------------------------
# Globale Ressourcen (lazy)
# ------------------------------------------------------------
MODEL_NAME = "ViT-B/32"
INDEX_PATH = "artifacts/faiss.index"
IDS_PATH   = "artifacts/ids.npy"

_device: str = "cpu"

# Lazy-Singletons
_model = None
_preprocess = None

# FAISS/ID-Dateien (nur für FAISS-Endpunkte)
_faiss = None
_index = None
_ids: Optional[np.ndarray] = None

# Torch/CLIP/PIL (nur wenn wir encodieren)
_torch = None
_clip = None
_Image = None


def db_connect():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL fehlt")
    return psycopg2.connect(dsn)


# ----------------------- Loader -----------------------------

def ensure_clip():
    """Lädt nur Torch, PIL, CLIP und das Modell/Preprocess."""
    global _torch, _clip, _Image, _model, _preprocess

    if _torch is None or _clip is None or _Image is None:
        try:
            import torch as _torch_mod
            import clip as _clip_mod  # OpenAI CLIP
            from PIL import Image as _PIL_Image
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"CLIP stack not available: {e}")

        _globals = globals()
        _globals["_torch"] = _torch_mod
        _globals["_clip"] = _clip_mod
        _globals["_Image"] = _PIL_Image

    if _model is None or _preprocess is None:
        cache_root = os.getenv("XDG_CACHE_HOME", "/tmp/.cache")
        _model_loaded, _preproc = _clip.load(MODEL_NAME, device=_device, download_root=cache_root)
        _model_loaded.eval()
        _globals = globals()
        _globals["_model"] = _model_loaded
        _globals["_preprocess"] = _preproc


def ensure_faiss_index():
    """Lädt FAISS + Index + IDs von Platte (nur für FAISS-Suche)."""
    global _faiss, _index, _ids
    if _faiss is None:
        try:
            import faiss as _faiss_mod
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"FAISS not available: {e}")
        globals()["_faiss"] = _faiss_mod

    if _index is None or _ids is None:
        if not (os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH)):
            raise HTTPException(status_code=503, detail="Indexdateien fehlen (faiss.index / ids.npy).")
        globals()["_index"] = _faiss.read_index(INDEX_PATH)
        globals()["_ids"] = np.load(IDS_PATH)


# -------------------- Utils --------------------------------

def _image_to_tensor(data: bytes):
    try:
        img = _Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Die Datei/URL liefert kein gültiges Bild.")
    return _preprocess(img).unsqueeze(0)


def _encode_image(tensor):
    with _torch.no_grad():
        vec = _model.encode_image(tensor.to(_device)).cpu().numpy().astype("float32")
    # Für Cosine-Ähnlichkeit:
    if _faiss is not None:
        _faiss.normalize_L2(vec)
    else:
        # Falls FAISS nicht geladen ist, normalisieren wir manuell:
        n = np.linalg.norm(vec, axis=1, keepdims=True)
        n[n == 0] = 1.0
        vec = vec / n
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


# ------------------ DB-KNN (pgvector) -----------------------

def _search_db_with_vector(vec, topk: int):
    """Sucht direkt in image_embeddings (pgvector)."""
    with db_connect() as conn:
        register_vector(conn)  # vector-Adapter für psycopg2
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Anzahl durchsuchter Listen im IVF-Index (falls vorhanden)
            vec_str = "[" + ",".join(str(float(x)) for x in vec) + "]"
            cur.execute("SET LOCAL ivfflat.probes = 10;")
            cur.execute(
                """
                SELECT 
                    ie.image_id,
                    p.id   AS product_id,
                    p.title,
                    i.image_url,
                    1 - (ie.embedding <=> %s) AS similarity
                FROM image_embeddings ie
                JOIN images   i ON i.id = ie.image_id
                JOIN products p ON p.id = i.product_id
                WHERE p.active = true
                ORDER BY ie.embedding <=> %s
                LIMIT %s
                """,
                (vec_str, vec_str, int(topk)),
            )
            return cur.fetchall()


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------

@router.post("/by-upload-db")
async def search_by_upload_db(
    file: UploadFile = File(...),
    topk: int = Form(5),
):
    """DB-basierte KNN-Suche (pgvector), durchsucht direkt die Datenbank."""
    ensure_clip()

    data = await file.read()
    q_tensor = _image_to_tensor(data)
    q_vec = _encode_image(q_tensor)[0].tolist()  # 512-dim, L2-normalisiert

    rows = _search_db_with_vector(q_vec, topk)

    result: List[Dict[str, Any]] = []
    for r in rows:
        result.append(
            {
                "product_id": r["product_id"],
                "score": float(r["similarity"]),
                "label": _label_for(float(r["similarity"])),
                "title": r["title"],
                "price": None,          # optional nachjoinen
                "currency": None,
                "merchant": None,
                "deeplink": None,
                "image_url": r["image_url"],
            }
        )
    return {"count": len(result), "results": result}


@router.post("/by-upload")
async def search_by_upload(
    file: UploadFile = File(...),
    topk: int = Form(5),
):
    """FAISS-basierte Suche (nutzt lokale faiss.index + ids.npy)."""
    ensure_clip()
    ensure_faiss_index()

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


@router.post("/image")
async def search_image_alias(
    file: UploadFile = File(...),
    topk: int = Form(5),
):
    """Alias auf die FAISS-Suche (Kompatibilität)."""
    return await search_by_upload(file=file, topk=topk)


if ENABLE_URL_SEARCH:
    @router.post("/by-url")
    async def search_by_url(
        image_url: str = Form(...),
        topk: int = Form(5),
    ):
        """FAISS-Suche via Bild-URL."""
        ensure_clip()
        ensure_faiss_index()

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
