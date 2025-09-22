from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import os, io, sys, subprocess, logging

# -----------------------------------------------------------------------------
# ENV & DB
# -----------------------------------------------------------------------------
load_dotenv()  # .env einmal laden

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL fehlt. Lege eine .env mit DATABASE_URL=... an.")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 5},
)

# -----------------------------------------------------------------------------
# APP
# -----------------------------------------------------------------------------
app = FastAPI()

# Beim Start FAISS-Index bauen, falls nicht vorhanden
@app.on_event("startup")
def _ensure_faiss_index():
    if os.getenv("SKIP_BUILD_INDEX_ON_START") == "1":
        logging.info("SKIP_BUILD_INDEX_ON_START=1 -> überspringe Index-Build beim Start.")
        return
    if not (os.path.exists("faiss.index") and os.path.exists("ids.npy")):
        logging.info("FAISS-Index nicht gefunden – baue neu …")
        try:
            subprocess.run([sys.executable, "build_index.py"], check=True)
            logging.info("Index-Build ok.")
        except Exception as e:
            # Wichtig: Service nicht crashen lassen (Health/Docs bleiben erreichbar)
            logging.warning(f"Index-Build fehlgeschlagen: {e}")

# CORS fürs MVP offen (später einschränken)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# HEALTH / DB
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/db_ping")
def db_ping():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"db": "ok"}
    except SQLAlchemyError as e:
        return {"db": "error", "detail": str(e)}

# -----------------------------------------------------------------------------
# PRODUCTS
# -----------------------------------------------------------------------------
@app.get("/products")
def get_products(limit: int = 20, q: str | None = None, merchant: str | None = None):
    where = ["p.active = true"]
    params: dict = {"limit": limit}

    if q:
        where.append("p.title ILIKE :q")
        params["q"] = f"%{q}%"
    if merchant:
        where.append("(m.domain = :merchant OR m.name = :merchant)")
        params["merchant"] = merchant

    sql = f"""
        SELECT 
            p.id, p.title, p.price_cents, p.currency, p.image_url, p.deeplink,
            m.name AS merchant_name, m.domain AS merchant_domain
        FROM products p
        JOIN merchants m ON m.id = p.merchant_id
        WHERE {' AND '.join(where)}
        ORDER BY p.updated_at DESC
        LIMIT :limit
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return {"items": [dict(r) for r in rows]}

# -----------------------------------------------------------------------------
# UPLOAD
# -----------------------------------------------------------------------------
try:
    from app.storage import save_file  # storage.py liegt unter outfit-backend/app/
    _STORAGE_READY = True
except Exception as e:
    logging.warning(f"Storage nicht aktiv (upload deaktiviert): {e}")
    _STORAGE_READY = False

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not _STORAGE_READY:
        return {"error": "storage_not_configured"}
    meta = save_file(io.BytesIO(await file.read()), file.filename)
    return {
        "filename": file.filename,
        "mime_type": file.content_type,
        "storage": meta.get("storage"),
        "location": meta.get("location"),
        "key_or_path": meta.get("key_or_path"),
    }

# -----------------------------------------------------------------------------
# SUCHE einhängen (aus app_search.py) – optional & fehlertolerant
# -----------------------------------------------------------------------------
def _mount_search():
    try:
        # app_search enthält eine FastAPI-App namens "app"
        from app_search import app as search_app  # Import heavy deps (faiss/clip) hier kapseln
        # Root mounten, damit /search/... Pfade erhalten bleiben
        app.mount("", search_app)
        logging.info("Mounted app_search at root")
    except Exception as e:
        logging.warning(f"Suchen-API (app_search) nicht aktiv: {e}")

_mount_search()
