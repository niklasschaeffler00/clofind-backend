from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
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

# CORS fürs MVP offen (später einschränken)
# CORS nur für dein Frontend + Localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://clofind.vercel.app",  # dein Vercel-Frontend
        "http://localhost:3000",       # fürs lokale Entwickeln
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# STARTUP: FAISS-Index bauen (optional) und Suche einhängen
# -----------------------------------------------------------------------------
@app.on_event("startup")
def _ensure_faiss_index():
    if os.getenv("SKIP_BUILD_INDEX_ON_START") == "1":
        logging.info("SKIP_BUILD_INDEX_ON_START=1 -> überspringe Index-Build beim Start.")
        _attach_search()
        return

    if not (os.path.exists("faiss.index") and os.path.exists("ids.npy")):
        logging.info("FAISS-Index nicht gefunden – baue neu …")
        try:
            subprocess.run([sys.executable, "build_index.py"], check=True)
            logging.info("Index-Build ok.")
        except Exception as e:
            # Wichtig: Service nicht crashen lassen (Health/Docs bleiben erreichbar)
            logging.warning(f"Index-Build fehlgeschlagen: {e}")

    # danach immer versuchen, die Suche einzubinden
    _attach_search()

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
    
@app.get("/version")
def version():
    import os
    return {
        "commit": os.getenv("RAILWAY_GIT_COMMIT_SHA", "unknown"),
        "env": os.getenv("RAILWAY_ENVIRONMENT", "dev"),
    }

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
# SUCHE einhängen (aus app_search.py) – optional & in /docs sichtbar
# -----------------------------------------------------------------------------
_SEARCH_ATTACHED = False

def _attach_search():
    """Versucht die Routen aus app_search in die Haupt-App zu übernehmen.
    Scheitert nie hart, sondern registriert bei Fehlern Stub-Routen (503 statt 404).
    """
    global _SEARCH_ATTACHED
    if _SEARCH_ATTACHED:
        return
    try:
        from app_search import router as search_router
        app.include_router(search_router)
        logging.info("Search API enabled")
        _SEARCH_ATTACHED = True
    except Exception as e:
        logging.warning(f"Suchen-API (app_search) nicht aktiv: {e}")

        # Fallback-Stub, damit das Frontend niemals 404 bekommt
        stub = APIRouter(prefix="/search", tags=["search"])

        @stub.post("/by-upload")
        async def search_stub(file: UploadFile = File(...), topk: int = Form(5)):
            raise HTTPException(status_code=503, detail=f"Search disabled on this deploy: {e}")

        @stub.post("/image")
        async def search_stub_alias(file: UploadFile = File(...), topk: int = Form(5)):
            return await search_stub(file, topk)

        app.include_router(stub)
        _SEARCH_ATTACHED = True
        logging.info("Registered search stub endpoints (503)")

# >>> NEU: sofort (best effort) anhängen – echte Suche ODER Stub, aber nie 404
_attach_search()
