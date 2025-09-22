from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import os, io

load_dotenv()  # .env einmal laden

# --- DB ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL fehlt. Lege eine .env mit DATABASE_URL=... an.")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 5}
)

# --- App ---
app = FastAPI()

# CORS fürs MVP offen (später einschränken)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health ---
@app.get("/health")
def health():
    return {"ok": True}

# --- DB Ping ---
@app.get("/db_ping")
def db_ping():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"db": "ok"}
    except SQLAlchemyError as e:
        return {"db": "error", "detail": str(e)}

# --- Products (dein bestehender Endpoint) ---
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

# --- Upload (einziger /upload Endpoint) ---
from app.storage import save_file  # Pfad ok, weil wir storage.py unter outfit-backend/app/ haben

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    meta = save_file(io.BytesIO(await file.read()), file.filename)
    return {
        "filename": file.filename,
        "mime_type": file.content_type,
        "storage": meta["storage"],
        "location": meta["location"],
        "key_or_path": meta["key_or_path"],
    }
from app.storage import S3_ENABLED, S3_BUCKET, S3_ENDPOINT
