import os
import io
import sys
import time
import faiss
import clip
import torch
import urllib.request
import tempfile
import numpy as np
from PIL import Image
from tqdm import tqdm
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# --- Settings ---
BATCH_SIZE = 16              # größer = schneller, aber mehr RAM
MODEL_NAME = "ViT-B/32"      # stabil & schnell für MVP
INDEX_PATH = "faiss.index"
IDS_PATH = "ids.npy"

def db_connect():
    load_dotenv()
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL fehlt (.env)!", file=sys.stderr)
        sys.exit(1)
    return psycopg2.connect(dsn)

def fetch_products(cur, limit=None):
    # hole nur Produkte mit Bild-URL
    sql = "SELECT id, image_url FROM products WHERE image_url IS NOT NULL"
    if limit:
        sql += " LIMIT %s"
        cur.execute(sql, (limit,))
    else:
        cur.execute(sql)
    return cur.fetchall()

def load_image_from_url(url, preprocess):
    # Lade Bild in RAM (keine Dateischreibrechte nötig)
    with urllib.request.urlopen(url, timeout=15) as r:
        data = r.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return preprocess(img).unsqueeze(0)

def main():
    device = "cpu"  # MVP: CPU-Modus
    print("Lade CLIP:", MODEL_NAME)
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    # DB
    print("Verbinde zur DB…")
    conn = db_connect()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    print("Lese Produkte (id, image_url)…")
    rows = fetch_products(cur)  # evtl. limit=1000 zum Testen
    if not rows:
        print("Keine Produkte mit image_url gefunden.")
        return

    ids = []
    all_vecs = []
    batch_imgs = []
    batch_ids = []

    def flush_batch():
        nonlocal batch_imgs, batch_ids, all_vecs, ids
        if not batch_imgs:
            return
        with torch.no_grad():
            imgs = torch.cat(batch_imgs, dim=0).to(device)
            feats = model.encode_image(imgs).cpu().numpy().astype("float32")
        all_vecs.append(feats)
        ids.extend(batch_ids)
        batch_imgs, batch_ids = [], []

    print(f"Embeddings berechnen für {len(rows)} Produkte…")
    for r in tqdm(rows):
        pid = r["id"]
        url = r["image_url"]
        try:
            img_tensor = load_image_from_url(url, preprocess)
            batch_imgs.append(img_tensor)
            batch_ids.append(pid)
            if len(batch_imgs) >= BATCH_SIZE:
                flush_batch()
        except Exception as e:
            # Bild nicht ladbar → überspringen
            # (Optional: hier könnte man in DB ein Flag setzen)
            continue

    flush_batch()

    if not ids:
        print("Keine gültigen Bilder konnten geladen werden.")
        return

    emb = np.vstack(all_vecs)  # [N, D]
    # Für Cosine-Similarity: Vektoren L2-normalisieren und IndexFlatIP (inner product) nutzen
    faiss.normalize_L2(emb)

    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)

    faiss.write_index(index, INDEX_PATH)
    np.save(IDS_PATH, np.array(ids, dtype=np.int64))
    print(f"Fertig. Gespeichert: {INDEX_PATH} und {IDS_PATH}")
    print(f"Vektoren: {emb.shape[0]}  Dim: {emb.shape[1]}")

if __name__ == "__main__":
    main()

