import os, io, sys, urllib.request
import numpy as np
import faiss, torch, clip
from PIL import Image
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Nutzung: python search_test.py <image_url> [topk]
# Beispiel: python search_test.py https://picsum.photos/seed/999/600/800.jpg 5

def db_connect():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL fehlt (.env)")
        sys.exit(1)
    return psycopg2.connect(dsn)

def load_image_from_url(url, preprocess):
    with urllib.request.urlopen(url, timeout=15) as r:
        data = r.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return preprocess(img).unsqueeze(0)

def main():
    load_dotenv()
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    query_url = sys.argv[1] if len(sys.argv) > 1 else None
    if not query_url:
        print("Bitte Bild-URL angeben. Beispiel:")
        print("  python search_test.py https://picsum.photos/seed/999/600/800.jpg 5")
        sys.exit(1)

    # 1) Modelle & Index laden
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    index = faiss.read_index("faiss.index")
    ids = np.load("ids.npy")

    # 2) Query-Embedding
    with torch.no_grad():
        q = load_image_from_url(query_url, preprocess).to(device)
        q_vec = model.encode_image(q).cpu().numpy().astype("float32")
    faiss.normalize_L2(q_vec)

    # 3) Suche
    D, I = index.search(q_vec, topk)  # Cosine-Sim via normalized IP
    sims = D[0]
    idxs = I[0]
    pid_hits = [int(ids[i]) for i in idxs]

    # 4) Produkte aus DB ziehen und anzeigen
    with db_connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, title, price_cents, currency, merchant_id, deeplink, image_url "
                "FROM products WHERE id = ANY(%s)",
                (pid_hits,)
            )
            rows = {r["id"]: r for r in cur.fetchall()}

    print("\nTop-Ergebnisse:")
    for rank, (pid, score) in enumerate(zip(pid_hits, sims), start=1):
        r = rows.get(pid)
        if not r:
            continue
        # einfache Label-Logik für MVP
        label = "Exact" if score >= 0.30 else ("Sehr ähnlich" if score >= 0.20 else "Alternative")
        price = f"{(r['price_cents'] or 0)/100:.2f} {r['currency'] or ''}".strip()
        print(f"{rank}. [{label}] PID={pid}  score={score:.3f}  {r['title']!r}  {price}  {r['deeplink']}\n    img: {r['image_url']}")

if __name__ == "__main__":
    main()
