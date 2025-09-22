import os, sys
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
dsn = os.getenv("DATABASE_URL")
if not dsn:
    print("ERROR: DATABASE_URL fehlt")
    sys.exit(1)

# Wir setzen für die ersten N Produkte eine garantiert ladbare Bild-URL:
# https://picsum.photos/seed/<pid>/600/800.jpg  -> konsistent & schnell
N = 20

with psycopg2.connect(dsn) as conn:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id FROM products
            ORDER BY id ASC
            LIMIT %s
        """, (N,))
        rows = cur.fetchall()

        for r in rows:
            pid = r["id"]
            new_url = f"https://picsum.photos/seed/{pid}/600/800.jpg"
            cur.execute("UPDATE products SET image_url=%s WHERE id=%s", (new_url, pid))

    conn.commit()

print(f"OK – {len(rows)} Produkte auf Picsum-URLs gesetzt.")
