import os, sys, io, ssl, urllib.request
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
dsn = os.getenv("DATABASE_URL")
if not dsn:
    print("ERROR: DATABASE_URL fehlt")
    sys.exit(1)

ctx = ssl.create_default_context()
# weniger strikt, falls es Self-signed/alte Ciphers gibt (nur für Tests!)
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

ok = 0
fail = 0
samples = []

with psycopg2.connect(dsn) as conn:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, image_url FROM products WHERE image_url IS NOT NULL LIMIT 50")
        rows = cur.fetchall()

for r in rows:
    pid, url = r["id"], r["image_url"]
    if not url or not url.startswith("http"):
        print(f"[{pid}] Ungültige URL: {url}")
        fail += 1
        continue
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = resp.read(2048)  # nur einen kleinen Chunk
            ctype = resp.headers.get("Content-Type", "")
        if not ctype.startswith("image/") and not (data[:4] in [b"\x89PNG", b"\xFF\xD8\xFF\xE0", b"\xFF\xD8\xFF\xE1"]):
            print(f"[{pid}] Kein Bild-Content-Type: {ctype} ({url})")
            fail += 1
        else:
            ok += 1
            if len(samples) < 5:
                samples.append((pid, url, ctype))
    except Exception as e:
        print(f"[{pid}] FEHLER: {e} ({url})")
        fail += 1

print(f"\nErgebnis: OK={ok}  FAIL={fail}  (gesamt: {len(rows)})")
if samples:
    print("Beispiele OK:")
    for s in samples:
        print(" ", s)
