# etl_import.py
import os, csv, psycopg2
from psycopg2.extras import execute_values

# 1) DB-Verbindung (liest DATABASE_URL aus deiner .env, wenn du via VS Code/Terminal mit dotenv arbeitest)
# Falls die Variable im Prozess nicht gesetzt ist, lies die .env manuell ein:
if "DATABASE_URL" not in os.environ:
    # einfache, sichere .env-Leseroutine (ohne extra Paket)
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL, "DATABASE_URL fehlt. Bitte in .env setzen."

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = False
cur = conn.cursor()

# 2) Tabellen anlegen (idempotent)
cur.execute("""
CREATE TABLE IF NOT EXISTS merchants (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    merchant_id INTEGER NOT NULL REFERENCES merchants(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    currency TEXT NOT NULL,
    deeplink TEXT,
    image_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""")

# 3) CSV lesen
rows_to_insert = []
with open("products.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Whitespace säubern
        merchant = (row["merchant"] or "").strip()
        title = (row["title"] or "").strip()
        currency = (row["currency"] or "").strip().upper()
        deeplink = (row["deeplink"] or "").strip()
        image_url = (row["image_url"] or "").strip()

        try:
            price_cents = int(str(row["price_cents"]).strip())
        except Exception:
            raise ValueError(f"Ungültiger price_cents-Wert in Zeile: {row}")

        # Händler einfügen/finden
        cur.execute(
            "INSERT INTO merchants(name) VALUES(%s) ON CONFLICT (name) DO NOTHING RETURNING id",
            [merchant],
        )
        if cur.rowcount:
            mid = cur.fetchone()[0]
        else:
            cur.execute("SELECT id FROM merchants WHERE name=%s", [merchant])
            mid = cur.fetchone()[0]

        rows_to_insert.append((
            mid, title, price_cents, currency, deeplink, image_url
        ))

from psycopg2.extras import execute_values

# Produkte in einem Rutsch einfügen (Upsert über UNIQUE Constraint)
sql = """
INSERT INTO products (merchant_id, title, price_cents, currency, deeplink, image_url)
VALUES %s
ON CONFLICT (merchant_id, title, deeplink) DO UPDATE
SET price_cents = EXCLUDED.price_cents,
    currency    = EXCLUDED.currency,
    image_url   = EXCLUDED.image_url
"""
execute_values(cur, sql, rows_to_insert)

# 5) Commit & kurzer Check
conn.commit()

cur.execute("SELECT COUNT(*) FROM products;")
count = cur.fetchone()[0]
print(f"Import abgeschlossen. Produkte in DB: {count}")

cur.execute("SELECT title, price_cents, currency FROM products ORDER BY id DESC LIMIT 5;")
sample = cur.fetchall()
print("Beispielzeilen:")
for t, p, c in sample:
    print(f"- {t} | {p} {c}")

cur.close()
conn.close()
print("Fertig.")
