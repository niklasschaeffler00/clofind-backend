import os
import uuid
import pathlib
import boto3
from typing import BinaryIO
from botocore.config import Config

# Werte aus .env laden
S3_ENABLED = os.getenv("S3_ENABLED", "false").lower() == "true"
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY_ID = os.getenv("S3_KEY_ID")
S3_APP_KEY = os.getenv("S3_APP_KEY")
LOCAL_UPLOAD_DIR = os.getenv("LOCAL_UPLOAD_DIR", "./uploads")  # Windows-freundlich

def _s3_client():
    """Erzeugt einen S3-Client für Backblaze (v4 Signatur + PATH-Style!)."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY_ID,
        aws_secret_access_key=S3_APP_KEY,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"}  # wichtig bei Groß-/Sonderzeichen im Bucketnamen
        ),
    )

def save_file(fileobj: BinaryIO, filename: str) -> dict:
    """
    Speichert eine Datei in S3 (wenn aktiviert) oder lokal.
    Gibt Metadaten zurück: {"location": "...", "storage": "s3|local", "key_or_path": "..."}
    """
    ext = pathlib.Path(filename).suffix.lower() or ""
    key = f"uploads/{uuid.uuid4().hex}{ext}"

    # Debug: aktuelle Storage-Konfiguration loggen
    print(f"[storage] S3_ENABLED={S3_ENABLED} bucket={S3_BUCKET} endpoint={S3_ENDPOINT}")

    if S3_ENABLED:
        try:
            s3 = _s3_client()
            fileobj.seek(0)
            # Optional: ContentType setzen (hilfreich für spätere Auslieferung/Preview)
            s3.upload_fileobj(
                fileobj, S3_BUCKET, key,
                ExtraArgs={"ContentType": _guess_mime(ext)}
            )
            print(f"[storage] S3 upload OK -> s3://{S3_BUCKET}/{key}")
            return {
                "location": f"s3://{S3_BUCKET}/{key}",
                "storage": "s3",
                "key_or_path": key,
            }
        except Exception as e:
            # Fallback auf lokal, aber Fehler sichtbar machen
            print(f"[storage] S3-Upload fehlgeschlagen, fallback auf lokal: {e}")

    # Lokal speichern
    pathlib.Path(LOCAL_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    dest_path = pathlib.Path(LOCAL_UPLOAD_DIR) / pathlib.Path(key).name
    fileobj.seek(0)
    with open(dest_path, "wb") as out:
        out.write(fileobj.read())

    print(f"[storage] Lokal gespeichert -> {dest_path}")
    return {
        "location": str(dest_path),
        "storage": "local",
        "key_or_path": str(dest_path),
    }

def _guess_mime(ext: str) -> str:
    """Sehr einfache MIME-Erkennung für gängige Bildtypen."""
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "application/octet-stream"
