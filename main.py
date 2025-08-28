from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()  # MVP: nur im Speicher
    # TODO: später in S3 speichern
    return {"upload_id": "temp-123"}
