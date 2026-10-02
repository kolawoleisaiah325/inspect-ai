"""InspectAI API. Uploads are decoded in memory; images are never persisted."""
import functools
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from model import Inspector, MAX_BYTES

ROOT = Path(__file__).parent
app = FastAPI(title="InspectAI", version="1.0.0", description="Noncommercial bottle benchmark prototype. Scores are not probabilities.")

@functools.lru_cache(maxsize=1)
def inspector(): return Inspector()

@app.get("/api/health")
def health():
    metadata = json.loads((ROOT / "artifacts/metadata.json").read_text())
    return {"status": "ready", "model": metadata["model_version"], "input_size": metadata["input_size"],
            "threshold": metadata["threshold"], "uploads_stored": False}

@app.post("/api/inspect", openapi_extra={"requestBody": {"required": True, "content": {
    "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
    "image/png": {"schema": {"type": "string", "format": "binary"}},
    "image/webp": {"schema": {"type": "string", "format": "binary"}}}}})
async def inspect(request: Request):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BYTES:
            return JSONResponse({"detail": "Choose an image smaller than 2 MB."}, status_code=413)
        chunks.append(chunk)
    try:
        result = await run_in_threadpool(inspector().inspect, b"".join(chunks))
    except ValueError as error:
        return JSONResponse({"detail": str(error)}, status_code=422)
    response = JSONResponse(result)
    response.headers["Cache-Control"] = "no-store"
    return response

# For local Uvicorn and Docker; Vercel serves the public directory as static assets.
# Vercel removes public/ from the function bundle, so it must not be mounted there.
if (ROOT / "public").is_dir():
    @app.get("/")
    def index(): return FileResponse(ROOT / "public/index.html")
    app.mount("/", StaticFiles(directory=ROOT / "public"), name="public")
