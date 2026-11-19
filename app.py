"""Worker API. Run: uvicorn app:app --host 0.0.0.0 --port 8000"""
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import pipeline

JOBS_DIR = Path(os.getenv("JOBS_DIR", "./jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL_JOBS", "1"))  # free CPU: one job at a time
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "1024"))
JOB_TTL_HOURS = float(os.getenv("JOB_TTL_HOURS", "6"))

app = FastAPI(title="Clipper Worker")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL)
jobs: dict[str, dict] = {}  # TODO Task 4: move to Supabase
lock = threading.Lock()


def _set(job_id: str, **kw):
    with lock:
        jobs[job_id].update(kw)


def _run(job_id: str, url: str | None, uploaded: Path | None, n: int):
    try:
        clips = pipeline.process(JOBS_DIR / job_id, url=url, uploaded=uploaded, n_clips=n,
                                 on_status=lambda s: _set(job_id, status=s))
        _set(job_id, status="done", clips=clips)
    except Exception as e:  # noqa: BLE001
        _set(job_id, status="error", error=str(e)[:500])


@app.post("/jobs")
async def create_job(url: str | None = Form(None), file: UploadFile | None = File(None),
                     n_clips: int = Form(5)):
    if not url and not file:
        raise HTTPException(400, "Provide a YouTube url or upload a video file")
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    uploaded = None
    if file:
        uploaded = job_dir / "source.mp4"
        size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
        with open(uploaded, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    f.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB)")
                f.write(chunk)
    with lock:
        jobs[job_id] = {"id": job_id, "status": "queued", "clips": []}
    executor.submit(_run, job_id, url, uploaded, max(1, min(n_clips, 10)))
    return jobs[job_id]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/jobs/{job_id}/clips/{name}")
def get_clip(job_id: str, name: str):
    path = (JOBS_DIR / job_id / name).resolve()
    if not path.is_file() or path.parent != (JOBS_DIR / job_id).resolve() or path.suffix != ".mp4":
        raise HTTPException(404, "Clip not found")
    return FileResponse(path, media_type="video/mp4", filename=name)


@app.get("/health")
def health():
    return {"ok": True}


def _cleanup_loop():
    """Free hosting has small disks: delete job folders older than JOB_TTL_HOURS."""
    while True:
        cutoff = time.time() - JOB_TTL_HOURS * 3600
        for d in JOBS_DIR.glob("*"):
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                with lock:
                    jobs.pop(d.name, None)
        time.sleep(600)


threading.Thread(target=_cleanup_loop, daemon=True).start()
@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")
