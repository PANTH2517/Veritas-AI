import os

import shutil

import uuid

import asyncio

import sys

import platform

from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks

from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from fastapi.responses import FileResponse

from app.services.scanner import FaceDeepfakeScannerService, UPLOAD_DIR

from app.database import init_db

@asynccontextmanager

async def lifespan(app: FastAPI):

    try:

        init_db()

    except Exception as e:

        print(f"Error initializing database on startup: {e}")

    yield

app = FastAPI(

    title="Veritas AI – Face Deepfake Detector",

    description="Forensic face-image analysis API that detects AI-generated faces.",

    version="2.0.0",

    lifespan=lifespan

)

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],

)

scanner_service = FaceDeepfakeScannerService()

static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True)

if os.environ.get("VERCEL") or not os.access(static_path, os.W_OK):
    cache_path = "/tmp/cache"
else:
    cache_path = os.path.join(static_path, "cache")
os.makedirs(cache_path, exist_ok=True)

app.mount("/static/cache", StaticFiles(directory=cache_path), name="static_cache")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")

async def read_index():

    index_file = os.path.join(static_path, "index.html")

    if os.path.exists(index_file):

        return FileResponse(index_file)

    return {"message": "Veritas AI Face Detector is active."}

@app.post("/api/scan/file")

async def scan_single_file(file: UploadFile = File(...)):

    temp_id = uuid.uuid4().hex

    ext = os.path.splitext(file.filename)[1].lower()

    temp_filename = f"upload_{temp_id}{ext}"

    temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)

    try:

        with open(temp_filepath, "wb") as buffer:

            shutil.copyfileobj(file.file, buffer)

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    try:

        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(

            scanner_service.executor,

            scanner_service.scan_media_file,

            temp_filepath,

            file.filename,

        )

        return result

    except ValueError as ve:

        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:

        import traceback

        traceback.print_exc()

        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:

        if os.path.exists(temp_filepath):

            try:

                os.remove(temp_filepath)

            except Exception:

                pass

@app.post("/api/scan/dual")

async def scan_dual_files(real_file: UploadFile = File(...), check_file: UploadFile = File(...)):

    temp_id = uuid.uuid4().hex

    real_ext = os.path.splitext(real_file.filename)[1].lower()

    check_ext = os.path.splitext(check_file.filename)[1].lower()

    real_temp_fn = f"dual_real_{temp_id}{real_ext}"

    check_temp_fn = f"dual_check_{temp_id}{check_ext}"

    real_temp_path = os.path.join(UPLOAD_DIR, real_temp_fn)

    check_temp_path = os.path.join(UPLOAD_DIR, check_temp_fn)

    try:

        with open(real_temp_path, "wb") as buffer:

            shutil.copyfileobj(real_file.file, buffer)

        with open(check_temp_path, "wb") as buffer:

            shutil.copyfileobj(check_file.file, buffer)

    except Exception as e:

        for path in [real_temp_path, check_temp_path]:

            if os.path.exists(path):

                os.remove(path)

        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files: {str(e)}")

    try:

        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(

            scanner_service.executor,

            scanner_service.scan_dual_faces,

            real_temp_path,

            real_file.filename,

            check_temp_path,

            check_file.filename

        )

        return result

    except ValueError as ve:

        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:

        import traceback

        traceback.print_exc()

        raise HTTPException(status_code=500, detail=f"Dual verification failed: {str(e)}")

    finally:

        for path in [real_temp_path, check_temp_path]:

            if os.path.exists(path):

                try:

                    os.remove(path)

                except Exception:

                    pass

@app.get("/api/history")

async def get_scan_history():

    return scanner_service.get_history()

@app.get("/api/history/{scan_id}")

async def get_single_history_item(scan_id: str):

    from app.database import get_scan_from_history

    result = get_scan_from_history(scan_id)

    if not result:

        raise HTTPException(status_code=404, detail="Historical scan not found.")

    return result

@app.delete("/api/history/{scan_id}")

async def delete_history_item(scan_id: str):

    from app.database import delete_scan_from_history

    deleted = delete_scan_from_history(scan_id)

    if not deleted:

        raise HTTPException(status_code=404, detail="Scan record not found.")

    scanner_service.history = [h for h in scanner_service.history if h.get("scan_id") != scan_id]

    return {"status": "deleted", "scan_id": scan_id}

@app.delete("/api/history")

async def clear_all_history():

    from app.database import clear_all_history as db_clear

    db_clear()

    scanner_service.history = []

    return {"status": "cleared"}

@app.get("/api/health")

async def get_health():

    return {

        "status": "ONLINE",

        "version": "2.0.0",

        "system": platform.system(),

        "python_version": sys.version.split()[0],

        "device": "CPU (TensorFlow / OpenCV)",

        "models_loaded": {

            "face_deepfake_classifier": "MobileNetV2 + Forensic Signals (22 channels)",

            "face_detector": "OpenCV Haar Cascade Classifier",

            "voice_deepfake_classifier": "Spectral Anomaly Neural Network (6 channels)",

        },

    }

