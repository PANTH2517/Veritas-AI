"""
Vercel entry point — wraps the FastAPI app with a top-level safety net.
If the main app fails to import, a minimal fallback app is used so
/api/health always returns 200 with the actual error message.
"""
import sys
import traceback

_import_error = None

try:
    from app.main import app
except Exception as _e:
    _import_error = traceback.format_exc()
    print(f"[index] IMPORT ERROR:\n{_import_error}")

    # Build a minimal fallback app so /api/health always answers
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Veritas AI — startup error")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    async def health_fallback():
        return {
            "status": "ONLINE",
            "version": "2.0.0",
            "startup_error": _import_error,
            "models_loaded": False,
        }

    @app.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT"])
    async def catch_all(path: str):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable due to startup error: {_import_error}"
        )
