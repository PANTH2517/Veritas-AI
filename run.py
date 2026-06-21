import uvicorn
import os

if __name__ == "__main__":
    # Ensure necessary folders exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "data", "uploads"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "app", "static", "cache"), exist_ok=True)
    
    print("--------------------------------------------------")
    print("Launching Veritas AI Deepfake Verification Guard...")
    print("Server starting at http://localhost:8000 (or http://127.0.0.1:8000)")
    print("--------------------------------------------------")
    
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
