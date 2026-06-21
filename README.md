# Veritas AI – Deepfake Verification Guard

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite)](https://sqlite.org/)
[![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel)](https://vercel.com/)

**Veritas AI** is a state-of-the-art forensic analysis and verification application designed to detect AI-generated media (deepfakes). Using a combination of deep neural networks (MobileNetV2) and hand-crafted mathematical forensic signal estimators, Veritas AI evaluates images, audio clips, and video frames to determine the probability of synthesis, complete with visual explainability maps.

---

## 🚀 Key Features

*   **Multi-Modal Analysis**: Supports scanning of face images, cloned audio segments (voice clones), and video files.
*   **Dual-Image Verification (Face-Swap Detector)**:
    *   Compares a **Reference Real Photo** against a **Suspect Check Image**.
    *   Computes identity similarity using deep embeddings and cosine similarity.
    *   Evaluates the suspect image for deepfake synthesis artifacts.
    *   Distinguishes between classic face-swaps (mismatched identity + synthesis), image-wide deepfakes (matched identity + synthesis), and verified genuine matches.
*   **22-Channel Forensic Face Classifier**:
    *   Blends deep model predictions with **22 mathematical forensic signals** capturing artifacts typical of GANs and diffusion models.
    *   Signals include: *Boundary Gradient Discontinuities, High-Freq Texture Residuals, DCT Grid (GAN Checkerboard) Artifacts, FFT Ring Energy, LBP Texture Entropy Deficit, Color Histogram Uniformity, Chromatic Aberration Absence, and Local Over-Smoothing*.
*   **6-Channel Spectral Audio Classifier**:
    *   Uses Short-Time Fourier Transform (STFT) to identify spectral anomalies typical of text-to-speech or voice cloning engines.
    *   Anomalies checked: *High-Frequency Energy, Phase Incoherence, Spectral Flux Deviation, Spectral Centroid, Spectral Rolloff, and Spectral Bandwidth*.
*   **Visual Anomaly Heatmaps**: Automatically highlights region-specific forensic anomalies on the analyzed face crop to provide explainable visual evidence.
*   **Persistent SQLite History**: Keeps a local record of recent scans, allowing users to view full detailed breakdowns of past runs or clear the database logs.
*   **Premium Web Experience**: Fully custom single-page application frontend featuring glassmorphism, responsive layouts, sleek neon-colored dark-mode dashboards, and micro-interactions.

---

## 📂 Project Structure

```text
├── api/
│   └── index.py            # Entry point wrapper for Vercel Serverless Functions
├── app/
│   ├── main.py             # FastAPI App definition, routes, middleware, and setup
│   ├── database.py         # SQLite storage interface & dummy signature seeds
│   ├── models/
│   │   ├── deepfake_face.py   # MobileNetV2 + 22-channel forensic signal classifier
│   │   └── deepfake_voice.py  # 6-channel spectral anomaly audio neural network
│   ├── services/
│   │   └── scanner.py      # Scanner pipeline orchestration for images, audio, video & dual-scans
│   ├── utils/
│   │   ├── heatmaps.py     # Forensic anomaly heatmap generator
│   │   └── media.py        # Face detection, audio transcoding, and frame extraction
│   └── static/             # Single-Page Frontend Application (HTML/CSS/JS)
│       ├── css/style.css   # Premium custom stylesheet
│       ├── js/app.js       # Dynamic AJAX communications and interactive UI updates
│       └── index.html      # Main landing & dashboard HTML
├── data/
│   ├── uploads/            # Temporary directory for uploaded files (.gitkeep)
│   ├── deep_model.keras    # Pre-trained MobileNetV2 classifier model (auto-generated if missing)
│   └── deepfake_signatures.db # SQLite database file (auto-generated on launch)
├── vercel.json             # Vercel deployment configuration
├── requirements.txt        # Package dependencies list
└── run.py                  # Local runner script
```

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.10 or 3.11 recommended.
*   Required libraries specified in `requirements.txt`.

### Step 1: Clone the Repository
```bash
git clone https://github.com/PANTH2517/Veritas-AI.git
cd Veritas-AI
```

### Step 2: Install Dependencies
It is highly recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 3: Run the Server Locally
To start the FastAPI web app locally, run:
```bash
python run.py
```
This launches the server at `http://localhost:8000`. You can visit this URL in your web browser to access the interactive dashboard.

*Note: On first run, if the deep learning model files (`.keras`) are missing from the `data/` folder, they will be built automatically on the fly.*

---

## 📡 API Endpoints

### 1. Single Media Scan
*   **Endpoint**: `POST /api/scan/file`
*   **Body**: `file` (multipart form-data file)
*   **Description**: Scans a single image, audio file, or video file for deepfake synthesis signatures.

### 2. Dual Face Verification
*   **Endpoint**: `POST /api/scan/dual`
*   **Body**: 
    *   `real_file` (Reference Real Photo)
    *   `check_file` (Suspect Check Photo)
*   **Description**: Verifies if the identity matches between the reference and the suspect while checking the suspect image for deepfake modifications (swaps/synthesis).

### 3. Scan History
*   **Endpoint**: `GET /api/history`
*   **Description**: Retrieves a list of the 30 most recent scan records.
*   **Endpoint**: `GET /api/history/{scan_id}`
*   **Description**: Retrieves the complete details of a specific past scan.
*   **Endpoint**: `DELETE /api/history/{scan_id}`
*   **Description**: Removes a scan record from the database.
*   **Endpoint**: `DELETE /api/history`
*   **Description**: Resets and clears the entire database scan history.

### 4. System Health
*   **Endpoint**: `GET /api/health`
*   **Description**: Returns system status, loaded classifiers, and host hardware metadata.

---

## 🌐 Deployment to Vercel

The application is pre-configured to deploy seamlessly to Vercel Serverless Functions:
1.  Make sure you have the [Vercel CLI](https://vercel.com/cli) installed.
2.  Run `vercel` in the project root.
3.  Deploy production using `vercel --prod`.

The routing is handled automatically via `vercel.json` pointing incoming requests to `api/index.py` for API endpoints and serving the `app/static/` directory for static pages.

---

## ⚖️ License & Disclaimer

Veritas AI is designed for forensic research and proof-of-concept verification. It relies on mathematical modeling of common GAN/diffusion artifact footprints and spectral speech pattern changes. Actual detection accuracy may vary depending on compression rates, resolution, and noise levels.
