# Veritas AI – Enterprise-Grade Deepfake Intelligence Platform

```text
  ██████╗   ███████╗   ██████╗    ██╗   ████████╗    ██████╗    ███████╗
  ██╔══██╗  ██╔════╝   ██╔══██╗   ██║   ╚══██╔══╝   ██╔═══██╗   ██╔════╝
  ██║  ██║  █████╗     ██████╔╝   ██║      ██║      ██║   ██║   ███████╗
  ██║  ██║  ██╔══╝     ██╔══██╗   ██║      ██║      ██║   ██║   ╚════██║
  ██████╔╝  ███████╗   ██║  ██║   ██║      ██║      ╚██████╔╝   ███████║
  ╚═════╝   ╚══════╝   ╚═╝  ╚═╝   ╚═╝      ╚═╝       ╚═════╝    ╚══════╝
  
  ═══════════ FORENSIC INTELLIGENCE & VERIFICATION GUARD ═══════════
```

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite)](https://sqlite.org/)
[![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel)](https://vercel.com/)

**Veritas AI** is an advanced, enterprise-grade deepfake forensic intelligence platform. Designed to combat the rapid proliferation of synthetic generative media, Veritas AI combines deep convolutional latent feature learning with 22 hand-crafted spatial, spectral, and geometric digital forensic estimators to verify images, audio recordings, and video frames. 

The platform delivers a rigorous verification standard calibrating vision models with classical image physics—providing granular probability breakdowns and visual anomaly heatmaps.

---

## 🎯 Project Motive & Performance Calibrations

Generative AI (GANs, Diffusion, TTS, and Voice Cloning models) can create hyper-realistic media that bypasses human detection and standard hash-based verification. Veritas AI serves as a forensic shield that decomposes media into high-frequency residues, compression histories, and biometric properties.

### Empirical Performance Metrics
*   **System Validation Accuracy**: **87.66%** (F1-Score: 0.88) on hybrid benchmarks.
*   **Ensemble Structure**: Blended decision matrix combining deep convolutional outputs (86% weight) with classical forensic signals (14% weight).
*   **Face Classifier Accuracy**: **71.75%** verified across 2,000+ benchmark face signatures.

---

## 🧬 Core Architecture & Signal Pipeline

Veritas AI fuses two distinct methodology paradigms into a single calibrated classification engine:

```text
[Input Media] ➔ [Pre-Processing (Face Crop / STFT Audio Spectrogram)]
                       │
                       ├➔ [MobileNetV2 CNN Engine] (86% Weight) ──┐
                       │                                         ├➔ [Hybrid Fusion] ➔ [Verdict & Heatmaps]
                       └➔ [22-Channel Forensic MLP] (14% Weight) ─┘
```

### 1. MobileNetV2 CNN Classifier
Operates as the primary neural feature extractor. Pre-trained on ImageNet and fine-tuned on custom real vs. synthetic datasets, it outputs a probability based on high-level latent feature anomalies.

### 2. The 22-Channel Forensic Signal Suite
To prevent neural network blindspots, the system extracts 22 hand-crafted digital forensic filters mapping pixel physics:

| Signal Identifier | Description | Forensic Significance |
| :--- | :--- | :--- |
| **Boundary Gradient Discontinuity** | Detects gradient differences between cropping boundaries. | Identifies blending/splicing in composite fakes. |
| **High-Freq Texture Residual** | Measures the variance of high-pass image residuals. | GANs and diffusion engines tend to over-smooth fine noise. |
| **Chroma Channel Noise Asymmetry** | Compares the variance discrepancy between U and V channels. | Identifies abnormal chrominance imbalances in synthesized faces. |
| **Eye/Mouth Sharpness Mismatch** | Compares Laplacian variances in eye vs. mouth regions. | Evaluates whether localized regions are synthetically blurred. |
| **Local Blur / Over-Smoothing** | Analyzes the inverse of the global Laplacian variance. | Catches low-pass filters used to hide synthesis artifacts. |
| **DCT Grid Artifacts** | Fourier-transforms block-wise cosine coefficients. | Exposes checkerboard patterns caused by transpose convolution. |
| **Bilateral Face Asymmetry** | Measures absolute difference between face halves. | Organic faces have organic asymmetry; fakes are often symmetric. |
| **Edge Density Deviation** | Calculates normalized deviation of Canny edge densities. | Detects missing or excess micro-edges in skin textures. |
| **FFT Ring Artifacts** | Computes energy concentration in mid-frequency radial rings. | Detects repeating frequency anomalies typical of GAN generators. |
| **Noise Floor Absence** | Computes the standard deviation of local Gaussian noise. | Generative faces often lack standard sensor noise floors. |
| **Color Histogram Uniformity** | Evaluates the entropy and variance of RGB channels. | Identifies flat or artificial color distributions. |
| **LBP Texture Entropy Deficit** | Computes Local Binary Pattern histogram entropy. | Captures the loss of natural micro-texture frequency. |
| **Skin Smoothness Excess** | Extracts Laplacian variance in targeted skin bounding boxes. | Exposes "plastic skin" typical of generative face models. |
| **JPEG Compression Inconsistency**| Analyzes block variance distributions across $8\times8$ grids. | Detects double-compression or splicing artifacts. |
| **Gradient Phase Coherence** | Computes phase variances of directional Sobel gradients. | Highlights unnatural pixel-alignment flows. |
| **Co-occurrence Matrix Irregularity** | Computes spatial gray-level co-occurrence matrix (GLCM). | Identifies artificial repetitions in pixel micro-layouts. |
| **Spectral Flatness Excess** | Compares geometric mean vs. arithmetic mean of FFT power. | Detects artificial frequency spikes. |
| **Specular Highlight Abnormality** | Evaluates intensity densities in high-luminosity areas. | Identifies abnormal reflections in eyes and skin. |
| **Facial Contour Over-Regularity** | Analyzes curvature changes in major outer contours. | GANs often generate overly smooth, mathematically perfect curves. |
| **Chromatic Aberration Absence** | Measures color misalignments near high-contrast edges. | True lenses cause aberration; synthesized images lack it. |
| **Shadow/Lighting Directionality** | Evaluates luminance gradients across facial quadrants. | Checks for conflicting lighting or missing directional shadows. |
| **Micro-Texture Frequency Deficit**| Measures high-frequency skin patch Fourier coefficients. | Exposes lack of pore-level details in synthetic textures. |

---

## 🛠️ Tech Stack & File Structure

```text
├── api/
│   └── index.py            # Vercel Serverless Function entry point wrapper
├── app/
│   ├── main.py             # FastAPI App definition, routing & static mounting
│   ├── database.py         # SQLite interface, session analytics & signature seeding
│   ├── models/
│   │   ├── deepfake_face.py   # MobileNetV2 + 22-channel forensic MLP pipeline
│   │   └── deepfake_voice.py  # 6-channel spectral voice cloning detector
│   ├── services/
│   │   └── scanner.py      # Orchestrator for file scanning, history logging & dual comparison
│   ├── utils/
│   │   ├── heatmaps.py     # Generates Grad-CAM-like visual anomaly overlays
│   │   └── media.py        # Face detection, audio transcoding, and video frame extraction
│   └── static/             # Single-Page Frontend Dashboard (HTML / CSS / JS)
│       ├── css/style.css   # Custom CSS with neon styling, grid overlays & glassmorphism
│       ├── js/app.js       # Dynamic AJAX handling, UI states, timeline & chart rendering
│       └── index.html      # Main dashboard HTML template
├── data/
│   ├── uploads/            # Bounded upload staging directory
│   ├── deep_model.keras    # Pre-trained MobileNetV2 neural model (auto-generated if missing)
│   └── deepfake_signatures.db # Persistent local SQLite store (auto-generated)
├── vercel.json             # Vercel deployment routing configurations
├── requirements.txt        # Python dependency manifest
└── run.py                  # Local development server bootstrapper
```

---

## 🚀 Setup & Installation

### Local Deployment
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/PANTH2517/Veritas-AI.git
    cd Veritas-AI
    ```
2.  **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Required Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Launch the Web Platform**:
    ```bash
    python run.py
    ```
    Open `http://localhost:8000` in your web browser. The app runs a boot diagnostics sequence and will automatically download or compile missing models.

### Cloud Deployment (Vercel)
The codebase includes pre-configured routing for serverless deployments on Vercel:
1.  Install the Vercel CLI: `npm install -g vercel`
2.  Run `vercel` in the project root to set up a new project.
3.  Run `vercel --prod` to deploy to production.

---

## 📡 REST API Specifications

### 1. Single Asset Scan
Scans a single image, audio clip, or video file for deepfake synthesis signatures.
*   **Route**: `POST /api/scan/file`
*   **Payload**: `file` (Multipart Form-Data)
*   **Sample Request**:
    ```bash
    curl -X POST -F "file=@face.jpg" http://localhost:8000/api/scan/file
    ```
*   **Sample Response**:
    ```json
    {
      "id": "c86a3458ef2140bb...",
      "filename": "face.jpg",
      "verdict": "AI_GENERATED",
      "risk_score": 0.9421,
      "confidence": 94.2,
      "risk_grade": "CRITICAL RISK",
      "media_type": "image",
      "face_count": 1,
      "overlay_url": "/static/cache/face_overlay_c86a3458ef2140bb.jpg",
      "heatmap_url": "/static/cache/face_heatmap_c86a3458ef2140bb.jpg",
      "analysis": {
        "deep_model_score": 0.9632,
        "forensic_model_score": 0.8745,
        "signals": {
          "Boundary Gradient Discontinuity": 0.8123,
          "DCT Grid Artifacts (GAN Checkerboard)": 0.9452,
          "FFT Ring Artifacts": 0.9103
        }
      }
    }
    ```

### 2. Dual Biometric Verification (Identity Match / Face-Swap)
Compares a verified reference image against a suspect image to detect identity spoofing and localized modifications.
*   **Route**: `POST /api/scan/dual`
*   **Payload**:
    *   `real_file`: Reference Real Image (Multipart Form-Data)
    *   `check_file`: Suspect Target Image (Multipart Form-Data)
*   **Sample Request**:
    ```bash
    curl -X POST \
      -F "real_file=@verified_id.png" \
      -F "check_file=@suspect_submission.png" \
      http://localhost:8000/api/scan/dual
    ```
*   **Sample Response Verdict Categories**:
    *   `VERIFIED_REAL_IDENTITY`: Identity matches reference, and no synthetic textures detected.
    *   `AI_GENERATED_MANIPULATION_DETECTED`: Identity matches, but synthesized face artifacts are present (image modification).
    *   `IDENTITY_MISMATCH_SUSPECTED_SWAP`: Identity does not match reference photo.
    *   `IDENTITY_MISMATCH_AND_AI_GENERATED`: Identity mismatch and suspect image contains synthetic anomalies (classic face-swap).

### 3. Historical Logs & Database Audit
*   **List Records**: `GET /api/history`
*   **Fetch Details**: `GET /api/history/{scan_id}`
*   **Delete Record**: `DELETE /api/history/{scan_id}`
*   **Reset History**: `DELETE /api/history`

---

## ⚖️ License & Research Disclaimer

Veritas AI is developed for digital forensic research, security auditing, and proof-of-concept verification. While the hybrid ensemble achieves high accuracy under laboratory settings, real-world performance may be impacted by extreme compression (e.g., social media uploads), adversarial noise, and low resolution.
