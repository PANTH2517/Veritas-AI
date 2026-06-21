import os
import json
import cv2
import numpy as np
# onnxruntime is imported lazily inside __init__ to avoid cold-start timeouts

MODEL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "deep_model.onnx"
)

BASE_MODEL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "base_model.onnx"
)

FORENSIC_WEIGHTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "face_model_weights.npz"
)

THRESHOLD_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "threshold.json"
)

N_FORENSIC = 22

class DeepfakeFaceDetector:
    def __init__(self):
        self.deep_session  = None
        self.base_session  = None
        self._onnx_mode    = False  # onnxruntime not installed; using forensic-only mode

        # Forensic weights (numpy MLP) — the only model used on Vercel
        if os.path.exists(FORENSIC_WEIGHTS_FILE):
            self.forensic_weights = np.load(FORENSIC_WEIGHTS_FILE)
            print("[DeepfakeFaceDetector] Forensic weights loaded — forensic-only mode active.")
        else:
            self.forensic_weights = None
            print(f"[DeepfakeFaceDetector] WARNING: Forensic weights not found at {FORENSIC_WEIGHTS_FILE}")

        # Threshold tuned for forensic-only mode
        self.threshold       = 0.42
        self.deep_weight     = 0.0
        self.forensic_weight = 1.0

        config_file = THRESHOLD_FILE
        locked_file = os.path.join(os.path.dirname(THRESHOLD_FILE), "threshold_locked.json")
        if os.path.exists(locked_file):
            config_file = locked_file
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                # Only use forensic threshold from config if present
                if "forensic_threshold" in data:
                    self.threshold = float(data["forensic_threshold"])
                print(f"[DeepfakeFaceDetector] threshold={self.threshold:.4f} (forensic-only)")
            except Exception as e:
                print(f"[DeepfakeFaceDetector] Config load failed ({e}). Using defaults.")


    def run_mlp(self, x, weights):
        # Ensure 2D shape (1, 22)
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)
            
        # Dense 0 + ReLU
        w0 = weights['w_0']
        b0 = weights['b_0']
        x = np.dot(x, w0) + b0
        x = np.maximum(0, x)
        
        # Dense 1 + ReLU
        w1 = weights['w_1']
        b1 = weights['b_1']
        x = np.dot(x, w1) + b1
        x = np.maximum(0, x)
        
        # Dense 2 + ReLU
        w2 = weights['w_2']
        b2 = weights['b_2']
        x = np.dot(x, w2) + b2
        x = np.maximum(0, x)
        
        # Dense 3 + Sigmoid
        w3 = weights['w_3']
        b3 = weights['b_3']
        x = np.dot(x, w3) + b3
        x = 1.0 / (1.0 + np.exp(-x))
        
        return float(x[0, 0])

    def extract_forensic_features(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return np.zeros(N_FORENSIC, dtype=np.float32)
        face_roi = cv2.resize(face_roi, (224, 224))
        h, w = face_roi.shape[:2]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)
        bw = max(1, int(w * 0.10))
        bh_b = max(1, int(h * 0.10))
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (w, h), 255, -1)
        cv2.rectangle(mask, (bw, bh_b), (w - bw, h - bh_b), 0, -1)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(sx**2 + sy**2)
        f0 = float(np.clip(np.mean(gmag[mask == 255]) / 128.0, 0.0, 1.0))
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = cv2.absdiff(gray, blurred)
        f1 = float(np.clip(np.var(residual.astype(np.float32)) / 80.0, 0.0, 1.0))
        yuv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YUV)
        f2 = float(np.clip(abs(float(np.var(yuv[:, :, 1])) - float(np.var(yuv[:, :, 2]))) / 150.0, 0.0, 1.0))
        eye_zone   = gray[int(h*0.28):int(h*0.52), int(w*0.20):int(w*0.80)]
        mouth_zone = gray[int(h*0.62):int(h*0.86), int(w*0.28):int(w*0.72)]
        eye_lap    = cv2.Laplacian(eye_zone, cv2.CV_64F).var() if eye_zone.size > 0 else 0
        mouth_lap  = cv2.Laplacian(mouth_zone, cv2.CV_64F).var() if mouth_zone.size > 0 else 0
        overall_lap = cv2.Laplacian(gray, cv2.CV_64F).var() + 1e-5
        ratio = (eye_lap + mouth_lap) / (2.0 * overall_lap)
        f3 = float(np.clip(abs(ratio - 1.0) / 2.0, 0.0, 1.0))
        f4 = float(np.clip(60.0 / (overall_lap + 1.0), 0.0, 1.0))
        dct_in = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
        dct = cv2.dct(dct_in)
        hf = dct[32:, 32:]
        f5 = float(np.clip((np.sum(np.abs(hf)) / (np.sum(np.abs(dct)) + 1e-6)) * 12.0, 0.0, 1.0))
        f6 = float(np.clip(np.mean(cv2.absdiff(gray, cv2.flip(gray, 1))) / 45.0, 0.0, 1.0))
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.size + 1e-6)
        f7 = float(np.clip(abs(edge_density - 0.12) / 0.12, 0.0, 1.0))
        fft = np.fft.fft2(gray_f)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))
        cy, cx = h // 2, w // 2
        ring_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(ring_mask, (cx, cy), min(cx, cy) // 3, 255, 4)
        ring_energy  = np.mean(magnitude[ring_mask == 255]) if np.any(ring_mask == 255) else 0.0
        total_energy = np.mean(magnitude) + 1e-6
        f8 = float(np.clip((ring_energy / total_energy - 1.0) / 3.0, 0.0, 1.0))
        noise_res = gray_f - cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        noise_std = float(np.std(noise_res))
        f9 = float(np.clip(1.0 - (noise_std / 8.0), 0.0, 1.0))
        hist_scores = []
        for c in range(3):
            hist = cv2.calcHist([face_roi], [c], None, [64], [0, 256]).flatten()
            hist /= (hist.sum() + 1e-6)
            hist_scores.append(float(np.sum(hist ** 2)))
        avg_uniformity = np.mean(hist_scores)
        f10 = float(np.clip(1.0 - (avg_uniformity * 64.0), 0.0, 1.0))
        lbp = np.zeros_like(gray_f)
        offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        for bit, (dy, dx) in enumerate(offsets):
            shifted = np.roll(np.roll(gray_f, dy, axis=0), dx, axis=1)
            lbp += (gray_f >= shifted).astype(np.float32) * (2 ** bit)
        lbp_hist, _ = np.histogram(lbp[1:-1, 1:-1], bins=256, range=(0, 255))
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
        entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
        f11 = float(np.clip(1.0 - (entropy / 8.0), 0.0, 1.0))
        skin_zone = face_roi[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]
        if skin_zone.size > 0:
            skin_lap_var = cv2.Laplacian(cv2.cvtColor(skin_zone, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            f12 = float(np.clip(30.0 / (skin_lap_var + 1.0), 0.0, 1.0))
        else:
            f12 = 0.5
        block_vars = [
            float(np.var(gray[r:r+8, c:c+8].astype(np.float32)))
            for r in range(0, h-8, 8) for c in range(0, w-8, 8)
        ]
        if block_vars:
            bv_arr = np.array(block_vars)
            cv_val = float(np.std(bv_arr) / (np.mean(bv_arr) + 1e-6))
            f13 = float(np.clip(abs(cv_val - 1.2) / 2.0, 0.0, 1.0))
        else:
            f13 = 0.5
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        phase = np.arctan2(gy, gx)
        phase_var = float(np.var(np.diff(phase.flatten())))
        f14 = float(np.clip(1.0 - phase_var / 2.5, 0.0, 1.0))
        gray_q = (gray // 32).astype(np.uint8)                         
        left = gray_q[:-1, :-1]
        right = gray_q[:-1, 1:]
        pairs = left.astype(np.int32) * 8 + right.astype(np.int32)
        glcm = np.bincount(pairs.flatten(), minlength=64).reshape(8, 8).astype(np.float32)
        glcm /= (glcm.sum() + 1e-6)
        i_idx, j_idx = np.mgrid[0:8, 0:8]
        contrast = float(np.sum(((i_idx - j_idx) ** 2) * glcm))
        f15 = float(np.clip(abs(contrast - 3.5) / 5.0, 0.0, 1.0))
        ps = np.abs(fft_shift) ** 2
        ps_flat = ps.flatten()
        geom_mean = np.exp(np.mean(np.log(ps_flat + 1e-10)))
        arith_mean = np.mean(ps_flat) + 1e-10
        spectral_flatness = geom_mean / arith_mean
        f16 = float(np.clip(spectral_flatness * 5.0, 0.0, 1.0))
        bright_mask = (gray > 220).astype(np.float32)
        bright_density = float(np.mean(bright_mask))
        f17 = float(np.clip(abs(bright_density - 0.04) / 0.06, 0.0, 1.0))
        edge_map = cv2.Canny(gray, 40, 120)
        contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            curvatures = []
            for c in contours:
                if len(c) > 10:
                    pts = c.reshape(-1, 2).astype(np.float32)
                    v1 = pts[1:-1] - pts[:-2]
                    v2 = pts[2:] - pts[1:-1]
                    cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
                    curvatures.extend(cross.tolist())
            if curvatures:
                curv_var = float(np.var(curvatures))
                f18 = float(np.clip(1.0 - curv_var / (curv_var + 50.0), 0.0, 1.0))
            else:
                f18 = 0.4
        else:
            f18 = 0.4
        b_ch, g_ch, r_ch = cv2.split(face_roi)
        edges_r = cv2.Canny(r_ch, 50, 150).astype(np.float32)
        edges_b = cv2.Canny(b_ch, 50, 150).astype(np.float32)
        if edges_r.sum() > 0 and edges_b.sum() > 0:
            rb_diff = float(np.mean(cv2.absdiff(edges_r.astype(np.uint8),
                                                   edges_b.astype(np.uint8))))
            f19 = float(np.clip(1.0 - rb_diff / 30.0, 0.0, 1.0))
        else:
            f19 = 0.5
        left_half  = gray[:, :w//2].astype(np.float32)
        right_half = gray[:, w//2:].astype(np.float32)
        upper_half = gray[:h//2, :].astype(np.float32)
        lower_half = gray[h//2:, :].astype(np.float32)
        lr_ratio = abs(np.mean(left_half) - np.mean(right_half)) / (np.mean(gray) + 1e-6)
        ud_ratio = abs(np.mean(upper_half) - np.mean(lower_half)) / (np.mean(gray) + 1e-6)
        f20 = float(np.clip(1.0 - (lr_ratio + ud_ratio), 0.0, 1.0))
        if skin_zone.size > 0:
            skin_gray = cv2.cvtColor(skin_zone, cv2.COLOR_BGR2GRAY).astype(np.float32)
            skin_fft  = np.fft.fft2(skin_gray)
            skin_mag  = np.abs(np.fft.fftshift(skin_fft))
            sh, sw    = skin_gray.shape
            hf_quad   = skin_mag[sh//2 - sh//4:, sw//2 - sw//4:]
            lf_total  = np.mean(skin_mag) + 1e-6
            hf_ratio  = float(np.mean(hf_quad) / lf_total)
            f21 = float(np.clip(1.0 - hf_ratio * 3.0, 0.0, 1.0))
        else:
            f21 = 0.5
        return np.array(
            [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9,
             f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21],
            dtype=np.float32
        )

    SIGNAL_NAMES = [
        "Boundary Gradient Discontinuity",
        "High-Freq Texture Residual",
        "Chroma Channel Noise Asymmetry",
        "Eye/Mouth Sharpness Mismatch",
        "Local Blur / Over-Smoothing",
        "DCT Grid Artifacts (GAN Checkerboard)",
        "Bilateral Face Asymmetry",
        "Edge Density Deviation",
        "FFT Ring Artifacts",
        "Noise Floor Absence",
        "Color Histogram Uniformity",
        "LBP Texture Entropy Deficit",
        "Skin Smoothness Excess",
        "JPEG Compression Inconsistency",
        "Gradient Phase Coherence Anomaly",
        "Co-occurrence Matrix Irregularity",
        "Spectral Flatness Excess",
        "Specular Highlight Abnormality",
        "Facial Contour Over-Regularity",
        "Chromatic Aberration Absence",
        "Shadow/Lighting Directionality Loss",
        "Micro-Texture Frequency Deficit",
    ]

    def predict(self, face_roi):
        features = self.extract_forensic_features(face_roi)
        forensic_score = self.run_mlp(features, self.forensic_weights) if self.forensic_weights is not None else 0.5

        if self._onnx_mode:
            # Full scoring: ONNX deep model + forensic MLP
            try:
                face_resized = cv2.resize(face_roi, (224, 224))
                rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
                deep_input = np.expand_dims(rgb, axis=0)
                input_name  = self.deep_session.get_inputs()[0].name
                output_name = self.deep_session.get_outputs()[0].name
                deep_score  = float(self.deep_session.run(
                    [output_name], {input_name: deep_input})[0][0][0])
            except Exception as e:
                print(f"[predict] ONNX inference failed ({e}), falling back to forensic-only.")
                deep_score = forensic_score   # graceful fallback

            combined = self.deep_weight * deep_score + self.forensic_weight * forensic_score
        else:
            # Forensic-only mode: pure numpy, runs in <10ms
            deep_score = forensic_score
            combined   = forensic_score

        combined = float(np.clip(combined, 0.0, 1.0))
        breakdown = {
            "deep_model_score":     round(deep_score, 4),
            "forensic_model_score": round(forensic_score, 4),
            "signals": {
                name: round(float(val), 4)
                for name, val in zip(self.SIGNAL_NAMES, features)
            }
        }
        return combined, breakdown

    def extract_deep_features(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return np.zeros(1280, dtype=np.float32)
        face_resized = cv2.resize(face_roi, (224, 224))
        rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # MobileNetV2 preprocessing: scale to [-1, 1]
        rgb_pp = (rgb / 127.5) - 1.0
        inp = np.expand_dims(rgb_pp, axis=0)
        
        input_name = self.base_session.get_inputs()[0].name
        output_name = self.base_session.get_outputs()[0].name
        return self.base_session.run([output_name], {input_name: inp})[0][0]
