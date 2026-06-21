

import os

import json

import cv2

import numpy as np

import tensorflow as tf

MODEL_FILE = os.path.join(

    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),

    "data", "face_model.keras"

)

DEEP_MODEL_FILE = os.path.join(

    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),

    "data", "deep_model.keras"

)

THRESHOLD_FILE = os.path.join(

    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),

    "data", "threshold.json"

)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

N_FORENSIC = 22                                     

class DeepfakeFaceDetector:

    DEEP_WEIGHT     = 1.0

    FORENSIC_WEIGHT = 0.0

    def __init__(self):

        if os.path.exists(DEEP_MODEL_FILE):

            try:

                self.deep_model = tf.keras.models.load_model(DEEP_MODEL_FILE)

                print(f"[DeepfakeFaceDetector] Deep model loaded.")

            except Exception as e:

                print(f"[DeepfakeFaceDetector] Deep model load failed ({e}). Rebuilding.")

                self.deep_model = self._build_deep_model()

        else:

            self.deep_model = self._build_deep_model()

        if os.path.exists(MODEL_FILE):

            try:

                self.forensic_model = tf.keras.models.load_model(MODEL_FILE)

                print(f"[DeepfakeFaceDetector] Forensic model loaded.")

            except Exception as e:

                print(f"[DeepfakeFaceDetector] Forensic model load failed ({e}). Rebuilding.")

                self.forensic_model = self._build_forensic_model()

                self._init_forensic_weights()

        else:

            self.forensic_model = self._build_forensic_model()

            self._init_forensic_weights()

        self.threshold = 0.7950

        self.deep_weight = 0.7600

        self.forensic_weight = 0.2400

        config_file = THRESHOLD_FILE

        locked_file = os.path.join(os.path.dirname(THRESHOLD_FILE), "threshold_locked.json")

        if os.path.exists(locked_file):

            config_file = locked_file

            print(f"[DeepfakeFaceDetector] Loading LOCKED configuration.")

        if os.path.exists(config_file):

            try:

                with open(config_file, "r") as f:

                    data = json.load(f)

                self.threshold = float(data.get("threshold", self.threshold))

                self.deep_weight = float(data.get("deep_weight", self.deep_weight))

                self.forensic_weight = float(data.get("forensic_weight", self.forensic_weight))

                print(f"[DeepfakeFaceDetector] Threshold: {self.threshold:.4f}, weights: deep={self.deep_weight:.2f}, forensic={self.forensic_weight:.2f}")

            except Exception as e:

                print(f"[DeepfakeFaceDetector] Configuration load failed ({e}). Using defaults.")

    def _build_deep_model(self):

        base = tf.keras.applications.MobileNetV2(

            input_shape=(224, 224, 3), include_top=False,

            weights='imagenet', pooling='avg'

        )

        base.trainable = False

        inp = tf.keras.Input(shape=(224, 224, 3))

        x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)

        x = base(x, training=False)

        x = tf.keras.layers.Dense(256, activation='relu')(x)

        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Dense(64, activation='relu')(x)

        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=inp, outputs=out)

        self._init_deep_head(model)

        return model

    def _init_deep_head(self, model):

        dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]

        if len(dense) < 3:

            return

        dense[0].set_weights([

            np.random.normal(0.0, 0.02, (1280, 256)).astype(np.float32),

            np.zeros(256, dtype=np.float32)

        ])

        dense[1].set_weights([

            np.random.normal(0.0, 0.05, (256, 64)).astype(np.float32),

            np.zeros(64, dtype=np.float32)

        ])

        dense[2].set_weights([

            np.random.normal(0.0, 0.05, (64, 1)).astype(np.float32),

            np.array([0.0], dtype=np.float32)

        ])

    def _build_forensic_model(self):

        model = tf.keras.Sequential([

            tf.keras.layers.Input(shape=(N_FORENSIC,)),

            tf.keras.layers.Dense(64, activation='relu'),

            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(32, activation='relu'),

            tf.keras.layers.Dense(16, activation='relu'),

            tf.keras.layers.Dense(1, activation='sigmoid')

        ])

        model.compile(optimizer='adam', loss='binary_crossentropy')

        return model

    def _init_forensic_weights(self):

        importances = [

            0.55,                                                       

            0.50,                                

            0.35,                              

            0.40,                                   

            0.30,                            

            0.60,                                                         

            0.25,                               

            0.30,                             

            0.65,                                                       

            0.55,                          

            0.35,                                 

            0.30,                          

            0.45,                             

            0.25,                                   

            0.50,                                              

            0.40,                                              

            0.45,                                              

            0.35,                                              

            0.40,                                              

            0.55,                                              

            0.45,                                              

            0.50,                                              

        ]

        w1 = np.zeros((N_FORENSIC, 64), dtype=np.float32)

        for i, imp in enumerate(importances):

            w1[i, :] = imp

        b1 = np.full((64,), -0.85, dtype=np.float32)

        w2 = np.ones((64, 32), dtype=np.float32) * 0.18

        b2 = np.full((32,), -0.08, dtype=np.float32)

        w3 = np.ones((32, 16), dtype=np.float32) * 0.22

        b3 = np.full((16,), -0.05, dtype=np.float32)

        w4 = np.ones((16, 1), dtype=np.float32) * 0.35

        b4 = np.array([-0.12], dtype=np.float32)

        np.random.seed(42)

        for w in [w1, w2, w3, w4]:

            w += np.random.normal(0.0, 0.015, size=w.shape).astype(np.float32)

        self.forensic_model.layers[0].set_weights([w1, b1])

        self.forensic_model.layers[2].set_weights([w2, b2])

        self.forensic_model.layers[3].set_weights([w3, b3])

        self.forensic_model.layers[4].set_weights([w4, b4])

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

        face_resized = cv2.resize(face_roi, (224, 224))

        rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32)

        deep_input = np.expand_dims(rgb, axis=0)

        deep_score = float(self.deep_model.predict(deep_input, verbose=0)[0][0])

        features = self.extract_forensic_features(face_roi)

        if self.forensic_weight > 0.0:

            forensic_input = np.expand_dims(features, axis=0)

            forensic_score = float(self.forensic_model.predict(forensic_input, verbose=0)[0][0])

        else:

            forensic_score = 0.0

        combined = self.deep_weight * deep_score + self.forensic_weight * forensic_score

        combined = float(np.clip(combined, 0.0, 1.0))

        breakdown = {

            "deep_model_score":    round(deep_score, 4),

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

        rgb_pp = tf.keras.applications.mobilenet_v2.preprocess_input(rgb)

        inp = np.expand_dims(rgb_pp, axis=0)

        if not hasattr(self, '_base_extractor'):

            self._base_extractor = tf.keras.applications.MobileNetV2(

                input_shape=(224, 224, 3), include_top=False,

                weights='imagenet', pooling='avg'

            )

        return self._base_extractor.predict(inp, verbose=0)[0]

