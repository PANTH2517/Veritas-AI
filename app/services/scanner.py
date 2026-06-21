import os

import uuid

import cv2

import numpy as np

from concurrent.futures import ThreadPoolExecutor

from app.utils.media import detect_faces

from app.utils.heatmaps import generate_face_heatmap

from app.models.deepfake_face import DeepfakeFaceDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if os.environ.get("VERCEL") or not os.access(os.path.join(BASE_DIR, "app", "static"), os.W_OK):
    CACHE_DIR = "/tmp/cache"
    UPLOAD_DIR = "/tmp/uploads"
else:
    CACHE_DIR = os.path.join(BASE_DIR, "app", "static", "cache")
    UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg'}

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS

AI_THRESHOLD = 0.30

class FaceDeepfakeScannerService:

    def __init__(self):

        self.detector = DeepfakeFaceDetector()

        self.executor = ThreadPoolExecutor(max_workers=4)

        self.history = []

        self.MAX_HISTORY = 30

    def scan_face_image(self, file_path: str, filename: str) -> dict:

        ext = os.path.splitext(filename)[1].lower()

        if ext not in ALLOWED_EXTENSIONS:

            raise ValueError(

                f"Unsupported format '{ext}'. "

                f"Please upload a face image: JPG, PNG, WEBP, or BMP."

            )

        img = cv2.imread(file_path)

        if img is None:

            raise ValueError("Could not read the image file. It may be corrupt or unsupported.")

        faces = detect_faces(img)

        if len(faces) == 0:

            raise ValueError(

                "No face detected in this image. "

                "Please upload a clear, front-facing photo."

            )

        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        primary_face = faces_sorted[0]

        faces_to_analyze = faces_sorted[:3]

        face_results = []

        for face_box in faces_to_analyze:

            x, y, fw, fh = face_box

            pad_x = int(fw * 0.10)

            pad_y = int(fh * 0.10)

            x1 = max(0, x - pad_x)

            y1 = max(0, y - pad_y)

            x2 = min(img.shape[1], x + fw + pad_x)

            y2 = min(img.shape[0], y + fh + pad_y)

            face_crop = img[y1:y2, x1:x2]

            score, breakdown = self.detector.predict(face_crop)

            face_results.append({

                "face_box": face_box,

                "score": score,

                "breakdown": breakdown

            })

        primary_result = face_results[0]

        risk_score = primary_result["score"]

        breakdown = primary_result["breakdown"]

        primary_box = primary_result["face_box"]

        ai_thresh = self.detector.threshold

        verdict = "AI_GENERATED" if risk_score >= ai_thresh else "REAL"

        confidence = risk_score if verdict == "AI_GENERATED" else (1.0 - risk_score)

        confidence = round(float(confidence) * 100, 1)

        overlay_img, heatmap_img = generate_face_heatmap(

            img, primary_box, risk_score, breakdown, threshold=self.detector.threshold

        )

        scan_id = uuid.uuid4().hex

        overlay_fn = f"face_overlay_{scan_id}.jpg"

        heatmap_fn = f"face_heatmap_{scan_id}.jpg"

        cv2.imwrite(os.path.join(CACHE_DIR, overlay_fn), overlay_img)

        cv2.imwrite(os.path.join(CACHE_DIR, heatmap_fn), heatmap_img)

        faces_summary = []

        for i, fr in enumerate(face_results):

            fb = fr["face_box"]

            faces_summary.append({

                "face_index": i + 1,

                "bounding_box": {"x": fb[0], "y": fb[1], "w": fb[2], "h": fb[3]},

                "ai_score": round(float(fr["score"]), 4),

                "verdict": "AI_GENERATED" if fr["score"] >= self.detector.threshold else "REAL"

            })

        result = {

            "id": scan_id,

            "filename": filename,

            "verdict": verdict,

            "ai_score": round(float(risk_score), 4),

            "risk_score": round(float(risk_score), 4),

            "media_type": "image",

            "confidence": confidence,

            "risk_grade": self._risk_grade(risk_score),

            "face_count": len(faces),

            "faces_analyzed": len(face_results),

            "primary_face_box": {

                "x": int(primary_box[0]),

                "y": int(primary_box[1]),

                "w": int(primary_box[2]),

                "h": int(primary_box[3])

            },

            "faces": faces_summary,

            "face_details": breakdown["signals"],

            "analysis": {

                "deep_model_score": breakdown["deep_model_score"],

                "forensic_model_score": breakdown["forensic_model_score"],

                "signals": breakdown["signals"]

            },

            "overlay_url": f"/static/cache/{overlay_fn}",

            "heatmap_url": f"/static/cache/{heatmap_fn}",

        }

        self.history.insert(0, {

            "id": scan_id,

            "filename": filename,

            "verdict": verdict,

            "ai_score": result["ai_score"],

            "risk_score": result["ai_score"],

            "media_type": "image",

            "confidence": confidence,

            "risk_grade": result["risk_grade"],

        })

        if len(self.history) > self.MAX_HISTORY:

            self.history.pop()

        return result

    def get_history(self):

        from app.database import get_recent_scans

        try:

            return get_recent_scans(limit=self.MAX_HISTORY)

        except Exception as e:

            print(f"Error loading scan history from DB: {e}")

            return self.history

    def scan_media_file(self, file_path: str, filename: str) -> dict:

        ext = os.path.splitext(filename)[1].lower()

        if ext in IMAGE_EXTENSIONS:

            result = self.scan_face_image(file_path, filename)

        elif ext in AUDIO_EXTENSIONS:

            result = self.scan_audio_file(file_path, filename)

        elif ext in VIDEO_EXTENSIONS:

            result = self.scan_video_file(file_path, filename)

        else:

            raise ValueError(

                f"Unsupported format '{ext}'. "

                f"Please upload a supported image, audio, or video file."

            )

        try:

            from app.database import save_scan_to_history

            save_scan_to_history(

                result["id"],

                result["filename"],

                result["media_type"],

                result["risk_score"],

                result["risk_grade"],

                result

            )

        except Exception as e:

            print(f"Error saving scan to history DB: {e}")

        return result

    def scan_audio_file(self, file_path: str, filename: str) -> dict:
        from app.models.deepfake_voice import _SCIPY_OK
        if not _SCIPY_OK:
            raise ValueError(
                "Audio deepfake verification is not supported in this serverless deployment (missing SciPy). "
                "Please run Veritas AI locally to use voice scanning."
            )

        scan_id = uuid.uuid4().hex

        wav_filename = f"voice_cache_{scan_id}.wav"

        wav_filepath = os.path.join(CACHE_DIR, wav_filename)

        from app.utils.media import decode_audio_file

        success = decode_audio_file(file_path, wav_filepath)

        if not success:

            raise ValueError("Failed to decode or process audio file.")

        import scipy.io.wavfile as wav

        try:

            rate, data = wav.read(wav_filepath)

        except Exception as e:

            raise ValueError(f"Failed to read processed WAV audio: {str(e)}")

        from app.models.deepfake_voice import DeepfakeVoiceDetector

        voice_detector = DeepfakeVoiceDetector()

        score, breakdown = voice_detector.predict(rate, data)

        verdict = "AI_GENERATED" if score >= AI_THRESHOLD else "REAL"

        confidence = score if verdict == "AI_GENERATED" else (1.0 - score)

        confidence = round(float(confidence) * 100, 1)

        voice_signals_mapping = {

            "high_frequency_energy_anomaly": "High-Frequency Energy Anomaly",

            "phase_incoherence": "Phase Incoherence",

            "spectral_flux_deviation": "Spectral Flux Deviation",

            "spectral_centroid_deviation": "Spectral Centroid Deviation",

            "spectral_rolloff_deviation": "Spectral Rolloff Deviation",

            "spectral_bandwidth_anomaly": "Spectral Bandwidth Anomaly"

        }

        mapped_signals = {

            voice_signals_mapping[k]: v for k, v in breakdown.items()

        }

        result = {

            "id": scan_id,

            "filename": filename,

            "verdict": verdict,

            "ai_score": round(float(score), 4),

            "risk_score": round(float(score), 4),

            "media_type": "audio",

            "confidence": confidence,

            "risk_grade": self._risk_grade(score),

            "face_count": 0,

            "faces_analyzed": 0,

            "audio_url": f"/static/cache/{wav_filename}",

            "face_details": None,

            "voice_details": mapped_signals,

            "analysis": {

                "forensic_model_score": round(float(score), 4),

                "signals": mapped_signals

            }

        }

        return result

    def scan_video_file(self, file_path: str, filename: str) -> dict:

        scan_id = uuid.uuid4().hex

        from app.utils.media import extract_frames

        frames_data = extract_frames(file_path, max_frames=45)

        if not frames_data:

            raise ValueError("Failed to extract frames from video or video is empty.")

        wav_filename = f"voice_cache_{scan_id}.wav"

        wav_filepath = os.path.join(CACHE_DIR, wav_filename)

        from app.utils.media import extract_audio_from_video

        audio_extracted = extract_audio_from_video(file_path, wav_filepath)

        voice_score = None

        voice_breakdown = None

        if audio_extracted:
            from app.models.deepfake_voice import _SCIPY_OK
            if not _SCIPY_OK:
                audio_extracted = False
            else:
                import scipy.io.wavfile as wav
                try:
                    rate, data = wav.read(wav_filepath)
                    from app.models.deepfake_voice import DeepfakeVoiceDetector
                    voice_detector = DeepfakeVoiceDetector()
                    voice_score, voice_breakdown = voice_detector.predict(rate, data)
                except Exception as e:
                    print(f"Error analyzing extracted video audio: {e}")
                    audio_extracted = False

        frames_list = []

        face_scores = []

        deep_scores = []

        forensic_scores = []

        signals_accum = {}

        total_faces_found = 0

        for idx, fd in enumerate(frames_data):

            frame_img = fd["frame"]

            timestamp = fd["timestamp"]

            faces = detect_faces(frame_img)

            total_faces_found += len(faces)

            if len(faces) > 0:

                faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

                primary_box = faces_sorted[0]

                x, y, fw, fh = primary_box

                pad_x = int(fw * 0.10)

                pad_y = int(fh * 0.10)

                x1 = max(0, x - pad_x)

                y1 = max(0, y - pad_y)

                x2 = min(frame_img.shape[1], x + fw + pad_x)

                y2 = min(frame_img.shape[0], y + fh + pad_y)

                face_crop = frame_img[y1:y2, x1:x2]

                score, breakdown = self.detector.predict(face_crop)

                face_scores.append(score)

                deep_scores.append(breakdown["deep_model_score"])

                forensic_scores.append(breakdown["forensic_model_score"])

                for s_name, s_val in breakdown["signals"].items():

                    signals_accum[s_name] = signals_accum.get(s_name, []) + [s_val]

                overlay_img, heatmap_img = generate_face_heatmap(frame_img, primary_box, score, breakdown, threshold=self.detector.threshold)

            else:

                score = 0.0

                overlay_img, heatmap_img = frame_img.copy(), frame_img.copy()

            overlay_fn = f"video_overlay_{scan_id}_{idx}.jpg"

            heatmap_fn = f"video_heatmap_{scan_id}_{idx}.jpg"

            cv2.imwrite(os.path.join(CACHE_DIR, overlay_fn), overlay_img)

            cv2.imwrite(os.path.join(CACHE_DIR, heatmap_fn), heatmap_img)

            frames_list.append({

                "timestamp": timestamp,

                "risk_score": round(float(score), 4),

                "overlay_url": f"/static/cache/{overlay_fn}",

                "heatmap_url": f"/static/cache/{heatmap_fn}"

            })

        face_score_val = np.max(face_scores) if face_scores else 0.0

        avg_deep_score = np.mean(deep_scores) if deep_scores else 0.0

        avg_forensic_score = np.mean(forensic_scores) if forensic_scores else 0.0

        avg_signals = {}

        for s_name, vals in signals_accum.items():

            avg_signals[s_name] = round(float(np.mean(vals)), 4)

        if audio_extracted and voice_score is not None:

            overall_score = max(face_score_val, voice_score)

        else:

            overall_score = face_score_val

        ai_thresh = self.detector.threshold

        verdict = "AI_GENERATED" if overall_score >= ai_thresh else "REAL"

        confidence = overall_score if verdict == "AI_GENERATED" else (1.0 - overall_score)

        confidence = round(float(confidence) * 100, 1)

        voice_signals_mapped = None

        if audio_extracted and voice_breakdown:

            voice_signals_mapping = {

                "high_frequency_energy_anomaly": "High-Frequency Energy Anomaly",

                "phase_incoherence": "Phase Incoherence",

                "spectral_flux_deviation": "Spectral Flux Deviation",

                "spectral_centroid_deviation": "Spectral Centroid Deviation",

                "spectral_rolloff_deviation": "Spectral Rolloff Deviation",

                "spectral_bandwidth_anomaly": "Spectral Bandwidth Anomaly"

            }

            voice_signals_mapped = {

                voice_signals_mapping[k]: v for k, v in voice_breakdown.items()

            }

        result = {

            "id": scan_id,

            "filename": filename,

            "verdict": verdict,

            "ai_score": round(float(overall_score), 4),

            "risk_score": round(float(overall_score), 4),

            "media_type": "video",

            "confidence": confidence,

            "risk_grade": self._risk_grade(overall_score),

            "face_count": total_faces_found,

            "faces_analyzed": len(face_scores),

            "frames": frames_list,

            "face_details": avg_signals if face_scores else None,

            "voice_details": voice_signals_mapped,

            "audio_url": f"/static/cache/{wav_filename}" if audio_extracted else None,

            "analysis": {

                "deep_model_score": round(float(avg_deep_score), 4),

                "forensic_model_score": round(float(avg_forensic_score), 4),

                "voice_model_score": round(float(voice_score), 4) if voice_score is not None else None,

                "signals": {**avg_signals, **(voice_signals_mapped or {})}

            }

        }

        return result

    def scan_dual_faces(self, real_file_path: str, real_filename: str, check_file_path: str, check_filename: str) -> dict:

        real_img = cv2.imread(real_file_path)

        check_img = cv2.imread(check_file_path)

        if real_img is None:

            raise ValueError("Could not read the Reference Real image.")

        if check_img is None:

            raise ValueError("Could not read the Suspect Checking image.")

        real_faces = detect_faces(real_img)

        check_faces = detect_faces(check_img)

        if len(real_faces) == 0:

            raise ValueError("No face detected in the Reference Real image.")

        if len(check_faces) == 0:

            raise ValueError("No face detected in the Suspect Checking image.")

        real_box = sorted(real_faces, key=lambda f: f[2] * f[3], reverse=True)[0]

        check_box = sorted(check_faces, key=lambda f: f[2] * f[3], reverse=True)[0]

        def get_padded_crop(img, box):

            x, y, fw, fh = box

            pad_x = int(fw * 0.10)

            pad_y = int(fh * 0.10)

            x1 = max(0, x - pad_x)

            y1 = max(0, y - pad_y)

            x2 = min(img.shape[1], x + fw + pad_x)

            y2 = min(img.shape[0], y + fh + pad_y)

            return img[y1:y2, x1:x2]

        real_crop = get_padded_crop(real_img, real_box)

        check_crop = get_padded_crop(check_img, check_box)

        real_emb = self.detector.extract_deep_features(real_crop)

        check_emb = self.detector.extract_deep_features(check_crop)

        real_norm = real_emb / (np.linalg.norm(real_emb) + 1e-8)

        check_norm = check_emb / (np.linalg.norm(check_emb) + 1e-8)

        cos_sim = float(np.dot(real_norm, check_norm))

        calib_match = np.clip((cos_sim - 0.40) / 0.40, 0.0, 1.0)

        similarity_pct = round(calib_match * 100, 1)

        deepfake_score, breakdown = self.detector.predict(check_crop)

        is_match = cos_sim >= 0.68

        is_ai = deepfake_score >= self.detector.threshold

        if is_ai:

            if is_match:

                verdict = "AI_GENERATED_MANIPULATION_DETECTED"

                risk_score = deepfake_score

                risk_grade = "High Risk (AI Manipulation)"

                summary = "Identity matches the Reference Photo, but face contains deepfake synthesis artifacts."

            else:

                verdict = "IDENTITY_MISMATCH_AND_AI_GENERATED"

                risk_score = max(1.0 - calib_match, deepfake_score)

                risk_grade = "Critical Risk (Face-Swap)"

                summary = "Identity does NOT match the Reference Photo, and checking face contains deepfake artifacts (classic face-swap)."

        else:

            if is_match:

                verdict = "VERIFIED_REAL_IDENTITY"

                risk_score = min(1.0 - calib_match, deepfake_score)            

                risk_grade = "Likely Real"

                summary = "Identity matches the Reference Photo and no deepfake artifacts were detected."

            else:

                verdict = "IDENTITY_MISMATCH_SUSPECTED_SWAP"

                risk_score = max(1.0 - calib_match, 0.50)

                risk_grade = "High Risk (Identity Mismatch)"

                summary = "No deepfake artifacts detected, but identity does NOT match the Reference Photo."

        scan_id = uuid.uuid4().hex

        real_crop_fn = f"dual_real_{scan_id}.jpg"

        check_crop_fn = f"dual_check_{scan_id}.jpg"

        from app.utils.heatmaps import generate_face_heatmap

        overlay_img, heatmap_img = generate_face_heatmap(check_img, check_box, deepfake_score, breakdown, threshold=self.detector.threshold)

        cv2.imwrite(os.path.join(CACHE_DIR, real_crop_fn), real_crop)

        cv2.imwrite(os.path.join(CACHE_DIR, check_crop_fn), get_padded_crop(overlay_img, check_box))

        result = {

            "id": scan_id,

            "filename": f"{real_filename} vs {check_filename}",

            "verdict": verdict,

            "risk_score": round(float(risk_score), 4),

            "risk_grade": risk_grade,

            "media_type": "dual",

            "confidence": similarity_pct,                                 

            "similarity": similarity_pct,

            "cos_sim": round(cos_sim, 4),

            "is_match": is_match,

            "deepfake_score": round(float(deepfake_score), 4),

            "summary": summary,

            "real_crop_url": f"/static/cache/{real_crop_fn}",

            "check_crop_url": f"/static/cache/{check_crop_fn}",

            "analysis": {

                "deep_model_score": breakdown["deep_model_score"],

                "forensic_model_score": breakdown["forensic_model_score"],

                "signals": breakdown["signals"]

            }

        }

        self.history.insert(0, {

            "id": scan_id,

            "filename": result["filename"],

            "verdict": verdict,

            "risk_score": result["risk_score"],

            "media_type": "dual",

            "confidence": similarity_pct,

            "risk_grade": risk_grade,

        })

        if len(self.history) > self.MAX_HISTORY:

            self.history.pop()

        try:

            from app.database import save_scan_to_history

            save_scan_to_history(

                result["id"],

                result["filename"],

                result["media_type"],

                result["risk_score"],

                result["risk_grade"],

                result

            )

        except Exception as e:

            print(f"Error saving dual scan to history DB: {e}")

        return result

    @staticmethod

    def _risk_grade(score: float) -> str:

        if score < 0.70:

            return "Likely Real"

        elif score < 0.9350:

            return "Uncertain"

        elif score < 0.97:

            return "Likely AI"

        else:

            return "Strongly AI"

