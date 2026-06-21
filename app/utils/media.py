"""
media.py — Fully crash-proof for Vercel serverless.
Every module-level operation is wrapped in try/except so this module
always imports successfully, regardless of environment.
"""
import os
import numpy as np

# ── Optional heavy imports ───────────────────────────────────────────────────
try:
    import cv2
    _CV2_OK = True
except ImportError:
    cv2 = None
    _CV2_OK = False
    print("[media] OpenCV not available.")

try:
    import av
    _AV_AVAILABLE = True
except ImportError:
    av = None
    _AV_AVAILABLE = False

try:
    import scipy.io.wavfile as _wav_module
    _SCIPY_OK = True
except ImportError:
    _wav_module = None
    _SCIPY_OK = False

import urllib.request

# ── Cascade paths ─────────────────────────────────────────────────────────────
# Strategy: use files bundled inside the cv2 pip package — zero downloads.
_haar_front = None
_haar_side  = None
_haar_alt2  = None
_haar_lbp   = None
_dnn_net    = None

def _get_cv2_data_dir():
    """Safely return the cv2 data directory path, or empty string."""
    try:
        # cv2.data.haarcascades is a string path ending in '/'
        p = getattr(cv2, 'data', None)
        if p is None:
            return ""
        return getattr(p, 'haarcascades', "")
    except Exception:
        return ""

def _load_cascade(path):
    """Load a single cascade classifier, returning None on any failure."""
    try:
        if path and os.path.exists(path):
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                return c
    except Exception as e:
        print(f"[media] Cascade load failed ({path}): {e}")
    return None

def _init_cascades():
    """Initialize Haar cascade classifiers from cv2 bundled data."""
    global _haar_front, _haar_side, _haar_alt2, _haar_lbp
    if not _CV2_OK:
        return
    try:
        d = _get_cv2_data_dir()
        _haar_front = _load_cascade(os.path.join(d, "haarcascade_frontalface_default.xml"))
        _haar_side  = _load_cascade(os.path.join(d, "haarcascade_profileface.xml"))
        _haar_alt2  = _load_cascade(os.path.join(d, "haarcascade_frontalface_alt2.xml"))
        _haar_lbp   = _load_cascade(os.path.join(d, "lbpcascade_frontalface_improved.xml"))
        loaded = sum(1 for c in [_haar_front, _haar_side, _haar_alt2, _haar_lbp] if c is not None)
        print(f"[media] Loaded {loaded}/4 cascade classifiers from {d or 'unknown'}")
    except Exception as e:
        print(f"[media] Cascade init error (detection will use fallback): {e}")

# Run at import time — all errors are swallowed
try:
    _init_cascades()
except Exception as _e:
    print(f"[media] _init_cascades failed: {_e}")


# ── Helper: NMS ───────────────────────────────────────────────────────────────
def _iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / (union + 1e-6)

def _nms_boxes(boxes, iou_thresh=0.40):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        keep.append(boxes[i])
        for j in range(i + 1, len(boxes)):
            if not used[j] and _iou(boxes[i], boxes[j]) > iou_thresh:
                used[j] = True
    return keep

def _haar_detect(gray, cascade, scale=1.05, min_nb=3, min_sz=20):
    if cascade is None:
        return []
    try:
        if cascade.empty():
            return []
        faces = cascade.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=min_nb,
            minSize=(min_sz, min_sz), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    except Exception:
        return []

def _full_image_fallback(image_bgr):
    h, w = image_bgr.shape[:2]
    x1 = int(w * 0.15)
    y1 = int(h * 0.10)
    x2 = int(w * 0.85)
    y2 = int(h * 0.90)
    return [(x1, y1, x2 - x1, y2 - y1)]


# ── Public API ────────────────────────────────────────────────────────────────
def detect_faces(image_bgr):
    """Detect faces in a BGR image. Always returns at least one box (fallback)."""
    if not _CV2_OK or image_bgr is None:
        h, w = image_bgr.shape[:2] if image_bgr is not None else (100, 100)
        return [(int(w*0.15), int(h*0.10), int(w*0.70), int(h*0.80))]

    try:
        h, w = image_bgr.shape[:2]
        all_boxes = []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        for casc in [_haar_front, _haar_alt2, _haar_lbp]:
            for g in [gray, gray_eq]:
                for (sc, nb) in [(1.05, 3), (1.10, 2), (1.15, 1)]:
                    all_boxes.extend(_haar_detect(g, casc, scale=sc, min_nb=nb, min_sz=25))

        for flip_code in [None, 1]:
            g = gray if flip_code is None else cv2.flip(gray, flip_code)
            boxes = _haar_detect(g, _haar_side, scale=1.05, min_nb=2, min_sz=25)
            if flip_code == 1:
                boxes = [(w - x - bw, y, bw, bh) for (x, y, bw, bh) in boxes]
            all_boxes.extend(boxes)

        all_boxes = _nms_boxes(all_boxes, iou_thresh=0.35)

        if not all_boxes:
            print("[media] All face detectors failed — using full-image fallback.")
            all_boxes = _full_image_fallback(image_bgr)

        return all_boxes

    except Exception as e:
        print(f"[media] detect_faces error: {e}")
        return _full_image_fallback(image_bgr)


def extract_frames(video_path, max_frames=45):
    """Extract frames from a video file using OpenCV."""
    if not _CV2_OK:
        return []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
        frames_data = []
        current_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in frame_indices:
                frames_data.append({
                    'frame': frame,
                    'timestamp': round(current_idx / fps, 2),
                    'index': current_idx
                })
            current_idx += 1
            if len(frames_data) >= len(frame_indices):
                break
        cap.release()
        return frames_data
    except Exception as e:
        print(f"[media] extract_frames error: {e}")
        return []


def extract_audio_from_video(video_path, output_wav_path):
    """Extract audio from video. Returns False if PyAV unavailable."""
    if not _AV_AVAILABLE:
        print("[media] PyAV not available — skipping audio extraction.")
        return False
    try:
        container = av.open(video_path)
        audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
        if not audio_stream:
            return False
        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        samples = []
        for frame in container.decode(audio_stream):
            resampled_frames = resampler.resample(frame)
            if resampled_frames:
                for rf in resampled_frames:
                    samples.append(np.frombuffer(rf.planes[0], dtype=np.int16))
        if not samples or not _SCIPY_OK:
            return False
        _wav_module.write(output_wav_path, 16000, np.concatenate(samples))
        return True
    except Exception as e:
        print(f"[media] extract_audio_from_video error: {e}")
        return False


def decode_audio_file(audio_path, output_wav_path):
    """Decode audio to WAV. Returns False if PyAV unavailable."""
    if not _AV_AVAILABLE or not _SCIPY_OK:
        return False
    try:
        if audio_path.lower().endswith('.wav'):
            try:
                import shutil
                rate, data = _wav_module.read(audio_path)
                if rate == 16000 and len(data.shape) == 1:
                    shutil.copy(audio_path, output_wav_path)
                    return True
            except Exception:
                pass
        container = av.open(audio_path)
        audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
        if not audio_stream:
            return False
        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        samples = []
        for frame in container.decode(audio_stream):
            resampled_frames = resampler.resample(frame)
            if resampled_frames:
                for rf in resampled_frames:
                    samples.append(np.frombuffer(rf.planes[0], dtype=np.int16))
        if not samples:
            return False
        _wav_module.write(output_wav_path, 16000, np.concatenate(samples))
        return True
    except Exception as e:
        print(f"[media] decode_audio_file error: {e}")
        return False
