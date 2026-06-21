import os

import urllib.request

import cv2

import numpy as np

import av

import scipy.io.wavfile as wav

CASCADE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")

os.makedirs(CASCADE_DIR, exist_ok=True)

CASCADE_PATH       = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")

CASCADE_SIDE_PATH  = os.path.join(CASCADE_DIR, "haarcascade_profileface.xml")

DNN_PROTO_PATH     = os.path.join(CASCADE_DIR, "deploy.prototxt")

DNN_MODEL_PATH     = os.path.join(CASCADE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

CASCADE_ALT2_PATH  = os.path.join(CASCADE_DIR, "haarcascade_frontalface_alt2.xml")

CASCADE_LBP_PATH   = os.path.join(CASCADE_DIR, "lbpcascade_frontalface_improved.xml")

_URLS = {

    CASCADE_PATH:

        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",

    CASCADE_SIDE_PATH:

        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_profileface.xml",

    CASCADE_ALT2_PATH:

        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml",

    CASCADE_LBP_PATH:

        "https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades/lbpcascade_frontalface_improved.xml",

    DNN_PROTO_PATH:

        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",

    DNN_MODEL_PATH:

        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",

}

def _download(url: str, dest: str) -> bool:

    try:

        print(f"[media] Downloading {os.path.basename(dest)}…")

        urllib.request.urlretrieve(url, dest)

        print(f"[media] Downloaded {os.path.basename(dest)} OK")

        return True

    except Exception as e:

        print(f"[media] Download failed for {os.path.basename(dest)}: {e}")

        return False

def _ensure_assets():

    for path, url in _URLS.items():

        if not os.path.exists(path) or os.path.getsize(path) < 1000:

            _download(url, path)

_ensure_assets()

_haar_front  = None

_haar_side   = None

_haar_alt2   = None

_haar_lbp    = None

_dnn_net     = None

def _load_cascades():

    global _haar_front, _haar_side, _haar_alt2, _haar_lbp

    if _haar_front is None and os.path.exists(CASCADE_PATH):

        _haar_front = cv2.CascadeClassifier(CASCADE_PATH)

    if _haar_side is None and os.path.exists(CASCADE_SIDE_PATH):

        _haar_side = cv2.CascadeClassifier(CASCADE_SIDE_PATH)

    if _haar_alt2 is None and os.path.exists(CASCADE_ALT2_PATH):

        _haar_alt2 = cv2.CascadeClassifier(CASCADE_ALT2_PATH)

    if _haar_lbp is None and os.path.exists(CASCADE_LBP_PATH):

        _haar_lbp = cv2.CascadeClassifier(CASCADE_LBP_PATH)

def _load_dnn():

    global _dnn_net

    if _dnn_net is None:

        if os.path.exists(DNN_PROTO_PATH) and os.path.exists(DNN_MODEL_PATH):

            try:

                _dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)

                print("[media] DNN SSD face detector loaded.")

            except Exception as e:

                print(f"[media] DNN load error: {e}")

_load_cascades()

_load_dnn()

def _haar_detect(gray, cascade, scale=1.05, min_nb=3, min_sz=20):

    if cascade is None or cascade.empty():

        return []

    faces = cascade.detectMultiScale(

        gray, scaleFactor=scale, minNeighbors=min_nb,

        minSize=(min_sz, min_sz), flags=cv2.CASCADE_SCALE_IMAGE

    )

    if len(faces) == 0:

        return []

    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def _dnn_detect(image_bgr, conf_thresh=0.30):

    if _dnn_net is None:

        return []

    h, w = image_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(

        cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300),

        (104.0, 177.0, 123.0), swapRB=False, crop=False

    )

    _dnn_net.setInput(blob)

    try:

        detections = _dnn_net.forward()

    except Exception:

        return []

    boxes = []

    for i in range(detections.shape[2]):

        confidence = float(detections[0, 0, i, 2])

        if confidence < conf_thresh:

            continue

        x1 = max(0, int(detections[0, 0, i, 3] * w))

        y1 = max(0, int(detections[0, 0, i, 4] * h))

        x2 = min(w, int(detections[0, 0, i, 5] * w))

        y2 = min(h, int(detections[0, 0, i, 6] * h))

        if x2 > x1 and y2 > y1:

            boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes

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

def _full_image_fallback(image_bgr):

    h, w = image_bgr.shape[:2]

    x1 = int(w * 0.15)

    y1 = int(h * 0.10)

    x2 = int(w * 0.85)

    y2 = int(h * 0.90)

    return [(x1, y1, x2 - x1, y2 - y1)]

def detect_faces(image_bgr):

    h, w = image_bgr.shape[:2]

    all_boxes = []

    dnn_boxes = _dnn_detect(image_bgr, conf_thresh=0.30)

    all_boxes.extend(dnn_boxes)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gray_eq = cv2.equalizeHist(gray)                                      

    for casc in [_haar_front, _haar_alt2, _haar_lbp]:

        for g in [gray, gray_eq]:

            for (sc, nb) in [(1.05, 3), (1.10, 2), (1.15, 1)]:

                boxes = _haar_detect(g, casc, scale=sc, min_nb=nb, min_sz=25)

                all_boxes.extend(boxes)

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

def extract_frames(video_path, max_frames=45):

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

            timestamp = current_idx / fps

            frames_data.append({

                'frame': frame,

                'timestamp': round(timestamp, 2),

                'index': current_idx

            })

        current_idx += 1

        if len(frames_data) >= len(frame_indices):

            break

    cap.release()

    return frames_data

def extract_audio_from_video(video_path, output_wav_path):

    try:

        container = av.open(video_path)

    except Exception as e:

        print(f"Error opening video for audio extraction: {e}")

        return False

    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)

    if not audio_stream:

        return False

    try:

        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)

    except Exception as e:

        print(f"Error creating audio resampler: {e}")

        return False

    samples = []

    try:

        for frame in container.decode(audio_stream):

            resampled_frames = resampler.resample(frame)

            if resampled_frames:

                for rf in resampled_frames:

                    data = np.frombuffer(rf.planes[0], dtype=np.int16)

                    samples.append(data)

    except Exception as e:

        print(f"Error decoding audio stream: {e}")

        if not samples:

            return False

    if not samples:

        return False

    try:

        audio_data = np.concatenate(samples)

        wav.write(output_wav_path, 16000, audio_data)

        return True

    except Exception as e:

        print(f"Error writing WAV file: {e}")

        return False

def decode_audio_file(audio_path, output_wav_path):

    try:

        if audio_path.lower().endswith('.wav'):

            try:

                rate, data = wav.read(audio_path)

                if rate == 16000 and len(data.shape) == 1:

                    import shutil

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

                    data = np.frombuffer(rf.planes[0], dtype=np.int16)

                    samples.append(data)

        if not samples:

            return False

        audio_data = np.concatenate(samples)

        wav.write(output_wav_path, 16000, audio_data)

        return True

    except Exception as e:

        print(f"Error decoding audio file {audio_path}: {e}")

        return False

