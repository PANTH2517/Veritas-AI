import cv2

import numpy as np

def generate_face_heatmap(face_img_bgr, face_box, risk_score, forensic_features=None, threshold=0.5):

    h, w = face_img_bgr.shape[:2]

    anomaly_map = np.zeros((h, w), dtype=np.float32)

    if face_box is not None:

        x, y, fw, fh = face_box

        x1, y1 = max(0, x), max(0, y)

        x2, y2 = min(w, x + fw), min(h, y + fh)

        if x2 > x1 and y2 > y1:

            face_roi = face_img_bgr[y1:y2, x1:x2]

            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            rh, rw = gray.shape

            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            hp = cv2.absdiff(gray, blurred)

            hp_f = hp.astype(np.float32)

            mean_sq = cv2.boxFilter(hp_f ** 2, -1, (15, 15))

            sq_mean = cv2.boxFilter(hp_f, -1, (15, 15)) ** 2

            local_var = np.clip(mean_sq - sq_mean, 0, None)

            local_std = np.sqrt(local_var)

            texture_layer = local_std / (local_std.max() + 1e-6)

            fft = np.fft.fft2(gray.astype(np.float32))

            fft_shift = np.fft.fftshift(fft)

            magnitude = np.log1p(np.abs(fft_shift))

            mag_norm = magnitude / (magnitude.max() + 1e-6)

            fft_spatial = np.abs(np.fft.ifft2(np.fft.ifftshift(

                fft_shift * mag_norm

            ))).astype(np.float32)

            fft_layer = cv2.resize(fft_spatial, (rw, rh))

            fft_layer = fft_layer / (fft_layer.max() + 1e-6)

            bw_px = max(2, int(rw * 0.08))

            bh_px = max(2, int(rh * 0.08))

            boundary_mask = np.zeros_like(gray, dtype=np.uint8)

            cv2.rectangle(boundary_mask, (0, 0), (rw, rh), 255, -1)

            cv2.rectangle(boundary_mask, (bw_px, bh_px),

                          (rw - bw_px, rh - bh_px), 0, -1)

            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)

            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

            gmag = np.sqrt(gx ** 2 + gy ** 2)

            gmag_norm = gmag / (gmag.max() + 1e-6)

            boundary_layer = gmag_norm * (boundary_mask.astype(np.float32) / 255.0)

            landmark_mask = np.zeros_like(gray, dtype=np.float32)

            cv2.ellipse(landmark_mask, (int(rw * 0.32), int(rh * 0.38)),

                        (int(rw * 0.13), int(rh * 0.07)), 0, 0, 360, 1.0, -1)

            cv2.ellipse(landmark_mask, (int(rw * 0.68), int(rh * 0.38)),

                        (int(rw * 0.13), int(rh * 0.07)), 0, 0, 360, 1.0, -1)

            cv2.ellipse(landmark_mask, (int(rw * 0.50), int(rh * 0.57)),

                        (int(rw * 0.08), int(rh * 0.06)), 0, 0, 360, 0.7, -1)

            cv2.ellipse(landmark_mask, (int(rw * 0.50), int(rh * 0.74)),

                        (int(rw * 0.20), int(rh * 0.09)), 0, 0, 360, 1.0, -1)

            tex_w = 0.35

            fft_w = 0.25

            bnd_w = 0.25

            lmk_w = 0.15

            if forensic_features:

                signals = forensic_features.get("signals", {})

                tex_w = 0.15 + 0.25 * signals.get("High-Freq Texture Residual", 0.5)

                fft_w = 0.10 + 0.30 * signals.get("FFT Ring Artifacts", 0.5)

                bnd_w = 0.10 + 0.25 * signals.get("Boundary Gradient Discontinuity", 0.5)

                lmk_w = 0.05 + 0.20 * signals.get("Eye/Mouth Sharpness Mismatch", 0.5)

                total = tex_w + fft_w + bnd_w + lmk_w

                tex_w, fft_w, bnd_w, lmk_w = (

                    tex_w / total, fft_w / total, bnd_w / total, lmk_w / total

                )

            landmark_layer = (hp_f / 255.0) * landmark_mask

            face_anomaly = (

                tex_w * texture_layer +

                fft_w * fft_layer +

                bnd_w * boundary_layer +

                lmk_w * landmark_layer

            )

            face_anomaly = np.clip(face_anomaly * (0.25 + 0.75 * risk_score), 0.0, 1.0)

            anomaly_map[y1:y2, x1:x2] = face_anomaly

    else:

        gray_full = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2GRAY)

        blurred_full = cv2.GaussianBlur(gray_full, (9, 9), 0)

        hp_full = cv2.absdiff(gray_full, blurred_full).astype(np.float32)

        anomaly_map = np.clip(hp_full / 255.0 * risk_score, 0.0, 1.0)

    anomaly_map = cv2.GaussianBlur(anomaly_map, (17, 17), 0)

    anomaly_uint8 = (anomaly_map * 255.0).astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(anomaly_uint8, cv2.COLORMAP_TURBO)

    threshold_mask = (anomaly_map > 0.06).astype(np.float32)[:, :, np.newaxis]

    heatmap_img = (heatmap_colored * threshold_mask).astype(np.uint8)

    alpha = 0.50 * anomaly_map[:, :, np.newaxis]

    overlay_img = np.clip(

        face_img_bgr.astype(np.float32) * (1.0 - alpha) +

        heatmap_colored.astype(np.float32) * alpha,

        0, 255

    ).astype(np.uint8)

    if face_box is not None:

        x, y, fw, fh = face_box

        color = (87, 61, 255) if risk_score >= threshold else (118, 230, 0)

        cv2.rectangle(overlay_img, (x, y), (x + fw, y + fh), color, 2)

    return overlay_img, heatmap_img

