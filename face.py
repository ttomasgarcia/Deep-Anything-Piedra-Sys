"""
Realce de detalle del rostro en el mapa de z-depth.
Detecta la cara (MediaPipe Face Detection) y amplifica el contraste de profundidad
DENTRO de esa región, con una máscara elíptica suavizada (feather) para no dejar borde.
"""
import cv2
import numpy as np
import mediapipe as mp


class FaceDetector:
    """Reusar 1 por job (no es thread-safe)."""

    def __init__(self, min_conf: float = 0.4):
        # model_selection=1 = 'full range' (caras chicas/lejanas también)
        self.fd = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_conf)

    def boxes(self, rgb_frame):
        """Devuelve lista de (x0,y0,x1,y1) en píxeles para cada cara detectada."""
        H, W = rgb_frame.shape[:2]
        res = self.fd.process(rgb_frame)
        out = []
        if res.detections:
            for det in res.detections:
                bb = det.location_data.relative_bounding_box
                x0 = int(bb.xmin * W); y0 = int(bb.ymin * H)
                x1 = int((bb.xmin + bb.width) * W); y1 = int((bb.ymin + bb.height) * H)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(W, x1), min(H, y1)
                if x1 > x0 and y1 > y0:
                    out.append((x0, y0, x1, y1))
        return out

    def close(self):
        self.fd.close()


def feather_mask(boxes, W, H, pad=0.35):
    """Máscara float 0..1 (H,W): elipse suave por cara. 0 si no hay caras."""
    m = np.zeros((H, W), np.float32)
    if not boxes:
        return m
    for (x0, y0, x1, y1) in boxes:
        bw, bh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax, ay = bw * (0.5 + pad), bh * (0.5 + pad)   # semiejes con padding
        cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
        sigma = 0.18 * max(bw, bh)
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.clip(m, 0, 1)


def enhance(raw_full, global_norm, mask, strength=1.0):
    """
    Le da al rostro su propio rango de grises usando el depth CRUDO (no el
    globalmente recortado), para revelar el relieve facial. Fuera de la máscara
    queda intacto el depth normal.

    raw_full:    depth crudo re-escalado a (H,W) float — NO recortado globalmente.
    global_norm: depth normalizado global 0..1 (H,W).
    mask:        feather 0..1 del rostro.
    strength:    contraste del relieve facial (1.0 = rango completo).
    """
    core = mask > 0.5
    if int(core.sum()) < 30:
        return global_norm
    face = raw_full[core]
    lo, hi = np.percentile(face, 2), np.percentile(face, 98)
    span = float(hi - lo)
    if span < 1e-9:
        return global_norm
    # relieve local del rostro, su rango propio -> 0..1 (0 = más cerca)
    local = np.clip((raw_full - lo) / span, 0.0, 1.0)
    # micro-contraste (CLAHE) para revelar el relieve suave que da DA3
    g8 = (local * 255).astype(np.uint8)
    clip_limit = 2.0 + 3.0 * float(np.clip(strength, 0.0, 2.0))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    local = clahe.apply(g8).astype(np.float32) / 255.0
    band = float(np.clip(0.6 + 0.5 * strength, 0.3, 1.6))
    local = np.clip((local - 0.5) * band + 0.5, 0.0, 1.0)   # contraste global del rostro
    # mezcla feather: adentro usa el relieve facial, afuera el depth normal
    return np.clip(global_norm * (1.0 - mask) + local * mask, 0.0, 1.0)
