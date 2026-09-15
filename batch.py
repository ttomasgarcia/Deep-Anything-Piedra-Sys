"""
Procesamiento por lotes de videos -> z-depth (grises), reusando el pipeline del app.
Config fija para este batch: process_res=672, rostro=aplanar, grises, cerca=blanco, fps original.
Reanudable: saltea clips cuyo output ya existe.

uso: python batch.py <archivo_lista.txt> <carpeta_salida>
     (archivo_lista.txt = un path de video por línea)
"""
import os, sys, time, shutil, subprocess
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import cv2
import torch
from PIL import Image

import app                      # reusa get_model, extract_frames, depth_normalize, norm_to_bgr, FFMPEG
import face as face_mod

PROCESS_RES = 672
INVERT = True
COLORMAP = "gray"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def process_one(model, video_path: Path, out_path: Path, work_dir: Path):
    t0 = time.time()
    frames, out_fps, (W, H) = app.extract_frames(video_path, None)
    n = len(frames)
    log(f"  {n} frames @ {out_fps:.3f} fps  ({W}x{H})")

    face_det = face_mod.FaceDetector()
    raw_depths, face_boxes = [], []
    for idx, fr in enumerate(frames):
        with torch.no_grad():
            pred = model.inference([Image.fromarray(fr)], process_res=PROCESS_RES,
                                   export_format="mini_npz", export_dir=None)
        raw_depths.append(pred.depth[0].astype(np.float32))
        face_boxes.append(face_det.boxes(fr))
        if (idx + 1) % 20 == 0 or idx + 1 == n:
            el = time.time() - t0
            eta = el / (idx + 1) * (n - idx - 1)
            log(f"    depth {idx+1}/{n}  ({el/(idx+1):.1f}s/frame, ETA {eta/60:.1f}min)")
    face_det.close()
    frames = None

    allv = np.concatenate([d.ravel() for d in raw_depths])
    lo, hi = float(np.percentile(allv, 1)), float(np.percentile(allv, 99))

    frames_dir = work_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    for idx in range(n):
        norm = app.depth_normalize(raw_depths[idx], W, H, lo, hi)
        mask = face_mod.feather_mask(face_boxes[idx], W, H)
        norm = face_mod.flatten(norm, mask)
        bgr = app.norm_to_bgr(norm, INVERT, COLORMAP)
        cv2.imwrite(str(frames_dir / f"f_{idx:06d}.png"), bgr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [app.FFMPEG, "-y", "-framerate", f"{out_fps:.6f}",
           "-i", str(frames_dir / "f_%06d.png"),
           "-i", str(video_path),                       # audio del original
           "-map", "0:v", "-map", "1:a?",
           "-c:a", "aac", "-b:a", "192k", "-shortest",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
           "-movflags", "+faststart", str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg falló:\n" + proc.stderr[-1200:])
    log(f"  OK -> {out_path.name}  ({time.time()-t0:.0f}s)")

def main():
    list_file, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(app.RUNS) / "_batch"
    work_dir.mkdir(parents=True, exist_ok=True)
    videos = [Path(l.strip()) for l in list_file.read_text().splitlines() if l.strip()]

    log(f"cargando modelo… ({len(videos)} clips, res {PROCESS_RES})")
    model = app.get_model()
    log("modelo listo")

    ok = skip = fail = 0
    for i, v in enumerate(videos, 1):
        out_path = out_dir / f"{v.stem}_zdepth.mp4"
        log(f"[{i}/{len(videos)}] {v.name}")
        if out_path.exists():
            log("  ya existe, saltea"); skip += 1; continue
        try:
            process_one(model, v, out_path, work_dir)
            ok += 1
        except Exception as e:
            log(f"  ERROR: {e}"); fail += 1
    log(f"FIN — ok={ok} saltados={skip} fallidos={fail} de {len(videos)}")

if __name__ == "__main__":
    main()
