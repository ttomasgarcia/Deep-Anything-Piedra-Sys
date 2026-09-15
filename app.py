"""
Depth Anything 3 — Web app local (Apple Silicon / MPS)
Subís un video -> devuelve un .mp4 de z-depth en escala de grises.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # ops no soportadas caen a CPU

import io
import time
import shutil
import threading
import subprocess
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from depth_anything_3.api import DepthAnything3
import pose as pose_mod

# ----------------------------------------------------------------------------
BASE = Path(__file__).parent
RUNS = BASE / "runs"
RUNS.mkdir(exist_ok=True)
STATIC = BASE / "static"

MODEL_ID = os.environ.get("DA3_MODEL", "depth-anything/DA3MONO-LARGE")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

# ----------------------------------------------------------------------------
# Estado global del modelo (carga perezosa, singleton)
_model = None
_model_lock = threading.Lock()   # una inferencia a la vez (MPS)
_model_state = {"status": "idle", "message": "modelo no cargado"}


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model_state.update(status="loading", message=f"cargando {MODEL_ID}…")
                t0 = time.time()
                m = DepthAnything3.from_pretrained(MODEL_ID)
                m = m.to(device=DEVICE)
                m.device = DEVICE
                m.eval()
                _model = m
                _model_state.update(
                    status="ready",
                    message=f"{MODEL_ID} listo en {DEVICE} ({time.time()-t0:.0f}s)",
                )
    return _model


# ----------------------------------------------------------------------------
# Jobs
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def new_job() -> str:
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {
            "status": "queued",
            "message": "en cola",
            "done": 0,
            "total": 0,
            "error": None,
            "out": None,
            "started": time.time(),
        }
    return jid


def upd(jid, **kw):
    with JOBS_LOCK:
        if jid in JOBS:
            JOBS[jid].update(kw)


# ----------------------------------------------------------------------------
def extract_frames(video_path: Path, target_fps: float | None):
    """Devuelve (lista_de_frames_RGB_uint8, fps_salida, (W,H) original)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video.")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if src_fps <= 0:
        src_fps = 24.0
    if target_fps and target_fps > 0 and target_fps < src_fps:
        step = max(1, round(src_fps / target_fps))
        out_fps = src_fps / step
    else:
        step = 1
        out_fps = src_fps

    frames = []
    W = H = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            if W is None:
                H, W = frame.shape[:2]
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames, out_fps, (W, H)


def grab_frame(video_path: Path, frac: float):
    """Devuelve (frame_RGB uint8, idx, total) del frame en la posición frac (0..1)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = int(round((total - 1) * max(0.0, min(1.0, frac)))) if total > 0 else 0
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:  # fallback: reabrir y leer el primero
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        idx = 0
    cap.release()
    if not ok:
        raise RuntimeError("No se pudo leer un frame del video.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), idx, total


def depth_to_bgr(d, W, H, invert, colormap, lo, hi):
    """d: depth crudo (h,w). Devuelve imagen BGR normalizada + coloreada a WxH."""
    rng = (hi - lo) or 1e-6
    nrm = np.clip((d - lo) / rng, 0, 1)
    if invert:
        nrm = 1.0 - nrm            # cerca = claro
    g = (nrm * 255).astype(np.uint8)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
    if colormap == "turbo":
        return cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
    if colormap == "magma":
        return cv2.applyColorMap(g, cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def render_single(frame_rgb, opts):
    """Procesa UN frame (depth/pose según modo) y devuelve imagen BGR. Para el preview."""
    mode = opts.get("mode", "depth")
    need_depth = mode in ("depth", "depth_pose")
    need_pose = mode in ("depth_pose", "pose")
    H, W = frame_rgb.shape[:2]

    if need_depth:
        model = get_model()
        with _model_lock:
            with torch.no_grad():
                pred = model.inference(
                    [Image.fromarray(frame_rgb)],
                    process_res=int(opts.get("process_res", 504)),
                    export_format="mini_npz", export_dir=None,
                )
        d = pred.depth[0].astype(np.float32)
        lo = float(np.percentile(d, 1))
        hi = float(np.percentile(d, 99))
        base = depth_to_bgr(d, W, H, bool(opts.get("invert", True)),
                            opts.get("colormap", "gray"), lo, hi)
    else:
        base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if need_pose:
        est = pose_mod.PoseEstimator(complexity=int(opts.get("pose_complexity", 1)))
        try:
            pose_mod.draw(base, est.detect(frame_rgb), draw_face=bool(opts.get("draw_face", False)))
        finally:
            est.close()
    return base


def run_job(jid, video_path: Path, opts: dict):
    estimator = None
    try:
        mode = opts.get("mode", "depth")          # depth | depth_pose | pose
        need_depth = mode in ("depth", "depth_pose")
        need_pose = mode in ("depth_pose", "pose")

        upd(jid, status="extracting", message="extrayendo frames…")
        frames, out_fps, (W, H) = extract_frames(video_path, opts.get("target_fps"))
        n = len(frames)
        if n == 0:
            raise RuntimeError("El video no tiene frames legibles.")
        upd(jid, total=n, message=f"{n} frames @ {out_fps:.2f} fps")

        process_res = int(opts.get("process_res", 504))
        invert = bool(opts.get("invert", True))
        colormap = opts.get("colormap", "gray")
        draw_face = bool(opts.get("draw_face", False))

        model = None
        if need_depth:
            upd(jid, status="loading_model", message="preparando modelo de depth…")
            model = get_model()
        if need_pose:
            estimator = pose_mod.PoseEstimator(complexity=int(opts.get("pose_complexity", 1)))

        raw_depths = []          # depth crudo por frame (resolución de proceso)
        pose_frames = []         # landmarks por frame
        upd(jid, status="inferring")
        for idx, fr in enumerate(frames):
            if need_depth:
                with _model_lock:
                    with torch.no_grad():
                        pred = model.inference(
                            [Image.fromarray(fr)],
                            process_res=process_res,
                            export_format="mini_npz",
                            export_dir=None,
                        )
                raw_depths.append(pred.depth[0].astype(np.float32))
            if need_pose:
                pose_frames.append(estimator.detect(fr))
            upd(jid, done=idx + 1, message=f"procesando {idx+1}/{n}")

        # Normalización global del depth (rango consistente -> menos flicker)
        lo = hi = rng = None
        if need_depth:
            allv = np.concatenate([d.ravel() for d in raw_depths])
            lo = float(np.percentile(allv, 1))
            hi = float(np.percentile(allv, 99))
            rng = (hi - lo) or 1e-6

        upd(jid, status="encoding", message="componiendo y codificando…")
        frames_dir = RUNS / jid / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(n):
            # --- imagen base (BGR) ---
            if need_depth:
                nrm = np.clip((raw_depths[idx] - lo) / rng, 0, 1)
                if invert:
                    nrm = 1.0 - nrm            # cerca = claro
                g = (nrm * 255).astype(np.uint8)
                g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
                if colormap == "turbo":
                    base = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
                elif colormap == "magma":
                    base = cv2.applyColorMap(g, cv2.COLORMAP_MAGMA)
                else:
                    base = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            else:
                base = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)  # video original

            # --- overlay de pose ---
            if need_pose:
                pose_mod.draw(base, pose_frames[idx], draw_face=draw_face)

            cv2.imwrite(str(frames_dir / f"f_{idx:06d}.png"), base)

        out_path = RUNS / jid / "output.mp4"
        cmd = [
            FFMPEG, "-y", "-framerate", f"{out_fps:.6f}",
            "-i", str(frames_dir / "f_%06d.png"),
        ]
        if opts.get("include_audio"):
            # segundo input = video original; se toma sólo su audio (opcional con ?)
            cmd += ["-i", str(video_path),
                    "-map", "0:v", "-map", "1:a?",
                    "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
            "-movflags", "+faststart", str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg falló:\n" + proc.stderr[-1500:])

        shutil.rmtree(frames_dir, ignore_errors=True)
        upd(jid, status="done", out=str(out_path),
            message=f"listo — {n} frames en {time.time()-JOBS[jid]['started']:.0f}s")
    except Exception as e:
        upd(jid, status="error", error=str(e),
            message="error: " + str(e))
        traceback.print_exc()
    finally:
        if estimator is not None:
            estimator.close()


# ----------------------------------------------------------------------------
app = FastAPI(title="Depth Anything 3 — local")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/model")
def model_status():
    return _model_state


@app.post("/api/preview")
async def preview(
    video: UploadFile = File(...),
    mode: str = Form("depth"),
    position: float = Form(0.5),            # 0..1 posición del frame
    process_res: int = Form(504),
    invert: bool = Form(True),
    colormap: str = Form("gray"),
    pose_complexity: int = Form(1),
    draw_face: bool = Form(False),
):
    if not video.filename:
        raise HTTPException(400, "Falta el archivo de video.")
    if mode not in ("depth", "depth_pose", "pose"):
        raise HTTPException(400, "modo inválido")
    tmp_dir = RUNS / ("preview_" + uuid.uuid4().hex[:8])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        vpath = tmp_dir / ("input" + Path(video.filename).suffix)
        with open(vpath, "wb") as f:
            shutil.copyfileobj(video.file, f)
        frame_rgb, idx, total = grab_frame(vpath, position)
        opts = dict(mode=mode, process_res=process_res, invert=invert,
                    colormap=colormap, pose_complexity=pose_complexity, draw_face=draw_face)
        bgr = render_single(frame_rgb, opts)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("no se pudo codificar el PNG")
        headers = {"x-frame": str(idx), "x-total": str(total)}
        return Response(content=buf.tobytes(), media_type="image/png", headers=headers)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    mode: str = Form("depth"),              # depth | depth_pose | pose
    process_res: int = Form(504),
    target_fps: float = Form(0),
    invert: bool = Form(True),
    colormap: str = Form("gray"),
    pose_complexity: int = Form(1),
    draw_face: bool = Form(False),
    include_audio: bool = Form(False),
):
    if not video.filename:
        raise HTTPException(400, "Falta el archivo de video.")
    if mode not in ("depth", "depth_pose", "pose"):
        raise HTTPException(400, "modo inválido")
    jid = new_job()
    job_dir = RUNS / jid
    job_dir.mkdir(parents=True, exist_ok=True)
    vpath = job_dir / ("input" + Path(video.filename).suffix)
    with open(vpath, "wb") as f:
        shutil.copyfileobj(video.file, f)

    opts = {
        "mode": mode,
        "process_res": process_res,
        "target_fps": target_fps if target_fps and target_fps > 0 else None,
        "invert": invert,
        "colormap": colormap,
        "pose_complexity": pose_complexity,
        "draw_face": draw_face,
        "include_audio": include_audio,
    }
    threading.Thread(target=run_job, args=(jid, vpath, opts), daemon=True).start()
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    with JOBS_LOCK:
        j = JOBS.get(jid)
        if not j:
            raise HTTPException(404, "job no encontrado")
        return {k: v for k, v in j.items() if k != "out"} | {"has_result": bool(j.get("out"))}


@app.get("/api/jobs/{jid}/result")
def job_result(jid: str):
    with JOBS_LOCK:
        j = JOBS.get(jid)
    if not j or not j.get("out"):
        raise HTTPException(404, "resultado no disponible")
    return FileResponse(j["out"], media_type="video/mp4", filename=f"output_{jid}.mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
