import os, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
import numpy as np
import torch
from PIL import Image
from depth_anything_3.api import DepthAnything3

MODEL = "depth-anything/DA3MONO-LARGE"
IMG = "repo/assets/examples/SOH/000.png"

dev = torch.device("mps")
print("loading model...", MODEL)
t0 = time.time()
model = DepthAnything3.from_pretrained(MODEL)
model = model.to(device=dev)
model.device = dev
model.eval()
print(f"model loaded in {time.time()-t0:.1f}s")

img = Image.open(IMG).convert("RGB")
print("input size:", img.size)

t0 = time.time()
with torch.no_grad():
    pred = model.inference([img], export_format="mini_npz", export_dir=None)
dt = time.time() - t0
d = pred.depth  # N,H,W
print(f"inference {dt:.2f}s | depth shape={d.shape} dtype={d.dtype} min={d.min():.4f} max={d.max():.4f} is_metric={pred.is_metric}")

# normalize to grayscale (near=white)
dd = d[0].astype(np.float32)
lo, hi = np.percentile(dd, 1), np.percentile(dd, 99)
n = np.clip((dd - lo) / (hi - lo + 1e-8), 0, 1)
n = 1.0 - n  # invert so near = bright (typical z-depth convention varies)
Image.fromarray((n * 255).astype(np.uint8)).save("test_depth.png")
print("saved test_depth.png")
