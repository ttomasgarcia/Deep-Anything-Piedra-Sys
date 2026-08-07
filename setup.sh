#!/bin/bash
# Instalación de la web app Depth Anything 3 en macOS (Apple Silicon / MPS).
# Reproduce el entorno completo desde cero.
set -e
cd "$(dirname "$0")"

echo "==> Requisitos del sistema (Homebrew)"
if ! command -v python3.12 >/dev/null 2>&1 && [ ! -x /opt/homebrew/bin/python3.12 ]; then
  brew install python@3.12
fi
command -v ffmpeg >/dev/null 2>&1 || brew install ffmpeg

PY=/opt/homebrew/bin/python3.12
[ -x "$PY" ] || PY=$(command -v python3.12)

echo "==> Creando venv (Python 3.12)"
"$PY" -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo "==> Instalando PyTorch (MPS)"
pip install "torch>=2" torchvision

echo "==> Clonando ByteDance/Depth-Anything-3"
[ -d repo ] || git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git repo

echo "==> Instalando el paquete DA3 SIN sus dependencias (evita xformers)"
pip install --no-deps -e repo

echo "==> Instalando dependencias compatibles con Mac"
pip install -r requirements.txt
pip install "numpy<2"   # re-fijar por si alguna dep la subió

echo ""
echo "✅ Listo. Para arrancar la app:  ./run.sh   ->  http://127.0.0.1:8000"
