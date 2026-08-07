#!/bin/bash
# Lanza la web app de Depth Anything 3 (local, MPS)
cd "$(dirname "$0")"
source venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1
echo "→ http://127.0.0.1:8000"
exec python -m uvicorn app:app --host 127.0.0.1 --port 8000
