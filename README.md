# Deep Anything Piedra Sys

Web app **local** para extraer el mapa de profundidad (**z-depth**) de un video, usando
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) de ByteDance.
Subís un video → te devuelve un `.mp4` de z-depth en escala de grises, listo para usar
como pase de profundidad en compositing (After Effects, Nuke, etc.).

Corre **100% local en Apple Silicon** (Mac M-series) vía **MPS** — sin nube, sin cuentas.

![z-depth](https://img.shields.io/badge/device-Apple%20MPS-blue) ![python](https://img.shields.io/badge/python-3.12-green)

## Características

- Interfaz web simple: arrastrás el video, elegís opciones, y ves el progreso frame a frame.
- **Tres modos:**
  - **Solo z-depth** — mapa de profundidad en grises o color.
  - **z-depth + pose encima** — esqueleto de cuerpo y manos sobre el depth.
  - **Solo pose** — esqueleto sobre el video original.
- **Estimación de pose** (cuerpo 33 pts + 21 pts por mano, cara opcional) con
  [MediaPipe Holistic](https://github.com/google/mediapipe), corriendo en CPU.
- Salida de depth en **grises** (para compositing) o **color** (Turbo / Magma).
- Controles de resolución, FPS de salida, inversión y calidad de pose.
- Normalización global del clip para reducir el *flicker* temporal.
- Modelo de depth por defecto: `depth-anything/DA3MONO-LARGE` (monocular relativo, por frame).

## Requisitos

- macOS con Apple Silicon (probado en M4 Pro, 24 GB).
- [Homebrew](https://brew.sh). `setup.sh` instala Python 3.12 y ffmpeg si faltan.

## Instalación

```bash
git clone https://github.com/ttomasgarcia/Deep-Anything-Piedra-Sys.git
cd Deep-Anything-Piedra-Sys
./setup.sh
```

`setup.sh` crea el `venv`, instala PyTorch (MPS), clona el repo de Depth Anything 3 y
todas las dependencias. La primera vez tarda unos minutos (descarga PyTorch y deps).

## Uso

```bash
./run.sh
```

Abrí **http://127.0.0.1:8000**, arrastrá un video y dale **Procesar**.
El modelo (~1.4 GB) se descarga automáticamente de HuggingFace la primera vez.

## Notas técnicas

Este proyecto está pensado para correr en Mac, donde el repo original asume CUDA. Los ajustes clave:

- **Sin `xformers`.** No compila en Apple Silicon; el modelo cae a PyTorch puro
  (`SwiGLUFFN` + `scaled_dot_product_attention`), por eso se instala con `--no-deps`.
- **`numpy<2`** (con `scipy==1.12.0` y `plyfile==1.0.3`, las versiones compatibles).
- `KMP_DUPLICATE_LIB_OK=TRUE` y `PYTORCH_ENABLE_MPS_FALLBACK=1` (ver `run.sh`).
- Las dependencias 3D pesadas (open3d, gsplat) **no** son necesarias para el pase de depth.
- **MediaPipe fijado en `0.10.14`**: usa la API `solutions` (CPU, estable en Mac).
  La serie 1.x sólo trae la Tasks API, que **crashea en el delegado Metal** en Apple Silicon.

## Rendimiento

| Resolución | Velocidad aprox. (M4 Pro) |
|-----------|----------------------------|
| 336 px    | ~0.5 s / frame            |
| 504 px    | ~3 s / frame              |

Para clips largos, bajá la resolución y el FPS de salida.

## Limitaciones

Es depth **relativo, no métrico**: la escala es arbitraria por frame, por lo que puede
haber *flicker* / deriva entre frames (mitigado con normalización global, no eliminado).

## Estructura

```
app.py              backend FastAPI (jobs + progreso + inferencia)
pose.py             estimación de pose (MediaPipe Holistic)
static/index.html   interfaz web
run.sh              arranca el servidor
setup.sh            instala todo desde cero
requirements.txt    dependencias (sin torch ni xformers)
```

## Créditos

Modelo y motor de inferencia: [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
(ByteDance Seed, Apache-2.0). Este repo es solo el envoltorio web local.
