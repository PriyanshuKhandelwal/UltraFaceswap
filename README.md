# UltraFaceswap

Ultra-realistic face swap software: swap a face from a photo onto a video using open-source AI models.

## Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg (`brew install ffmpeg` on macOS)
- GPU recommended (CUDA) for faster processing

### Install

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python scripts/download_models.py  # ~550MB for InSwapper
```

### CLI

```bash
python swap.py --source photo.jpg --target video.mp4 --output output.mp4
```

Options:
- `--enhance` – GFPGAN face restoration
- `--swap-model inswapper|simswap` – InSwapper (faster) or SimSwap (sharper)
- `--det-size 320|640` – Face detection size (640 for HD)
- `--upscale 1|2|4` – Output upscaling factor

### API + Web App

```bash
# Terminal 1: Start API
python run_api.py

# Terminal 2: Start frontend (dev)
cd frontend && npm install && npm run dev
```

Open http://localhost:3000. The frontend proxies API requests to port 8000.

### Docker

```bash
# CPU only (no GPU)
docker build -f Dockerfile.cpu -t ultrafaceswap .
docker run -p 8000:8000 -v $(pwd)/models:/app/models ultrafaceswap

# GPU (requires nvidia-docker)
docker-compose up --build
```

## Pipeline

1. **Extract** – FFmpeg extracts frames
2. **Detect** – InsightFace finds faces (configurable 320/640)
3. **Swap** – InSwapper 128 or SimSwap 256
4. **Restore** (optional) – GFPGAN enhancement
5. **Upscale** (optional) – Real-ESRGAN 2× or 4×
6. **Merge** – FFmpeg reassembles video + audio

## Project Structure

```
UltraFaceswap/
├── backend/api/      # FastAPI routes
├── backend/core/     # extractor, face_swap, enhancer, merger
├── backend/queue/    # Job queue
├── frontend/         # React + Vite
├── scripts/          # download_models
├── swap.py           # CLI
├── run_api.py        # API server
├── Dockerfile
└── docker-compose.yml
```

## License

Uses InsightFace, InSwapper. Check model licenses for commercial use.
