# UltraFaceswap - Face swap API with GPU support
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python deps (CPU-only for smaller image; use onnxruntime-gpu for GPU)
RUN pip install --no-cache-dir \
    torch torchvision \
    numpy opencv-python Pillow \
    insightface onnxruntime-gpu \
    fastapi uvicorn python-multipart \
    tqdm

COPY backend/ backend/
COPY run_api.py .
COPY scripts/ scripts/

# Create models dir
RUN mkdir -p /app/models
ENV ULTRAFACESWAP_MODELS=/app/models
ENV ULTRAFACESWAP_OUTPUT=/tmp/output

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
