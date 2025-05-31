# Face Detection Service with CUDA Support
# Base image with CUDA 12.9 and cuDNN
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY docker/requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install ONNX Runtime GPU (specific version for CUDA 12.9)
RUN python3 -m pip install --no-cache-dir \
    onnxruntime-gpu==1.16.3 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install additional GPU-accelerated packages
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY app/ ./app/
COPY models/ ./models/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/temp

# Set permissions
RUN chmod +x scripts/*.py

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Environment variables for the application
ENV MODEL_PATH=/app/models
ENV CONFIG_PATH=/app/config
ENV LOG_PATH=/app/logs
ENV CACHE_PATH=/app/cache
ENV TEMP_PATH=/app/temp

# GPU Memory Settings (6GB limit)
ENV CUDA_MEMORY_FRACTION=0.8
ENV VRAM_LIMIT_MB=4800
ENV MAX_BATCH_SIZE=4
ENV MODEL_CACHE_SIZE=2

# ONNX Runtime settings
ENV ORT_DISABLE_ALL_OPTIMIZATIONS=0
ENV ORT_ENABLE_BASIC_OPTIMIZATIONS=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]