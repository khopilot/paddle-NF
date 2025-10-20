# Optimized Dockerfile for PaddleOCR-VL on Northflank H100 GPU
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip3.11 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all Python dependencies in one layer
RUN pip3.11 install --no-cache-dir \
    transformers>=4.44 \
    accelerate \
    einops \
    sentencepiece \
    protobuf \
    pillow \
    pymupdf \
    python-multipart \
    uvicorn[standard] \
    fastapi \
    pydantic \
    python-Levenshtein \
    pandas \
    numpy \
    tqdm \
    requests \
    huggingface-hub[hf_transfer]

# Set working directory
WORKDIR /app

# Download model using huggingface-hub (more reliable than git clone)
RUN python3.11 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('PaddlePaddle/PaddleOCR-VL', local_dir='/app/model', local_dir_use_symlinks=False)"

# Copy application files
COPY app.py /app/
COPY ocr_service.py /app/
COPY healthcheck.py /app/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python3.11 -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# Run the application
CMD ["python3.11", "-u", "app.py"]
