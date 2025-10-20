# Dockerfile for PaddleOCR-VL on Northflank H100 GPU
# Uses PaddlePaddle framework as intended by the model creators
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create python3 symlink
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PaddlePaddle GPU version (CUDA 12.6)
RUN python3 -m pip install --no-cache-dir paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Install safetensors for PaddlePaddle
RUN python3 -m pip install --no-cache-dir https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl || \
    python3 -m pip install --no-cache-dir safetensors

# Install PaddleOCR with doc-parser support (this is the key!)
RUN python3 -m pip install --no-cache-dir "paddleocr[doc-parser]"

# Install API dependencies
RUN python3 -m pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    python-Levenshtein \
    pandas \
    numpy \
    requests

# Set working directory
WORKDIR /app

# Copy application files
COPY app_paddleocr.py /app/app.py
COPY healthcheck.py /app/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# Run the application
CMD ["python3", "-u", "app.py"]
