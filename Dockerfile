# Dockerfile for PaddleOCR-VL on Northflank H100 GPU
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    git-lfs \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip3.11 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and dependencies
RUN pip3.11 install \
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
    tqdm

# Set working directory
WORKDIR /app

# Initialize git-lfs
RUN git lfs install

# Clone PaddleOCR-VL model from HuggingFace
# Using GIT_LFS_SKIP_SMUDGE first to avoid downloading large files during build
ENV GIT_LFS_SKIP_SMUDGE=1
RUN git clone https://huggingface.co/PaddlePaddle/PaddleOCR-VL /app/model

# Download the actual model files using huggingface-cli
RUN pip3.11 install huggingface-hub[cli] && \
    cd /app/model && \
    unset GIT_LFS_SKIP_SMUDGE && \
    git lfs pull

# Copy application files
COPY app.py /app/
COPY ocr_service.py /app/
COPY healthcheck.py /app/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3.11 healthcheck.py || exit 1

# Run the application
CMD ["python3.11", "app.py"]
