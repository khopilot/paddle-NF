#!/usr/bin/env python3
"""
FastAPI service for PaddleOCR-VL on Northflank H100 GPU
Provides REST API for OCR extraction and quality testing
"""

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import io
import time
import base64

# Initialize FastAPI
app = FastAPI(
    title="PaddleOCR-VL OCR Service",
    description="GPU-accelerated OCR service using PaddleOCR-VL-0.9B",
    version="1.0.0"
)

# Global model and processor
model = None
processor = None
device = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, processor, device

    print("="*80)
    print("LOADING PADDLEOCR-VL MODEL...")
    print("="*80)

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("⚠ Using CPU (should not happen on Northflank!)")

    # Load model
    model_path = "/app/model/PaddleOCR-VL-0.9B"
    print(f"\nLoading from: {model_path}")

    start = time.time()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16  # FP16 for speed
    )
    model = model.to(device)
    model.eval()

    load_time = time.time() - start
    print(f"✓ Model loaded in {load_time:.2f}s")
    print("="*80)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "PaddleOCR-VL OCR Service",
        "version": "1.0.0",
        "device": device,
        "status": "ready" if model is not None else "loading"
    }


@app.get("/health")
async def health():
    """Health check endpoint for Northflank"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "device": device}


@app.post("/ocr/extract")
async def extract_text(
    file: UploadFile = File(...),
    max_tokens: int = 512,
    resize_max: int = 1200
):
    """
    Extract text from uploaded image

    Parameters:
    - file: Image file (PNG, JPG, etc.)
    - max_tokens: Maximum tokens to generate (default: 512)
    - resize_max: Maximum image dimension (default: 1200)

    Returns:
    - extracted_text: The OCR extracted text
    - processing_time: Time taken in seconds
    - image_size: Original and processed image size
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        orig_size = image.size

        # Resize if needed to prevent GPU OOM
        if max(image.size) > resize_max:
            ratio = resize_max / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Process
        prompt = "<|IMAGE_PLACEHOLDER|>"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1
            )
        gen_time = time.time() - start

        # Decode
        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        text = text.replace(prompt, "").strip()

        return {
            "extracted_text": text,
            "processing_time": gen_time,
            "image_size": {
                "original": orig_size,
                "processed": image.size
            },
            "tokens_generated": len(outputs[0]) - len(inputs['input_ids'][0]),
            "device": device
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/ocr/batch")
async def batch_extract(
    files: list[UploadFile] = File(...),
    max_tokens: int = 512
):
    """
    Process multiple images in batch

    Parameters:
    - files: List of image files
    - max_tokens: Maximum tokens per image

    Returns:
    - results: List of OCR results
    - total_time: Total processing time
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    total_start = time.time()

    for i, file in enumerate(files):
        try:
            result = await extract_text(file, max_tokens=max_tokens)
            result['file_index'] = i
            result['filename'] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                "file_index": i,
                "filename": file.filename,
                "error": str(e)
            })

    total_time = time.time() - total_start

    return {
        "results": results,
        "total_files": len(files),
        "total_time": total_time,
        "avg_time_per_file": total_time / len(files) if files else 0
    }


@app.get("/status")
async def status():
    """Get service status and GPU info"""
    gpu_info = {}
    if device == "cuda":
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "allocated_memory_gb": torch.cuda.memory_allocated(0) / 1e9,
            "cached_memory_gb": torch.cuda.memory_reserved(0) / 1e9
        }

    return {
        "model_loaded": model is not None,
        "device": device,
        "gpu_info": gpu_info,
        "model_dtype": str(model.dtype) if model else None
    }


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
