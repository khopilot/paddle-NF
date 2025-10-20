#!/usr/bin/env python3
"""
FastAPI service for PaddleOCR-VL - FINAL CORRECT VERSION
Uses official PaddleOCR API exactly as documented
"""

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
import time
import json
import os

app = FastAPI(
    title="PaddleOCR-VL Service",
    description="Official PaddleOCR-VL with PaddlePaddle",
    version="1.0.0"
)

pipeline = None

@app.on_event("startup")
async def load_model():
    """Load PaddleOCR-VL pipeline"""
    global pipeline

    print("="*80)
    print("LOADING PADDLEOCR-VL PIPELINE...")
    print("="*80)

    try:
        from paddleocr import PaddleOCRVL
        import paddle

        # Check GPU
        if paddle.is_compiled_with_cuda():
            print(f"✓ Using NVIDIA GPU with CUDA")
            print(f"✓ GPU Count: {paddle.device.cuda.device_count()}")
            for i in range(paddle.device.cuda.device_count()):
                props = paddle.device.cuda.get_device_properties(i)
                print(f"✓ GPU {i}: {props.name}")
                print(f"✓ Total Memory: {props.total_memory / 1e9:.2f} GB")
        else:
            print("⚠ Running on CPU")

        # Initialize pipeline
        start = time.time()
        pipeline = PaddleOCRVL()
        load_time = time.time() - start

        print(f"✓ Pipeline loaded in {load_time:.2f}s")
        print("="*80)

    except Exception as e:
        print(f"✗ Failed to load pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "PaddleOCR-VL Service",
        "version": "1.0.0",
        "backend": "PaddlePaddle",
        "status": "ready" if pipeline is not None else "loading"
    }


@app.get("/health")
async def health():
    """Health check"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return {"status": "healthy", "backend": "PaddlePaddle"}


@app.post("/ocr/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from image using PaddleOCR-VL

    The official API uses res.save_to_json() and res.save_to_markdown()
    which save files to a directory
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    temp_image_path = None
    output_dir = Path("/tmp/paddleocr_output")
    output_dir.mkdir(exist_ok=True)

    try:
        # Save uploaded image
        temp_image_path = Path(f"/tmp/{file.filename}")
        contents = await file.read()
        with open(temp_image_path, 'wb') as f:
            f.write(contents)

        # Process with PaddleOCR-VL using official API
        start = time.time()
        output = pipeline.predict(str(temp_image_path))
        proc_time = time.time() - start

        # Extract results using official methods
        all_text = []
        all_json = []

        for idx, res in enumerate(output):
            # Save to JSON (creates files in the directory)
            json_dir = output_dir / f"json_{idx}"
            json_dir.mkdir(exist_ok=True)
            res.save_to_json(save_path=str(json_dir))

            # Save to Markdown
            md_dir = output_dir / f"md_{idx}"
            md_dir.mkdir(exist_ok=True)
            res.save_to_markdown(save_path=str(md_dir))

            # Read the generated JSON file
            # The filename is based on the input image name (without extension)
            base_name = temp_image_path.stem
            json_file = json_dir / f"{base_name}.json"

            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_json.append(data)

                    # Extract text from JSON structure
                    # The JSON contains the parsed document structure
                    if isinstance(data, dict):
                        # Try to get text content
                        if 'content' in data:
                            all_text.append(data['content'])
                        elif 'text' in data:
                            all_text.append(data['text'])
                        else:
                            # Fallback: convert entire JSON to text
                            all_text.append(json.dumps(data, ensure_ascii=False, indent=2))

            # Read the generated markdown file
            md_file = md_dir / f"{base_name}.md"
            if md_file.exists() and not json_file.exists():
                with open(md_file, 'r', encoding='utf-8') as f:
                    all_text.append(f.read())

        extracted_text = '\n\n'.join(all_text) if all_text else "No text extracted"

        # Cleanup
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink()

        return {
            "extracted_text": extracted_text,
            "processing_time": proc_time,
            "backend": "PaddlePaddle",
            "results_count": len(output),
            "detailed_results": all_json
        }

    except Exception as e:
        # Cleanup on error
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink()
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


@app.get("/status")
async def status():
    """Get service status"""
    import paddle

    gpu_info = {}
    if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        props = paddle.device.cuda.get_device_properties(0)
        gpu_info = {
            "gpu_count": paddle.device.cuda.device_count(),
            "gpu_name": props.name,
            "total_memory_gb": props.total_memory / 1e9
        }

    return {
        "pipeline_loaded": pipeline is not None,
        "backend": "PaddlePaddle",
        "cuda_available": paddle.is_compiled_with_cuda(),
        "gpu_info": gpu_info
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
