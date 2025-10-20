#!/usr/bin/env python3
"""
FastAPI service for PaddleOCR-VL using the OFFICIAL PaddleOCR API
This is the correct way to use PaddleOCR-VL!
"""

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import time
from pathlib import Path
import json

app = FastAPI(
    title="PaddleOCR-VL Service",
    description="Official PaddleOCR-VL API using PaddlePaddle",
    version="1.0.0"
)

# Global pipeline
pipeline = None

@app.on_event("startup")
async def load_model():
    """Load PaddleOCR-VL pipeline on startup"""
    global pipeline

    print("="*80)
    print("LOADING PADDLEOCR-VL PIPELINE...")
    print("="*80)

    try:
        from paddleocr import PaddleOCRVL
        import paddle

        # Check if GPU is available
        if paddle.is_compiled_with_cuda():
            print(f"✓ Using NVIDIA GPU with CUDA")
            print(f"✓ GPU Count: {paddle.device.cuda.device_count()}")
            for i in range(paddle.device.cuda.device_count()):
                props = paddle.device.cuda.get_device_properties(i)
                print(f"✓ GPU {i}: {props.name}")
                print(f"✓ Total Memory: {props.total_memory / 1e9:.2f} GB")
        else:
            print("⚠ Running on CPU")

        # Initialize pipeline (will auto-download model if needed)
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
    Extract text from uploaded image using PaddleOCR-VL

    Returns:
    - extracted_text: OCR text output
    - processing_time: Time taken
    - format: Output format (markdown/json)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{file.filename}")
        contents = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(contents)

        # Process with PaddleOCR-VL
        start = time.time()
        output = pipeline.predict(str(temp_path))
        proc_time = time.time() - start

        # DEBUG: Save output structure to temp file for inspection
        import tempfile
        debug_dir = Path("/tmp/debug_output")
        debug_dir.mkdir(exist_ok=True)

        # Try to use built-in save methods
        text_parts = []
        json_results = []

        for idx, res in enumerate(output):
            try:
                # Save to JSON using built-in method
                json_save_path = str(debug_dir / f"result_{idx}")
                res.save_to_json(save_path=json_save_path)

                # Also try markdown
                md_save_path = str(debug_dir / f"result_{idx}")
                res.save_to_markdown(save_path=md_save_path)

                # Read the saved JSON
                json_file = Path(json_save_path) / f"{temp_path.stem}.json"
                if json_file.exists():
                    with open(json_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        json_results.append(result_data)

                # Read the saved markdown
                md_file = Path(md_save_path) / f"{temp_path.stem}.md"
                if md_file.exists():
                    with open(md_file, 'r', encoding='utf-8') as f:
                        md_text = f.read()
                        text_parts.append(md_text)

            except Exception as e:
                # Fallback: try to get any string representation
                text_parts.append(f"Error extracting result {idx}: {str(e)}\nResult type: {type(res)}\nResult: {str(res)[:500]}")

        extracted_text = '\n'.join(text_parts) if text_parts else "No text extracted"

        # Clean up temp file
        temp_path.unlink()

        return {
            "extracted_text": extracted_text,
            "processing_time": proc_time,
            "backend": "PaddlePaddle",
            "results_count": len(output),
            "detailed_results": json_results
        }

    except Exception as e:
        # Clean up on error
        if temp_path.exists():
            temp_path.unlink()
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
