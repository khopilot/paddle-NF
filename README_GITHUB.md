# PaddleOCR-VL Northflank H100 GPU Deployment

Deploy PaddleOCR-VL on Northflank's NVIDIA H100 GPU for fast OCR processing.

## Files

- `Dockerfile` - CUDA-enabled container with PaddlePaddle GPU
- `app_paddleocr.py` - FastAPI service using official PaddleOCR API
- `healthcheck.py` - Health check script
- `client_test_northflank.py` - Client to process PDFs via deployed service

## Deploy on Northflank

**Configuration:**
- GPU: NVIDIA H100, 80GB VRAM
- Region: Asia-Southeast
- Port: 8080 (HTTP)
- Health Check: `/health`, 180s initial delay

**Steps:**
1. Fork/clone this repo
2. Create Northflank service
3. Connect this GitHub repository
4. Add port 8080 and health check
5. Deploy!

## Usage

Once deployed:

```bash
# Health check
curl https://your-service.nf.run/health

# Extract text from image
curl -X POST https://your-service.nf.run/ocr/extract \
  -F "file=@image.png"
```

## License

Apache 2.0 (same as PaddleOCR-VL)
