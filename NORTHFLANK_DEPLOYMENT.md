# Northflank H100 GPU Deployment Guide

## Overview

Deploy PaddleOCR-VL on Northflank's NVIDIA H100 GPU for fast OCR testing (50-100x faster than CPU).

**Estimated Performance:**
- **CPU (M4 Pro):** 3+ hours per page
- **H100 GPU:** 1-5 seconds per page
- **9,249 pages:** 2-4 hours on GPU vs 3+ years on CPU!

**Cost:** $2.74/hour × 4 hours = ~$11 for complete test

## Files Created for Deployment

✅ **Dockerfile** - GPU-enabled container with model cloning
✅ **app.py** - FastAPI OCR service with REST API
✅ **ocr_service.py** - OCR processing module
✅ **healthcheck.py** - Health check script
✅ **.dockerignore** - Exclude unnecessary files

## Your Northflank Configuration (Perfect!)

```
Service name: ocr-V1
Region: Asia - Southeast
GPU: NVIDIA H100, 80GB VRAM
Price: $2.74/hour
Compute: 26 vCPU, 234 GB RAM
Storage: 500 GB ephemeral, 234 GB SHM
Instances: 1
```

## What You Need to Add in Northflank

### 1. Port Configuration
**Add this port:**
- **Port:** 8080
- **Protocol:** HTTP
- **Public:** Yes (or Private if you prefer)

### 2. Health Check
**Add HTTP health check:**
- **Path:** `/health`
- **Port:** 8080
- **Initial delay:** 120 seconds (model needs time to load)
- **Interval:** 30 seconds
- **Timeout:** 10 seconds

### 3. Environment Variables (Optional)
You can add these if needed:
- `PYTORCH_CUDA_ALLOC_CONF`: `max_split_size_mb:512`
- `TRANSFORMERS_CACHE`: `/tmp/transformers_cache`

## Deployment Steps

### Option A: Deploy from Git Repository (Recommended)

**Step 1: Create Git Repository**
```bash
cd /Users/nicolas.consultant/Downloads/PaddleOCR-VL

# Initialize git if not already
git init
git add Dockerfile app.py ocr_service.py healthcheck.py .dockerignore
git commit -m "Add Northflank deployment files"

# Push to GitHub/GitLab
git remote add origin <your-repo-url>
git push -u origin main
```

**Step 2: Configure Northflank**
1. In Northflank, select "Combined - Build and deploy a Git repo"
2. Connect your Git repository
3. Set build context: `/`
4. Dockerfile path: `Dockerfile`
5. Add port 8080
6. Add health check `/health`
7. Click "Create service"

**Step 3: Wait for Build & Deployment**
- Build time: ~10-15 minutes (downloads model from HuggingFace)
- Model loading: ~2-3 minutes
- Total startup: ~15-20 minutes

**Step 4: Test the Service**
```bash
# Get your Northflank service URL (e.g., https://ocr-v1-xxxxx.nf.run)
SERVICE_URL="https://your-service-url"

# Check health
curl $SERVICE_URL/health

# Test OCR on an image
curl -X POST $SERVICE_URL/ocr/extract \
  -F "file=@page_1_test/temp_page_0.png" \
  -F "max_tokens=512"
```

### Option B: Deploy from External Docker Image

**Step 1: Build Locally (Optional)**
```bash
cd /Users/nicolas.consultant/Downloads/PaddleOCR-VL

# Build Docker image
docker build -t paddleocr-vl:latest .

# Test locally (if you have Docker with GPU support)
docker run --gpus all -p 8080:8080 paddleocr-vl:latest
```

**Step 2: Push to Registry**
```bash
# Tag for Docker Hub
docker tag paddleocr-vl:latest your-username/paddleocr-vl:latest

# Push
docker push your-username/paddleocr-vl:latest
```

**Step 3: Deploy on Northflank**
1. Select "External image"
2. Image: `your-username/paddleocr-vl:latest`
3. Configure GPU, ports, health checks as above
4. Deploy!

## API Endpoints

Once deployed, your service will have these endpoints:

### 1. Root - Service Info
```bash
GET /
```
Response:
```json
{
  "service": "PaddleOCR-VL OCR Service",
  "version": "1.0.0",
  "device": "cuda",
  "status": "ready"
}
```

### 2. Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "device": "cuda"
}
```

### 3. Extract Text from Image
```bash
POST /ocr/extract
Content-Type: multipart/form-data
```

Parameters:
- `file`: Image file (required)
- `max_tokens`: Max tokens to generate (default: 512)
- `resize_max`: Max image dimension (default: 1200)

Example:
```bash
curl -X POST https://your-service.nf.run/ocr/extract \
  -F "file=@image.png" \
  -F "max_tokens=512"
```

Response:
```json
{
  "extracted_text": "សៀមរាប ៖ ត្រីងៀតស្ងួត...",
  "processing_time": 2.34,
  "image_size": {
    "original": [1700, 2200],
    "processed": [927, 1200]
  },
  "tokens_generated": 256,
  "device": "cuda"
}
```

### 4. Batch Processing
```bash
POST /ocr/batch
```

Process multiple images at once.

### 5. Status & GPU Info
```bash
GET /status
```

Returns GPU memory usage and model status.

## Processing Your 9,249-Page PDF

### Method 1: Manual Page-by-Page (Simplest)

```python
import requests
import fitz  # PyMuPDF
from pathlib import Path

SERVICE_URL = "https://your-service.nf.run"
PDF_PATH = "ams_khmer_os_battambang.pdf"
OUTPUT_DIR = Path("ocr_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Convert PDF to images and process
pdf = fitz.open(PDF_PATH)
results = []

for page_num in range(pdf.page_count):
    # Convert page to image
    page = pdf[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
    img_path = OUTPUT_DIR / f"page_{page_num}.png"
    pix.save(str(img_path))

    # Send to OCR service
    with open(img_path, 'rb') as f:
        response = requests.post(
            f"{SERVICE_URL}/ocr/extract",
            files={"file": f},
            data={"max_tokens": 512}
        )

    result = response.json()
    result['page_num'] = page_num + 1
    results.append(result)

    # Save progress
    if (page_num + 1) % 100 == 0:
        print(f"Processed {page_num + 1} pages...")

pdf.close()

# Save all results
import json
with open(OUTPUT_DIR / "all_results.json", 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✓ Completed all {len(results)} pages!")
```

### Method 2: Upload PDF and Process on Server (Advanced)

Would require adding a `/ocr/process_pdf` endpoint that handles the PDF server-side.

## Expected Performance on H100

Based on similar VLM deployments:
- **Model loading:** 20-30 seconds (one time)
- **Per page:** 1-5 seconds (depending on text length)
- **9,249 pages:**
  - Best case: 2.5 hours (1s/page)
  - Average case: 3-4 hours (1.5-2s/page)
  - Worst case: 13 hours (5s/page)

**Much better than 3+ years on CPU!**

## Cost Estimate

**Scenario 1: 4 hours total**
- 4 hours × $2.74/hr = **$10.96**

**Scenario 2: 8 hours (conservative)**
- 8 hours × $2.74/hr = **$21.92**

**Scenario 3: Add development/testing time**
- Testing: 1 hour
- Full run: 4 hours
- Buffer: 1 hour
- Total: 6 hours × $2.74/hr = **$16.44**

## Monitoring Your Deployment

Once deployed, you can monitor via:

1. **Northflank Dashboard:**
   - View logs in real-time
   - Monitor GPU usage
   - Check memory consumption

2. **API Status Endpoint:**
   ```bash
   curl https://your-service.nf.run/status
   ```

3. **Logs:**
   ```bash
   # Via Northflank CLI
   northflank logs service ocr-v1
   ```

## Troubleshooting

### Build Fails
- Check Dockerfile syntax
- Ensure all files (app.py, ocr_service.py, healthcheck.py) are committed
- Check Northflank build logs

### Model Download Fails
- Git LFS timeout: Increase timeout in Northflank settings
- Network issues: Try building again

### GPU Out of Memory
- Reduce `resize_max` to 800 or 1000
- Reduce `max_tokens` to 256 or 128
- Process in smaller batches

### Slow Performance
- Verify GPU is being used: check `/status` endpoint
- Check GPU utilization in Northflank dashboard
- Ensure FP16 is enabled (it is in the code)

## Next Steps

1. ✅ **Files created** - All deployment files ready
2. **Create Git repo** - Push files to GitHub/GitLab
3. **Deploy on Northflank** - Use your configuration settings
4. **Test with 1 image** - Verify it works
5. **Process all 9,249 pages** - Run the full test
6. **Download results** - Get OCR output
7. **Calculate quality metrics** - Compare with ground truth locally

## Alternative: Pre-built Image

If you don't want to build on Northflank, I can provide a Docker Hub image link (requires local Docker build first).

## Files Summary

| File | Purpose |
|------|---------|
| Dockerfile | Container definition with CUDA, model cloning |
| app.py | FastAPI web service with OCR endpoints |
| ocr_service.py | OCR processing logic |
| healthcheck.py | Health check for Northflank |
| .dockerignore | Exclude local test files from build |

**Ready to deploy!** 🚀

Just push these files to a Git repo and connect it to Northflank with your config settings.
