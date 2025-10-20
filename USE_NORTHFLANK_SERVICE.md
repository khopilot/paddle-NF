# 🚀 Using Your Northflank H100 GPU OCR Service

## 📍 Your Service Details

**Service Name:** ocr-v1
**Public URL:** `https://p01--ocr-v1--jzknhfqkxn84.code.run`
**Port:** 8080 (HTTP)
**GitHub:** https://github.com/khopilot/paddle-NF
**GPU:** NVIDIA H100, 80GB VRAM
**Cost:** $2.74/hour

## ⏳ Deployment Status

Your Northflank service is currently **building**.

**Monitor in Northflank dashboard:**
- Go to service `ocr-v1`
- Click "Logs" to see build progress
- Wait for: `✓ Model loaded on CUDA`
- Service status will show **Healthy** when ready

**Build stages (~15-20 min total):**
1. ✅ Install Python 3.11 and system packages
2. ⏳ Install PyTorch with CUDA support
3. ⏳ Install transformers and dependencies
4. ⏳ Download PaddleOCR-VL model (~2 GB from HuggingFace)
5. ⏳ Start service and load model on GPU

## ✅ Once Deployed - Test Commands

### 1. Health Check

```bash
curl https://p01--ocr-v1--jzknhfqkxn84.code.run/health
```

**Expected:**
```json
{"status": "healthy", "device": "cuda"}
```

### 2. Check GPU Status

```bash
curl https://p01--ocr-v1--jzknhfqkxn84.code.run/status
```

**Expected:**
```json
{
  "model_loaded": true,
  "device": "cuda",
  "gpu_info": {
    "gpu_name": "NVIDIA H100 80GB HBM3",
    "total_memory_gb": 80.0,
    "allocated_memory_gb": 3.8
  }
}
```

### 3. Test Single Image OCR

```bash
curl -X POST https://p01--ocr-v1--jzknhfqkxn84.code.run/ocr/extract \
  -F "file=@page_1_test/temp_page_0.png" \
  -F "max_tokens=512"
```

**Expected response:**
```json
{
  "extracted_text": "សៀមរាប ៖ ត្រីងៀតស្ងួត...",
  "processing_time": 2.3,
  "image_size": {
    "original": [1700, 2200],
    "processed": [927, 1200]
  },
  "tokens_generated": 256,
  "device": "cuda"
}
```

### 4. Process Your Full PDF (9,249 pages)

**First, test with 10 pages:**
```bash
python3 client_test_northflank.py \
  --service-url "https://p01--ocr-v1--jzknhfqkxn84.code.run" \
  --pdf ams_khmer_os_battambang.pdf \
  --ground-truth ams_khmer_os_battambang.txt \
  --num-pages 10 \
  --output-dir test_10_pages
```

**If successful, run full test:**
```bash
python3 client_test_northflank.py \
  --service-url "https://p01--ocr-v1--jzknhfqkxn84.code.run" \
  --pdf ams_khmer_os_battambang.pdf \
  --ground-truth ams_khmer_os_battambang.txt \
  --output-dir northflank_full_results
```

**Expected:**
- **Time:** 2-4 hours for all 9,249 pages
- **Cost:** ~$11 total
- **Speed:** 1-5 seconds per page
- **Output:** Quality metrics (CER, WER, Accuracy, BLEU)

## 📊 Monitor Progress

The client script shows real-time progress:
```
Processing pages with H100 acceleration...
Extracting text: 100%|████| 10/10 [00:23<00:00, 2.3s/it]

Page 1: 2.1s, Accuracy: 0.945
Page 2: 2.3s, Accuracy: 0.932
...
Processed 10 pages. Avg: 2.2s/page. ETA: 5.6 hours
```

## 💰 Cost Tracking

| Pages | Time (est) | Cost |
|-------|-----------|------|
| 10 pages | ~30 sec | $0.02 |
| 100 pages | ~5 min | $0.23 |
| 1,000 pages | ~50 min | $2.28 |
| 9,249 pages | ~4 hours | $10.96 |

**Remember to STOP the service when done!**

## 🎯 After Processing Completes

Your results will be in `northflank_full_results/`:

```bash
# View summary
cat northflank_full_results/results_detailed.json

# Open in Excel
open northflank_full_results/results.csv

# View all extracted text
cat northflank_full_results/extracted_text_all_pages.txt
```

## 🔧 Troubleshooting

### Service Not Responding
- Check Northflank dashboard for service status
- Look at logs for errors
- Wait 2-3 minutes after "Healthy" status (model loads after container starts)

### Slow Performance
- Should be 1-5 seconds per page on H100
- If slower, check `/status` to verify GPU is being used
- Check Northflank logs for memory warnings

### Errors
- Check Northflank logs
- Verify model downloaded correctly
- Rebuild if needed

## ⚠️ IMPORTANT - Stop Service When Done

To avoid ongoing charges:
1. Go to Northflank dashboard
2. Select service `ocr-v1`
3. Click "Stop" or "Delete"
4. Confirm

## 📋 Quick Command Reference

```bash
# Set URL variable for easy copy-paste
export SERVICE_URL="https://p01--ocr-v1--jzknhfqkxn84.code.run"

# Health check
curl $SERVICE_URL/health

# Status (includes GPU info)
curl $SERVICE_URL/status

# Test 1 image
curl -X POST $SERVICE_URL/ocr/extract -F "file=@your_image.png"

# Test 10 pages
python3 client_test_northflank.py \
  --service-url "$SERVICE_URL" \
  --pdf ams_khmer_os_battambang.pdf \
  --ground-truth ams_khmer_os_battambang.txt \
  --num-pages 10

# Full test (9,249 pages)
python3 client_test_northflank.py \
  --service-url "$SERVICE_URL" \
  --pdf ams_khmer_os_battambang.pdf \
  --ground-truth ams_khmer_os_battambang.txt
```

---

**Status:** Northflank building... ⏳
**Next:** Wait for deployment, then test!
**Your URL:** https://p01--ocr-v1--jzknhfqkxn84.code.run 🎯
