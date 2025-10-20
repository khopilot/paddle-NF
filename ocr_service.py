#!/usr/bin/env python3
"""
OCR Service Module for batch processing PDFs on Northflank GPU
"""

import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import fitz  # PyMuPDF
from pathlib import Path
import time
from typing import List, Dict
from tqdm import tqdm


class NorthflankOCRService:
    """GPU-accelerated OCR service for Northflank H100"""

    def __init__(self, model_path: str = "/app/model/PaddleOCR-VL-0.9B"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None

    def load_model(self):
        """Load model on GPU"""
        print(f"Loading PaddleOCR-VL model on {self.device.upper()}...")
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device.upper()}")

    def extract_from_image(self, image: Image.Image, max_tokens: int = 512) -> Dict:
        """Extract text from single image"""
        # Resize if too large
        if max(image.size) > 1200:
            ratio = 1200 / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        prompt = "<|IMAGE_PLACEHOLDER|>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1
            )
        gen_time = time.time() - start

        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        text = text.replace(prompt, "").strip()

        return {
            "text": text,
            "time": gen_time,
            "tokens": len(outputs[0]) - len(inputs['input_ids'][0])
        }

    def process_pdf(
        self,
        pdf_path: str,
        start_page: int = 0,
        num_pages: int = None,
        dpi: int = 150
    ) -> List[Dict]:
        """Process PDF and extract text from all pages"""
        pdf_document = fitz.open(pdf_path)
        total_pages = pdf_document.page_count

        end_page = start_page + num_pages if num_pages else total_pages
        end_page = min(end_page, total_pages)

        results = []

        print(f"Processing {end_page - start_page} pages...")
        for page_num in tqdm(range(start_page, end_page), desc="Pages"):
            page = pdf_document[page_num]
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            # Extract text
            result = self.extract_from_image(image)
            result['page_num'] = page_num + 1

            results.append(result)

            # Free GPU memory periodically
            if (page_num - start_page + 1) % 10 == 0:
                torch.cuda.empty_cache()

        pdf_document.close()
        return results


# For standalone testing
if __name__ == "__main__":
    import io
    service = NorthflankOCRService()
    service.load_model()
    print("Service ready!")
