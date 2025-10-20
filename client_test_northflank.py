#!/usr/bin/env python3
"""
Client script to process PDF using deployed Northflank H100 GPU service
This runs on your local machine and sends images to the cloud service
"""

import requests
import fitz  # PyMuPDF
from pathlib import Path
import json
import time
from tqdm import tqdm
import Levenshtein
import pandas as pd
import argparse


class NorthflankOCRClient:
    """Client for Northflank OCR service"""

    def __init__(self, service_url: str, ground_truth_path: str = None):
        self.service_url = service_url.rstrip('/')
        self.ground_truth_path = ground_truth_path
        self.ground_truth_lines = []

    def check_service_health(self):
        """Check if service is healthy"""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=10)
            if response.status_code == 200:
                print(f"✓ Service is healthy: {response.json()}")
                return True
            else:
                print(f"✗ Service unhealthy: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot reach service: {e}")
            return False

    def get_service_status(self):
        """Get detailed service status"""
        try:
            response = requests.get(f"{self.service_url}/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                print("\n" + "="*80)
                print("SERVICE STATUS")
                print("="*80)
                print(f"Model Loaded: {status.get('model_loaded')}")
                print(f"Device: {status.get('device')}")
                if 'gpu_info' in status and status['gpu_info']:
                    gpu = status['gpu_info']
                    print(f"\nGPU Information:")
                    print(f"  Name: {gpu.get('gpu_name')}")
                    print(f"  Total VRAM: {gpu.get('total_memory_gb'):.2f} GB")
                    print(f"  Allocated: {gpu.get('allocated_memory_gb'):.2f} GB")
                    print(f"  Cached: {gpu.get('cached_memory_gb'):.2f} GB")
                print("="*80)
                return status
        except Exception as e:
            print(f"Error getting status: {e}")
            return None

    def load_ground_truth(self):
        """Load ground truth if provided"""
        if self.ground_truth_path and Path(self.ground_truth_path).exists():
            with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
                self.ground_truth_lines = f.readlines()
            print(f"✓ Loaded {len(self.ground_truth_lines)} lines of ground truth")

    def extract_from_image_file(self, image_path: str, max_tokens: int = 512) -> dict:
        """Send image to service for OCR"""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'max_tokens': max_tokens}

            response = requests.post(
                f"{self.service_url}/ocr/extract",
                files=files,
                data=data,
                timeout=120
            )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"OCR failed: {response.status_code} - {response.text}")

    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str = "northflank_ocr_results",
        start_page: int = 0,
        num_pages: int = None,
        dpi: int = 150,
        max_tokens: int = 512
    ):
        """Process entire PDF using Northflank service"""

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print("\n" + "="*80)
        print("PROCESSING PDF WITH NORTHFLANK H100 GPU SERVICE")
        print("="*80)

        # Load ground truth
        self.load_ground_truth()

        # Open PDF
        print(f"\nOpening PDF: {pdf_path}")
        pdf = fitz.open(pdf_path)
        total_pages = pdf.page_count

        end_page = start_page + num_pages if num_pages else total_pages
        end_page = min(end_page, total_pages)

        print(f"Processing pages {start_page} to {end_page-1} (total: {end_page - start_page})")

        # Process each page
        results = []
        all_text = []

        for page_num in tqdm(range(start_page, end_page), desc="Processing pages"):
            try:
                # Convert page to image
                page = pdf[page_num]
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)

                # Save temp image
                temp_img = output_dir / f"temp_page_{page_num}.png"
                pix.save(str(temp_img))

                # Send to OCR service
                result = self.extract_from_image_file(str(temp_img), max_tokens)

                # Add page number
                result['page_num'] = page_num + 1

                # Store text
                all_text.append(f"=== Page {page_num + 1} ===\n{result['extracted_text']}\n")

                # Calculate quality metrics if ground truth available
                if page_num < len(self.ground_truth_lines):
                    gt_text = self.ground_truth_lines[page_num]
                    metrics = self.calculate_metrics(gt_text, result['extracted_text'])
                    result.update(metrics)

                results.append(result)

                # Clean up temp image
                temp_img.unlink()

                # Progress update
                if (page_num - start_page + 1) % 10 == 0:
                    avg_time = sum(r['processing_time'] for r in results) / len(results)
                    remaining = end_page - page_num - 1
                    eta = remaining * avg_time
                    print(f"\n  Processed {page_num - start_page + 1} pages. Avg: {avg_time:.2f}s/page. ETA: {eta/60:.1f} min")

            except Exception as e:
                print(f"\n  Error on page {page_num + 1}: {e}")
                results.append({
                    'page_num': page_num + 1,
                    'error': str(e)
                })

        pdf.close()

        # Save results
        self.save_results(results, all_text, output_dir)

        return results

    def calculate_metrics(self, reference: str, hypothesis: str) -> dict:
        """Calculate quality metrics"""
        distance = Levenshtein.distance(reference, hypothesis)
        cer = distance / len(reference) if len(reference) > 0 else 0
        accuracy = 1.0 - (distance / max(len(reference), len(hypothesis))) if max(len(reference), len(hypothesis)) > 0 else 1.0

        return {
            'cer': cer,
            'accuracy': accuracy,
            'edit_distance': distance,
            'ref_length': len(reference),
            'hyp_length': len(hypothesis)
        }

    def save_results(self, results: list, all_text: list, output_dir: Path):
        """Save all results"""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)

        # Save extracted text
        text_path = output_dir / "extracted_text_all_pages.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.writelines(all_text)
        print(f"✓ Extracted text: {text_path}")

        # Save detailed JSON
        json_path = output_dir / "results_detailed.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Detailed results: {json_path}")

        # Save CSV
        df = pd.DataFrame(results)
        csv_path = output_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ CSV results: {csv_path}")

        # Print summary
        self.print_summary(df)

    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        print(f"\nTotal pages: {len(df)}")
        print(f"Total time: {df['processing_time'].sum():.2f}s")
        print(f"Avg time/page: {df['processing_time'].mean():.2f}s")

        if 'cer' in df.columns:
            print(f"\nQuality Metrics:")
            print(f"  CER Mean: {df['cer'].mean():.4f}")
            print(f"  CER Median: {df['cer'].median():.4f}")
            print(f"  Accuracy Mean: {df['accuracy'].mean():.4f}")
            print(f"  Accuracy Median: {df['accuracy'].median():.4f}")

        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Process PDF using Northflank OCR service')
    parser.add_argument('--service-url', required=True, help='Northflank service URL (e.g., https://ocr-v1-xxxxx.nf.run)')
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--ground-truth', help='Path to ground truth text file (optional)')
    parser.add_argument('--output-dir', default='northflank_ocr_results', help='Output directory')
    parser.add_argument('--start-page', type=int, default=0, help='Starting page')
    parser.add_argument('--num-pages', type=int, help='Number of pages (default: all)')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for PDF rendering')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens per page')

    args = parser.parse_args()

    # Create client
    client = NorthflankOCRClient(args.service_url, args.ground_truth)

    # Check service
    print("Checking Northflank service...")
    if not client.check_service_health():
        print("\n✗ Service is not ready. Please check your Northflank deployment.")
        return 1

    # Get status
    client.get_service_status()

    # Process PDF
    results = client.process_pdf(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        start_page=args.start_page,
        num_pages=args.num_pages,
        dpi=args.dpi,
        max_tokens=args.max_tokens
    )

    print("\n✓ Processing complete!")
    print(f"Results saved to: {args.output_dir}/")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
