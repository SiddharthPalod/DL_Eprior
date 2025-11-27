# Batch PDF Processor

A scalable, standalone script for batch processing PDFs into CSV format with text extraction, OCR, and embeddings.

## Features

- **Batch Processing**: Process hundreds of PDFs from a folder
- **Text Extraction**: Uses PyMuPDF and CRF-based heading detection
- **OCR Support**: Extracts text from images using Tesseract OCR
- **Embedding Generation**: Generates 384-dimensional embeddings using all-MiniLM-L6-v2
- **CSV Output**: Produces CSV files matching the system's export format
- **Progress Tracking**: Shows progress and statistics during processing
- **Error Handling**: Continues processing even if individual files fail

## Requirements

The script uses the same dependencies as the main backend. Install them with:

```bash
cd backend
pip install -r requirements.txt
```

Key dependencies:
- `PyMuPDF` (fitz) - PDF parsing
- `pytesseract` - OCR
- `sentence-transformers` - Embedding generation
- `sklearn-crfsuite` (optional) - Better heading detection
- `Pillow` - Image processing
- `opencv-python` - Image preprocessing for OCR

## Usage

### Basic Usage

Process all PDFs in a folder:

```bash
python batch_pdf_processor.py --input_folder ./pdfs --output_folder ./output
```

### Advanced Options

```bash
# Process only first 10 PDFs
python batch_pdf_processor.py --input_folder ./pdfs --output_folder ./output --max_files 10

# Process without embeddings (much faster)
python batch_pdf_processor.py --input_folder ./pdfs --output_folder ./output --no-embeddings

# Use different OCR language
python batch_pdf_processor.py --input_folder ./pdfs --output_folder ./output --ocr_language spa
```

### Command Line Arguments

- `--input_folder` (required): Path to folder containing PDF files
- `--output_folder` (required): Path to folder for output CSV files
- `--ocr_language` (optional): OCR language code (default: 'eng')
- `--no-embeddings` (optional): Skip embedding generation for faster processing
- `--max_files` (optional): Maximum number of PDFs to process (default: all)

## Output Format

Each PDF generates a CSV file with the following columns:

- `source`: Source type ("text", "image_ocr")
- `page_number`: Page number (0-based)
- `chunk_index`: Index of the chunk
- `image_index`: Image index (if applicable)
- `content`: Text content
- `section_title`: Section title/heading
- `chunk_type`: Type of chunk ("content", "heading", "image_ocr", etc.)
- `embedding_dim`: Embedding dimension (384 for all-MiniLM-L6-v2)
- `embedding_preview`: First 8 dimensions of embedding (comma-separated)

## Performance

Processing speed depends on:
- PDF size and complexity
- Number of images (OCR is slower)
- Whether embeddings are generated

Typical performance:
- **With embeddings**: ~5-15 seconds per PDF
- **Without embeddings**: ~2-5 seconds per PDF
- **Large PDFs with many images**: May take 30+ seconds

## Example

```bash
# Setup
mkdir pdfs output

# Add your PDFs to the pdfs folder
cp *.pdf pdfs/

# Process all PDFs
python batch_pdf_processor.py --input_folder ./pdfs --output_folder ./output

# Output:
# [1/10] Processing: document1.pdf
#   Extracting text and images from document1.pdf...
#   Extracted 45 chunks from document1.pdf
#   Generated 52 CSV rows from document1.pdf
#   Saved 52 rows to document1_chunks.csv
#   Completed in 8.23s
# ...
#
# ============================================================
# BATCH PROCESSING SUMMARY
# ============================================================
# Total PDFs found: 10
# Successfully processed: 10
# Failed: 0
# Total chunks extracted: 450
# Total CSV rows generated: 520
# Total processing time: 82.30s
# Average time per PDF: 8.23s
# Processing rate: 0.12 PDFs/second
# ============================================================
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the `synapse-docs-main` directory:

```bash
cd synapse-docs-main
python batch_pdf_processor.py --input_folder ./pdfs --output_folder ./output
```

### OCR Errors

If OCR fails, ensure Tesseract is installed:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Memory Issues

For very large batches, process in smaller chunks:

```bash
# Process 50 at a time
python batch_pdf_processor.py --input_folder ./pdfs --output_folder ./output --max_files 50
```

Or disable embeddings for faster processing:

```bash
python batch_pdf_processor.py --input_folder ./pdfs --output_folder ./output --no-embeddings
```

## Notes

- The script uses singleton pattern for services, so models are loaded once and reused
- Processing is sequential (one PDF at a time) to manage memory
- Failed PDFs are logged but don't stop the batch process
- CSV files are named as `{pdf_name}_chunks.csv`

