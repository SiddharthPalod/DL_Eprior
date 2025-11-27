#!/usr/bin/env python3
"""
Batch PDF Processor - Scalable PDF to CSV Converter

This script processes multiple PDFs from a folder, extracts text and OCR content,
generates embeddings, and outputs CSV files matching the system's export format.

Usage:
    python batch_pdf_processor.py --input_folder /path/to/pdfs --output_folder /path/to/output
"""

import os
import sys
import argparse
import logging
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend to path to import services
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Check critical dependencies before importing services
def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    
    # Check PyMuPDF (fitz)
    try:
        import fitz
        logger.info("✓ PyMuPDF (fitz) is available")
    except ImportError:
        missing.append("PyMuPDF (install with: pip install PyMuPDF)")
        logger.error("✗ PyMuPDF (fitz) is NOT available")
    
    # Check sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("✓ sentence-transformers is available")
        # Try a quick test to ensure it can actually load
        try:
            test_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Model can be loaded successfully")
            del test_model
        except Exception as model_error:
            logger.warning(f"⚠ Model loading test failed: {model_error}")
            logger.warning("  This may cause issues during processing")
    except ImportError as e:
        missing.append("sentence-transformers (install with: pip install sentence-transformers)")
        logger.error(f"✗ sentence-transformers is NOT available: {e}")
    
    # Check pytesseract for OCR
    try:
        import pytesseract
        logger.info("✓ pytesseract is available")
    except ImportError:
        missing.append("pytesseract (install with: pip install pytesseract)")
        logger.warning("✗ pytesseract is NOT available (OCR will be skipped)")
    
    if missing:
        logger.warning(f"\nMissing dependencies: {', '.join(missing)}")
        logger.warning("The script may not work correctly without these packages.\n")
    
    return len(missing) == 0

try:
    from app.services.document_parser import DocumentParser
    from app.services.embedding_service import EmbeddingService
except ImportError as e:
    print(f"Error importing services: {e}")
    print("Make sure you're running this from the synapse-docs-main directory")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent}")
    sys.exit(1)


class BatchPDFProcessor:
    """Batch processor for converting PDFs to CSV format."""
    
    def __init__(self, ocr_language: str = 'eng', include_embeddings: bool = True):
        """
        Initialize the batch processor.
        
        Args:
            ocr_language: Language code for OCR (default: 'eng')
            include_embeddings: Whether to generate embeddings (default: True)
        """
        self.ocr_language = ocr_language
        self.include_embeddings = include_embeddings
        
        # Check dependencies first
        logger.info("Checking dependencies...")
        deps_ok = check_dependencies()
        
        # Check if PyMuPDF is available (critical)
        try:
            import fitz
            fitz_available = True
        except ImportError:
            fitz_available = False
            logger.error("CRITICAL: PyMuPDF (fitz) is required but not available!")
            logger.error("Please install it with: pip install PyMuPDF")
            raise ImportError("PyMuPDF (fitz) is required for PDF processing")
        
        # Initialize services (singleton pattern ensures single model instance)
        logger.info("Initializing DocumentParser...")
        self.document_parser = DocumentParser()
        
        if self.include_embeddings:
            logger.info("Initializing EmbeddingService with embeddings enabled...")
            # Direct check for sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("✓ sentence-transformers import successful")
                
                # Try to actually load the model to verify it works
                logger.info("  Testing model loading (this may take a moment on first run)...")
                test_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✓ Model loaded successfully")
                del test_model  # Free memory
                
                # Now initialize the service
                self.embedding_service = EmbeddingService()
                
                # Force reload if model is None (bypass the SENTENCE_TRANSFORMERS_AVAILABLE check)
                if self.embedding_service.model is None:
                    logger.warning("  Model not loaded in service, forcing manual load...")
                    # Directly set and load the model
                    from sentence_transformers import SentenceTransformer
                    self.embedding_service.model_name = "all-MiniLM-L6-v2"
                    self.embedding_service.model = SentenceTransformer(self.embedding_service.model_name)
                    self.embedding_service.model.eval()
                    logger.info("✓ Model manually loaded successfully")
                    
                if self.embedding_service.model is None:
                    raise RuntimeError("Failed to load embedding model - embeddings are required!")
                
                # Test that embeddings actually work
                test_embedding = self.embedding_service.create_embedding("test")
                if not test_embedding or len(test_embedding) != 384:
                    raise RuntimeError(f"Embedding test failed - got {len(test_embedding) if test_embedding else 0} dimensions, expected 384")
                
                logger.info("✓ EmbeddingService initialized and tested successfully")
                
            except ImportError as e:
                logger.error(f"✗ sentence-transformers import failed: {e}")
                logger.error("Please install: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"✗ Failed to initialize EmbeddingService: {e}", exc_info=True)
                logger.error("Embeddings are required but initialization failed!")
                raise
        else:
            self.embedding_service = None
        
        logger.info("Batch processor initialized successfully")
    
    def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Process a single PDF file and return CSV rows.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries representing CSV rows
        """
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        try:
            # Read PDF file
            with open(pdf_path, 'rb') as f:
                file_bytes = f.read()
            
            # Verify PDF can be opened
            try:
                import fitz
                test_doc = fitz.open(stream=file_bytes, filetype="pdf")
                page_count = len(test_doc)
                test_doc.close()
                logger.info(f"  PDF verified: {page_count} pages")
            except Exception as e:
                logger.error(f"  Failed to open PDF: {e}")
                return []
            
            # Extract text chunks and images with OCR
            logger.info(f"  Extracting text and images from {pdf_path.name}...")
            try:
                chunks = self.document_parser.get_text_chunks_with_images(
                    file_bytes,
                    include_ocr=True,
                    include_image_embeddings=False,  # We'll generate text embeddings only
                    ocr_language=self.ocr_language,
                    include_ocr_as_chunks=True,  # OCR text as separate chunks
                    include_blip_captions=False,  # Skip BLIP-2 for speed
                    include_caption_embeddings=False
                )
            except Exception as e:
                logger.error(f"  Error during extraction: {e}", exc_info=True)
                # Try fallback: just text extraction without images
                logger.info(f"  Attempting fallback text extraction...")
                try:
                    chunks = self.document_parser.get_text_chunks(file_bytes)
                    if chunks:
                        logger.info(f"  Fallback extraction successful: {len(chunks)} chunks")
                    else:
                        logger.warning(f"  Fallback extraction returned no chunks")
                        return []
                except Exception as fallback_error:
                    logger.error(f"  Fallback extraction also failed: {fallback_error}")
                    return []
            
            if not chunks:
                logger.warning(f"  No chunks extracted from {pdf_path.name}")
                return []
            
            logger.info(f"  Extracted {len(chunks)} chunks from {pdf_path.name}")
            
            # Convert chunks to CSV rows
            rows = []
            seen_keys = set()
            
            for idx, chunk in enumerate(chunks):
                text_val = (chunk.get('text_chunk') or '').strip()
                
                # Generate embedding if requested
                embed_preview = None
                embed_dim = None
                if self.include_embeddings and self.embedding_service and text_val:
                    try:
                        vec = self.embedding_service.create_embedding(text_val)
                        if isinstance(vec, list) and vec:
                            embed_dim = len(vec)
                            embed_preview = ",".join([f"{x:.4f}" for x in vec[:8]])
                    except Exception as e:
                        logger.warning(f"  Failed to generate embedding for chunk {idx}: {e}")
                
                # Determine source and chunk type
                chunk_type_value = chunk.get('chunk_type', 'content')
                source_value = "text"
                if chunk_type_value == "image_ocr":
                    source_value = "image_ocr"
                
                # Create main row
                row = {
                    "source": source_value,
                    "page_number": chunk.get('page_number', 0),
                    "chunk_index": idx,
                    "image_index": chunk.get('image_index'),
                    "content": text_val,
                    "section_title": chunk.get('section_title'),
                    "chunk_type": chunk_type_value,
                    "embedding_dim": embed_dim,
                    "embedding_preview": embed_preview,
                }
                
                # Avoid duplicates
                key = (row["source"], row["page_number"], row["image_index"], row["content"] or "")
                if key not in seen_keys:
                    rows.append(row)
                    seen_keys.add(key)
                
                # Process images attached to chunks (for additional OCR rows)
                for img in chunk.get('images', []) or []:
                    ocr_text = (img.get('ocr_text') or '').strip()
                    if ocr_text:
                        # Generate embedding for OCR text if requested
                        ocr_embed_preview = None
                        ocr_embed_dim = None
                        if self.include_embeddings and self.embedding_service:
                            try:
                                vec = self.embedding_service.create_embedding(ocr_text)
                                if isinstance(vec, list) and vec:
                                    ocr_embed_dim = len(vec)
                                    ocr_embed_preview = ",".join([f"{x:.4f}" for x in vec[:8]])
                            except Exception as e:
                                logger.warning(f"  Failed to generate embedding for OCR text: {e}")
                        
                        ocr_row = {
                            "source": "image_ocr",
                            "page_number": chunk.get('page_number', 0),
                            "chunk_index": idx,
                            "image_index": img.get('image_index'),
                            "content": ocr_text,
                            "section_title": chunk.get('section_title'),
                            "chunk_type": "image_ocr",
                            "embedding_dim": ocr_embed_dim,
                            "embedding_preview": ocr_embed_preview,
                        }
                        
                        ocr_key = (ocr_row["source"], ocr_row["page_number"], ocr_row["image_index"], ocr_row["content"] or "")
                        if ocr_key not in seen_keys:
                            rows.append(ocr_row)
                            seen_keys.add(ocr_key)
            
            logger.info(f"  Generated {len(rows)} CSV rows from {pdf_path.name}")
            return rows
            
        except Exception as e:
            logger.error(f"  Error processing {pdf_path.name}: {e}", exc_info=True)
            return []
    
    def save_csv(self, rows: List[Dict[str, Any]], output_path: Path):
        """
        Save rows to CSV file.
        
        Args:
            rows: List of dictionaries representing CSV rows
            output_path: Path to output CSV file
        """
        if not rows:
            logger.warning(f"  No rows to save to {output_path.name}")
            return
        
        try:
            # Get header from first row
            header = list(rows[0].keys())
            
            # Write CSV file
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                
                for row in rows:
                    # Convert None to empty string for CSV compatibility
                    cleaned_row = {k: (v if v is not None else '') for k, v in row.items()}
                    writer.writerow(cleaned_row)
            
            logger.info(f"  Saved {len(rows)} rows to {output_path.name}")
            
        except Exception as e:
            logger.error(f"  Error saving CSV to {output_path.name}: {e}", exc_info=True)
            raise
    
    def process_folder(
        self, 
        input_folder: Path, 
        output_folder: Path,
        max_files: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process all PDFs in a folder.
        
        Args:
            input_folder: Path to folder containing PDFs
            output_folder: Path to folder for output CSV files
            max_files: Maximum number of files to process (None for all)
            
        Returns:
            Dictionary with processing statistics
        """
        # Create output folder if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files
        pdf_files = sorted(list(input_folder.glob("*.pdf")))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_folder}")
            return {
                "total_files": 0,
                "processed": 0,
                "failed": 0,
                "total_chunks": 0,
                "total_rows": 0
            }
        
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process statistics
        stats = {
            "total_files": len(pdf_files),
            "processed": 0,
            "failed": 0,
            "total_chunks": 0,
            "total_rows": 0,
            "processing_times": []
        }
        
        start_time = time.time()
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            file_start_time = time.time()
            logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            
            try:
                # Process PDF
                rows = self.process_pdf(pdf_path)
                
                if rows:
                    # Generate output filename
                    output_filename = pdf_path.stem + "_chunks.csv"
                    output_path = output_folder / output_filename
                    
                    # Save CSV
                    self.save_csv(rows, output_path)
                    
                    stats["processed"] += 1
                    stats["total_rows"] += len(rows)
                    stats["total_chunks"] += len(set(r.get('chunk_index', 0) for r in rows))
                else:
                    logger.warning(f"  No rows generated for {pdf_path.name}")
                    stats["failed"] += 1
                
                file_time = time.time() - file_start_time
                stats["processing_times"].append(file_time)
                logger.info(f"  Completed in {file_time:.2f}s")
                
            except Exception as e:
                logger.error(f"  Failed to process {pdf_path.name}: {e}", exc_info=True)
                stats["failed"] += 1
        
        total_time = time.time() - start_time
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total PDFs found: {stats['total_files']}")
        logger.info(f"Successfully processed: {stats['processed']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Total chunks extracted: {stats['total_chunks']}")
        logger.info(f"Total CSV rows generated: {stats['total_rows']}")
        logger.info(f"Total processing time: {total_time:.2f}s")
        if stats['processed'] > 0:
            avg_time = sum(stats['processing_times']) / len(stats['processing_times'])
            logger.info(f"Average time per PDF: {avg_time:.2f}s")
            logger.info(f"Processing rate: {stats['processed']/total_time:.2f} PDFs/second")
        logger.info("="*60)
        
        return stats


def main():
    """Main entry point for the batch processor."""
    parser = argparse.ArgumentParser(
        description="Batch process PDFs to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PDFs in a folder
  python batch_pdf_processor.py --input_folder ./pdfs --output_folder ./output
  
  # Process first 10 PDFs only
  python batch_pdf_processor.py --input_folder ./pdfs --output_folder ./output --max_files 10
  
  # Process without embeddings (faster)
  python batch_pdf_processor.py --input_folder ./pdfs --output_folder ./output --no-embeddings
        """
    )
    
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='Path to folder containing PDF files'
    )
    
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='Path to folder for output CSV files'
    )
    
    parser.add_argument(
        '--ocr_language',
        type=str,
        default='eng',
        help='OCR language code (default: eng)'
    )
    
    parser.add_argument(
        '--no-embeddings',
        action='store_true',
        help='Skip embedding generation for faster processing'
    )
    
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maximum number of PDFs to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    if not input_folder.is_dir():
        logger.error(f"Input path is not a directory: {input_folder}")
        sys.exit(1)
    
    # Create output folder
    output_folder = Path(args.output_folder)
    
    # Initialize processor
    processor = BatchPDFProcessor(
        ocr_language=args.ocr_language,
        include_embeddings=not args.no_embeddings
    )
    
    # Process folder
    stats = processor.process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        max_files=args.max_files
    )
    
    # Exit with error code if all failed
    if stats['processed'] == 0 and stats['total_files'] > 0:
        logger.error("No files were successfully processed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

