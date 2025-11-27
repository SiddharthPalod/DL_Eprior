from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class Chunk:
    source: str
    page_number: int
    chunk_index: int
    content: str
    section_title: str


def load_chunks(pdf_embeddings_dir: Path) -> List[Chunk]:
    if not pdf_embeddings_dir.exists():
        raise FileNotFoundError(f"Missing embeddings folder: {pdf_embeddings_dir}")

    chunks: List[Chunk] = []
    
    # Try to load from CSV files first (more common)
    csv_files = sorted(pdf_embeddings_dir.glob("*.csv"))
    xlsx_files = sorted(pdf_embeddings_dir.glob("*.xlsx"))
    
    files_to_process = csv_files if csv_files else xlsx_files
    
    if not files_to_process:
        raise FileNotFoundError(
            f"No .csv or .xlsx files found in {pdf_embeddings_dir}. "
            f"Found files: {list(pdf_embeddings_dir.iterdir())[:5]}..."
        )
    
    for file in files_to_process:
        try:
            # Read CSV or Excel file
            if file.suffix.lower() == ".csv":
                frame = pd.read_csv(file)
            else:
                frame = pd.read_excel(file)
            
            for row in frame.itertuples(index=False):
                content = (getattr(row, "content", "") or "").strip()
                if not content:
                    continue
                
                chunks.append(
                    Chunk(
                        source=str(getattr(row, "source", "")),
                        page_number=int(getattr(row, "page_number", 0)),
                        chunk_index=int(getattr(row, "chunk_index", 0)),
                        content=content,
                        section_title=(getattr(row, "section_title", "") or "").strip(),
                    )
                )
        except Exception as e:
            print(f"Warning: Failed to load {file.name}: {e}")
            continue
    
    if not chunks:
        raise ValueError("No textual chunks were loaded from the embeddings directory")
    
    print(f"Loaded {len(chunks)} chunks from {len(files_to_process)} files")
    return chunks
