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
    for file in sorted(pdf_embeddings_dir.glob("*.xlsx")):
        frame = pd.read_excel(file)
        for row in frame.itertuples(index=False):
            content = (row.content or "").strip()
            if not content:
                continue
            chunks.append(
                Chunk(
                    source=str(row.source),
                    page_number=int(row.page_number),
                    chunk_index=int(row.chunk_index),
                    content=content,
                    section_title=(row.section_title or "").strip(),
                )
            )
    if not chunks:
        raise ValueError("No textual chunks were loaded from the embeddings directory")
    return chunks
