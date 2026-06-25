"""RAG Text Splitter — mirrors the Java splitter modules.

Provides multiple splitting strategies:
- Recursive character text splitter (LangChain)
- Paragraph-based splitting
- Sentence window splitting
- Overlapping window splitting
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    """Multi-strategy text chunking for RAG."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        min_chunk_length: int = 80,
        max_chunks: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.max_chunks = max_chunks

    def split(
        self,
        text: str,
        strategy: str = "recursive",
        metadata: dict | None = None,
    ) -> list[dict]:
        """Split text into chunks using the specified strategy.

        Strategies:
        - "recursive": LangChain RecursiveCharacterTextSplitter (default)
        - "paragraph": Split by paragraphs, then recursively if too large
        - "sentence": Sentence window with context sentences
        - "overlap": Fixed-size overlapping windows
        """
        if not text or not text.strip():
            return []

        if strategy == "paragraph":
            chunks = self._split_by_paragraph(text)
        elif strategy == "sentence":
            chunks = self._split_by_sentence_window(text)
        elif strategy == "overlap":
            chunks = self._split_overlapping(text)
        else:
            chunks = self._split_recursive(text)

        # Filter and enrich chunks
        enriched = []
        meta = metadata or {}
        for i, chunk_text in enumerate(chunks):
            cleaned = chunk_text.strip()
            if len(cleaned) < self.min_chunk_length:
                continue
            enriched.append({
                "content": cleaned,
                "chunk_index": i,
                "char_count": len(cleaned),
                "token_estimate": len(cleaned) // 2,  # rough estimate
                "metadata": {**meta, "chunk_index": i},
            })
            if len(enriched) >= self.max_chunks:
                break

        return enriched

    def _split_recursive(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", ". ", " ", ""],
            length_function=len,
        )
        return splitter.split_text(text)

    def _split_by_paragraph(self, text: str) -> list[str]:
        """Split by paragraphs, then recursively split large paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.chunk_size:
                chunks.append(para)
            else:
                # Recursively split large paragraphs
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                chunks.extend(sub_splitter.split_text(para))
        return chunks

    def _split_by_sentence_window(self, text: str) -> list[str]:
        """Split by sentences with context window."""
        # Split into sentences (Chinese + English patterns)
        sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        window_size = 3  # sentences per chunk
        for i in range(0, len(sentences), window_size):
            chunk = " ".join(sentences[i : i + window_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _split_overlapping(self, text: str) -> list[str]:
        """Fixed-size overlapping windows."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks
