"""RAG Document Parser — mirrors the Java parser modules (PDFBox, POI, etc.).

Handles: PDF, DOCX, PPTX, XLSX, Markdown, plain text, images (OCR).
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parse various document formats into plain text."""

    async def parse(self, file_bytes: bytes, file_name: str, file_type: str | None = None) -> str:
        """Parse file bytes into plain text based on file extension."""
        ext = self._resolve_extension(file_name, file_type)

        parsers = {
            "pdf": self._parse_pdf,
            "docx": self._parse_docx,
            "doc": self._parse_doc,
            "pptx": self._parse_pptx,
            "ppt": self._parse_ppt,
            "xlsx": self._parse_xlsx,
            "xls": self._parse_xls,
            "md": self._parse_markdown,
            "txt": self._parse_text,
            "csv": self._parse_csv,
            "json": self._parse_json,
            "xml": self._parse_xml,
            "png": self._parse_image,
            "jpg": self._parse_image,
            "jpeg": self._parse_image,
            "gif": self._parse_image,
        }

        parser = parsers.get(ext)
        if not parser:
            logger.warning(f"No parser for extension: {ext}, trying text fallback")
            return self._parse_text(file_bytes)

        try:
            text = await parser(file_bytes)
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Failed to parse {file_name} ({ext}): {e}")
            raise

    def _resolve_extension(self, file_name: str, file_type: str | None) -> str:
        if file_type:
            return file_type.lower().lstrip(".")
        return Path(file_name).suffix.lower().lstrip(".")

    # ── PDF ────────────────────────────────────────────────────

    async def _parse_pdf(self, file_bytes: bytes) -> str:
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            # Fallback to PyPDF2
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    # ── DOCX ───────────────────────────────────────────────────

    async def _parse_docx(self, file_bytes: bytes) -> str:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)

        # Also extract tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text for cell in row.cells)
                if row_text.strip():
                    paragraphs.append(row_text)

        return "\n\n".join(paragraphs)

    # ── DOC (legacy) ───────────────────────────────────────────

    async def _parse_doc(self, file_bytes: bytes) -> str:
        # Try antiword or textract if available; otherwise skip
        logger.warning("Legacy .doc format — limited support, extracting as raw text")
        return self._parse_text(file_bytes)

    # ── PPTX ───────────────────────────────────────────────────

    async def _parse_pptx(self, file_bytes: bytes) -> str:
        from pptx import Presentation
        prs = Presentation(io.BytesIO(file_bytes))
        slides_text = []
        for i, slide in enumerate(prs.slides):
            slide_parts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        if para.text.strip():
                            slide_parts.append(para.text)
            if slide_parts:
                slides_text.append(f"--- Slide {i+1} ---\n" + "\n".join(slide_parts))
        return "\n\n".join(slides_text)

    async def _parse_ppt(self, file_bytes: bytes) -> str:
        logger.warning("Legacy .ppt format — limited support")
        return self._parse_text(file_bytes)

    # ── XLSX ───────────────────────────────────────────────────

    async def _parse_xlsx(self, file_bytes: bytes) -> str:
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
        sheets_text = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = []
            for row in ws.iter_rows(values_only=True):
                row_text = "\t".join(str(cell) if cell is not None else "" for cell in row)
                if row_text.strip():
                    rows.append(row_text)
            if rows:
                sheets_text.append(f"--- Sheet: {sheet_name} ---\n" + "\n".join(rows))
        wb.close()
        return "\n\n".join(sheets_text)

    async def _parse_xls(self, file_bytes: bytes) -> str:
        logger.warning("Legacy .xls format — limited support")
        return self._parse_text(file_bytes)

    # ── Text-based ─────────────────────────────────────────────

    async def _parse_markdown(self, file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8", errors="replace")

    async def _parse_text(self, file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8", errors="replace")

    async def _parse_csv(self, file_bytes: bytes) -> str:
        import csv
        text = file_bytes.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        return "\n".join(", ".join(row) for row in reader)

    async def _parse_json(self, file_bytes: bytes) -> str:
        import json
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
        return json.dumps(data, indent=2, ensure_ascii=False)

    async def _parse_xml(self, file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8", errors="replace")

    # ── Image OCR ──────────────────────────────────────────────

    async def _parse_image(self, file_bytes: bytes) -> str:
        try:
            from PIL import Image
            import pytesseract

            image = Image.open(io.BytesIO(file_bytes))
            # Try multiple languages: Chinese simplified + traditional + English
            text = pytesseract.image_to_string(image, lang="chi_sim+chi_tra+eng")
            return text.strip()
        except ImportError:
            logger.warning("pytesseract not available — OCR skipped")
            return "[Image — OCR not available]"
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return "[Image — OCR failed]"

    # ── Text Cleaning ──────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Clean extracted text: normalize whitespace, remove control chars."""
        if not text:
            return ""
        # Remove null bytes and other control characters (except newlines/tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Strip leading/trailing whitespace
        return text.strip()


# Singleton
document_parser = DocumentParser()
