"""Bounded PDF ingestion with non-sensitive validation errors.

Resource limits are operational safeguards, not a complete sandbox for untrusted
PDF parsers. Deployments should also enforce process memory/time limits.
"""
import io
import os
import tempfile

from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader


class PDFValidationError(ValueError):
    """An expected, user-safe PDF validation failure."""


def _limit(name, default):
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        raise RuntimeError(f"Configuração inválida: {name}") from None


def validate_pdf_bytes(data):
    """Reject unsuitable PDFs before writing a temporary file or embedding."""
    if not isinstance(data, bytes) or not data:
        raise PDFValidationError("O arquivo PDF está vazio ou não pôde ser lido.")
    if len(data) > _limit("RAG_MAX_PDF_BYTES", 10 * 1024 * 1024):
        raise PDFValidationError("O PDF excede o tamanho máximo permitido.")
    if not data.startswith(b"%PDF-"):
        raise PDFValidationError("O arquivo não é um PDF válido.")

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise PDFValidationError("PDF protegido por senha não é aceito.")
        page_count = len(reader.pages)
        if page_count == 0:
            raise PDFValidationError("O PDF não contém páginas.")
        if page_count > _limit("RAG_MAX_PDF_PAGES", 100):
            raise PDFValidationError("O PDF excede o limite de páginas.")
    except PDFValidationError:
        raise
    except Exception:
        raise PDFValidationError("O PDF está corrompido ou não é suportado.") from None
    return data


def load_pdf(uploaded_file):
    """Validate, parse and clean up a temporary PDF under every outcome."""
    data = validate_pdf_bytes(uploaded_file.getvalue())
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            path = tmp.name
            tmp.write(data)
        try:
            documents = PyPDFLoader(path).load_and_split()
        except Exception:
            raise PDFValidationError("Não foi possível extrair o texto do PDF.") from None
        max_chars = _limit("RAG_MAX_EXTRACTED_CHARS", 1_000_000)
        if sum(len(doc.page_content) for doc in documents) > max_chars:
            raise PDFValidationError("O texto extraído excede o limite permitido.")
        if not any(doc.page_content.strip() for doc in documents):
            raise PDFValidationError("O PDF não contém texto extraível. PDFs digitalizados precisam de OCR.")
        return documents
    finally:
        if path and os.path.exists(path):
            os.remove(path)
