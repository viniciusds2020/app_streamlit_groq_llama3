"""Security regression tests: synthetic PDFs and no live API credentials."""
import io
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from ingestion import PDFValidationError, load_pdf, validate_pdf_bytes


class Upload:
    def __init__(self, content):
        self.content = content

    def getvalue(self):
        return self.content


def pdf_bytes(pages=1):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_rejects_invalid_header():
    with pytest.raises(PDFValidationError, match="não é um PDF"):
        validate_pdf_bytes(b"this-is-not-a-pdf")


def test_rejects_oversized_before_parser(monkeypatch):
    monkeypatch.setenv("RAG_MAX_PDF_BYTES", "5")
    with pytest.raises(PDFValidationError, match="tamanho"):
        validate_pdf_bytes(b"%PDF-" + b"x" * 5)


def test_rejects_excess_pages(monkeypatch):
    monkeypatch.setenv("RAG_MAX_PDF_PAGES", "1")
    with pytest.raises(PDFValidationError, match="páginas"):
        validate_pdf_bytes(pdf_bytes(2))


def test_rejects_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("password")
    out = io.BytesIO()
    writer.write(out)
    with pytest.raises(PDFValidationError, match="senha"):
        validate_pdf_bytes(out.getvalue())


def test_rejects_no_extractable_text_and_cleans_up(monkeypatch):
    import ingestion
    visited = []

    class FakeLoader:
        def __init__(self, path):
            visited.append(path)

        def load_and_split(self):
            return [SimpleNamespace(page_content="  ")]

    monkeypatch.setattr(ingestion, "PyPDFLoader", FakeLoader)
    with pytest.raises(PDFValidationError, match="OCR"):
        load_pdf(Upload(pdf_bytes()))
    assert visited
    assert not __import__("os").path.exists(visited[0])


def test_rejects_extraction_limit_and_cleans_up(monkeypatch):
    import ingestion
    import os
    visited = []
    monkeypatch.setenv("RAG_MAX_EXTRACTED_CHARS", "3")

    class FakeLoader:
        def __init__(self, path):
            visited.append(path)

        def load_and_split(self):
            return [SimpleNamespace(page_content="too long")]

    monkeypatch.setattr(ingestion, "PyPDFLoader", FakeLoader)
    with pytest.raises(PDFValidationError, match="texto extraído"):
        load_pdf(Upload(pdf_bytes()))
    assert not os.path.exists(visited[0])


def test_parser_exception_is_sanitized_and_temp_cleaned(monkeypatch):
    import ingestion
    import os
    visited = []

    class FakeLoader:
        def __init__(self, path):
            visited.append(path)

        def load_and_split(self):
            raise RuntimeError("private document content / secret token")

    monkeypatch.setattr(ingestion, "PyPDFLoader", FakeLoader)
    with pytest.raises(PDFValidationError) as caught:
        load_pdf(Upload(pdf_bytes()))
    assert "private document" not in str(caught.value)
    assert not os.path.exists(visited[0])


def test_accepted_pdf_has_pages():
    assert validate_pdf_bytes(pdf_bytes(1)).startswith(b"%PDF-")
