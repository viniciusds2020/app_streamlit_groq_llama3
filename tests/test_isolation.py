import importlib
import io
import sys
import types
from pathlib import Path

import pytest
from pypdf import PdfWriter


def test_document_switch_clears_index(monkeypatch):
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.session_state = {}
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    sys.modules.pop("chat_state", None)
    module = importlib.import_module("chat_state")
    module.initialize_chat_state("document-a")
    fake_streamlit.session_state["vector_store"] = object()
    fake_streamlit.session_state["indexed_document"] = "document-a"
    fake_streamlit.session_state["history"].append(("pergunta", "resposta"))
    module.initialize_chat_state("document-a")
    assert len(fake_streamlit.session_state["history"]) == 1
    module.initialize_chat_state("document-b")
    assert fake_streamlit.session_state["history"] == []
    assert "vector_store" not in fake_streamlit.session_state
    assert "indexed_document" not in fake_streamlit.session_state


def test_pdf_temp_file_removed_after_loader_failure(monkeypatch):
    # Patch the imported class, not sys.modules, to prevent order-dependent tests.
    import ingestion

    observed = {}

    class BrokenLoader:
        def __init__(self, path):
            observed["path"] = path

        def load_and_split(self):
            raise ValueError("sensitive parser detail")

    monkeypatch.setattr(ingestion, "PyPDFLoader", BrokenLoader)

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    stream = io.BytesIO()
    writer.write(stream)

    class Upload:
        def getvalue(self):
            return stream.getvalue()

    with pytest.raises(ingestion.PDFValidationError, match="Não foi possível extrair"):
        ingestion.load_pdf(Upload())
    assert observed, "the valid PDF should reach the loader"
    assert not Path(observed["path"]).exists()
