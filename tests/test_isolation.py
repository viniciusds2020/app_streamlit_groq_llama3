import importlib
import sys
import types
from unittest.mock import MagicMock

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
    loader_module = types.ModuleType("langchain_community.document_loaders")
    observed = {}
    class BrokenLoader:
        def __init__(self, path):
            observed["path"] = path
        def load_and_split(self):
            raise ValueError("invalid pdf")
    loader_module.PyPDFLoader = BrokenLoader
    package = types.ModuleType("langchain_community")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "langchain_community", package)
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", loader_module)
    sys.modules.pop("ingestion", None)
    ingestion = importlib.import_module("ingestion")
    upload = MagicMock()
    upload.getvalue.return_value = b"not-a-pdf"
    import pytest
    with pytest.raises(ValueError, match="invalid pdf"):
        ingestion.load_pdf(upload)
    from pathlib import Path
    assert not Path(observed["path"]).exists()
