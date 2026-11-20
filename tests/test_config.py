import pytest
from config import get_groq_config

def test_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_groq_config()

def test_config(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    assert get_groq_config() == ("test-key", "test-model")
