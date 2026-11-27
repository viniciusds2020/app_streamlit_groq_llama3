import contextlib
import sys
from types import SimpleNamespace

from scripts import langfuse_smoke


class FakeClient:
    def __init__(self):
        self.authenticated = False
        self.flushed = False
        self.span_name = None

    def auth_check(self):
        self.authenticated = True

    @contextlib.contextmanager
    def start_as_current_observation(self, **kwargs):
        self.span_name = kwargs["name"]
        yield SimpleNamespace(update=lambda **values: None)

    def flush(self):
        self.flushed = True


def test_smoke_requires_credentials(monkeypatch, capsys):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert langfuse_smoke.main() == 2
    assert "LANGFUSE_PUBLIC_KEY" in capsys.readouterr().err


def test_smoke_authenticates_and_flushes_without_content(monkeypatch, capsys):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-test")
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(get_client=lambda: client))

    assert langfuse_smoke.main() == 0
    assert client.authenticated
    assert client.flushed
    assert client.span_name == "rag.langfuse_smoke"
    assert "accepted" in capsys.readouterr().out
