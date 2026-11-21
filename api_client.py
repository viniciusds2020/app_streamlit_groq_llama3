"""Small HTTP client for the FastAPI backend; never show raw provider errors."""
import os
import requests

BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")


class RAGAPIError(Exception):
    pass


def _request(method, path, **kwargs):
    try:
        response = requests.request(method, BASE_URL + path, timeout=(5, 120), **kwargs)
    except requests.RequestException:
        raise RAGAPIError("Não foi possível conectar ao serviço RAG. Verifique se a FastAPI está em execução.") from None

    if not response.ok:
        try:
            detail = response.json().get("detail", {})
            message = detail.get("message") if isinstance(detail, dict) else None
        except (ValueError, TypeError, AttributeError):
            message = None
        # Avoid surfacing unexpected arbitrary server responses.
        approved = {
            "Sessão não encontrada.",
            "Envie um PDF antes de perguntar.",
            "O PDF excede o tamanho máximo permitido.",
            "O arquivo PDF está vazio ou não pôde ser lido.",
            "O arquivo não é um PDF válido.",
            "PDF protegido por senha não é aceito.",
            "O PDF não contém páginas.",
            "O PDF excede o limite de páginas.",
            "O PDF está corrompido ou não é suportado.",
            "Não foi possível extrair o texto do PDF.",
            "O texto extraído excede o limite permitido.",
            "O PDF não contém texto extraível. PDFs digitalizados precisam de OCR.",
        }
        raise RAGAPIError(message if message in approved else "Não foi possível concluir a solicitação.")
    return response.json() if response.content else {}


def create_session():
    return _request("POST", "/api/v1/sessions")["session_id"]


def upload_pdf(session_id, pdf_bytes, filename="documento.pdf"):
    return _request(
        "POST", f"/api/v1/sessions/{session_id}/documents",
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )


def ask(session_id, question):
    return _request("POST", f"/api/v1/sessions/{session_id}/questions", json={"question": question})


def reset(session_id):
    try:
        _request("DELETE", f"/api/v1/sessions/{session_id}")
    except RAGAPIError:
        pass  # Backend session may already have expired or restarted.
