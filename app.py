"""Streamlit presentation layer. RAG execution lives in the FastAPI backend."""
import hashlib
import logging

import streamlit as st

from api_client import RAGAPIError, ask, create_session, reset, upload_pdf
from styles import CUSTOM_CSS

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Assistente de documentos", page_icon="💬", layout="centered")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def clear_conversation():
    session_id = st.session_state.pop("api_session_id", None)
    if session_id:
        reset(session_id)
    st.session_state.pop("indexed_document", None)
    st.session_state["history"] = []


with st.sidebar:
    st.markdown('<div class="sidebar-brand">💬 Assistente IA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Converse com o conteúdo de um PDF</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Carregue um PDF", type="pdf", label_visibility="collapsed")
    if uploaded_file is not None:
        st.markdown('<span class="status-pill">📄 Documento carregado</span>', unsafe_allow_html=True)
    if st.session_state.get("history"):
        st.button("🗑️ Nova conversa", use_container_width=True, on_click=clear_conversation)

st.markdown(
    """
    <div class="app-hero">
        <div class="app-icon">🤖</div>
        <div>
            <h1>Assistente de documentos</h1>
            <p>Pergunte qualquer coisa sobre o PDF carregado</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is None:
    # No active document should remain reachable through the UI.
    if st.session_state.get("api_session_id"):
        clear_conversation()
    st.markdown(
        """
        <div class="empty-state">
            <div class="big-icon">📄</div>
            <strong>Nenhum documento carregado</strong>
            <span>Carregue um PDF na barra lateral para começar a conversa.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    try:
        pdf_bytes = uploaded_file.getvalue()
        fingerprint = hashlib.sha256(pdf_bytes).hexdigest()
        if st.session_state.get("indexed_document") != fingerprint:
            clear_conversation()
            with st.spinner("Processando PDF..."):
                session_id = create_session()
                st.session_state["api_session_id"] = session_id
                upload_pdf(session_id, pdf_bytes, uploaded_file.name)
                st.session_state["indexed_document"] = fingerprint

        for question, answer in st.session_state.get("history", []):
            with st.chat_message("user", avatar="🧑"):
                st.markdown(question)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(answer)

        question = st.chat_input("Pergunte algo sobre o documento...")
        if question and question.strip():
            with st.spinner("Consultando documento..."):
                result = ask(st.session_state["api_session_id"], question.strip())
            answer = result["answer"]
            pages = result.get("pages", [])
            if pages:
                answer += "\n\nPáginas recuperadas: " + ", ".join(map(str, pages))
            st.session_state.setdefault("history", []).append((question.strip(), answer))
            st.rerun()
    except RAGAPIError as exc:
        st.error(str(exc))
    except Exception as exc:
        logger.error("Streamlit API client failure exception_type=%s", type(exc).__name__)
        st.error("Não foi possível concluir a solicitação. Verifique o serviço RAG.")
