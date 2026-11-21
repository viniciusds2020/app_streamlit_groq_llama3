import hashlib
import logging
import uuid
import streamlit as st
from langchain.chains import ConversationalRetrievalChain

from chat_state import initialize_chat_state
from ingestion import load_pdf, validate_pdf_bytes, PDFValidationError
from retrieval import create_vector_store
from llm import load_llm
from styles import CUSTOM_CSS

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Assistente de documentos", page_icon="💬", layout="centered")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-brand">💬 Assistente IA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Converse com o conteúdo de um PDF</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Carregue um PDF", type="pdf", label_visibility="collapsed")
    if uploaded_file is not None:
        st.markdown('<span class="status-pill">📄 Documento carregado</span>', unsafe_allow_html=True)
    if st.session_state.get("history"):
        st.button(
            "🗑️ Nova conversa",
            use_container_width=True,
            on_click=lambda: st.session_state.update(history=[]),
        )

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
        # Validate before indexing or building a chain. No provider exception or
        # document content is shown in the interface or application logs.
        pdf_bytes = validate_pdf_bytes(uploaded_file.getvalue())
        document_id = hashlib.sha256(pdf_bytes).hexdigest()
        initialize_chat_state(document_id)
        if st.session_state.get("indexed_document") != document_id:
            with st.spinner("Processando PDF..."):
                documents = load_pdf(uploaded_file)
                st.session_state["vector_store"] = create_vector_store(documents)
                st.session_state["indexed_document"] = document_id
        chain = ConversationalRetrievalChain.from_llm(
            llm=load_llm(),
            retriever=st.session_state["vector_store"].as_retriever(),
            return_source_documents=True,
        )
        for question, answer in st.session_state["history"]:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(question)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(answer)

        question = st.chat_input("Pergunte algo sobre o documento...")
        if question and question.strip():
            with st.spinner("Consultando documento..."):
                result = chain.invoke({
                    "question": question,
                    "chat_history": st.session_state["history"],
                })
            answer = result["answer"]
            sources = sorted({
                str(doc.metadata["page"] + 1)
                for doc in result.get("source_documents", [])
                if isinstance(doc.metadata.get("page"), int)
            }, key=int)
            if sources:
                answer += "\n\nPáginas recuperadas: " + ", ".join(sources)
            st.session_state["history"].append((question, answer))
            st.rerun()
    except PDFValidationError as exc:
        st.error(str(exc))  # Only controlled, explicitly safe messages.
    except Exception as exc:
        incident_id = uuid.uuid4().hex[:12]
        logger.error("RAG request failed: incident=%s exception_type=%s",
                     incident_id, type(exc).__name__)
        st.error(f"Não foi possível concluir a solicitação. Tente novamente. Referência: {incident_id}")
