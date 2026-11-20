import hashlib
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message

from chat_state import initialize_chat_state
from ingestion import load_pdf
from retrieval import create_vector_store
from llm import load_llm

st.set_page_config(page_title="Assistente para análise de documentos", page_icon="🤖")
st.header("Assistente IA | Análise de documentos")
uploaded_file = st.sidebar.file_uploader("Carregue um PDF", type="pdf")

if uploaded_file is None:
    st.info("Por favor, carregue um arquivo PDF para começar.")
else:
    document_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    initialize_chat_state(document_id)
    try:
        if st.session_state.get("indexed_document") != document_id:
            with st.spinner("Processando PDF..."):
                documents = load_pdf(uploaded_file)
                if not documents:
                    raise ValueError("PDF sem texto extraível")
                st.session_state["vector_store"] = create_vector_store(documents)
                st.session_state["indexed_document"] = document_id
        chain = ConversationalRetrievalChain.from_llm(
            llm=load_llm(),
            retriever=st.session_state["vector_store"].as_retriever(),
            return_source_documents=True,
        )
        for i, (question, answer) in enumerate(st.session_state["history"]):
            message(question, is_user=True, key=f"user_{i}")
            message(answer, key=f"assistant_{i}")
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input("Pergunta:")
            submitted = st.form_submit_button("Enviar")
        if submitted and question.strip():
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
    except Exception as exc:
        st.error(f"Não foi possível processar a solicitação: {type(exc).__name__}. Verifique a configuração e o PDF.")
