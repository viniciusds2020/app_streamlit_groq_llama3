import streamlit as st

def initialize_chat_state(document_id):
    if st.session_state.get("document_name") != document_id:
        for key in ("vector_store", "indexed_document", "history", "past", "generated"):
            st.session_state.pop(key, None)
        st.session_state["document_name"] = document_id
    for name in ("history", "past", "generated"):
        st.session_state.setdefault(name, [])
