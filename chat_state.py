import streamlit as st

def initialize_chat_state(filename):
    if st.session_state.get("document_name") != filename:
        st.session_state["document_name"] = filename
        st.session_state["history"] = []
        st.session_state["past"] = []
        st.session_state["generated"] = []
    for name in ("history", "past", "generated"):
        st.session_state.setdefault(name, [])
