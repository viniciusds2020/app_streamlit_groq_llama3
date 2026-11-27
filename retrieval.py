"""FAISS embedding adapter independent of the Streamlit UI."""
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _load_embeddings():
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_vector_store(documents):
    return FAISS.from_documents(documents, _load_embeddings())
