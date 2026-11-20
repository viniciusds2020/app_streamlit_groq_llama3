from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )
    return FAISS.from_documents(documents, embeddings)
