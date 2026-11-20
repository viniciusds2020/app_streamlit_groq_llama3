import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(uploaded_file):
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            path = tmp.name
            tmp.write(uploaded_file.getvalue())
        return PyPDFLoader(path).load_and_split()
    finally:
        if path and os.path.exists(path):
            os.remove(path)
