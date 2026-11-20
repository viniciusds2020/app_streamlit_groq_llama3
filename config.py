import os

def get_groq_config():
    api_key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY não configurada")
    if not model:
        raise RuntimeError("GROQ_MODEL não configurado")
    return api_key, model
