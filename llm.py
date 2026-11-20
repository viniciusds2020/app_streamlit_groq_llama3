from langchain_groq import ChatGroq
from config import get_groq_config

def load_llm():
    api_key, model = get_groq_config()
    return ChatGroq(temperature=0.7, groq_api_key=api_key, model_name=model)
