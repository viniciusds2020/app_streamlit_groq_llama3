import tempfile
import streamlit as st
from streamlit_chat import message
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader

DB_FAISS_PATH = 'vectorstore/db_faiss'
GROQ_API_KEY = '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'
MODEL_NAME = 'llama3-70b-8192'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

def load_llm():
    return ChatGroq(temperature=0.7, groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)

def initialize_chat_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Olá! Sou o seu assistente para ajudar sobre " + uploaded_file.name]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Olá!"]

def handle_file_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def create_vector_store(data):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

def create_conversational_chain(llm, db):
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

def conversational_chat(chain, query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Aplicativo streamlit:

st.set_page_config(page_title="Assistente para análise de dados", page_icon=":robot_face:", layout="wide")
st.header(":male-technologist: Assistente IA | Análise de docs | DFS :male-technologist:")
st.markdown('''
**Assistente IA de leitura de documentos**

O app permite importar um arquivo pdf e fazer perguntas para o assistente de dados um expert
no seu documento. Analise seus dados, extraia insights, crie resumos e
tome decisões informadas com facilidade e agilizando processos de tomada de decisão.
''')
st.markdown('''
**Modelo LLM: Llama3 - 70b**

O LLaMA 3 (Large Language Model Meta AI) é uma série de modelos de linguagem desenvolvidos 
pela Meta (antiga Facebook). A versão de 70 bilhões de parâmetros (70B) é a maior da terceira 
geração, destacando-se pela sua capacidade de gerar texto e compreender linguagem natural com 
alta precisão. Projetado para várias aplicações, desde assistência virtual até pesquisa avançada, 
o LLaMA 3-70B oferece uma performance robusta em tarefas complexas de processamento de linguagem natural.
''')

st.sidebar.header("Menu")

uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf")

if uploaded_file:
    tmp_file_path = handle_file_upload(uploaded_file)
    loader = PyPDFLoader(file_path=tmp_file_path)
    data = loader.load_and_split()

    db = create_vector_store(data)
    llm = load_llm()
    chain = create_conversational_chain(llm, db)

    initialize_chat_state()

    # Container para histórico de respostas
    response_container = st.container()
    # Espaço reservado para manter o campo de entrada sempre na parte inferior
    input_container = st.empty()

    # Função para renderizar o histórico de mensagens
    def render_chat_history():
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user_msg', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i) + '_bot_msg', avatar_style="thumbs")

    # Campo de entrada de perguntas
    with input_container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Pergunta:", placeholder="Converse com o seu documento 👉 (:", key='input')
            submit_button = st.form_submit_button(label='Enviar')

        if submit_button and user_input:
            output = conversational_chat(chain, user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            render_chat_history()  # Atualizar histórico após nova mensagem

else:
    st.info("Por favor, carregue um arquivo PDF para começar.")
