# Assistente RAG de PDFs com Streamlit, FAISS e Groq

Aplicação para carregar um PDF e conversar com seu conteúdo. A busca utiliza embeddings multilíngues via fastembed e FAISS; a geração usa um modelo disponível na Groq.

## Executar

Recomendado: Python 3.10 ou 3.11.

```bash
git clone https://github.com/viniciusds2020/app_streamlit_groq_llama3.git
cd app_streamlit_groq_llama3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="sua_chave_nova"
export GROQ_MODEL="nome_de_um_modelo_disponivel_na_sua_conta"
streamlit run app.py
```

No Windows, configure as variáveis de ambiente com os comandos correspondentes ao terminal. Não envie credenciais ao GitHub.

## Arquitetura

- `app.py`: interface e fluxo de perguntas
- `config.py`: validação da configuração externa
- `chat_state.py`: histórico e troca de documento por sessão
- `ingestion.py`: leitura de PDF e limpeza do arquivo temporário
- `retrieval.py`: embeddings e índice FAISS em memória
- `llm.py`: cliente Groq

A aplicação mostra páginas recuperadas, que não devem ser confundidas com citações verificadas de cada afirmação. PDFs digitalizados sem camada de texto exigem OCR, ainda fora do escopo.

## Testes

```bash
pytest -q
```

A instalação e o teste ponta a ponta devem ser verificados em ambiente limpo. A credencial antiga foi exposta em histórico público; revogá-la é necessário, mas removê-la do código atual não elimina os commits antigos.

## Especificações

Consulte `openspec/changes/secure-and-refactor-rag/`.
