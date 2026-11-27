# Assistente RAG de PDFs com Streamlit, FastAPI, Qdrant e Groq

Aplicação para carregar um PDF e conversar com seu conteúdo. A interface Streamlit chama uma API FastAPI; embeddings multilíngues são gerados localmente com FastEmbed, vetores ficam no Qdrant e metadados/histórico no DuckDB. A geração usa Groq.

## Executar

Recomendado: Python 3.11.

```bash
git clone https://github.com/viniciusds2020/app_streamlit_groq_llama3.git
cd app_streamlit_groq_llama3
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="sua_chave"
export GROQ_MODEL="nome_de_um_modelo_disponivel_na_sua_conta"
```

Inicie o backend e o Streamlit em terminais separados:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
streamlit run app.py
```

Por padrão, o Qdrant grava em `data/qdrant` e o DuckDB em `data/rag.duckdb`. Para um servidor Qdrant, configure `QDRANT_URL` e, se necessário, `QDRANT_API_KEY`. Consulte [FASTAPI.md](FASTAPI.md) para Langfuse, configurações e limites. Nunca exponha a API local sem autenticação e autorização.

## Arquitetura

- `app.py` e `api_client.py`: interface e cliente HTTP
- `api.py`: endpoints FastAPI, request IDs e limites de upload
- `rag_service.py`: ingestão, recuperação, histórico e limites de sessão
- `retrieval.py`: embeddings FastEmbed e persistência de vetores no Qdrant
- `metadata_store.py`: metadados e histórico persistidos em DuckDB
- `observability.py`: tracing opcional de respostas no Langfuse
- `llm.py`: cliente Groq
- `ingestion.py`: leitura de PDF e limpeza do arquivo temporário

A aplicação mostra páginas recuperadas, que não devem ser confundidas com citações verificadas de cada afirmação. PDFs digitalizados sem camada de texto exigem OCR, ainda fora do escopo.

## Testes

```bash
python -m pytest -q
```

Os testes não exigem credenciais Groq ou Langfuse. O backend local é single-process; os diretórios de Qdrant e o arquivo DuckDB guardam conteúdo dos documentos e não são criptografados automaticamente.

## Limites e segurança do upload

O processamento valida bytes e estrutura do PDF, rejeita PDFs protegidos por senha, limita páginas e caracteres extraídos e exclui o arquivo temporário ao terminar ou falhar. Limites operacionais configuráveis:

| Variável | Padrão | Efeito |
| --- | ---: | --- |
| `RAG_MAX_PDF_BYTES` | `10485760` (10 MiB) | Rejeita uploads acima do limite |
| `RAG_MAX_PDF_PAGES` | `100` | Rejeita documentos com mais páginas |
| `RAG_MAX_EXTRACTED_CHARS` | `1000000` | Rejeita texto excessivo antes de indexar |
| `RAG_MAX_CONCURRENT_OPERATIONS` | `2` | Limita uploads e perguntas simultâneos |

Essas verificações não substituem limites de CPU, memória e tempo no processo/contêiner, nem autenticação, autorização e políticas de retenção adequadas.

Consulte `openspec/changes/` para os critérios de arquitetura e segurança.
