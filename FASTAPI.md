# FastAPI RAG — execução local

Esta branch separa a interface Streamlit da orquestração RAG. O backend mantém FAISS/FastEmbed, Groq e validação dos PDFs; a UI usa HTTP. **Somente para localhost**: tokens de sessão são credenciais bearer, o índice fica em memória de um processo e não há autenticação/autorização de usuários. Nunca exponha a API na internet assim.

## Preparação

Recomendado Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="sua_chave"
export GROQ_MODEL="modelo-disponivel-na-sua-conta"
```

Em Windows PowerShell: `.venv\Scripts\Activate.ps1` e `$env:GROQ_API_KEY="..."`, `$env:GROQ_MODEL="..."`.

## Iniciar em dois terminais

Terminal 1, backend (ambiente ativado):

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Terminal 2, Streamlit (mesmo ambiente):

```bash
streamlit run app.py
```

Documentação interativa do backend: http://127.0.0.1:8000/docs. Para outra porta, configure `RAG_API_URL` no processo Streamlit.

## Endpoints

- `POST /api/v1/sessions`: cria sessão local de acesso por token.
- `POST /api/v1/sessions/{session_id}/documents`: upload de um PDF; substitui índice e histórico da sessão.
- `POST /api/v1/sessions/{session_id}/questions`: JSON `{"question":"..."}` retorna `answer`, `pages` e `document_id`.
- `DELETE /api/v1/sessions/{session_id}`: elimina estado da sessão.
- `GET /health/live`, `GET /health/ready`: estado básico do processo (não testam serviços externos).

PDFs devem respeitar as restrições preexistentes de tamanho, páginas e texto. Sessões expiram após inatividade (padrão: 3600 s); limite padrão de 100 sessões no processo. Reiniciar o backend apaga índices e conversas. A resposta apresenta páginas recuperadas, não citações verificadas em nível de afirmação.

## Testes

```bash
python -m pytest -q
```

O CI usa um modelo simulado para não precisar da chave Groq. Para validar ponta a ponta com a Groq real, rode ambos os processos e teste: (1) carregar PDF, (2) perguntar, (3) trocar PDF, (4) reiniciar conversa, (5) rejeitar PDF inválido. A execução manual real e o comportamento sob carga ainda precisam de evidência antes do merge.

## Limitações arquiteturais

O backend é local, single-process, sem armazenamento persistente ou autenticação. Para escala, serão necessários mecanismos de identidade e autorização, limite de taxa, processamento isolado/assíncrono, store compartilhado e observabilidade. Qdrant, DuckDB e Langfuse continuam fora do escopo desta mudança.
