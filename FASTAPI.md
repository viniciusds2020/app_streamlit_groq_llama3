# FastAPI RAG — execução local

Esta branch separa a interface Streamlit da orquestração RAG. O backend usa Qdrant para vetores, DuckDB para metadados e histórico, FastEmbed para embeddings e Groq para geração; a UI usa HTTP. **Somente para localhost**: tokens de sessão são credenciais bearer e não há autenticação/autorização de usuários. Nunca exponha a API na internet assim.

## Preparação

Recomendado Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="sua_chave"
export GROQ_MODEL="modelo-disponivel-na-sua-conta"
```

Em Windows PowerShell: `.venv\Scripts\Activate.ps1`; configure `$env:GROQ_API_KEY`, `$env:GROQ_MODEL` e, se usar Qdrant remoto, `$env:QDRANT_URL` e `$env:QDRANT_API_KEY`.

Langfuse é opcional. Sem as duas chaves, o RAG não cria traces e continua funcionando.

### Langfuse self-hosted local

No Windows, use Docker Desktop com WSL 2 e rode estes passos em WSL ou Git Bash. O procedimento oficial mantém o Langfuse em outro diretório/repositório, separado deste app:

1. Siga o [guia oficial de Docker Compose](https://langfuse.com/self-hosting/deployment/docker-compose) para clonar o repositório Langfuse e gerar o arquivo local `.env` com segredos aleatórios. Não reutilize segredos de exemplo.
2. Na pasta do clone do Langfuse, inicie a stack com `docker compose up -d`. Aguarde o health check responder e abra `http://localhost:3000`.
3. Crie o primeiro usuário, uma organização e um projeto na interface. Copie as chaves públicas e secretas do projeto. Não as envie no chat nem as adicione ao Git.
4. No PowerShell do app, configure as variáveis na sessão local:

```powershell
$env:LANGFUSE_PUBLIC_KEY="pk-lf-..."
$env:LANGFUSE_SECRET_KEY="sk-lf-..."
$env:LANGFUSE_BASE_URL="http://localhost:3000"
python scripts/langfuse_smoke.py
```

O script valida autenticação e envia o span `rag.langfuse_smoke`, sem conteúdo do usuário. Confira o trace no projeto Langfuse antes de iniciar a API; mantenha a stack self-hosted acessível apenas localmente.

Para parar o Langfuse sem apagar os dados, rode `docker compose down` no clone do Langfuse. Evite `docker compose down -v`: a opção `-v` remove os volumes persistentes. A documentação oficial recomenda Docker Compose para uso local/teste em máquina única; não oferece alta disponibilidade nem backup automatizado.

A integração do RAG envia apenas metadados operacionais: ID da requisição, impressão digital do ID da sessão, tamanho da pergunta, duração, quantidade de páginas recuperadas e resultado. Texto da pergunta, resposta e conteúdo do PDF não são enviados por esse trace.

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

## Persistência Qdrant e DuckDB

Por padrão, o Qdrant local mantém vetores em `data/qdrant` e o DuckDB mantém sessões e histórico em `data/rag.duckdb`. Essas pastas são locais e ignoradas pelo Git. O Qdrant em disco guarda os trechos de texto extraídos e seus vetores, não o PDF original; o DuckDB guarda o histórico de perguntas e respostas. Proteja ambos como dados do usuário e faça backup se precisar manter o índice e as conversas.

Para usar um servidor Qdrant, defina `QDRANT_URL` e, quando necessário, `QDRANT_API_KEY`; `QDRANT_PATH` é usado quando `QDRANT_URL` está vazio. Configure `RAG_METADATA_DB` para mover o arquivo DuckDB. IDs de sessão são armazenados como hash. O backend continua single-process; não use o modo local com múltiplos workers.

## Endpoints

- `POST /api/v1/sessions`: cria sessão local de acesso por token.
- `POST /api/v1/sessions/{session_id}/documents`: upload de um PDF; substitui índice e histórico da sessão.
- `POST /api/v1/sessions/{session_id}/questions`: JSON `{"question":"..."}` retorna `answer`, `pages`, `document_id` e `request_id`.
- `DELETE /api/v1/sessions/{session_id}`: elimina estado da sessão.
- `GET /health/live`, `GET /health/ready`: estado básico do processo (não testam serviços externos).

Todas as respostas HTTP incluem o cabeçalho `X-Request-ID`; respostas de sucesso da API também incluem `request_id` no JSON quando há corpo. Erros usam o mesmo ID para facilitar diagnóstico.

## Limites e observabilidade

PDFs devem respeitar as restrições preexistentes de tamanho, páginas e texto. O arquivo é lido com limite de bytes, e requisições de upload com `Content-Length` acima do limite mais a margem multipart são recusadas antes do parsing. Sessões expiram após inatividade (padrão: 3600 s); limite padrão de 100 sessões no processo. Operações pesadas simultâneas são limitadas a 2 por processo por padrão; configure `RAG_MAX_CONCURRENT_OPERATIONS` para ajustar. Os vetores, metadados da sessão e histórico sobrevivem ao reinício do backend. A expiração de sessão remove também a coleção Qdrant associada.

Com Langfuse configurado, cada resposta gera um span `rag.answer` com duração, páginas recuperadas e erros sanitizados. Não há tracing de ingestão, conteúdo ou custo/token nesta etapa. A resposta apresenta páginas recuperadas, não citações verificadas em nível de afirmação.

## Testes

```bash
python -m pytest -q
```

O CI usa um modelo simulado, Qdrant em memória e DuckDB temporário; não precisa de chave Groq nem de chaves Langfuse. Para validar ponta a ponta com a Groq real, rode ambos os processos e teste: (1) carregar PDF, (2) perguntar, (3) trocar PDF, (4) reiniciar conversa, (5) rejeitar PDF inválido.

## Limitações arquiteturais

O backend é local, single-process e sem autenticação/autorização de usuários; Qdrant e DuckDB persistem os dados localmente. O limite via `Content-Length` não substitui um limite de corpo no proxy/servidor, especialmente para uploads sem esse cabeçalho. Para escala, serão necessários mecanismos de identidade e autorização, limite de taxa e processamento isolado/assíncrono. O modo Qdrant local e o arquivo DuckDB não são criptografados automaticamente.
