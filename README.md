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


## Limites e segurança do upload (OpenSpec P0)

O processamento aplica validação de bytes e estrutura do PDF, rejeita PDFs protegidos
por senha, limita páginas e caracteres extraídos e exclui o arquivo temporário ao
terminar ou falhar. São limites operacionais configuráveis por variáveis de ambiente:

| Variável | Padrão | Efeito |
| --- | ---: | --- |
| `RAG_MAX_PDF_BYTES` | `10485760` (10 MiB) | Rejeita uploads acima do limite antes de analisar o PDF |
| `RAG_MAX_PDF_PAGES` | `100` | Rejeita documentos com mais páginas |
| `RAG_MAX_EXTRACTED_CHARS` | `1000000` | Rejeita texto excessivo antes de indexar |

O Streamlit mantém seu próprio limite de upload, que deve ser alinhado à
configuração do aplicativo. **Essas verificações não são um sandbox de PDF:**
em produção, configure limites de CPU, memória e tempo no processo/contêiner,
além de autenticação, autorização e políticas de retenção adequadas.

Exceções inesperadas são apresentadas ao usuário sem detalhes internos; logs
registram apenas tipo da exceção e identificador de incidente, sem traceback,
chave Groq, prompt ou conteúdo dos documentos. O comportamento de referência
por página ainda não garante fidelidade de cada afirmação do modelo.

Testes de segurança: `pytest -q tests/test_pdf_security.py`.
Critérios e demais tarefas: `openspec/changes/harden-and-validate-rag/`.
