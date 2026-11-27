import hashlib

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from metadata_store import MetadataStore
from retrieval import PersistentQdrantStore


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(len(text)), float(text.count("a") + 1), 1.0]


def test_qdrant_store_adds_and_searches_documents():
    client = QdrantClient(":memory:")
    store = PersistentQdrantStore(
        client=client,
        collection_name="rag_storage_test",
        embeddings=FakeEmbeddings(),
    )

    store.add_texts(
        ["texto sobre dados", "conteúdo sobre saúde"],
        metadatas=[{"page": 0}, {"page": 1, "source": "/tmp/private.pdf"}],
    )
    results = store.similarity_search("dados", k=2)

    assert len(results) == 2
    assert all(isinstance(result.page_content, str) for result in results)
    assert all("source" not in result.metadata for result in results)
    client.close()


def test_duckdb_persists_session_metadata_and_history(tmp_path):
    database_path = str(tmp_path / "rag.duckdb")
    token = "bearer-session-token"
    session_key = hashlib.sha256(token.encode()).hexdigest()

    store = MetadataStore(database_path)
    store.create_session(session_key)
    store.replace_document(session_key, "doc-hash", "rag_abc_index")
    store.save_history(session_key, [("pergunta", "resposta")])
    store.close()

    reopened = MetadataStore(database_path)
    stored = reopened.get_session(session_key)
    assert stored["document_id"] == "doc-hash"
    assert stored["collection_name"] == "rag_abc_index"
    assert stored["history"] == [["pergunta", "resposta"]]
    assert reopened.delete_session(session_key) == "rag_abc_index"
    reopened.close()
