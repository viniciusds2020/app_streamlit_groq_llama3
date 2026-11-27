"""Qdrant vector store adapter with local on-disk persistence by default."""
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient, models

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_PREFIX = "rag_"


@lru_cache(maxsize=1)
def _load_embeddings():
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _load_qdrant_client():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY") or None
    if url:
        return QdrantClient(url=url, api_key=api_key)
    path = os.getenv("QDRANT_PATH", "data/qdrant")
    if path != ":memory:":
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=path)


class PersistentQdrantStore(VectorStore):
    """Minimal LangChain VectorStore adapter for a single isolated collection."""

    def __init__(self, client, collection_name, embeddings=None):
        self.client = client
        self.collection_name = collection_name
        self._embeddings = embeddings or _load_embeddings()

    def add_texts(self, texts: Iterable[str], metadatas=None, ids=None, **kwargs):
        texts = list(texts)
        if not texts:
            return []
        vectors = self._embeddings.embed_documents(texts)
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]),
                    distance=models.Distance.COSINE,
                ),
            )
        identifiers = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in texts]
        metadata_values = list(metadatas) if metadatas is not None else [{} for _ in texts]
        points = [
            models.PointStruct(
                id=identifiers[index],
                vector=vectors[index],
                payload={
                    "page_content": text,
                    "metadata": {
                        key: value
                        for key, value in dict(metadata_values[index] or {}).items()
                        if key != "source"
                    },
                },
            )
            for index, text in enumerate(texts)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return identifiers

    @classmethod
    def from_texts(
        cls,
        texts,
        embedding,
        metadatas=None,
        *,
        collection_name,
        client=None,
        ids=None,
        **kwargs,
    ):
        """Create a store and index texts in the requested Qdrant collection."""
        qdrant_client = client or _load_qdrant_client()
        if qdrant_client.collection_exists(collection_name):
            qdrant_client.delete_collection(collection_name)
        store = cls(qdrant_client, collection_name, embeddings=embedding)
        store.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return store

    def similarity_search(self, query, k=4, filter=None, **kwargs):
        if not self.client.collection_exists(self.collection_name):
            return []
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=self._embeddings.embed_query(query),
            query_filter=filter,
            limit=k,
            with_payload=True,
        ).points
        documents = []
        for hit in hits:
            payload = hit.payload or {}
            documents.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata", {}),
                )
            )
        return documents


def collection_name_for(session_key, index_id):
    """Build an opaque, Qdrant-safe collection name without storing the bearer token."""
    return f"{COLLECTION_PREFIX}{session_key[:16]}_{index_id[:16]}"


def create_vector_store(documents, collection_name):
    client = _load_qdrant_client()
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    store = PersistentQdrantStore(client, collection_name)
    try:
        store.add_documents(documents)
    except Exception:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        raise
    return store


def load_vector_store(collection_name):
    client = _load_qdrant_client()
    if not client.collection_exists(collection_name):
        raise ValueError("Vector collection is unavailable")
    return PersistentQdrantStore(client, collection_name)


def delete_vector_store(collection_name):
    if not collection_name:
        return
    client = _load_qdrant_client()
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
