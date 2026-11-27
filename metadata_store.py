"""DuckDB-backed metadata and conversation persistence for the local RAG MVP."""
import json
import os
import threading
import time
from pathlib import Path

import duckdb


class MetadataStore:
    """Small serialized DuckDB repository; bearer tokens are stored only as hashes."""

    def __init__(self, path=None):
        self.path = path or os.getenv("RAG_METADATA_DB", "data/rag.duckdb")
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(self.path)
        self._lock = threading.RLock()
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_sessions (
                    session_key VARCHAR PRIMARY KEY,
                    document_id VARCHAR,
                    collection_name VARCHAR,
                    history_json VARCHAR NOT NULL,
                    touched DOUBLE NOT NULL
                )
                """
            )

    def create_session(self, session_key):
        now = time.time()
        with self._lock:
            self._connection.execute(
                "INSERT INTO rag_sessions VALUES (?, NULL, NULL, ?, ?)",
                [session_key, "[]", now],
            )

    def count_sessions(self):
        with self._lock:
            return self._connection.execute("SELECT COUNT(*) FROM rag_sessions").fetchone()[0]

    def get_session(self, session_key):
        with self._lock:
            row = self._connection.execute(
                """
                SELECT document_id, collection_name, history_json, touched
                FROM rag_sessions WHERE session_key = ?
                """,
                [session_key],
            ).fetchone()
        if row is None:
            return None
        return {
            "document_id": row[0],
            "collection_name": row[1],
            "history": json.loads(row[2]),
            "touched": row[3],
        }

    def touch(self, session_key, touched=None):
        with self._lock:
            self._connection.execute(
                "UPDATE rag_sessions SET touched = ? WHERE session_key = ?",
                [touched or time.time(), session_key],
            )

    def replace_document(self, session_key, document_id, collection_name):
        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                UPDATE rag_sessions
                SET document_id = ?, collection_name = ?, history_json = '[]', touched = ?
                WHERE session_key = ?
                """,
                [document_id, collection_name, now, session_key],
            )
            row = self._connection.execute(
                "SELECT COUNT(*) FROM rag_sessions WHERE session_key = ?",
                [session_key],
            ).fetchone()
        if row[0] != 1:
            raise KeyError("Session no longer exists")

    def save_history(self, session_key, history):
        with self._lock:
            self._connection.execute(
                "UPDATE rag_sessions SET history_json = ?, touched = ? WHERE session_key = ?",
                [json.dumps(history, ensure_ascii=False), time.time(), session_key],
            )

    def delete_session(self, session_key):
        with self._lock:
            row = self._connection.execute(
                "SELECT collection_name FROM rag_sessions WHERE session_key = ?",
                [session_key],
            ).fetchone()
            self._connection.execute(
                "DELETE FROM rag_sessions WHERE session_key = ?",
                [session_key],
            )
        return None if row is None else row[0]

    def delete_expired(self, cutoff):
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT session_key, collection_name
                FROM rag_sessions WHERE touched < ?
                """,
                [cutoff],
            ).fetchall()
            self._connection.execute("DELETE FROM rag_sessions WHERE touched < ?", [cutoff])
        return rows

    def close(self):
        with self._lock:
            self._connection.close()
