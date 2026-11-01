"""SQLite WAL Database Manager with FTS5 lexical search and write-through API cache.

Provides deterministic relational subspace pre-filtering and sub-millisecond
BM25 lexical scoring over official ICAR agronomic guidance documents.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

DEFAULT_DB_PATH = os.path.join("artifacts", "icar_knowledge.db")


class SQLiteKnowledgeManager:
    """Thread-safe SQLite manager configured for WAL mode concurrency."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a connection configured with WAL journal mode and NORMAL synchronous."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Creates tables, indexes, triggers, and FTS5 virtual tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Main ICAR guidance chunks table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS icardocs_chunks (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crop_name TEXT NOT NULL,
                    disease_or_pest TEXT NOT NULL,
                    growth_stage TEXT,
                    agro_climatic_zone TEXT,
                    raw_content TEXT NOT NULL,
                    chemical_dosage_json TEXT NOT NULL,
                    source_doc_title TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    gov_url TEXT NOT NULL
                );
                """
            )

            # Composite index for deterministic subspace filtering
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_crop_disease 
                ON icardocs_chunks(crop_name, disease_or_pest);
                """
            )

            # 2. FTS5 Virtual Table for BM25 keyword matching
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    raw_content,
                    content='icardocs_chunks',
                    content_rowid='chunk_id'
                );
                """
            )

            # FTS Triggers for automatic sync
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS icardocs_ai AFTER INSERT ON icardocs_chunks BEGIN
                    INSERT INTO chunks_fts(rowid, raw_content) VALUES (new.chunk_id, new.raw_content);
                END;
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS icardocs_ad AFTER DELETE ON icardocs_chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, raw_content) VALUES('delete', old.chunk_id, old.raw_content);
                END;
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS icardocs_au AFTER UPDATE ON icardocs_chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, raw_content) VALUES('delete', old.chunk_id, old.raw_content);
                    INSERT INTO chunks_fts(rowid, raw_content) VALUES (new.chunk_id, new.raw_content);
                END;
                """
            )

            # 3. Write-Through API Resilience Cache table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS api_resilience_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    last_updated_utc TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def insert_chunk(
        self,
        crop_name: str,
        disease_or_pest: str,
        raw_content: str,
        chemical_dosage_json: dict[str, Any],
        source_doc_title: str,
        page_number: int,
        gov_url: str,
        growth_stage: str | None = "All Stages",
        agro_climatic_zone: str | None = "Pan-India",
    ) -> int:
        """Inserts an ICAR document chunk and returns the chunk_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO icardocs_chunks (
                    crop_name, disease_or_pest, growth_stage, agro_climatic_zone,
                    raw_content, chemical_dosage_json, source_doc_title,
                    page_number, gov_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    crop_name.strip().lower(),
                    disease_or_pest.strip().lower(),
                    growth_stage,
                    agro_climatic_zone,
                    raw_content,
                    json.dumps(chemical_dosage_json),
                    source_doc_title,
                    page_number,
                    gov_url,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_candidate_chunk_ids(
        self, crop_name: str, disease_or_pest: str
    ) -> list[int]:
        """Subspace Filter: Returns exact candidate chunk IDs for (crop_name, disease_or_pest).

        Guarantees zero cross-contamination before vector/BM25 scoring.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT chunk_id FROM icardocs_chunks
                WHERE crop_name = ? AND disease_or_pest = ?
                ORDER BY chunk_id ASC;
                """,
                (crop_name.strip().lower(), disease_or_pest.strip().lower()),
            )
            rows = cursor.fetchall()
            return [row["chunk_id"] for row in rows]

    def search_bm25(
        self,
        query: str,
        candidate_ids: list[int] | None = None,
        limit: int = 10,
    ) -> list[tuple[int, float]]:
        """Performs BM25 lexical rank search on chunks_fts.

        Restricts results strictly to candidate_ids if provided.
        Returns list of (chunk_id, bm25_score) where score is higher for better match.
        """
        if candidate_ids is not None and len(candidate_ids) == 0:
            return []

        # Sanitize query for FTS5 syntax
        clean_tokens = [
            w for w in query.replace('"', "").replace("'", "").split() if w.isalnum()
        ]
        if not clean_tokens:
            return []
        fts_query = " OR ".join(clean_tokens)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if candidate_ids is not None:
                placeholders = ",".join(["?"] * len(candidate_ids))
                sql = f"""
                    SELECT rowid, bm25(chunks_fts) as score
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ? AND rowid IN ({placeholders})
                    ORDER BY score ASC
                    LIMIT ?;
                """
                params = [fts_query] + candidate_ids + [limit]
            else:
                sql = """
                    SELECT rowid, bm25(chunks_fts) as score
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY score ASC
                    LIMIT ?;
                """
                params = [fts_query, limit]

            try:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                # In SQLite FTS5, bm25() returns negative values where more negative is better match.
                # Convert to positive rank score: -score
                return [(row["rowid"], -float(row["score"])) for row in rows]
            except sqlite3.OperationalError:
                # Fallback if FTS5 query syntax fails
                return []

    def get_chunk_by_id(self, chunk_id: int) -> dict[str, Any] | None:
        """Retrieves complete chunk record by chunk_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM icardocs_chunks WHERE chunk_id = ?;", (chunk_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["chemical_dosage"] = json.loads(data["chemical_dosage_json"])
            return data

    def cache_get(self, cache_key: str) -> dict[str, Any] | None:
        """Retrieves cached API response payload if available, with latest fallback."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT payload_json FROM api_resilience_cache WHERE cache_key = ?;",
                (cache_key,),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row["payload_json"])
            cursor.execute(
                "SELECT payload_json FROM api_resilience_cache ORDER BY rowid DESC LIMIT 1;"
            )
            fallback_row = cursor.fetchone()
            if fallback_row:
                return json.loads(fallback_row["payload_json"])
            return None

    def cache_set(self, cache_key: str, payload: dict[str, Any]) -> None:
        """Stores or updates API response payload in write-through cache."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO api_resilience_cache (cache_key, payload_json, last_updated_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    last_updated_utc = excluded.last_updated_utc;
                """,
                (cache_key, json.dumps(payload), now_utc),
            )
            conn.commit()

    def get_all_chunks(self) -> list[dict[str, Any]]:
        """Retrieves all chunks for index building."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM icardocs_chunks ORDER BY chunk_id ASC;")
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["chemical_dosage"] = json.loads(d["chemical_dosage_json"])
                results.append(d)
            return results
