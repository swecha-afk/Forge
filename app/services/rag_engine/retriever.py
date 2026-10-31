"""Deterministic Hybrid Subspace RAG Retriever.

Combines relational SQLite pre-filtering, FAISS IDSelector dense search,
FTS5 BM25 lexical ranking, Reciprocal Rank Fusion (RRF), and an Out-of-Distribution
Confidence Gate to eliminate LLM hallucinations and cross-crop chemical contamination.
"""

import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas.swarm_schemas import (
    ICARCitation,
    RAGQueryResponse,
)
from app.services.rag_engine.db_manager import SQLiteKnowledgeManager
from app.services.risk_engine.constants import (
    CONFIDENCE_GATE_TAU,
    DEFAULT_DB_PATH,
    DEFAULT_FAISS_PATH,
    EMBEDDING_MODEL_NAME,
    RRF_K,
)


class DeterministicICARRAG:
    """Subspace-isolated agronomic retriever over verified ICAR protocols."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        faiss_path: str = DEFAULT_FAISS_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.db = SQLiteKnowledgeManager(db_path)
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{faiss_path}'. Please run ingest_icar.py first."
            )
        self.index = faiss.read_index(faiss_path)
        self.embedder = SentenceTransformer(model_name)

    def retrieve(
        self,
        crop_name: str,
        disease_or_pest: str,
        farmer_query: str,
        top_k: int = 3,
    ) -> RAGQueryResponse:
        """Executes strict 5-stage deterministic retrieval:

        Stage 1: Relational Subspace Pre-Filter via SQLite B-Tree (crop_name, disease_or_pest).
        Stage 2: Dense Vector Search restricted strictly to candidate IDs via FAISS IDSelectorBatch.
        Stage 3: BM25 Lexical Keyword Search on FTS5 within candidate IDs.
        Stage 4: Reciprocal Rank Fusion (RRF) combining dense and lexical ranks.
        Stage 5: Confidence Gate: If best cosine similarity < tau (0.60), block chemical synthesis.
        """
        crop_clean = crop_name.strip().lower()
        disease_clean = disease_or_pest.strip().lower()

        # --- Stage 1: Relational Subspace Pre-Filter ---
        candidate_ids = self.db.get_candidate_chunk_ids(crop_clean, disease_clean)
        if not candidate_ids:
            return RAGQueryResponse(
                status="SUBSPACE_EMPTY",
                crop_name=crop_clean,
                disease_or_pest=disease_clean,
                citations=[],
                top_similarity_score=0.0,
                chemical_synthesis_blocked=True,
                guardrail_message=(
                    f"Subspace Empty: No verified ICAR protocols indexed for crop '{crop_clean}' "
                    f"and condition '{disease_clean}'. Chemical dosage strictly withheld."
                ),
            )

        # --- Stage 2: Dense Vector Search with Subspace ID Restriction ---
        query_vec = self.embedder.encode(
            [farmer_query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        query_vec = np.array(query_vec, dtype=np.float32)

        # Subspace selector restricts FAISS to candidate_ids only
        id_selector = faiss.IDSelectorBatch(candidate_ids)
        search_params = faiss.SearchParameters()
        search_params.sel = id_selector

        k_search = min(len(candidate_ids), max(top_k * 2, 4))
        dense_distances, dense_indices = self.index.search(
            query_vec, k_search, params=search_params
        )

        dense_scores: dict[int, float] = {}
        dense_ranks: dict[int, int] = {}
        rank_idx = 1
        for dist, idx in zip(dense_distances[0], dense_indices[0]):
            if idx >= 0 and idx in candidate_ids:
                dense_scores[int(idx)] = float(dist)
                dense_ranks[int(idx)] = rank_idx
                rank_idx += 1

        top_dense_similarity = max(dense_scores.values()) if dense_scores else 0.0

        # --- Stage 5 Early Check: Confidence Gate ---
        if top_dense_similarity < CONFIDENCE_GATE_TAU:
            return RAGQueryResponse(
                status="LOW_CONFIDENCE_GUARDRAIL_TRIGGERED",
                crop_name=crop_clean,
                disease_or_pest=disease_clean,
                citations=[],
                top_similarity_score=float(top_dense_similarity),
                chemical_synthesis_blocked=True,
                guardrail_message=(
                    f"Low Confidence Guardrail Triggered: Top semantic similarity "
                    f"({top_dense_similarity:.3f}) is below confidence threshold ({CONFIDENCE_GATE_TAU:.2f}). "
                    f"Out-of-distribution or ambiguous query detected. "
                    f"Chemical dosage and spray synthesis blocked to prevent crop phytotoxicity."
                ),
            )

        # --- Stage 3: BM25 Lexical Keyword Search on FTS5 ---
        bm25_results = self.db.search_bm25(
            query=farmer_query,
            candidate_ids=candidate_ids,
            limit=k_search,
        )

        bm25_ranks: dict[int, int] = {}
        rank_bm25 = 1
        for cid, _ in bm25_results:
            bm25_ranks[int(cid)] = rank_bm25
            rank_bm25 += 1

        # --- Stage 4: Reciprocal Rank Fusion (RRF) ---
        all_scored_ids = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))
        rrf_scores: dict[int, float] = {}

        for cid in all_scored_ids:
            r_dense = dense_ranks.get(cid, 999)
            r_bm25 = bm25_ranks.get(cid, 999)
            rrf_score = (1.0 / (RRF_K + r_dense)) + (1.0 / (RRF_K + r_bm25))
            rrf_scores[cid] = rrf_score

        # Sort candidate chunks by fused RRF score descending
        sorted_candidates = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        citations: list[ICARCitation] = []
        for cid, score in sorted_candidates:
            chunk = self.db.get_chunk_by_id(cid)
            if not chunk:
                continue
            chem = chunk.get("chemical_dosage", {})
            citations.append(
                ICARCitation(
                    chunk_id=chunk["chunk_id"],
                    source_doc_title=chunk["source_doc_title"],
                    page_number=chunk["page_number"],
                    gov_url=chunk["gov_url"],
                    crop_name=chunk["crop_name"],
                    disease_or_pest=chunk["disease_or_pest"],
                    active_ingredient=chem.get("active_ingredient", "N/A"),
                    dilution_dosage=chem.get("dilution_dosage", "N/A"),
                    phi_days=int(chem.get("phi_days", 0)),
                    raw_content=chunk["raw_content"],
                    score=float(score),
                )
            )

        return RAGQueryResponse(
            status="SUCCESS",
            crop_name=crop_clean,
            disease_or_pest=disease_clean,
            citations=citations,
            top_similarity_score=float(top_dense_similarity),
            chemical_synthesis_blocked=False,
            guardrail_message=None,
        )
