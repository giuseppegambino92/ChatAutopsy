"""Embedding reali + ricerca ibrida (saltato se il modello leggero non è nella cache locale)."""
from __future__ import annotations

import pytest

from chatautopsy import config, demo, rag_engine
from chatautopsy.ingestion import run_ingestion

pytestmark = pytest.mark.embeddings


def test_hybrid_search_with_real_embeddings(data_dir):
    pytest.importorskip("sentence_transformers")
    if not rag_engine.is_model_cached(config.EMBEDDING_LIGHT_MODEL):
        pytest.skip("modello di embedding non in cache")
    db = data_dir / "emb.db"
    res = run_ingestion(None, db, file_bytes=demo.couple_chat(days=60).encode(), source_name="emb.txt",
                        build_embeddings=True, embedding_model=config.EMBEDDING_LIGHT_MODEL)
    assert res["embedded_sessions"] == res["sessions"] and not res["embedding_error"]
    hits = rag_engine.HybridRAG(db).hybrid_search("litigio e scuse", top_k=5)
    assert hits and all(h.get("vector_rank") is not None or h.get("bm25_rank") is not None for h in hits)


def test_missing_model_falls_back_to_bm25_without_network():
    with pytest.raises(rag_engine.EmbedderNotCached):
        rag_engine.load_embedder("example-org/modello-inesistente")
