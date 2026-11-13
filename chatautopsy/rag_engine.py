"""
ChatAutopsy — Hybrid RAG Engine.

  * Indicizzazione duale:
      - FTS5 (BM25, tokenizer unicode61) su sessioni e singoli messaggi.
      - Embedding densi delle sessioni con sentence-transformers su MPS (fallback CUDA/CPU),
        salvati come BLOB float32 + tabella vec0 di sqlite-vec (se disponibile).
  * Ricerca ibrida: BM25 e similarità coseno eseguite in parallelo e fuse con
    Reciprocal Rank Fusion:  RRF(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_vec(d)),  k = 60.
"""
from __future__ import annotations

import gc
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from chatautopsy import config, database

_EMBEDDER_CACHE: Dict[Tuple[str, str], Any] = {}
_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER_FAILURES: Dict[str, Tuple[float, str]] = {}


class EmbedderNotCached(RuntimeError):
    """Il modello di embedding non è nella cache locale e il download non è consentito (fase di query)."""
_MATRIX_CACHE: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}
_MATRIX_LOCK = threading.Lock()

_FTS_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Privacy: nessuna telemetria verso Hugging Face.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Device & modello
# ---------------------------------------------------------------------------
def get_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_embedder(model_name: Optional[str] = None, device: Optional[str] = None, allow_download: bool = False):
    """Carica il modello di embedding.

    - Sempre prima dalla cache locale (zero rete).
    - allow_download=True solo in ingestione: download una tantum se il modello non è in cache.
    - In fase di query (allow_download=False) un modello assente fallisce subito (RuntimeError) e la
      ricerca ibrida ripiega su BM25, invece di restare appesa alla rete.
    """
    model_name = model_name or config.EMBEDDING_MODEL
    device = device or get_device()
    key = (model_name, device)
    if key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[key]
    if not allow_download and not is_model_cached(model_name):
        # verifica leggera sulla cache HF: evita di importare torch/sentence-transformers (10-30s a freddo)
        raise EmbedderNotCached(
            f"Modello di embedding '{model_name}' non presente nella cache locale: "
            "ricerca solo BM25 (rigenera gli embedding dall'ingestione per scaricarlo)."
        )

    from sentence_transformers import SentenceTransformer

    def _load(dev: str):
        try:
            return SentenceTransformer(model_name, device=dev, local_files_only=True)
        except Exception:
            if not allow_download:
                raise EmbedderNotCached(
                    f"Modello di embedding '{model_name}' non presente nella cache locale: "
                    "ricerca solo BM25 (rigenera gli embedding dall'ingestione per scaricarlo)."
                )
            return SentenceTransformer(model_name, device=dev)

    with _EMBEDDER_LOCK:
        if key in _EMBEDDER_CACHE:
            return _EMBEDDER_CACHE[key]
        last_fail = _EMBEDDER_FAILURES.get(model_name)
        if not allow_download and last_fail and time.time() - last_fail[0] < 300:
            raise EmbedderNotCached(last_fail[1])
        try:
            try:
                model = _load(device)
            except EmbedderNotCached:
                raise
            except Exception:
                if device == "cpu":
                    raise
                model = _load("cpu")  # fallback CPU se il device accelerato non è utilizzabile
        except Exception as exc:
            _EMBEDDER_FAILURES[model_name] = (time.time(), str(exc))
            raise
        _EMBEDDER_FAILURES.pop(model_name, None)
        _apply_max_seq_length(model)
        _EMBEDDER_CACHE[key] = model  # chiave richiesta: niente ricaricamenti se si è ripiegato su CPU
        return model


def is_model_cached(model_name: str) -> bool:
    """True se il modello è una cartella locale o è già nella cache Hugging Face (nessun accesso di rete)."""
    if Path(model_name).expanduser().is_dir():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return True  # impossibile verificare: lascia decidere a SentenceTransformer(local_files_only=True)
    def cached(fname: str) -> bool:
        try:
            path = try_to_load_from_cache(model_name, fname)
            return isinstance(path, str) and os.path.getsize(path) > 0
        except Exception:
            return False

    # servono sia la configurazione sia i pesi (un download interrotto lascia solo i .json)
    has_config = cached("modules.json") or cached("config.json")
    has_weights = cached("model.safetensors") or cached("pytorch_model.bin")
    return has_config and has_weights


_PREWARM_STARTED: set = set()


def prewarm_embedder(model_name: Optional[str]) -> bool:
    """Carica in background (thread daemon) il modello di embedding di un DB appena selezionato,
    così la prima ricerca semantica non paga import di torch + caricamento pesi. Idempotente."""
    if not model_name or model_name in _PREWARM_STARTED or not is_model_cached(model_name):
        return False
    _PREWARM_STARTED.add(model_name)

    def _run() -> None:
        try:
            load_embedder(model_name)
        except Exception:
            pass  # la ricerca ripiegherà su BM25 come di consueto

    threading.Thread(target=_run, name=f"prewarm-{model_name}", daemon=True).start()
    return True


def stored_embedding_model(db_path: str | Path) -> Optional[str]:
    """Modello usato per gli embedding del DB (None se il DB non ha vettori)."""
    try:
        with database.connect(db_path, read_only=True) as conn:
            if conn.execute("SELECT COUNT(*) FROM (SELECT 1 FROM session_embeddings LIMIT 1)").fetchone()[0] == 0:
                return None
            return database.get_meta(conn, "embedding_model")
    except Exception:
        return None


def _model_position_limit(model) -> int:
    """Limite posizionale reale del transformer (8192 per bge-m3, 512 per e5-large, 128 per MiniLM)."""
    try:
        return int(model[0].auto_model.config.max_position_embeddings) - 2
    except Exception:
        return int(getattr(model, "max_seq_length", 8192) or 8192)


def _apply_max_seq_length(model) -> None:
    """Fissa esplicitamente max_seq_length = 1024 (EMBEDDING_MAX_SEQ_LENGTH).

    bge-m3 accetta 8192 token ma l'attenzione è quadratica: su MPS sessioni lunghe saturano la
    Unified Memory. 1024 token coprono abbondantemente una sessione da 45 messaggi. Per modelli con
    limite posizionale inferiore (es. e5-large = 512) si usa quel limite per non andare fuori range.
    """
    model.max_seq_length = min(config.EMBEDDING_MAX_SEQ_LENGTH, _model_position_limit(model))


def _release_memory(device: str) -> None:
    """Restituisce la memoria al sistema: utile mentre gira in parallelo l'LLM locale."""
    gc.collect()
    try:
        import torch

        if device.startswith("mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def build_fts_query(query: str) -> str:
    """Trasforma testo libero in una query FTS5 sicura.

    - frasi tra virgolette => match di frase esatta
    - altrimenti token OR (privilegiando i non-stopword); i token lunghi usano anche il prefisso.
    """
    phrases = re.findall(r'"([^"]+)"', query)
    rest = re.sub(r'"[^"]+"', " ", query)
    parts: List[str] = []
    for ph in phrases:
        toks = _FTS_TOKEN_RE.findall(ph.lower())
        if toks:
            parts.append('"' + " ".join(toks) + '"')
    tokens = [t.lower() for t in _FTS_TOKEN_RE.findall(rest)]
    stop = set(config.STOPWORDS)
    content = [t for t in tokens if t not in stop and len(t) > 1] or [t for t in tokens if len(t) > 1]
    for t in dict.fromkeys(content):
        parts.append(f'"{t}"*' if len(t) >= 5 else f'"{t}"')
    return " OR ".join(parts)


# ---------------------------------------------------------------------------
# Motore
# ---------------------------------------------------------------------------
class HybridRAG:
    def __init__(self, db_path: str | Path, model_name: Optional[str] = None):
        self.db_path = str(db_path)
        self.model_name = model_name or self._stored_model() or config.EMBEDDING_MODEL

    def _stored_model(self) -> Optional[str]:
        try:
            with database.connect(self.db_path, read_only=True) as conn:
                return database.get_meta(conn, "embedding_model")
        except Exception:
            return None

    # ------------------------------------------------------------ indexing
    def index_fts(self) -> Dict[str, int]:
        with database.connect(self.db_path) as conn:
            conn.execute("DELETE FROM fts_messages")
            conn.execute("DELETE FROM fts_sessions")
            conn.execute(
                "INSERT INTO fts_messages(rowid, text, sender) SELECT id, text, sender FROM messages WHERE is_media = 0 AND text <> ''"
            )
            conn.execute("INSERT INTO fts_sessions(rowid, transcript) SELECT session_id, transcript FROM sessions")
            conn.execute("INSERT INTO fts_messages(fts_messages) VALUES('optimize')")
            conn.execute("INSERT INTO fts_sessions(fts_sessions) VALUES('optimize')")
            n_msg = conn.execute("SELECT COUNT(*) FROM fts_messages").fetchone()[0]
            n_ses = conn.execute("SELECT COUNT(*) FROM fts_sessions").fetchone()[0]
        return {"messages": int(n_msg), "sessions": int(n_ses)}

    @staticmethod
    def _embedding_text(transcript: str) -> str:
        # Rimuove i timestamp per risparmiare token: "[YYYY-MM-DD HH:MM] Nome: testo" -> "Nome: testo"
        return re.sub(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] ", "", transcript, flags=re.MULTILINE)

    def build_vector_index(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        force: bool = True,
    ) -> int:
        model = load_embedder(self.model_name, allow_download=True)  # ingestione: download una tantum consentito
        _apply_max_seq_length(model)
        device = str(model.device)
        with database.connect(self.db_path) as conn:
            if force:
                conn.execute("DELETE FROM session_embeddings")
            rows = conn.execute(
                "SELECT session_id, transcript FROM sessions WHERE session_id NOT IN (SELECT session_id FROM session_embeddings)"
            ).fetchall()
        total = len(rows)
        if total == 0:
            return 0
        prefix = "passage: " if _is_e5(self.model_name) else ""
        items = [(r["session_id"], prefix + self._embedding_text(r["transcript"])) for r in rows]
        items.sort(key=lambda x: len(x[1]), reverse=True)  # meno padding per batch

        conn = database.get_connection(self.db_path)
        vec_ok = database.load_sqlite_vec(conn)
        dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        dim = dim_fn()
        if vec_ok:
            try:
                database.ensure_vec_table(conn, dim)
                if force:
                    conn.execute("DELETE FROM vec_sessions")
            except Exception:
                vec_ok = False
        database.set_meta(conn, "embedding_model", self.model_name)
        database.set_meta(conn, "embedding_dim", str(dim))
        database.set_meta(conn, "vector_backend", "sqlite-vec" if vec_ok else "numpy")
        conn.commit()

        done = 0
        pending = 0
        since_clear = 0
        try:
            for start in range(0, total, batch_size):
                batch = items[start:start + batch_size]
                texts = [t for _, t in batch]
                try:
                    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=False)
                except RuntimeError:
                    # OOM su MPS con le sessioni più lunghe: libera memoria e riprova una alla volta
                    _release_memory(device)
                    vecs = np.vstack([
                        model.encode([t], batch_size=1, normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False) for t in texts
                    ])
                vecs = vecs.astype(np.float32)
                conn.executemany(
                    "INSERT OR REPLACE INTO session_embeddings(session_id, model, dim, embedding) VALUES (?, ?, ?, ?)",
                    [(int(sid), self.model_name, int(dim), vecs[i].tobytes()) for i, (sid, _) in enumerate(batch)],
                )
                if vec_ok:
                    conn.executemany(
                        "INSERT OR REPLACE INTO vec_sessions(rowid, embedding) VALUES (?, ?)",
                        [(int(sid), vecs[i].tobytes()) for i, (sid, _) in enumerate(batch)],
                    )
                del vecs
                done += len(batch)
                pending += len(batch)
                since_clear += len(batch)
                if pending >= config.EMBEDDING_COMMIT_EVERY:
                    conn.commit()
                    pending = 0
                if since_clear >= config.EMBEDDING_CACHE_CLEAR_EVERY:
                    _release_memory(device)
                    since_clear = 0
                if progress_callback:
                    progress_callback(done, total, device)
            conn.commit()
        finally:
            conn.close()
            _release_memory(device)  # fine indicizzazione: niente memoria bloccata per l'LLM locale
        with _MATRIX_LOCK:
            _MATRIX_CACHE.pop(self.db_path, None)
        return done

    def has_vectors(self) -> bool:
        with database.connect(self.db_path, read_only=True) as conn:
            return conn.execute("SELECT COUNT(*) FROM session_embeddings").fetchone()[0] > 0

    # ------------------------------------------------------------- filtri
    @staticmethod
    def _filter_sql(sender_filter: Optional[str], start_date: Optional[str], end_date: Optional[str],
                    min_conflict_score: float, alias: str = "s") -> Tuple[str, List[Any]]:
        clauses, params = [], []
        if sender_filter:
            clauses.append(f"EXISTS (SELECT 1 FROM json_each({alias}.participants) WHERE value = ?)")
            params.append(sender_filter)
        if start_date:
            clauses.append(f"{alias}.start_time >= ?")
            params.append(f"{str(start_date)[:10]} 00:00:00")
        if end_date:
            clauses.append(f"{alias}.start_time <= ?")
            params.append(f"{str(end_date)[:10]} 23:59:59")
        if min_conflict_score and float(min_conflict_score) > 0:
            clauses.append(f"{alias}.conflict_score >= ?")
            params.append(float(min_conflict_score))
        return (" AND ".join(clauses) if clauses else "1=1"), params

    # -------------------------------------------------------------- BM25
    def bm25_search(self, query: str, limit: int = config.RAG_CANDIDATE_POOL, sender_filter: Optional[str] = None,
                    start_date: Optional[str] = None, end_date: Optional[str] = None,
                    min_conflict_score: float = 0) -> List[Dict[str, Any]]:
        fts_q = build_fts_query(query)
        if not fts_q:
            return []
        where, params = self._filter_sql(sender_filter, start_date, end_date, min_conflict_score)
        sql = (
            "SELECT s.session_id, bm25(fts_sessions) AS bm25_score, "
            "snippet(fts_sessions, 0, '«', '»', ' … ', 24) AS snippet "
            "FROM fts_sessions JOIN sessions s ON s.session_id = fts_sessions.rowid "
            f"WHERE fts_sessions MATCH ? AND {where} ORDER BY bm25_score LIMIT ?"
        )
        with database.connect(self.db_path, read_only=True) as conn:
            try:
                rows = conn.execute(sql, [fts_q, *params, int(limit)]).fetchall()
            except Exception:
                return []
        return [{"session_id": int(r["session_id"]), "bm25_score": float(r["bm25_score"]), "snippet": r["snippet"]} for r in rows]

    def search_messages(self, query: str, limit: int = 30, sender_filter: Optional[str] = None,
                        start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        fts_q = build_fts_query(query)
        if not fts_q:
            return []
        clauses, params = [], []
        if sender_filter:
            clauses.append("m.sender = ?")
            params.append(sender_filter)
        if start_date:
            clauses.append("m.timestamp >= ?")
            params.append(f"{str(start_date)[:10]} 00:00:00")
        if end_date:
            clauses.append("m.timestamp <= ?")
            params.append(f"{str(end_date)[:10]} 23:59:59")
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = (
            "SELECT m.id, m.session_id, m.timestamp, m.sender, m.text, bm25(fts_messages) AS score "
            "FROM fts_messages JOIN messages m ON m.id = fts_messages.rowid "
            f"WHERE fts_messages MATCH ? AND {where} ORDER BY score LIMIT ?"
        )
        with database.connect(self.db_path, read_only=True) as conn:
            try:
                rows = conn.execute(sql, [fts_q, *params, int(limit)]).fetchall()
            except Exception:
                return []
        return [
            {"message_id": int(r["id"]), "session_id": int(r["session_id"]),
             "citation": f'[{r["timestamp"][:16]}] {r["sender"]}: "{r["text"][:400]}"', "bm25_score": float(r["score"])}
            for r in rows
        ]

    # ------------------------------------------------------------ vettori
    def _load_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        sig = database.db_signature(self.db_path)
        with _MATRIX_LOCK:
            cached = _MATRIX_CACHE.get(self.db_path)
            if cached and cached[0] == sig:
                return cached[1], cached[2]
        with database.connect(self.db_path, read_only=True) as conn:
            rows = conn.execute("SELECT session_id, dim, embedding FROM session_embeddings ORDER BY session_id").fetchall()
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 1), dtype=np.float32)
        ids = np.array([r["session_id"] for r in rows], dtype=np.int64)
        dim = int(rows[0]["dim"])
        mat = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32).reshape(len(rows), dim)
        with _MATRIX_LOCK:
            _MATRIX_CACHE[self.db_path] = (sig, ids, mat)
        return ids, mat

    def embed_query(self, query: str) -> np.ndarray:
        model = load_embedder(self.model_name)
        prefix = "query: " if _is_e5(self.model_name) else ""
        vec = model.encode([prefix + query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        return vec.astype(np.float32)

    def vector_search(self, query: str, limit: int = config.RAG_CANDIDATE_POOL, sender_filter: Optional[str] = None,
                      start_date: Optional[str] = None, end_date: Optional[str] = None,
                      min_conflict_score: float = 0) -> List[Dict[str, Any]]:
        if not self.has_vectors():
            return []
        qvec = self.embed_query(query)
        has_filters = bool(sender_filter or start_date or end_date or (min_conflict_score and min_conflict_score > 0))

        if not has_filters:
            with database.connect(self.db_path, read_only=True) as conn:
                if database.load_sqlite_vec(conn):
                    try:
                        rows = conn.execute(
                            "SELECT rowid, distance FROM vec_sessions WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                            (qvec.tobytes(), int(limit)),
                        ).fetchall()
                        if rows:
                            return [{"session_id": int(r[0]), "similarity": round(1.0 - float(r[1]), 5)} for r in rows]
                    except Exception:
                        pass

        ids, mat = self._load_matrix()
        if len(ids) == 0 or mat.shape[1] != qvec.shape[0]:
            return []
        if has_filters:
            where, params = self._filter_sql(sender_filter, start_date, end_date, min_conflict_score)
            with database.connect(self.db_path, read_only=True) as conn:
                allowed = np.array([r[0] for r in conn.execute(f"SELECT s.session_id FROM sessions s WHERE {where}", params)],
                                   dtype=np.int64)
            mask = np.isin(ids, allowed)
            if not mask.any():
                return []
            sub_ids, sub_mat = ids[mask], mat[mask]
        else:
            sub_ids, sub_mat = ids, mat
        sims = sub_mat @ qvec
        k = min(int(limit), len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [{"session_id": int(sub_ids[i]), "similarity": round(float(sims[i]), 5)} for i in top]

    # ------------------------------------------------------------- ibrida
    def hybrid_search(self, query: str, top_k: int = 8, sender_filter: Optional[str] = None,
                      start_date: Optional[str] = None, end_date: Optional[str] = None,
                      min_conflict_score: float = 0, include_transcript: bool = True) -> List[Dict[str, Any]]:
        pool = max(config.RAG_CANDIDATE_POOL, top_k * 6)
        kwargs = dict(limit=pool, sender_filter=sender_filter, start_date=start_date, end_date=end_date,
                      min_conflict_score=min_conflict_score)
        vector_error = None
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_bm25 = ex.submit(self.bm25_search, query, **kwargs)
            f_vec = ex.submit(self.vector_search, query, **kwargs)
            bm25 = f_bm25.result()
            try:
                vec = f_vec.result()
            except Exception as exc:  # modello embedding non disponibile => solo BM25
                vec, vector_error = [], f"{type(exc).__name__}: {exc}"

        k = config.RRF_K
        fused: Dict[int, Dict[str, Any]] = {}
        for rank, r in enumerate(bm25, start=1):
            d = fused.setdefault(r["session_id"], {"session_id": r["session_id"], "rrf_score": 0.0})
            d["rrf_score"] += 1.0 / (k + rank)
            d["bm25_rank"] = rank
            d["bm25_score"] = round(r["bm25_score"], 4)
            d["snippet"] = r["snippet"]
        for rank, r in enumerate(vec, start=1):
            d = fused.setdefault(r["session_id"], {"session_id": r["session_id"], "rrf_score": 0.0})
            d["rrf_score"] += 1.0 / (k + rank)
            d["vector_rank"] = rank
            d["similarity"] = r["similarity"]
        ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]
        if not ranked:
            return []
        ids = [r["session_id"] for r in ranked]
        cols = "session_id, start_time, end_time, message_count, participants, initiator, conflict_score, emotional_intensity_score"
        if include_transcript:
            cols += ", transcript"
        with database.connect(self.db_path, read_only=True) as conn:
            meta = {
                int(r["session_id"]): dict(r)
                for r in conn.execute(f"SELECT {cols} FROM sessions WHERE session_id IN ({','.join('?' * len(ids))})", ids)
            }
        out = []
        for r in ranked:
            m = meta.get(r["session_id"])
            if not m:
                continue
            try:
                m["participants"] = json.loads(m["participants"])
            except (TypeError, json.JSONDecodeError):
                pass
            m.update({
                "rrf_score": round(r["rrf_score"], 6),
                "bm25_rank": r.get("bm25_rank"),
                "vector_rank": r.get("vector_rank"),
                "similarity": r.get("similarity"),
                "snippet": r.get("snippet"),
            })
            if vector_error:
                m["vector_search_note"] = vector_error
            out.append(m)
        return out


def hybrid_search(db_path: str | Path, query: str, top_k: int = 8, sender_filter: Optional[str] = None,
                  start_date: Optional[str] = None, end_date: Optional[str] = None,
                  min_conflict_score: float = 0) -> List[Dict[str, Any]]:
    return HybridRAG(db_path).hybrid_search(query, top_k=top_k, sender_filter=sender_filter, start_date=start_date,
                                            end_date=end_date, min_conflict_score=min_conflict_score)
