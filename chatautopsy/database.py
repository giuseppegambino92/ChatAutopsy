"""
ChatAutopsy — Layer di persistenza SQLite.

Un file .db per ogni chat analizzata (in DATA_DIR). Ogni database contiene:
  * messages            — messaggi raw + feature per messaggio
  * sessions            — sessioni conversazionali + score euristici + trascrizione teatrale
  * fts_messages        — FTS5 (BM25) sui singoli messaggi
  * fts_sessions        — FTS5 (BM25) sulle trascrizioni di sessione
  * session_embeddings  — embedding float32 (BLOB) delle sessioni (sempre presente, fallback NumPy)
  * vec_sessions        — tabella vec0 di sqlite-vec (se l'estensione è caricabile)
  * meta                — metadati di ingestione (sorgente, modello embedding, dimensione, ...)
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from chatautopsy import config

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id                INTEGER PRIMARY KEY,
    start_time                TEXT NOT NULL,
    end_time                  TEXT NOT NULL,
    first_message_id          INTEGER NOT NULL,
    last_message_id           INTEGER NOT NULL,
    message_count             INTEGER NOT NULL,
    overlap_count             INTEGER NOT NULL DEFAULT 0,
    is_continuation           INTEGER NOT NULL DEFAULT 0,
    participant_count         INTEGER NOT NULL,
    participants              TEXT NOT NULL,
    initiator                 TEXT,
    duration_minutes          REAL,
    messages_per_minute       REAL,
    wall_of_text_count        INTEGER,
    cold_reply_count          INTEGER,
    multi_punct_count         INTEGER,
    shouting_count            INTEGER,
    conflict_lexicon_hits     INTEGER,
    accusation_lexicon_hits   INTEGER,
    affection_lexicon_hits    INTEGER,
    laughter_ratio            REAL,
    conflict_score            REAL NOT NULL DEFAULT 0,
    emotional_intensity_score REAL NOT NULL DEFAULT 0,
    transcript                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id                 INTEGER PRIMARY KEY,
    session_id         INTEGER REFERENCES sessions(session_id),
    timestamp          TEXT NOT NULL,
    date               TEXT NOT NULL,
    month              TEXT NOT NULL,
    hour               INTEGER NOT NULL,
    weekday            INTEGER NOT NULL,
    sender             TEXT NOT NULL,
    text               TEXT,
    char_count         INTEGER NOT NULL DEFAULT 0,
    word_count         INTEGER NOT NULL DEFAULT 0,
    emoji_list         TEXT NOT NULL DEFAULT '[]',
    emoji_count        INTEGER NOT NULL DEFAULT 0,
    question_count     INTEGER NOT NULL DEFAULT 0,
    exclamation_count  INTEGER NOT NULL DEFAULT 0,
    uppercase_ratio    REAL NOT NULL DEFAULT 0,
    is_night_message   INTEGER NOT NULL DEFAULT 0,
    is_media           INTEGER NOT NULL DEFAULT 0,
    media_type         TEXT,
    reply_time_seconds REAL,
    gap_seconds        REAL,
    is_forwarded       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_messages_ts       ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_sender   ON messages(sender);
CREATE INDEX IF NOT EXISTS idx_messages_session  ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_month    ON messages(month, sender);
CREATE INDEX IF NOT EXISTS idx_sessions_start    ON sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_sessions_conflict ON sessions(conflict_score);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_messages USING fts5(
    text,
    sender UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_sessions USING fts5(
    transcript,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS session_embeddings (
    session_id INTEGER PRIMARY KEY REFERENCES sessions(session_id),
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    embedding  BLOB NOT NULL
);
"""

# Descrizione dello schema fornita all'agente LLM per scrivere query SQL corrette.
SCHEMA_DESCRIPTION_FOR_LLM = """
TABELLA messages (un record per messaggio, ordinati cronologicamente per id):
  id INTEGER PK, session_id INTEGER, timestamp TEXT 'YYYY-MM-DD HH:MM:SS' (ora locale dell'export),
  date TEXT 'YYYY-MM-DD', month TEXT 'YYYY-MM', hour INTEGER 0-23, weekday INTEGER (0=Lunedì ... 6=Domenica),
  sender TEXT, text TEXT, char_count INTEGER, word_count INTEGER, emoji_list TEXT (JSON array), emoji_count INTEGER,
  question_count INTEGER, exclamation_count INTEGER, uppercase_ratio REAL (0-1), is_night_message INTEGER (1 se 01:00-05:30),
  is_media INTEGER (1 = media/evento non testuale), media_type TEXT (image, audio, video, sticker, gif, document, deleted, call, ...),
  is_forwarded INTEGER (1 = messaggio inoltrato da terzi, solo Telegram: non è testo scritto dal mittente),
  reply_time_seconds REAL (secondi dal messaggio precedente SOLO se il mittente è cambiato, altrimenti NULL),
  gap_seconds REAL (secondi dal messaggio precedente, chiunque l'abbia scritto)
TABELLA sessions (sessioni conversazionali: nuova sessione dopo 45 min di silenzio o ogni 45 messaggi con overlap 5):
  session_id INTEGER PK, start_time TEXT, end_time TEXT, first_message_id INTEGER, last_message_id INTEGER
  (i messaggi di una sessione sono: messages.id BETWEEN first_message_id AND last_message_id),
  message_count INTEGER, is_continuation INTEGER (1 = sessione nata per split dimensionale), participant_count INTEGER,
  participants TEXT (JSON array), initiator TEXT, duration_minutes REAL, messages_per_minute REAL,
  wall_of_text_count INTEGER, cold_reply_count INTEGER, multi_punct_count INTEGER, shouting_count INTEGER,
  conflict_lexicon_hits INTEGER, accusation_lexicon_hits INTEGER, affection_lexicon_hits INTEGER,
  laughter_ratio REAL (quota msg con risate/emoji positive; se >= 0.30 il conflict_score è smorzato 0.4x),
  conflict_score REAL 0-100, emotional_intensity_score REAL 0-100, transcript TEXT ('[YYYY-MM-DD HH:MM] Nome: Testo' per riga)
TABELLE FTS5 (full-text BM25): fts_messages(text, sender) con rowid = messages.id;
  fts_sessions(transcript) con rowid = sessions.session_id. Uso: SELECT rowid FROM fts_messages WHERE fts_messages MATCH '"parola"'.
Funzioni utili SQLite: strftime, date, julianday, json_each, LIKE (case-insensitive per ASCII), lower().
""".strip()


# ---------------------------------------------------------------------------
# Connessioni
# ---------------------------------------------------------------------------
def get_connection(db_path: str | Path, read_only: bool = False, check_same_thread: bool = True) -> sqlite3.Connection:
    """Ogni connessione (anche read-only) usa WAL + synchronous=NORMAL + busy_timeout.

    WAL permette letture concorrenti (dashboard Streamlit, agente, RAG) mentre un writer
    indicizza: i lettori non bloccano lo scrittore e viceversa, quindi niente "database is locked".
    """
    db_path = Path(db_path)
    if read_only:
        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=check_same_thread, timeout=30)
    else:
        conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL").fetchone()
    except sqlite3.OperationalError:
        # Una connessione read-only non può convertire un db non-WAL: lo farà il primo writer.
        conn.execute("PRAGMA journal_mode").fetchone()
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-64000")
    return conn


@contextmanager
def connect(db_path: str | Path, read_only: bool = False) -> Iterator[sqlite3.Connection]:
    conn = get_connection(db_path, read_only=read_only)
    try:
        yield conn
        if not read_only:
            conn.commit()
    finally:
        conn.close()


def load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Carica l'estensione sqlite-vec. Ritorna False se non disponibile (fallback NumPy)."""
    try:
        import sqlite_vec  # type: ignore

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("SELECT vec_version()").fetchone()
        return True
    except Exception:
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass
        return False


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> bool:
    """Crea (o ricrea se la dimensione cambia) la tabella vec0. Richiede sqlite-vec già caricato."""
    current = get_meta(conn, "vec_dim")
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_sessions'"
    ).fetchone() is not None
    if exists and current is not None and int(current) != int(dim):
        conn.execute("DROP TABLE vec_sessions")
        exists = False
    if not exists:
        conn.execute(
            f"CREATE VIRTUAL TABLE vec_sessions USING vec0(embedding float[{int(dim)}] distance_metric=cosine)"
        )
    set_meta(conn, "vec_dim", str(int(dim)))
    return True


# ---------------------------------------------------------------------------
# Schema & meta
# ---------------------------------------------------------------------------
def init_db(db_path: str | Path, reset: bool = False) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        for suffix in ("", "-wal", "-shm"):
            target = Path(str(db_path) + suffix)
            if target.exists():
                target.unlink()
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        migrate_schema(conn)


def migrate_schema(conn: sqlite3.Connection) -> None:
    """Aggiunge le colonne introdotte dopo la creazione del DB (DB esistenti restano leggibili)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
    if cols and "is_forwarded" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN is_forwarded INTEGER NOT NULL DEFAULT 0")


def table_columns(db_path: str | Path, table: str) -> set:
    with connect(db_path, read_only=True) as conn:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def set_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def get_meta(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return default
    return row[0] if row else default


def get_all_meta(db_path: str | Path) -> Dict[str, str]:
    with connect(db_path, read_only=True) as conn:
        return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}


# ---------------------------------------------------------------------------
# Gestione file database
# ---------------------------------------------------------------------------
def secure_wipe_database(db_path: str | Path) -> List[str]:
    """Cancella il database e i suoi file WAL/SHM, sovrascrivendone prima il contenuto con zeri.

    Nota: su SSD/APFS (copy-on-write, snapshot Time Machine) la sovrascrittura non garantisce
    l'irrecuperabilità fisica dei blocchi; garantisce che il file non sia più leggibile dall'app.
    """
    db_path = Path(db_path)
    removed: List[str] = []
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path), timeout=30)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
        except sqlite3.Error:
            pass
    for suffix in ("", "-wal", "-shm", "-journal"):
        target = Path(str(db_path) + suffix)
        if not target.exists():
            continue
        size = target.stat().st_size
        with open(target, "r+b") as fh:
            chunk = b"\0" * (1024 * 1024)
            written = 0
            while written < size:
                n = min(len(chunk), size - written)
                fh.write(chunk[:n])
                written += n
            fh.flush()
            os.fsync(fh.fileno())
        target.unlink()
        removed.append(target.name)
    return removed


def slugify(name: str) -> str:
    norm = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    norm = re.sub(r"[^A-Za-z0-9]+", "_", norm).strip("_").lower()
    return norm or f"chat_{datetime.now():%Y%m%d_%H%M%S}"


def db_path_for_name(name: str) -> Path:
    return config.DATA_DIR / f"{slugify(name)}.db"


def list_databases() -> List[Path]:
    dbs = []
    for p in sorted(config.DATA_DIR.glob("*.db"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with connect(p, read_only=True) as conn:
                if conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] > 0:
                    dbs.append(p)
        except sqlite3.Error:
            continue
    return dbs


def db_signature(db_path: str | Path) -> float:
    """Firma di cache: mtime del db e del WAL (cambia quando i dati cambiano)."""
    p = Path(db_path)
    sig = p.stat().st_mtime if p.exists() else 0.0
    wal = Path(str(p) + "-wal")
    if wal.exists():
        sig = max(sig, wal.stat().st_mtime)
    return sig


def get_db_summary(db_path: str | Path) -> Dict[str, Any]:
    with connect(db_path, read_only=True) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n, MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts, COUNT(DISTINCT sender) AS senders "
            "FROM messages"
        ).fetchone()
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        embedded = conn.execute("SELECT COUNT(*) FROM session_embeddings").fetchone()[0]
        participants = [
            r[0] for r in conn.execute("SELECT sender FROM messages GROUP BY sender ORDER BY COUNT(*) DESC")
        ]
        meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}
    return {
        "messages": int(row["n"] or 0),
        "first_ts": row["first_ts"],
        "last_ts": row["last_ts"],
        "participant_count": int(row["senders"] or 0),
        "participants": participants,
        "sessions": int(sessions),
        "embedded_sessions": int(embedded),
        "meta": meta,
    }


def get_participants(db_path: str | Path) -> List[str]:
    with connect(db_path, read_only=True) as conn:
        return [r[0] for r in conn.execute("SELECT sender FROM messages GROUP BY sender ORDER BY COUNT(*) DESC")]


def fetch_session(db_path: str | Path, session_id: int) -> Optional[Dict[str, Any]]:
    with connect(db_path, read_only=True) as conn:
        row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (int(session_id),)).fetchone()
        if row is None:
            return None
        data = dict(row)
        msgs = conn.execute(
            "SELECT id, timestamp, sender, text, char_count, is_media, media_type, reply_time_seconds "
            "FROM messages WHERE id BETWEEN ? AND ? ORDER BY id",
            (data["first_message_id"], data["last_message_id"]),
        ).fetchall()
        data["messages"] = [dict(m) for m in msgs]
        try:
            data["participants"] = json.loads(data["participants"])
        except (TypeError, json.JSONDecodeError):
            data["participants"] = []
    return data


def fetch_messages_window(db_path: str | Path, center_message_id: int, before: int = 6, after: int = 8) -> List[Dict[str, Any]]:
    with connect(db_path, read_only=True) as conn:
        rows = conn.execute(
            "SELECT id, timestamp, sender, text, char_count, is_media, media_type FROM messages "
            "WHERE id BETWEEN ? AND ? ORDER BY id",
            (int(center_message_id) - before, int(center_message_id) + after),
        ).fetchall()
    return [dict(r) for r in rows]
