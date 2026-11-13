"""
ChatAutopsy — Motore analitico deterministico (SQL + Pandas + NumPy + scikit-learn).

Nessun LLM: tutte le metriche psicometriche, relazionali, lessicali, di rete e
cronobiologiche sono calcolate qui in pochi secondi anche su centinaia di migliaia
di messaggi. Ogni metodo pubblico restituisce DataFrame / dict serializzabili,
consumati sia dalla dashboard Streamlit sia dall'agente (dossier JSON).
"""
from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from chatautopsy import config, database
from chatautopsy.ingestion import compile_lexicon

AFFECTION_RE = compile_lexicon(config.AFFECTION_LEXICON)
APOLOGY_RE = compile_lexicon(config.APOLOGY_LEXICON)
BLAME_RE = compile_lexicon(config.BLAME_LEXICON)
GASLIGHTING_RE = compile_lexicon(config.GASLIGHTING_LEXICON)
PASSIVE_AGGRESSIVE_RE = compile_lexicon(config.PASSIVE_AGGRESSIVE_LEXICON)
GUILT_TRIPPING_RE = compile_lexicon(config.GUILT_TRIPPING_LEXICON)
WE_RE = compile_lexicon(config.WE_PRONOUNS)
I_RE = compile_lexicon(config.I_PRONOUNS)
I_RE_EN = compile_lexicon(config.I_PRONOUNS + config.I_PRONOUNS_EN_ONLY)  # "i" solo su chat in inglese
YOU_RE = compile_lexicon(config.YOU_PRONOUNS)
ABSOLUTE_RE = compile_lexicon(config.ABSOLUTE_TERMS)
YOU_ABSOLUTE_RE = re.compile(
    r"(?<!\w)(?:tu|te|tuo|tua|voi|you)(?!\w)[^.?!\n]{0,40}?(?<!\w)(?:sempre|mai|always|never)(?!\w)", re.IGNORECASE
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
LAUGHTER_RE = re.compile(config.LAUGHTER_PATTERN, re.IGNORECASE)

LEXICONS = {
    "apology": APOLOGY_RE,
    "blame": BLAME_RE,
    "gaslighting": GASLIGHTING_RE,
    "passive_aggressive": PASSIVE_AGGRESSIVE_RE,
    "guilt_tripping": GUILT_TRIPPING_RE,
    "affection": AFFECTION_RE,
    "conflict": compile_lexicon(config.CONFLICT_LEXICON),
}

MEDIA_LABELS_IT = {
    "testo": "Testo", "inoltrato": "Inoltri", "image": "Foto", "audio": "Vocali / audio", "video": "Video",
    "sticker": "Sticker", "gif": "GIF", "document": "Documenti", "location": "Posizioni", "contact": "Contatti",
    "poll": "Sondaggi", "deleted": "Eliminati", "call": "Chiamate", "view_once": "Visualizza una volta",
    "media": "Altri media",
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _norm_date(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0 or b is None or (isinstance(b, float) and not math.isfinite(b)):
            return default
        out = a / b
        return out if math.isfinite(out) else default
    except (TypeError, ZeroDivisionError):
        return default


def _month_range(first: str, last: str) -> List[str]:
    if not first or not last:
        return []
    periods = pd.period_range(pd.Period(first, "M"), pd.Period(last, "M"), freq="M")
    return [str(p) for p in periods]


def to_jsonable(obj: Any) -> Any:
    """Converte ricorsivamente DataFrame/NumPy/Timestamp in tipi JSON nativi."""
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return {str(k): to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else round(f, 4)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, date):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    return obj


class cached_property_locked:
    """Come functools.cached_property ma thread-safe per istanza: il precalcolo in background e la
    sessione dell'utente non calcolano due volte la stessa serie (es. il DataFrame da 1M+ righe)."""

    def __init__(self, func):
        self.func = func
        self.attrname = func.__name__
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        self.attrname = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        cache = instance.__dict__
        if self.attrname in cache:
            return cache[self.attrname]
        with instance._key_lock(("prop", self.attrname)):
            if self.attrname not in cache:
                cache[self.attrname] = self.func(instance)
        return cache[self.attrname]


def _memo(fn):
    """Memorizza il risultato del metodo sull'istanza (l'engine è condiviso e i dati non cambiano).
    I risultati vanno trattati come sola lettura dai chiamanti interni."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        key = (fn.__name__, args, tuple(sorted(kwargs.items())))
        cache = self.__dict__.setdefault("_memo_cache", {})
        if key in cache:
            return cache[key]
        # un lock per chiave: se il precalcolo in background sta già calcolando questa analisi,
        # il chiamante aspetta il risultato invece di ricalcolarlo in parallelo (doppio lavoro + GIL)
        with self._key_lock(key):
            if key not in cache:
                cache[key] = fn(self, *args, **kwargs)
        return cache[key]

    return wrapper


def _minutes(seconds: Any) -> Optional[float]:
    try:
        f = float(seconds)
    except (TypeError, ValueError):
        return None
    return round(f / 60.0, 2) if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# Motore
# ---------------------------------------------------------------------------
class AnalyticsEngine:
    def __init__(self, db_path: str | Path, start_date: Any = None, end_date: Any = None):
        import threading

        self._locks_guard = threading.Lock()
        self._locks: Dict[Any, Any] = {}
        self.db_path = str(db_path)
        self.start_date = _norm_date(start_date)
        self.end_date = _norm_date(end_date)

    def _key_lock(self, key: Any):
        import threading

        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = self._locks[key] = threading.RLock()
        return lock

    # ------------------------------------------------------------------ load
    def _date_clause(self, column: str) -> tuple:
        clauses, params = [], []
        if self.start_date:
            clauses.append(f"{column} >= ?")
            params.append(f"{self.start_date} 00:00:00")
        if self.end_date:
            clauses.append(f"{column} <= ?")
            params.append(f"{self.end_date} 23:59:59")
        return (" AND ".join(clauses) if clauses else "1=1"), params

    @cached_property_locked
    def df(self) -> pd.DataFrame:
        where, params = self._date_clause("timestamp")
        fwd_col = "is_forwarded" if "is_forwarded" in database.table_columns(self.db_path, "messages") else "0 AS is_forwarded"
        with database.connect(self.db_path, read_only=True) as conn:
            df = pd.read_sql_query(
                "SELECT id, session_id, timestamp, date, month, hour, weekday, sender, text, char_count, word_count, "
                "emoji_list, emoji_count, question_count, exclamation_count, uppercase_ratio, is_night_message, "
                f"is_media, media_type, {fwd_col} FROM messages WHERE {where} ORDER BY id",
                conn,
                params=params,
            )
        if df.empty:
            return df.assign(ts=pd.Series(dtype="datetime64[ns]"), gap=pd.Series(dtype=float),
                             reply=pd.Series(dtype=float), prev_sender=pd.Series(dtype=object),
                             turn_id=pd.Series(dtype=int), lower=pd.Series(dtype=object))
        df["ts"] = pd.to_datetime(df["timestamp"])
        df["text"] = df["text"].fillna("")
        df["gap"] = df["ts"].diff().dt.total_seconds()
        df["prev_sender"] = df["sender"].shift()
        changed = df["sender"].ne(df["prev_sender"])
        df["reply"] = df["gap"].where(changed & df["prev_sender"].notna())
        df["turn_id"] = changed.cumsum().astype(int)
        # testo "proprio" del mittente: esclusi media e messaggi inoltrati da terzi (articoli, catene)
        df["lower"] = np.where(df["is_media"].eq(0) & df["is_forwarded"].eq(0), df["text"].str.lower(), "")
        return df

    @cached_property_locked
    def chat_language(self) -> str:
        """'it' / 'en' / 'other' dalla frequenza di parole funzione su un campione di messaggi."""
        it_words = {"che", "non", "di", "il", "la", "sono", "per", "ma", "mi", "ti", "è", "un", "una", "anche", "ho", "hai"}
        en_words = {"the", "and", "you", "to", "is", "it", "that", "for", "my", "what", "are", "have", "was", "this", "i'm", "don't"}
        sample = self.df["lower"].head(30000)
        it_n = en_n = 0
        for text in sample:
            for w in text.split():
                if w in it_words:
                    it_n += 1
                elif w in en_words:
                    en_n += 1
        if it_n == en_n == 0:
            return "other"
        return "en" if en_n > it_n * 1.2 else "it"

    @cached_property_locked
    def total_participants(self) -> int:
        """Partecipanti dell'intera chat (indipendente dal filtro date): decide se è una chat di gruppo."""
        with database.connect(self.db_path, read_only=True) as conn:
            return int(conn.execute("SELECT COUNT(DISTINCT sender) FROM messages").fetchone()[0])

    @cached_property_locked
    def text_df(self) -> pd.DataFrame:
        """Messaggi di testo scritti dal mittente (esclusi media e inoltri)."""
        return self.df[(self.df["is_media"] == 0) & (self.df["is_forwarded"] == 0)]

    @cached_property_locked
    def conflict_calibration(self) -> Dict[str, Any]:
        """Soglia di "sessione tesa" calibrata sulla chat intera (indipendente dal filtro date)."""
        with database.connect(self.db_path, read_only=True) as conn:
            scores = np.array([r[0] for r in conn.execute("SELECT conflict_score FROM sessions")], dtype=float)
        if scores.size == 0:
            return {"threshold": float(config.HIGH_CONFLICT_THRESHOLD), "mode": "assoluta", "pctl_value": None,
                    "absolute": config.HIGH_CONFLICT_THRESHOLD, "max": None, "median": None, "p90": None,
                    "above_absolute": 0, "above_threshold": 0, "sessions": 0,
                    "percentile": config.CONFLICT_RELATIVE_PERCENTILE}
        pctl = float(np.percentile(scores, config.CONFLICT_RELATIVE_PERCENTILE))
        thr = float(min(max(pctl, config.CONFLICT_RELATIVE_MIN), config.HIGH_CONFLICT_THRESHOLD))
        return {
            "threshold": round(thr, 1),
            "mode": "relativa" if thr < config.HIGH_CONFLICT_THRESHOLD else "assoluta",
            "percentile": config.CONFLICT_RELATIVE_PERCENTILE,
            "pctl_value": round(pctl, 1),
            "absolute": config.HIGH_CONFLICT_THRESHOLD,
            "max": round(float(scores.max()), 1),
            "median": round(float(np.median(scores)), 1),
            "p90": round(float(np.percentile(scores, 90)), 1),
            "above_absolute": int((scores > config.HIGH_CONFLICT_THRESHOLD).sum()),
            "above_threshold": int((scores > thr).sum()),
            "sessions": int(scores.size),
        }

    @property
    def high_conflict_threshold(self) -> float:
        return float(self.conflict_calibration["threshold"])

    def conflict_calibration_info(self) -> Dict[str, Any]:
        return dict(self.conflict_calibration)

    @cached_property_locked
    def participants(self) -> List[str]:
        if self.df.empty:
            return []
        return self.df["sender"].value_counts().index.tolist()

    def participant_list(self) -> List[str]:
        return list(self.participants)

    @cached_property_locked
    def months(self) -> List[str]:
        if self.df.empty:
            return []
        return _month_range(self.df["month"].min(), self.df["month"].max())

    @cached_property_locked
    def turns(self) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame(columns=["turn_id", "sender", "start", "end", "n", "chars", "first_pos", "last_pos"])
        d = self.df.reset_index(drop=True).assign(pos=lambda x: np.arange(len(x)))
        g = d.groupby("turn_id", sort=True)
        turns = pd.DataFrame(
            {
                "sender": g["sender"].first(),
                "start": g["ts"].first(),
                "end": g["ts"].last(),
                "n": g.size(),
                "chars": g["char_count"].sum(),
                "first_pos": g["pos"].first(),
                "last_pos": g["pos"].last(),
            }
        ).reset_index()
        return turns

    @cached_property_locked
    def sessions_df(self) -> pd.DataFrame:
        where, params = self._date_clause("start_time")
        with database.connect(self.db_path, read_only=True) as conn:
            sdf = pd.read_sql_query(
                "SELECT session_id, start_time, end_time, first_message_id, last_message_id, message_count, "
                "is_continuation, participant_count, participants, initiator, duration_minutes, messages_per_minute, "
                "wall_of_text_count, cold_reply_count, multi_punct_count, shouting_count, conflict_lexicon_hits, "
                "accusation_lexicon_hits, affection_lexicon_hits, conflict_score, emotional_intensity_score "
                f"FROM sessions WHERE {where} ORDER BY session_id",
                conn,
                params=params,
            )
        if not sdf.empty:
            sdf["start"] = pd.to_datetime(sdf["start_time"])
        return sdf

    @cached_property_locked
    def emoji_series(self) -> pd.Series:
        """Lista emoji per messaggio (solo righe con emoji, per efficienza)."""
        sub = self.df.loc[self.df["emoji_count"] > 0, "emoji_list"]
        return sub.apply(lambda s: json.loads(s) if s else [])

    @cached_property_locked
    def affection_hits(self) -> pd.Series:
        return self.df["lower"].str.count(AFFECTION_RE)

    @cached_property_locked
    def heart_hits(self) -> pd.Series:
        hearts = config.HEART_EMOJIS
        counts = pd.Series(0, index=self.df.index, dtype=int)
        if not self.emoji_series.empty:
            counts.loc[self.emoji_series.index] = self.emoji_series.apply(
                lambda lst: sum(1 for e in lst if e in hearts or e.replace("️", "") in hearts)
            )
        return counts

    def lexicon_counts(self, name: str) -> pd.Series:
        rx = LEXICONS[name]
        cache = self.__dict__.setdefault("_lex_cache", {})
        if name not in cache:
            with self._key_lock(("lexicon", name)):
                if name not in cache:
                    cache[name] = self.df["lower"].str.count(rx)
        return cache[name]

    # ------------------------------------------------------------- overview
    def overview_kpis(self) -> Dict[str, Any]:
        df = self.df
        if df.empty:
            return {"total_messages": 0}
        first, last = df["ts"].min(), df["ts"].max()
        days = max((last.normalize() - first.normalize()).days + 1, 1)
        init = self.initiation_rate()["per_user"]
        lat = self.response_latency()["per_responder"]
        top_init = init.iloc[0]["sender"] if not init.empty else None
        top_wait = None
        if not lat.empty:
            row = lat.sort_values("median_minutes", ascending=False).iloc[0]
            top_wait = row["responder"]
        return {
            "total_messages": int(len(df)),
            "text_messages": int((df["is_media"] == 0).sum()),
            "media_messages": int(df["is_media"].sum()),
            "participants": len(self.participants),
            "first_message": first.strftime("%Y-%m-%d %H:%M"),
            "last_message": last.strftime("%Y-%m-%d %H:%M"),
            "days_analyzed": int(days),
            "active_days": int(df["date"].nunique()),
            "messages_per_day": round(len(df) / days, 2),
            "messages_per_active_day": round(len(df) / max(df["date"].nunique(), 1), 2),
            "top_initiator": top_init,
            "top_initiator_pct": float(init.iloc[0]["pct"]) if not init.empty else None,
            "slowest_responder": top_wait,
            "slowest_responder_median_min": float(lat.sort_values("median_minutes", ascending=False).iloc[0]["median_minutes"]) if not lat.empty else None,
            "sessions": int(len(self.sessions_df)),
            "message_share": {k: round(v * 100, 2) for k, v in df["sender"].value_counts(normalize=True).items()},
        }

    def volume_timeseries(self, freq: str = "M") -> pd.DataFrame:
        df = self.df
        if df.empty:
            return pd.DataFrame(columns=["period", "sender", "messages"])
        if freq == "W":
            period = df["ts"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
        elif freq == "D":
            period = df["date"]
        else:
            period = df["month"]
        out = df.groupby([period.rename("period"), "sender"]).size().rename("messages").reset_index()
        return out.sort_values("period")

    # ------------------------------------------------------ potere & interesse
    @_memo
    def initiation_rate(self) -> Dict[str, pd.DataFrame]:
        df = self.df
        empty = {"per_user": pd.DataFrame(columns=["sender", "initiations", "pct"]),
                 "monthly": pd.DataFrame(columns=["month", "sender", "initiations", "pct"])}
        if df.empty:
            return empty
        mask = df["gap"].isna() | (df["gap"] > config.INITIATION_GAP_SECONDS)
        init = df[mask]
        counts = init["sender"].value_counts()
        per_user = pd.DataFrame({"sender": counts.index, "initiations": counts.values})
        for p in self.participants:
            if p not in set(per_user["sender"]):
                per_user = pd.concat([per_user, pd.DataFrame([{"sender": p, "initiations": 0}])], ignore_index=True)
        total = per_user["initiations"].sum()
        per_user["pct"] = (per_user["initiations"] / total * 100).round(2) if total else 0.0
        per_user = per_user.sort_values("initiations", ascending=False).reset_index(drop=True)
        monthly = init.groupby(["month", "sender"]).size().rename("initiations").reset_index()
        month_tot = monthly.groupby("month")["initiations"].transform("sum")
        monthly["pct"] = (monthly["initiations"] / month_tot * 100).round(2)
        return {"per_user": per_user, "monthly": monthly}

    @_memo
    def double_texting(self) -> Dict[str, pd.DataFrame]:
        df = self.df
        cols = ["sender", "chains", "double_text_events", "triple_text_events", "double_ratio_pct", "triple_ratio_pct", "max_attempts"]
        if df.empty:
            return {"per_user": pd.DataFrame(columns=cols), "monthly": pd.DataFrame(columns=["month", "sender", "events"])}
        same_turn = df["turn_id"].eq(df["turn_id"].shift())
        gap = df["gap"].fillna(0)
        breaks = ~same_turn | (gap > config.DOUBLE_TEXT_MAX_GAP_SECONDS)
        new_attempt = same_turn & (gap >= config.DOUBLE_TEXT_MIN_GAP_SECONDS) & (gap <= config.DOUBLE_TEXT_MAX_GAP_SECONDS)
        chain_id = breaks.cumsum()
        chains = pd.DataFrame({"chain": chain_id, "sender": df["sender"], "new": new_attempt.astype(int), "month": df["month"]})
        agg = chains.groupby("chain").agg(sender=("sender", "first"), attempts=("new", "sum"), month=("month", "first"))
        agg["attempts"] = agg["attempts"] + 1
        rows = []
        for s in self.participants:
            sub = agg[agg["sender"] == s]
            n = len(sub)
            dbl = int((sub["attempts"] >= 2).sum())
            tri = int((sub["attempts"] >= config.DOUBLE_TEXT_MIN_ATTEMPTS).sum())
            rows.append({
                "sender": s, "chains": n, "double_text_events": dbl, "triple_text_events": tri,
                "double_ratio_pct": round(_safe_div(dbl, n) * 100, 2), "triple_ratio_pct": round(_safe_div(tri, n) * 100, 3),
                "max_attempts": int(sub["attempts"].max()) if n else 0,
            })
        per_user = pd.DataFrame(rows, columns=cols).sort_values("triple_text_events", ascending=False).reset_index(drop=True)
        events = agg[agg["attempts"] >= config.DOUBLE_TEXT_MIN_ATTEMPTS]
        monthly = events.groupby(["month", "sender"]).size().rename("events").reset_index()
        return {"per_user": per_user, "monthly": monthly}

    @_memo
    def effort_ratio(self) -> Dict[str, pd.DataFrame]:
        df = self.df
        if df.empty:
            return {"per_user": pd.DataFrame(), "monthly": pd.DataFrame(columns=["month", "sender", "avg_chars"])}
        tdf = self.text_df
        total_chars = df["char_count"].sum()
        rows = []
        for s in self.participants:
            sub = df[df["sender"] == s]
            tsub = tdf[tdf["sender"] == s]
            rows.append({
                "sender": s,
                "messages": int(len(sub)),
                "text_messages": int(len(tsub)),
                "total_chars": int(sub["char_count"].sum()),
                "chars_share_pct": round(_safe_div(sub["char_count"].sum(), total_chars) * 100, 2),
                "avg_chars": round(float(tsub["char_count"].mean()) if len(tsub) else 0.0, 2),
                "median_chars": float(tsub["char_count"].median()) if len(tsub) else 0.0,
                "avg_words": round(float(tsub["word_count"].mean()) if len(tsub) else 0.0, 2),
                "questions": int(sub["question_count"].sum()),
                "questions_per_100_msgs": round(_safe_div(sub["question_count"].sum(), len(tsub)) * 100, 2),
                "media_sent": int(sub["is_media"].sum()),
                "emoji_sent": int(sub["emoji_count"].sum()),
                "walls_of_text": int((tsub["char_count"] > config.LONG_MESSAGE_CHARS).sum()),
            })
        per_user = pd.DataFrame(rows)
        monthly = tdf.groupby(["month", "sender"]).agg(
            avg_chars=("char_count", "mean"), total_chars=("char_count", "sum"), questions=("question_count", "sum"),
            messages=("id", "count"),
        ).reset_index()
        monthly["avg_chars"] = monthly["avg_chars"].round(2)
        return {"per_user": per_user, "monthly": monthly}

    # -------------------------------------------------------------- latenza
    def _reply_events(self) -> pd.DataFrame:
        df = self.df
        if df.empty:
            return pd.DataFrame(columns=["writer", "responder", "reply", "month", "ts"])
        mask = df["reply"].notna() & (df["reply"] <= config.LATENCY_MAX_SECONDS)
        ev = df.loc[mask, ["prev_sender", "sender", "reply", "month", "ts"]].rename(
            columns={"prev_sender": "writer", "sender": "responder"}
        )
        return ev

    @_memo
    def response_latency(self) -> Dict[str, pd.DataFrame]:
        ev = self._reply_events()
        if ev.empty:
            empty = pd.DataFrame(columns=["writer", "responder", "replies", "median_minutes", "mean_minutes", "p90_minutes"])
            return {"pairs": empty, "matrix_median": pd.DataFrame(),
                    "per_responder": pd.DataFrame(columns=["responder", "replies", "median_minutes", "mean_minutes"])}
        pairs = ev.groupby(["writer", "responder"])["reply"].agg(
            replies="count", median="median", mean="mean", p90=lambda x: float(np.percentile(x, 90))
        ).reset_index()
        pairs["median_minutes"] = (pairs["median"] / 60).round(2)
        pairs["mean_minutes"] = (pairs["mean"] / 60).round(2)
        pairs["p90_minutes"] = (pairs["p90"] / 60).round(2)
        pairs = pairs.drop(columns=["median", "mean", "p90"]).sort_values("replies", ascending=False)
        matrix = pairs.pivot(index="writer", columns="responder", values="median_minutes")
        per_resp = ev.groupby("responder")["reply"].agg(replies="count", median="median", mean="mean").reset_index()
        per_resp["median_minutes"] = (per_resp["median"] / 60).round(2)
        per_resp["mean_minutes"] = (per_resp["mean"] / 60).round(2)
        per_resp = per_resp.drop(columns=["median", "mean"]).sort_values("median_minutes", ascending=False)
        return {"pairs": pairs.reset_index(drop=True), "matrix_median": matrix, "per_responder": per_resp.reset_index(drop=True)}

    @_memo
    def latency_monthly(self) -> pd.DataFrame:
        ev = self._reply_events()
        if ev.empty:
            return pd.DataFrame(columns=["month", "responder", "replies", "median_minutes", "mean_minutes"])
        m = ev.groupby(["month", "responder"])["reply"].agg(replies="count", median="median", mean="mean").reset_index()
        m = m[m["replies"] >= config.LATENCY_MIN_REPLIES_PER_MONTH]
        m["median_minutes"] = (m["median"] / 60).round(2)
        m["mean_minutes"] = (m["mean"] / 60).round(2)
        return m.drop(columns=["median", "mean"]).sort_values("month").reset_index(drop=True)

    @_memo
    def fading_detector(self) -> List[Dict[str, Any]]:
        lm = self.latency_monthly()
        results = []
        for resp in self.participants:
            series = lm[lm["responder"] == resp].sort_values("month")["median_minutes"].to_numpy(dtype=float)
            months = lm[lm["responder"] == resp].sort_values("month")["month"].tolist()
            needed = config.FADING_MIN_BASELINE_MONTHS + config.FADING_RECENT_MONTHS
            if len(series) < needed:
                results.append({"responder": resp, "status": "dati insufficienti", "months_available": len(series), "flagged": False})
                continue
            base = series[: -config.FADING_RECENT_MONTHS]
            recent = series[-config.FADING_RECENT_MONTHS:]
            base_med = float(np.median(base))
            mad = float(np.median(np.abs(base - base_med)) * 1.4826)
            scale = mad if mad > 1e-6 else (float(np.std(base)) if np.std(base) > 1e-6 else max(base_med * 0.25, 0.5))
            recent_med = float(np.median(recent))
            ratio = _safe_div(recent_med, base_med, default=float("nan"))
            z = (recent_med - base_med) / scale
            tail = series[-6:]
            slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) >= 3 else 0.0
            flagged = bool(math.isfinite(ratio) and ratio >= config.FADING_RATIO_THRESHOLD and z >= config.FADING_Z_THRESHOLD)
            if flagged:
                status = "FADING: latenza recente anomala rispetto alla baseline"
            elif math.isfinite(ratio) and ratio <= 1 / config.FADING_RATIO_THRESHOLD and z <= -config.FADING_Z_THRESHOLD:
                status = "AVVICINAMENTO: latenza recente molto più bassa della baseline"
            else:
                status = "stabile"
            results.append({
                "responder": resp, "status": status, "flagged": flagged,
                "baseline_median_min": round(base_med, 2), "recent_median_min": round(recent_med, 2),
                "ratio": round(ratio, 3) if math.isfinite(ratio) else None, "robust_z": round(float(z), 2),
                "trend_slope_min_per_month_last6": round(slope, 3),
                "recent_months": months[-config.FADING_RECENT_MONTHS:], "months_available": len(series),
            })
        return results

    @_memo
    def love_bombing_devaluation(self) -> Dict[str, Any]:
        df = self.df
        if df.empty:
            return {"per_user": pd.DataFrame(), "windows": {}}
        first, last = df["ts"].min(), df["ts"].max()
        span_days = max((last - first).days, 1)
        window = timedelta(days=config.LOVE_BOMBING_WINDOW_DAYS)
        if span_days < 2 * config.LOVE_BOMBING_WINDOW_DAYS:
            window = timedelta(days=max(span_days / 3.0, 1))
        early_mask = df["ts"] <= first + window
        late_mask = df["ts"] >= last - window
        win_days = max(window.days, 1)
        aff = self.affection_hits
        hearts = self.heart_hits
        rows = []
        for s in self.participants:
            def stats(mask):
                sub = df[mask & (df["sender"] == s)]
                text_n = max(int((sub["is_media"] == 0).sum()), 1)
                replies = sub["reply"].dropna()
                replies = replies[replies <= config.LATENCY_MAX_SECONDS]
                return {
                    "msgs_per_day": len(sub) / win_days,
                    "median_latency_min": float(replies.median() / 60) if len(replies) else float("nan"),
                    "affection_per_100": aff[sub.index].sum() / text_n * 100,
                    "hearts_per_100": hearts[sub.index].sum() / max(len(sub), 1) * 100,
                    "avg_chars": float(sub.loc[sub["is_media"] == 0, "char_count"].mean()) if text_n else 0.0,
                }
            e, l = stats(early_mask), stats(late_mask)

            def pct_change(a, b):
                if not (math.isfinite(a) and math.isfinite(b)):
                    return float("nan")
                if a == 0:
                    return 100.0 if b > 0 else 0.0
                return (b - a) / abs(a) * 100

            changes = {k: pct_change(e[k], l[k]) for k in e}
            components = [
                -changes["msgs_per_day"],
                -changes["affection_per_100"],
                changes["median_latency_min"],
                -changes["hearts_per_100"],
                -changes["avg_chars"],
            ]
            comps = [float(np.clip(c, -100, 100)) for c in components if math.isfinite(c)]
            index = float(np.mean(comps)) if comps else 0.0
            if index > 40:
                label = "Svalutazione marcata (forte calo di investimento e affetto)"
            elif index > 15:
                label = "Raffreddamento progressivo"
            elif index >= -15:
                label = "Stabile"
            else:
                label = "Intensificazione (investimento in crescita)"
            row = {"sender": s, "devaluation_index": round(index, 2), "interpretation": label}
            for k in e:
                row[f"early_{k}"] = round(e[k], 2) if math.isfinite(e[k]) else None
                row[f"late_{k}"] = round(l[k], 2) if math.isfinite(l[k]) else None
                row[f"change_pct_{k}"] = round(changes[k], 1) if math.isfinite(changes[k]) else None
            rows.append(row)
        return {
            "per_user": pd.DataFrame(rows),
            "windows": {
                "early": [first.strftime("%Y-%m-%d"), (first + window).strftime("%Y-%m-%d")],
                "late": [(last - window).strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")],
                "window_days": int(win_days),
            },
        }

    # ----------------------------------------------------- red flag & conflitto
    @_memo
    def stonewalling(self) -> Dict[str, pd.DataFrame]:
        df = self.df.reset_index(drop=True)
        cols = ["long_message_id", "timestamp", "author", "long_chars", "long_text", "responder", "reply_timestamp",
                "reply_text", "reply_chars", "silence_minutes", "type", "session_id"]
        if df.empty:
            return {"events": pd.DataFrame(columns=cols), "per_responder": pd.DataFrame(), "per_author": pd.DataFrame()}
        turns = self.turns
        turn_last_pos = dict(zip(turns["turn_id"], turns["last_pos"]))
        long_mask = (df["is_media"] == 0) & (df["char_count"] > config.LONG_MESSAGE_CHARS)
        # un messaggio lungo conta solo se è un'apertura scritta di proprio pugno: esclusi inoltri e
        # link/articoli incollati (testo senza URL sotto la soglia)
        own_chars = pd.Series(np.iinfo(np.int32).max, index=df.index)
        own_chars[long_mask] = df.loc[long_mask, "text"].str.replace(URL_RE, "", regex=True).str.len()
        shared = long_mask & ((df["is_forwarded"] == 1) | (own_chars <= config.LONG_MESSAGE_CHARS))
        excluded_shared = int(shared.sum())
        cand = df[long_mask & ~shared]
        if cand.empty:
            return {"events": pd.DataFrame(columns=cols), "per_responder": pd.DataFrame(), "per_author": pd.DataFrame()}
        # un evento per turno: il messaggio lungo più lungo del turno
        cand = cand.loc[cand.groupby("turn_id")["char_count"].idxmax()]
        n = len(df)
        ts = df["ts"].to_numpy()
        senders_arr = df["sender"].to_numpy()
        group_chat = self.total_participants > 2
        session_senders: Dict[int, set] = (
            df.groupby("session_id")["sender"].agg(lambda x: set(x)).to_dict() if group_chat else {}
        )
        events = []
        received_long_turns: Counter = Counter()
        excluded_group_noise = 0
        for pos, row in cand.iterrows():
            last_pos = turn_last_pos[row["turn_id"]]
            nxt = last_pos + 1
            if nxt >= n:
                continue
            reply = df.iloc[nxt]
            if group_chat:
                # Nei gruppi conta solo se A e B stanno dialogando 1-a-1: nella sessione o nella finestra
                # (6 messaggi prima del messaggio lungo -> risposta) non devono aver scritto altri membri.
                pair = {row["sender"], reply["sender"]}
                window = set(senders_arr[max(0, pos - config.STONEWALL_GROUP_WINDOW_MESSAGES): nxt + 1])
                in_session = session_senders.get(int(row["session_id"]), window)
                if not (window <= pair or in_session <= pair):
                    excluded_group_noise += 1
                    continue
            received_long_turns[reply["sender"]] += 1
            silence = (ts[nxt] - ts[last_pos]) / np.timedelta64(1, "s")
            cold = (reply["is_media"] == 0 and reply["char_count"] < config.COLD_REPLY_MAX_CHARS) or (
                reply["media_type"] in config.COLD_MEDIA_TYPES
            )
            long_silence = silence > config.STONEWALL_SILENCE_SECONDS
            if not (cold or long_silence):
                continue
            etype = "Risposta glaciale + silenzio" if (cold and long_silence) else ("Risposta glaciale" if cold else "Silenzio > 4h")
            events.append({
                "long_message_id": int(row["id"]), "timestamp": row["ts"].strftime("%Y-%m-%d %H:%M"),
                "author": row["sender"], "long_chars": int(row["char_count"]),
                "long_text": row["text"][:600], "responder": reply["sender"],
                "reply_timestamp": reply["ts"].strftime("%Y-%m-%d %H:%M"),
                "reply_text": reply["text"][:200], "reply_chars": int(reply["char_count"]),
                "silence_minutes": round(float(silence) / 60, 1), "type": etype, "session_id": int(row["session_id"]),
            })
        ev = pd.DataFrame(events, columns=cols)
        per_resp_rows = []
        for p in self.participants:
            sub = ev[ev["responder"] == p]
            recv = received_long_turns.get(p, 0)
            per_resp_rows.append({
                "responder": p, "stonewalling_events": int(len(sub)),
                "cold_replies": int(sub["type"].str.contains("glaciale").sum()),
                "long_silences": int(sub["type"].str.contains("ilenzio").sum()),
                "long_messages_received": int(recv),
                "stonewalling_rate_pct": round(_safe_div(len(sub), recv) * 100, 2),
            })
        per_resp = pd.DataFrame(per_resp_rows).sort_values("stonewalling_events", ascending=False)
        per_author = ev.groupby("author").size().rename("times_stonewalled").reset_index() if not ev.empty else pd.DataFrame(columns=["author", "times_stonewalled"])
        return {"events": ev.sort_values("timestamp", ascending=False).reset_index(drop=True),
                "per_responder": per_resp.reset_index(drop=True), "per_author": per_author,
                "group_filter_applied": bool(group_chat), "excluded_group_noise": int(excluded_group_noise),
                "excluded_shared_content": excluded_shared}

    @_memo
    def pronoun_shift(self) -> Dict[str, pd.DataFrame]:
        tdf = self.text_df
        if tdf.empty:
            return {"monthly": pd.DataFrame(), "per_user": pd.DataFrame()}
        lower = tdf["lower"]
        counts = pd.DataFrame({
            "month": tdf["month"], "sender": tdf["sender"], "words": tdf["word_count"],
            "we": lower.str.count(WE_RE), "i": lower.str.count(I_RE_EN if self.chat_language == "en" else I_RE),
            "you": lower.str.count(YOU_RE),
            "absolutes": lower.str.count(ABSOLUTE_RE), "you_absolute": lower.str.count(YOU_ABSOLUTE_RE),
        })

        def finish(g: pd.DataFrame) -> pd.DataFrame:
            pron = g["we"] + g["i"] + g["you"]
            g["we_index_pct"] = (g["we"] / pron.replace(0, np.nan) * 100).round(2)
            g["you_index_pct"] = (g["you"] / pron.replace(0, np.nan) * 100).round(2)
            g["i_index_pct"] = (g["i"] / pron.replace(0, np.nan) * 100).round(2)
            g["we_per_1000_words"] = (g["we"] / g["words"].replace(0, np.nan) * 1000).round(2)
            g["you_per_1000_words"] = (g["you"] / g["words"].replace(0, np.nan) * 1000).round(2)
            g["you_absolute_per_1000_words"] = (g["you_absolute"] / g["words"].replace(0, np.nan) * 1000).round(3)
            return g

        monthly_user = finish(counts.groupby(["month", "sender"]).sum(numeric_only=True).reset_index())
        monthly_all = finish(counts.drop(columns=["sender"]).groupby("month").sum(numeric_only=True).reset_index())
        monthly_all["sender"] = "Totale"
        monthly = pd.concat([monthly_all, monthly_user], ignore_index=True).sort_values("month")
        per_user = finish(counts.groupby("sender").sum(numeric_only=True).reset_index())
        return {"monthly": monthly, "per_user": per_user}

    @_memo
    def lexicon_report(self, name: str, examples: int = 12) -> Dict[str, Any]:
        df = self.df
        if df.empty:
            return {"per_user": pd.DataFrame(), "monthly": pd.DataFrame(), "examples": []}
        hits = self.lexicon_counts(name)
        tmask = df["is_media"] == 0
        rows = []
        for s in self.participants:
            m = df["sender"] == s
            n_text = int((m & tmask).sum())
            h = int(hits[m].sum())
            rows.append({"sender": s, "hits": h, "messages_with_hits": int((hits[m] > 0).sum()),
                         "per_1000_msgs": round(_safe_div(h, n_text) * 1000, 2)})
        per_user = pd.DataFrame(rows).sort_values("hits", ascending=False).reset_index(drop=True)
        hm = df.loc[hits > 0]
        monthly = hm.assign(hits=hits[hits > 0]).groupby(["month", "sender"])["hits"].sum().reset_index()
        rx = LEXICONS[name]
        ex_rows = hm.sort_values("ts", ascending=False).head(examples * 3)
        ex = []
        for _, r in ex_rows.iterrows():
            ex.append({
                "message_id": int(r["id"]), "session_id": int(r["session_id"]),
                "citation": f'[{r["ts"].strftime("%Y-%m-%d %H:%M")}] {r["sender"]}: "{r["text"][:300]}"',
                "matched": sorted(set(m.group(0).lower() for m in rx.finditer(r["lower"])))[:5],
            })
        # varietà: massimo examples, privilegiando match diversi
        seen, picked = set(), []
        for e in ex:
            key = tuple(e["matched"])
            if key in seen and len(picked) < examples // 2:
                continue
            seen.add(key)
            picked.append(e)
            if len(picked) >= examples:
                break
        return {"per_user": per_user, "monthly": monthly, "examples": picked}

    @_memo
    def apology_vs_blame(self) -> Dict[str, Any]:
        apo = self.lexicon_report("apology", examples=8)
        bla = self.lexicon_report("blame", examples=8)
        if apo["per_user"].empty:
            return {"per_user": pd.DataFrame(), "monthly": pd.DataFrame(), "apology_examples": [], "blame_examples": []}
        pu = apo["per_user"].rename(columns={"hits": "apologies", "messages_with_hits": "apology_msgs",
                                             "per_1000_msgs": "apologies_per_1000"})
        pb = bla["per_user"].rename(columns={"hits": "blame", "messages_with_hits": "blame_msgs",
                                             "per_1000_msgs": "blame_per_1000"})
        merged = pu.merge(pb, on="sender", how="outer").fillna(0)
        merged["apology_blame_ratio"] = merged.apply(
            lambda r: round(_safe_div(r["apologies"], r["blame"], default=float(r["apologies"])), 2), axis=1
        )
        am = apo["monthly"].rename(columns={"hits": "apologies"})
        bm = bla["monthly"].rename(columns={"hits": "blame"})
        monthly = am.merge(bm, on=["month", "sender"], how="outer").fillna(0).sort_values("month")
        return {"per_user": merged, "monthly": monthly, "apology_examples": apo["examples"], "blame_examples": bla["examples"]}

    @_memo
    def conflict_recovery(self) -> Dict[str, Any]:
        sdf = self.sessions_df
        df = self.df.reset_index(drop=True)
        cols = ["session_id", "conflict_start", "conflict_score", "conflict_participants", "peacemaker",
                "repair_time", "hours_of_silence", "with_apology", "repair_text"]
        if sdf.empty or df.empty:
            return {"per_user": pd.DataFrame(), "events": pd.DataFrame(columns=cols)}
        high = sdf[sdf["conflict_score"] > self.high_conflict_threshold]
        id_index = pd.Index(df["id"])
        apology = self.lexicon_counts("apology").reset_index(drop=True)
        events = []
        for _, s in high.iterrows():
            pos = id_index.get_indexer([s["last_message_id"]])[0]
            if pos < 0 or pos + 1 >= len(df):
                continue
            nxt = df.iloc[pos + 1]
            silence = (nxt["ts"] - df.iloc[pos]["ts"]).total_seconds()
            if silence <= config.REPAIR_SILENCE_SECONDS:
                continue
            events.append({
                "session_id": int(s["session_id"]), "conflict_start": s["start_time"][:16],
                "conflict_score": float(s["conflict_score"]), "conflict_participants": s["participants"],
                "peacemaker": nxt["sender"], "repair_time": nxt["ts"].strftime("%Y-%m-%d %H:%M"),
                "hours_of_silence": round(silence / 3600, 2), "with_apology": bool(apology.iloc[pos + 1] > 0),
                "repair_text": nxt["text"][:250],
            })
        ev = pd.DataFrame(events, columns=cols)
        rows = []
        total = len(ev)
        for p in self.participants:
            sub = ev[ev["peacemaker"] == p]
            rows.append({"sender": p, "repairs_initiated": int(len(sub)),
                         "repair_share_pct": round(_safe_div(len(sub), total) * 100, 2),
                         "with_apology": int(sub["with_apology"].sum()) if len(sub) else 0,
                         "median_hours_before_repair": round(float(sub["hours_of_silence"].median()), 2) if len(sub) else None})
        per_user = pd.DataFrame(rows).sort_values("repairs_initiated", ascending=False).reset_index(drop=True)
        return {"per_user": per_user, "events": ev, "high_conflict_sessions": int(len(high)),
                "followed_by_silence": int(total)}

    # ------------------------------------------------------------- lessico
    @_memo
    def tfidf_signature(self, top_n: int = config.TFIDF_TOP_N) -> Dict[str, pd.DataFrame]:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

        tdf = self.text_df
        if tdf.empty:
            return {}
        senders = [s for s in self.participants if (tdf["sender"] == s).any()]
        # URL e risate di qualsiasi lunghezza (ahahahah, hahaha, lol, xd…) non sono parole identitarie
        docs = [LAUGHTER_RE.sub(" ", URL_RE.sub(" ", "\n".join(tdf.loc[tdf["sender"] == s, "lower"].tolist())))
                for s in senders]
        stop = set(config.STOPWORDS) | config.MEDIA_NOISE_WORDS
        for p in self.participants:
            stop |= {t.lower() for t in re.findall(r"[^\W\d_]+", p)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = CountVectorizer(
                lowercase=True, token_pattern=r"(?u)\b[^\W\d_]{3,}\b", stop_words=sorted(stop),
                ngram_range=(1, 2), max_features=150000,
            )
            try:
                counts = cv.fit_transform(docs)
            except ValueError:
                return {s: pd.DataFrame(columns=["term", "score", "count", "exclusivity_pct"]) for s in senders}
        tfidf = TfidfTransformer(sublinear_tf=True, smooth_idf=False).fit_transform(counts)
        vocab = np.array(cv.get_feature_names_out())
        counts_arr = counts.toarray() if counts.shape[1] < 400000 else counts
        totals = np.asarray(counts.sum(axis=0)).ravel()
        out = {}
        for i, s in enumerate(senders):
            row_counts = np.asarray(counts_arr[i]).ravel()
            row_tfidf = np.asarray(tfidf[i].todense()).ravel()
            excl = np.divide(row_counts, totals, out=np.zeros_like(row_tfidf), where=totals > 0)
            score = row_tfidf * (excl if len(senders) > 1 else 1.0)
            score[row_counts < config.TFIDF_MIN_TERM_COUNT] = 0
            top = np.argsort(-score)[:top_n]
            top = [j for j in top if score[j] > 0]
            out[s] = pd.DataFrame({
                "term": vocab[top], "score": np.round(score[top], 5), "count": row_counts[top].astype(int),
                "exclusivity_pct": np.round(excl[top] * 100, 1) if len(senders) > 1 else 100.0,
            })
        return out

    def word_trend(self, terms: Sequence[str]) -> pd.DataFrame:
        cols = ["term", "month", "sender", "count", "per_1000_msgs"]
        tdf = self.text_df
        terms = [t.strip() for t in terms if t and t.strip()]
        if tdf.empty or not terms:
            return pd.DataFrame(columns=cols)
        msgs_per_ms = tdf.groupby(["month", "sender"]).size()
        regexes = {}
        for term in terms:
            t = term.lower()
            if t.endswith("*"):
                regexes[term] = re.compile(r"(?<!\w)" + re.escape(t[:-1]) + r"\w*", re.IGNORECASE)
            else:
                regexes[term] = compile_lexicon([t])
        # Prefiltro: una sola passata con l'unione dei termini individua le righe candidate;
        # i conteggi per termine girano solo su quelle (identici, ma senza N passate sull'intero corpus).
        exact = [t.lower() for t in terms if not t.endswith("*")]
        prefixes = [t[:-1].lower() for t in terms if t.endswith("*")]
        union_parts = []
        if exact:
            union_parts.append(compile_lexicon(exact).pattern)
        if prefixes:
            union_parts.append(r"(?<!\w)(?:" + "|".join(re.escape(p) for p in sorted(prefixes, key=len, reverse=True)) + ")")
        union = re.compile("|".join(union_parts), re.IGNORECASE | re.UNICODE)
        cand_lower = tdf["lower"][tdf["lower"].str.contains(union)]
        frames = []
        for term in terms:
            rx = regexes[term]
            c = cand_lower.str.count(rx)
            g = c.groupby([tdf.loc[c.index, "month"], tdf.loc[c.index, "sender"]]).sum()
            full_index = pd.MultiIndex.from_product([self.months, self.participants], names=["month", "sender"])
            g = g.reindex(full_index, fill_value=0)
            denom = msgs_per_ms.reindex(full_index, fill_value=0)
            f = pd.DataFrame({"count": g.values, "per_1000_msgs": np.round(
                np.divide(g.values * 1000.0, denom.values, out=np.zeros(len(g), dtype=float), where=denom.values > 0), 3)},
                index=full_index).reset_index()
            f.insert(0, "term", term)
            frames.append(f)
        return pd.concat(frames, ignore_index=True)[cols]

    @_memo
    def affection_timeline(self) -> Dict[str, Any]:
        from sklearn.feature_extraction.text import CountVectorizer

        tdf = self.text_df
        if tdf.empty:
            return {"monthly": pd.DataFrame(), "summary": pd.DataFrame(), "detected_terms": []}
        detected: List[Dict[str, Any]] = []
        marker = (self.affection_hits[tdf.index] > 0) | (self.heart_hits[tdf.index] > 0)
        if marker.sum() >= 10 and len(tdf) >= 200:
            exclude = (set(config.STOPWORDS) | config.MEDIA_NOISE_WORDS | set(config.NICKNAME_EXCLUDE)
                       | {w for t in config.AFFECTION_LEXICON for w in t.split()})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    cv = CountVectorizer(binary=True, token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
                                         min_df=config.NICKNAME_MIN_COUNT, max_features=30000)
                    X = cv.fit_transform(tdf["lower"].tolist())
                    vocab = np.array(cv.get_feature_names_out())
                    y = marker.to_numpy()
                    tot = np.asarray(X.sum(axis=0)).ravel().astype(float)
                    pos = np.asarray(X[y].sum(axis=0)).ravel().astype(float)
                    base = y.mean()
                    lift = np.divide(pos / np.maximum(tot, 1), base) if base > 0 else np.zeros_like(tot)
                    order = np.argsort(-(lift * np.log1p(tot)))
                    for j in order:
                        term = vocab[j]
                        if term in exclude or lift[j] < config.NICKNAME_MIN_LIFT:
                            continue
                        detected.append({"term": term, "lift": round(float(lift[j]), 2), "messages": int(tot[j])})
                        if len(detected) >= config.NICKNAME_MAX_DETECTED:
                            break
                except ValueError:
                    detected = []
        terms = list(dict.fromkeys(config.PET_NAME_TERMS + [d["term"] for d in detected]))
        trend = self.word_trend(terms)
        trend = trend[trend["count"] > 0] if not trend.empty else trend
        summary_rows = []
        for term in terms:
            sub = trend[trend["term"] == term] if not trend.empty else pd.DataFrame()
            if sub.empty:
                continue
            for s in sub["sender"].unique():
                ss = sub[sub["sender"] == s]
                peak = ss.loc[ss["count"].idxmax()]
                summary_rows.append({
                    "term": term, "sender": s, "total": int(ss["count"].sum()), "first_seen": ss["month"].min(),
                    "last_seen": ss["month"].max(), "peak_month": peak["month"], "peak_count": int(peak["count"]),
                    "detected_from_corpus": term in {d["term"] for d in detected},
                })
        summary = pd.DataFrame(summary_rows)
        if not summary.empty:
            summary = summary.sort_values("total", ascending=False).reset_index(drop=True)
        return {"monthly": trend, "summary": summary, "detected_terms": detected}

    @_memo
    def emoji_fingerprint(self, top_n: int = config.EMOJI_TOP_N) -> Dict[str, Any]:
        df = self.df
        if df.empty:
            return {"top": pd.DataFrame(), "ratio": pd.DataFrame(), "monthly_groups": pd.DataFrame(), "shift": pd.DataFrame()}
        es = self.emoji_series
        exploded = pd.DataFrame({"idx": es.index.repeat(es.str.len()), "emoji": [e for lst in es for e in lst]})
        if exploded.empty:
            ratio = df.groupby("sender").size().rename("messages").reset_index()
            ratio["emoji_total"] = 0
            ratio["emoji_per_msg"] = 0.0
            return {"top": pd.DataFrame(columns=["sender", "emoji", "count", "share_pct"]), "ratio": ratio,
                    "monthly_groups": pd.DataFrame(), "shift": pd.DataFrame()}
        exploded["key"] = exploded["emoji"].str.replace("️", "", regex=False)
        display = exploded.groupby("key")["emoji"].agg(lambda x: x.value_counts().index[0])
        exploded["emoji"] = exploded["key"].map(display)
        exploded["sender"] = df.loc[exploded["idx"], "sender"].to_numpy()
        exploded["month"] = df.loc[exploded["idx"], "month"].to_numpy()
        exploded["ts"] = df.loc[exploded["idx"], "ts"].to_numpy()
        top_rows = []
        for s in self.participants:
            sub = exploded[exploded["sender"] == s]
            vc = sub["emoji"].value_counts()
            for e, c in vc.head(top_n).items():
                top_rows.append({"sender": s, "emoji": e, "count": int(c), "share_pct": round(c / max(len(sub), 1) * 100, 2)})
        top = pd.DataFrame(top_rows)
        msgs = df.groupby("sender").size()
        em = exploded.groupby("sender").size()
        ratio = pd.DataFrame({"messages": msgs, "emoji_total": em.reindex(msgs.index, fill_value=0)}).reset_index()
        ratio = ratio.rename(columns={"index": "sender"})
        ratio["emoji_per_msg"] = (ratio["emoji_total"] / ratio["messages"]).round(3)
        group_of = {}
        for g, members in config.TRACKED_EMOJI_GROUPS.items():
            for m in members:
                group_of[m.replace("️", "")] = g
        exploded["group"] = exploded["key"].map(group_of)
        mg = exploded.dropna(subset=["group"]).groupby(["month", "sender", "group"]).size().rename("count").reset_index()
        per_ms = df.groupby(["month", "sender"]).size().rename("messages").reset_index()
        mg = mg.merge(per_ms, on=["month", "sender"], how="left")
        mg["per_100_msgs"] = (mg["count"] / mg["messages"] * 100).round(2)
        # shift: quota delle emoji nel primo vs ultimo quarto della timeline
        t0, t1 = df["ts"].min(), df["ts"].max()
        q = (t1 - t0) / 4
        shift_rows = []
        for s in self.participants:
            sub = exploded[exploded["sender"] == s]
            early = sub[sub["ts"] <= t0 + q]["emoji"].value_counts(normalize=True)
            late = sub[sub["ts"] >= t1 - q]["emoji"].value_counts(normalize=True)
            keys = set(early.head(10).index) | set(late.head(10).index)
            for k in keys:
                a, b = float(early.get(k, 0.0)) * 100, float(late.get(k, 0.0)) * 100
                shift_rows.append({"sender": s, "emoji": k, "early_share_pct": round(a, 2), "late_share_pct": round(b, 2),
                                   "delta_pp": round(b - a, 2)})
        shift = pd.DataFrame(shift_rows)
        if not shift.empty:
            shift = shift.reindex(shift["delta_pp"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        return {"top": top, "ratio": ratio, "monthly_groups": mg, "shift": shift}

    # --------------------------------------------------------- grafo sociale
    @_memo
    def markov_matrix(self, window_seconds: int = config.MARKOV_WINDOW_SECONDS) -> Dict[str, Any]:
        turns = self.turns
        if turns.empty:
            return {"matrix": pd.DataFrame(), "edges": pd.DataFrame(columns=["source", "target", "probability", "count", "median_latency_s"]), "turns": {}}
        senders = turns["sender"].to_numpy()
        starts = turns["start"].to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9
        ends = turns["end"].to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9
        n = len(turns)
        counts: Dict[str, Counter] = defaultdict(Counter)
        lats: Dict[tuple, List[float]] = defaultdict(list)
        n_turns = Counter(senders.tolist())
        for i in range(n):
            a = senders[i]
            t_end = ends[i]
            j = i + 1
            seen = set()
            while j < n and starts[j] - t_end <= window_seconds:
                b = senders[j]
                if b != a and b not in seen:
                    seen.add(b)
                    counts[a][b] += 1
                    lats[(a, b)].append(starts[j] - t_end)
                j += 1
        parts = self.participants
        matrix = pd.DataFrame(0.0, index=parts, columns=parts)
        edges = []
        for a in parts:
            for b in parts:
                if a == b:
                    continue
                c = counts[a][b]
                p = _safe_div(c, n_turns.get(a, 0))
                matrix.loc[a, b] = round(p, 4)
                if c > 0:
                    edges.append({"source": a, "target": b, "probability": round(p, 4), "count": int(c),
                                  "median_latency_s": round(float(np.median(lats[(a, b)])), 1)})
        edges_df = pd.DataFrame(edges, columns=["source", "target", "probability", "count", "median_latency_s"])
        if not edges_df.empty:
            edges_df = edges_df.sort_values("probability", ascending=False).reset_index(drop=True)
        return {"matrix": matrix, "edges": edges_df, "turns": dict(n_turns)}

    @_memo
    def catalysts_killers(self) -> Dict[str, pd.DataFrame]:
        df = self.df.reset_index(drop=True)
        if df.empty:
            return {"catalysts": pd.DataFrame(), "killers": pd.DataFrame()}
        conv_id = (df["gap"].isna() | (df["gap"] > config.SESSION_GAP_SECONDS)).cumsum()
        conv = df.groupby(conv_id).agg(
            starter=("sender", "first"), messages=("id", "count"), participants=("sender", "nunique"),
            start=("ts", "first"), end=("ts", "last"),
        )
        conv["duration_min"] = (conv["end"] - conv["start"]).dt.total_seconds() / 60
        cat = conv.groupby("starter").agg(
            conversations_started=("messages", "count"), avg_messages=("messages", "mean"),
            median_messages=("messages", "median"), avg_participants=("participants", "mean"),
            avg_duration_min=("duration_min", "mean"),
        ).reset_index().rename(columns={"starter": "sender"})
        cat["raw"] = cat["avg_messages"] * cat["avg_participants"]
        eligible = cat["conversations_started"] >= config.CATALYST_MIN_STARTS
        mx = cat.loc[eligible, "raw"].max() if eligible.any() else cat["raw"].max()
        cat["catalyst_score"] = (cat["raw"] / mx * 100).round(1) if mx and mx > 0 else 0.0
        cat.loc[~eligible, "catalyst_score"] = (cat.loc[~eligible, "catalyst_score"] * 0.5).round(1)
        cat = cat.drop(columns=["raw"]).round({"avg_messages": 1, "avg_participants": 2, "avg_duration_min": 1})
        cat = cat.sort_values("catalyst_score", ascending=False).reset_index(drop=True)

        next_gap = df["gap"].shift(-1)
        turn_end = df["turn_id"].ne(df["turn_id"].shift(-1))
        hours = df["hour"]
        if config.KILLER_EXCLUDE_NIGHT:
            night = (hours >= config.KILLER_NIGHT_START_HOUR) | (hours < config.KILLER_NIGHT_END_HOUR)
        else:
            night = pd.Series(False, index=df.index)
        valid = turn_end & ~night & next_gap.notna()
        killed = valid & (next_gap > config.KILLER_SILENCE_SECONDS)
        k = pd.DataFrame({"sender": df["sender"], "valid": valid, "killed": killed})
        killers = k.groupby("sender").agg(turns_evaluated=("valid", "sum"), conversations_killed=("killed", "sum")).reset_index()
        killers["killer_rate_pct"] = np.where(
            killers["turns_evaluated"] > 0, (killers["conversations_killed"] / killers["turns_evaluated"].replace(0, np.nan) * 100), 0.0
        ).round(2)
        killers = killers.sort_values("killer_rate_pct", ascending=False).reset_index(drop=True)
        return {"catalysts": cat, "killers": killers}

    # ------------------------------------------------- serie per riga (cache)
    @cached_property_locked
    def laughter_hits(self) -> pd.Series:
        return self.df["lower"].str.contains(LAUGHTER_RE).astype(int)

    @cached_property_locked
    def kind_series(self) -> pd.Series:
        """Tipo di contenuto per messaggio: testo / inoltrato / image / audio / sticker / …"""
        df = self.df
        kind = np.where(df["is_media"].eq(1), df["media_type"].fillna("media"),
                        np.where(df["is_forwarded"].eq(1), "inoltrato", "testo"))
        return pd.Series(kind, index=df.index)

    @cached_property_locked
    def year_series(self) -> pd.Series:
        return self.df["month"].str[:4]

    # ----------------------------------------------------- equilibrio nel tempo
    @_memo
    def share_timeseries(self) -> pd.DataFrame:
        """Quota mensile di messaggi e caratteri per persona (50% = perfetta parità in una coppia)."""
        df = self.df
        cols = ["month", "sender", "messages", "chars", "share_pct", "chars_share_pct"]
        if df.empty:
            return pd.DataFrame(columns=cols)
        g = df.groupby(["month", "sender"]).agg(messages=("id", "count"), chars=("char_count", "sum")).reset_index()
        g["share_pct"] = g["messages"] / g.groupby("month")["messages"].transform("sum") * 100
        g["chars_share_pct"] = g["chars"] / g.groupby("month")["chars"].transform("sum").replace(0, np.nan) * 100
        return g[cols].round(2)

    # ------------------------------------------------------------- media mix
    @_memo
    def media_mix(self) -> Dict[str, pd.DataFrame]:
        df = self.df
        empty = {"per_user": pd.DataFrame(columns=["sender", "tipo", "count", "pct"]),
                 "monthly": pd.DataFrame(columns=["month", "sender", "tipo", "count", "pct"])}
        if df.empty:
            return empty
        kind = self.kind_series.map(lambda k: MEDIA_LABELS_IT.get(k, str(k).capitalize()))
        per = pd.DataFrame({"sender": df["sender"], "tipo": kind}).value_counts().rename("count").reset_index()
        per["pct"] = (per["count"] / per.groupby("sender")["count"].transform("sum") * 100).round(2)
        mon = pd.DataFrame({"month": df["month"], "sender": df["sender"], "tipo": kind}).value_counts().rename("count").reset_index()
        mon["pct"] = (mon["count"] / mon.groupby(["month", "sender"])["count"].transform("sum") * 100).round(2)
        return {"per_user": per.sort_values(["sender", "count"], ascending=[True, False]).reset_index(drop=True),
                "monthly": mon.sort_values("month").reset_index(drop=True)}

    # ------------------------------------------------------ distribuzioni
    @_memo
    def length_distribution(self) -> Dict[str, pd.DataFrame]:
        tdf = self.text_df
        if tdf.empty:
            return {"buckets": pd.DataFrame(columns=["sender", "fascia", "count", "pct"]), "stats": pd.DataFrame()}
        edges = [b[0] for b in config.LENGTH_BUCKETS] + [config.LENGTH_BUCKETS[-1][1] + 1]
        labels = [b[2] for b in config.LENGTH_BUCKETS]
        fascia = pd.cut(tdf["char_count"].clip(lower=1), bins=[e - 0.5 for e in edges], labels=labels)
        b = pd.DataFrame({"sender": tdf["sender"], "fascia": fascia}).value_counts().rename("count").reset_index()
        b["pct"] = (b["count"] / b.groupby("sender")["count"].transform("sum") * 100).round(2)
        b["fascia"] = pd.Categorical(b["fascia"], categories=labels, ordered=True)
        b = b.sort_values(["sender", "fascia"]).reset_index(drop=True)
        b["fascia"] = b["fascia"].astype(str)
        st_ = tdf.groupby("sender")["char_count"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
        st_ = st_.rename(columns={"count": "messaggi", "mean": "media", "25%": "p25", "50%": "mediana", "75%": "p75",
                                  "90%": "p90", "99%": "p99", "max": "max"})[["messaggi", "media", "p25", "mediana", "p75", "p90", "p99", "max"]]
        st_ = st_.round(1)
        for c in ("messaggi", "p25", "mediana", "p75", "p90", "p99", "max"):
            st_[c] = st_[c].astype(int)
        return {"buckets": b, "stats": st_.reset_index()}

    @_memo
    def latency_distribution(self) -> Dict[str, pd.DataFrame]:
        ev = self._reply_events()
        if ev.empty:
            return {"buckets": pd.DataFrame(columns=["responder", "fascia", "count", "pct"]),
                    "by_hour": pd.DataFrame(columns=["responder", "hour", "median_minutes", "replies"]),
                    "speed": pd.DataFrame(columns=["responder", "within_1m_pct", "within_5m_pct", "over_30m_pct", "over_2h_pct"])}
        edges = [b[0] for b in config.LATENCY_BUCKETS] + [config.LATENCY_BUCKETS[-1][1] + 1]
        labels = [b[2] for b in config.LATENCY_BUCKETS]
        fascia = pd.cut(ev["reply"], bins=edges, labels=labels, right=False)
        b = pd.DataFrame({"responder": ev["responder"], "fascia": fascia}).value_counts().rename("count").reset_index()
        b["pct"] = (b["count"] / b.groupby("responder")["count"].transform("sum") * 100).round(2)
        b["fascia"] = pd.Categorical(b["fascia"], categories=labels, ordered=True)
        b = b.sort_values(["responder", "fascia"]).reset_index(drop=True)
        b["fascia"] = b["fascia"].astype(str)
        hour = ev["ts"].dt.hour
        bh = ev.assign(hour=hour).groupby(["responder", "hour"])["reply"].agg(replies="count", median="median").reset_index()
        bh["median_minutes"] = (bh["median"] / 60).round(2)
        bh = bh.drop(columns=["median"])
        r = ev["reply"]
        speed = pd.DataFrame({"responder": ev["responder"], "w1": r <= 60, "w5": r <= 300, "o30": r > 1800, "o2h": r > 7200})
        speed = speed.groupby("responder").mean().mul(100).round(2).reset_index().rename(
            columns={"w1": "within_1m_pct", "w5": "within_5m_pct", "o30": "over_30m_pct", "o2h": "over_2h_pct"})
        return {"buckets": b, "by_hour": bh, "speed": speed}

    # ------------------------------------------------- calendario & silenzi
    @_memo
    def daily_activity(self, top_silences: int = 15) -> Dict[str, Any]:
        df = self.df
        if df.empty:
            return {"daily": pd.DataFrame(columns=["date", "total"]), "streaks": {}, "silences": pd.DataFrame()}
        daily = df.groupby(["date", "sender"]).size().unstack(fill_value=0)
        idx = pd.date_range(df["ts"].min().normalize(), df["ts"].max().normalize(), freq="D").strftime("%Y-%m-%d")
        daily = daily.reindex(idx, fill_value=0)
        daily["total"] = daily.sum(axis=1)
        active = (daily["total"] > 0).to_numpy()
        # serie di giorni consecutivi con almeno un messaggio
        best = cur = 0
        best_end = 0
        for i, a in enumerate(active):
            cur = cur + 1 if a else 0
            if cur > best:
                best, best_end = cur, i
        current = 0
        for a in active[::-1]:
            if not a:
                break
            current += 1
        silent_days = int((~active).sum())
        # silenzi più lunghi tra due messaggi consecutivi
        gaps = df["gap"].to_numpy()
        order = np.argsort(-np.nan_to_num(gaps, nan=-1))[:top_silences]
        rows = []
        for pos in order:
            if pos == 0 or not np.isfinite(gaps[pos]) or gaps[pos] < 3600:
                continue
            prev, nxt = df.iloc[pos - 1], df.iloc[pos]
            rows.append({"dal": prev["ts"].strftime("%Y-%m-%d %H:%M"), "al": nxt["ts"].strftime("%Y-%m-%d %H:%M"),
                         "durata_ore": round(float(gaps[pos]) / 3600, 1), "durata_giorni": round(float(gaps[pos]) / 86400, 2),
                         "ultimo_a_scrivere": prev["sender"], "ha_rotto_il_silenzio": nxt["sender"],
                         "messaggio_di_ripresa": (nxt["text"] or "")[:160]})
        busiest = daily["total"].sort_values(ascending=False).head(10)
        return {
            "daily": daily.reset_index().rename(columns={"index": "date"}),
            "streaks": {
                "longest_streak_days": int(best),
                "longest_streak_end": str(idx[best_end]) if best else None,
                "longest_streak_start": str(idx[best_end - best + 1]) if best else None,
                "current_streak_days": int(current),
                "silent_days": silent_days,
                "total_days": int(len(active)),
                "active_days_pct": round(float(active.mean() * 100), 2),
            },
            "silences": pd.DataFrame(rows),
            "busiest_days": [{"date": d, "messages": int(v)} for d, v in busiest.items()],
        }

    @_memo
    def conversation_closers(self) -> Dict[str, pd.DataFrame]:
        """Chi manda l'ultimo messaggio prima di una pausa > 6h (chi "chiude" la conversazione)."""
        df = self.df
        if df.empty:
            return {"per_user": pd.DataFrame(columns=["sender", "closings", "pct"]), "monthly": pd.DataFrame()}
        next_gap = df["gap"].shift(-1)
        closing = next_gap > config.INITIATION_GAP_SECONDS
        sub = df.loc[closing, ["sender", "month"]]
        per = sub["sender"].value_counts().rename_axis("sender").rename("closings").reset_index()
        per["pct"] = (per["closings"] / per["closings"].sum() * 100).round(2) if len(per) else 0.0
        mon = sub.groupby(["month", "sender"]).size().rename("closings").reset_index()
        mon["pct"] = (mon["closings"] / mon.groupby("month")["closings"].transform("sum") * 100).round(2)
        return {"per_user": per, "monthly": mon}

    @_memo
    def chronotype(self) -> Dict[str, Any]:
        df = self.df
        if df.empty:
            return {"hourly": pd.DataFrame(), "bands": pd.DataFrame(), "weekday": pd.DataFrame()}
        h = df.groupby(["sender", "hour"]).size().rename("messages").reset_index()
        h["pct"] = (h["messages"] / h.groupby("sender")["messages"].transform("sum") * 100).round(2)
        band = pd.cut(df["hour"], bins=[-1, 5, 11, 17, 23], labels=["Notte (0-6)", "Mattina (6-12)", "Pomeriggio (12-18)", "Sera (18-24)"])
        bands = pd.DataFrame({"sender": df["sender"], "fascia": band}).value_counts().rename("messages").reset_index()
        bands["pct"] = (bands["messages"] / bands.groupby("sender")["messages"].transform("sum") * 100).round(2)
        wd = df.groupby(["sender", "weekday"]).size().rename("messages").reset_index()
        wd["pct"] = (wd["messages"] / wd.groupby("sender")["messages"].transform("sum") * 100).round(2)
        wd["giorno"] = wd["weekday"].map(dict(enumerate(config.WEEKDAY_NAMES_IT)))
        peaks = h.loc[h.groupby("sender")["messages"].idxmax(), ["sender", "hour"]].rename(columns={"hour": "peak_hour"})
        return {"hourly": h, "bands": bands, "weekday": wd, "peaks": peaks.reset_index(drop=True)}

    # --------------------------------------------------------- clima mensile
    @_memo
    def climate_monthly(self) -> pd.DataFrame:
        """Per mese e persona: affetto, conflitto, scuse, colpe (per 1000 msg di testo) e % messaggi con risate."""
        df = self.df
        cols = ["month", "sender", "text_messages", "affection_per_1000", "conflict_per_1000", "apology_per_1000",
                "blame_per_1000", "laughter_pct", "affection_conflict_ratio"]
        if df.empty:
            return pd.DataFrame(columns=cols)
        own = (df["is_media"] == 0) & (df["is_forwarded"] == 0)
        d = pd.DataFrame({
            "month": df["month"], "sender": df["sender"], "own": own.astype(int),
            "aff": self.affection_hits + self.heart_hits, "conf": self.lexicon_counts("conflict"),
            "apo": self.lexicon_counts("apology"), "bla": self.lexicon_counts("blame"), "laugh": self.laughter_hits,
        })
        g = d.groupby(["month", "sender"]).sum().reset_index()
        n = g["own"].replace(0, np.nan)
        out = pd.DataFrame({
            "month": g["month"], "sender": g["sender"], "text_messages": g["own"],
            "affection_per_1000": g["aff"] / n * 1000, "conflict_per_1000": g["conf"] / n * 1000,
            "apology_per_1000": g["apo"] / n * 1000, "blame_per_1000": g["bla"] / n * 1000,
            "laughter_pct": g["laugh"] / n * 100,
        })
        out["affection_conflict_ratio"] = out["affection_per_1000"] / out["conflict_per_1000"].replace(0, np.nan)
        return out[cols].round(2)

    def conflict_distribution(self, bin_width: float = 2.5) -> pd.DataFrame:
        """Istogramma dei punteggi di tensione delle sessioni nel periodo selezionato."""
        sdf = self.sessions_df
        if sdf.empty:
            return pd.DataFrame(columns=["bin_start", "sessions"])
        edges = np.arange(0, 100 + bin_width, bin_width)
        counts, _ = np.histogram(sdf["conflict_score"].clip(0, 100), bins=edges)
        return pd.DataFrame({"bin_start": edges[:-1], "sessions": counts})

    # ------------------------------------------------------- pagella annuale
    @_memo
    def yearly_scorecard(self) -> pd.DataFrame:
        df = self.df
        if df.empty:
            return pd.DataFrame()
        year = self.year_series
        own = ((df["is_media"] == 0) & (df["is_forwarded"] == 0)).astype(int)
        kind = self.kind_series
        init_mask = (df["gap"].isna() | (df["gap"] > config.INITIATION_GAP_SECONDS)).astype(int)
        reply = df["reply"].where(df["reply"] <= config.LATENCY_MAX_SECONDS)
        d = pd.DataFrame({
            "year": year, "sender": df["sender"], "msg": 1, "chars": df["char_count"].where(own == 1, np.nan),
            "own": own, "init": init_mask, "q": df["question_count"],
            "aff": self.affection_hits + self.heart_hits, "conf": self.lexicon_counts("conflict"),
            "apo": self.lexicon_counts("apology"), "bla": self.lexicon_counts("blame"),
            "emoji": df["emoji_count"], "night": df["is_night_message"],
            "voice": (kind == "audio").astype(int), "media": df["is_media"], "reply": reply,
        })
        g = d.groupby(["year", "sender"])
        agg = g.agg(messages=("msg", "sum"), own=("own", "sum"), avg_chars=("chars", "mean"), inits=("init", "sum"),
                    q=("q", "sum"), aff=("aff", "sum"), conf=("conf", "sum"), apo=("apo", "sum"), bla=("bla", "sum"),
                    emoji=("emoji", "sum"), night=("night", "sum"), voice=("voice", "sum"), media=("media", "sum"),
                    median_reply_s=("reply", "median")).reset_index()
        tot_y = agg.groupby("year")["messages"].transform("sum")
        ini_y = agg.groupby("year")["inits"].transform("sum").replace(0, np.nan)
        own_n = agg["own"].replace(0, np.nan)
        out = pd.DataFrame({
            "Anno": agg["year"], "Persona": agg["sender"], "Messaggi": agg["messages"],
            "Quota messaggi %": agg["messages"] / tot_y * 100,
            "Avvii conversazione %": agg["inits"] / ini_y * 100,
            "Latenza mediana (min)": agg["median_reply_s"] / 60,
            "Caratteri medi": agg["avg_chars"],
            "Domande /100 msg": agg["q"] / own_n * 100,
            "Affetto /1000 msg": agg["aff"] / own_n * 1000,
            "Conflitto /1000 msg": agg["conf"] / own_n * 1000,
            "Scuse /1000 msg": agg["apo"] / own_n * 1000,
            "Colpe /1000 msg": agg["bla"] / own_n * 1000,
            "Emoji /msg": agg["emoji"] / agg["messages"],
            "Notturni %": agg["night"] / agg["messages"] * 100,
            "Vocali %": agg["voice"] / agg["messages"] * 100,
            "Media %": agg["media"] / agg["messages"] * 100,
        })
        return out.round(2).sort_values(["Anno", "Persona"]).reset_index(drop=True)

    # ----------------------------------------------------- pagella equilibrio
    @_memo
    def balance_scorecard(self) -> Dict[str, Any]:
        """Ogni metrica estraibile come indicatore valutabile: valore per persona, chi prevale, rapporto, verdetto."""
        parts = self.participants
        if not parts:
            return {"indicators": [], "reciprocity_index": None, "insights": []}
        df = self.df
        own = (df["is_media"] == 0) & (df["is_forwarded"] == 0)
        n_msgs = df["sender"].value_counts()
        n_own = df.loc[own, "sender"].value_counts()

        def per1000(series: pd.Series) -> Dict[str, float]:
            tot = series.groupby(df["sender"]).sum()
            return {p: _safe_div(float(tot.get(p, 0)), float(n_own.get(p, 0))) * 1000 for p in parts}

        def from_df(frame: pd.DataFrame, key: str, col: str) -> Dict[str, float]:
            if frame is None or frame.empty or col not in frame:
                return {p: float("nan") for p in parts}
            m = frame.set_index(key)[col].to_dict()
            return {p: float(m.get(p, 0) if m.get(p) is not None else float("nan")) for p in parts}

        init = self.initiation_rate()["per_user"]
        dt = self.double_texting()["per_user"]
        eff = self.effort_ratio()["per_user"]
        lat = self.response_latency()["per_responder"]
        spd = self.latency_distribution()["speed"]
        clos = self.conversation_closers()["per_user"]
        sw = self.stonewalling()["per_responder"]
        cr = self.conflict_recovery()["per_user"]
        ps = self.pronoun_shift()["per_user"]
        vamp = self.vampire_index()
        kil = self.catalysts_killers()["killers"]
        kind = self.kind_series
        kind_pct = lambda kinds: {p: _safe_div(float(((df["sender"] == p) & kind.isin(kinds)).sum()), float(n_msgs.get(p, 0))) * 100 for p in parts}  # noqa: E731
        emoji_rate = {p: _safe_div(float(df.loc[df["sender"] == p, "emoji_count"].sum()), float(n_msgs.get(p, 0))) for p in parts}
        hearts = {p: _safe_div(float(self.heart_hits[df["sender"] == p].sum()), float(n_msgs.get(p, 0))) * 100 for p in parts}
        you_abs = from_df(ps, "sender", "you_absolute_per_1000_words") if not ps.empty else {p: float("nan") for p in parts}

        I = []  # noqa: E741

        def add(cat, name, values, unit, meaning, higher="neutral", share=False):
            I.append({"categoria": cat, "indicatore": name, "valori": values, "unita": unit, "significato": meaning,
                      "orientamento": higher, "quota": share})

        add("Investimento", "Quota dei messaggi", {p: _safe_div(float(n_msgs.get(p, 0)), float(len(df))) * 100 for p in parts}, "%",
            "Chi scrive di più tiene in piedi il volume della conversazione.", share=True)
        add("Investimento", "Quota dei caratteri scritti", from_df(eff, "sender", "chars_share_pct"), "%",
            "Volume di testo prodotto: misura lo sforzo meglio del numero di messaggi.", share=True)
        add("Investimento", "Lunghezza media dei messaggi", from_df(eff, "sender", "avg_chars"), "caratteri",
            "Messaggi più lunghi = più elaborazione e apertura; molto corti = stile telegrafico.")
        add("Investimento", "Domande ogni 100 messaggi", from_df(eff, "sender", "questions_per_100_msgs"), "",
            "Fare domande segnala interesse attivo per l'altro.")
        add("Iniziativa", "Avvio delle conversazioni", from_df(init, "sender", "pct"), "%",
            "Chi scrive per primo dopo 6h di silenzio: chi cerca il contatto.", share=True)
        add("Iniziativa", "Chiusura delle conversazioni", from_df(clos, "sender", "pct"), "%",
            "Chi manda l'ultimo messaggio prima di una pausa > 6h (spesso: chi resta senza risposta o chi saluta).", share=True)
        add("Iniziativa", "Inseguimento (catene di ≥2 tentativi)", from_df(dt, "sender", "double_ratio_pct"), "% dei turni",
            "Messaggi ripetuti senza risposta a distanza di ≥3 minuti: bisogno di attenzione/ansia di risposta.", higher="low")
        add("Reattività", "Tempo di risposta mediano", from_df(lat, "responder", "median_minutes"), "min",
            "Quanto fa aspettare l'altro (pause > 8h escluse).", higher="low")
        add("Reattività", "Risposte entro 1 minuto", from_df(spd, "responder", "within_1m_pct"), "%",
            "Prontezza: quota di risposte quasi immediate.", higher="high")
        add("Reattività", "Risposte dopo oltre 30 minuti", from_df(spd, "responder", "over_30m_pct"), "%",
            "Quota di risposte lente: distanza o impegni.", higher="low")
        add("Affetto", "Lessico affettivo", {p: v for p, v in per1000(self.affection_hits).items()}, "/1000 msg",
            "Parole d'amore e tenerezza scritte di proprio pugno.", higher="high")
        add("Affetto", "Cuori ed emoji affettuose", hearts, "/100 msg", "Affetto espresso con emoji.", higher="high")
        add("Affetto", "Emoji per messaggio", emoji_rate, "", "Espressività emotiva generale.")
        add("Conflitto", "Scuse", per1000(self.lexicon_counts("apology")), "/1000 msg",
            "Capacità di riconoscere un torto. Molto alte = possibile ipercompiacenza.")
        add("Conflitto", "Accuse e attribuzione di colpa", per1000(self.lexicon_counts("blame")), "/1000 msg",
            "Frasi che mettono la colpa sull'altro.", higher="low")
        add("Conflitto", "Lessico di conflitto", per1000(self.lexicon_counts("conflict")), "/1000 msg",
            "Rabbia, litigio, insulti, ultimatum.", higher="low")
        add("Conflitto", "\"Tu sempre / tu mai\"", you_abs, "/1000 parole",
            "Generalizzazioni accusatorie (critica, uno dei 4 cavalieri di Gottman).", higher="low")
        add("Conflitto", "Stonewalling verso messaggi lunghi", from_df(sw, "responder", "stonewalling_rate_pct"), "% ricevuti",
            "Risposte glaciali o silenzi > 4h a un messaggio lungo e personale dell'altro.", higher="low")
        add("Conflitto", "Paci avviate dopo una sessione tesa", from_df(cr, "sender", "repairs_initiated"), "volte",
            "Chi rompe il silenzio dopo un litigio (soglia di tensione calibrata su questa chat).", higher="high")
        add("Linguaggio", "Indice \"Noi\"", from_df(ps, "sender", "we_index_pct"), "% pronomi",
            "Pensiero di coppia: quota di 'noi' su noi+io+tu.", higher="high")
        add("Linguaggio", "Indice \"Tu\"", from_df(ps, "sender", "you_index_pct"), "% pronomi",
            "Focus sull'altro: alto in contesti di cura ma anche di critica.")
        add("Abitudini", "Messaggi notturni (01:00-05:30)", from_df(vamp, "sender", "vampire_index_pct"), "%",
            "Quota di messaggi nelle ore piccole.")
        add("Abitudini", "Vocali", kind_pct(["audio"]), "% dei messaggi", "Preferenza per la voce rispetto al testo.")
        add("Abitudini", "Foto e video", kind_pct(["image", "video"]), "% dei messaggi", "Condivisione visiva della propria giornata.")
        add("Abitudini", "Sticker e GIF", kind_pct(["sticker", "gif"]), "% dei messaggi",
            "Risposte 'leggere' non verbali (in eccesso possono sostituire il dialogo).")
        add("Abitudini", "Contenuti inoltrati", kind_pct(["inoltrato"]), "% dei messaggi", "Condivisione di contenuti altrui (news, meme, catene).")
        add("Abitudini", "Conversation killer", from_df(kil, "sender", "killer_rate_pct"), "% dei turni",
            "Turni dopo cui cala un silenzio > 4h di giorno.", higher="low")

        for ind in I:
            vals = {p: v for p, v in ind["valori"].items() if v is not None and math.isfinite(v)}
            if len(vals) < 2:
                ind.update(verdetto="Dati insufficienti", prevale=None, rapporto=None, squilibrio=0.0)
                continue
            hi = max(vals, key=vals.get)
            lo = min(vals, key=vals.get)
            vmax, vmin = vals[hi], vals[lo]
            if vmax <= 0:
                ind.update(verdetto="Assente per tutti", prevale=None, rapporto=None, squilibrio=0.0)
                continue
            ratio = vmax / vmin if vmin > 0 else float("inf")
            imbalance = (vmax - vmin) / (vmax + vmin) if (vmax + vmin) > 0 else 0.0
            if ratio < config.BALANCE_MILD_RATIO:
                verdict = "Equilibrato"
            elif ratio < config.BALANCE_STRONG_RATIO:
                verdict = f"Lieve prevalenza: {hi}"
            else:
                verdict = f"Forte prevalenza: {hi}"
            ind.update(verdetto=verdict, prevale=hi if ratio >= config.BALANCE_MILD_RATIO else None,
                       rapporto=round(ratio, 2) if math.isfinite(ratio) else None, squilibrio=round(imbalance, 3))
            ind["valori"] = {p: round(v, 2) for p, v in ind["valori"].items() if v is not None and math.isfinite(v)}

        recip_keys = {"Quota dei messaggi", "Quota dei caratteri scritti", "Avvio delle conversazioni",
                      "Tempo di risposta mediano", "Domande ogni 100 messaggi", "Lessico affettivo", "Chiusura delle conversazioni"}
        recip = [ind["squilibrio"] for ind in I if ind["indicatore"] in recip_keys and ind.get("rapporto") is not None]
        # la reciprocità ha senso solo tra due persone (nei gruppi max/min è sempre estremo)
        reciprocity = round(100 * (1 - float(np.mean(recip))), 1) if (recip and len(parts) == 2) else None
        ranked = sorted([i for i in I if i.get("rapporto")], key=lambda i: i["squilibrio"], reverse=True)
        insights = []
        insight_items: List[Dict[str, Any]] = []  # stessa informazione, strutturata (la UI la traduce)
        for ind in ranked:
            if len(insights) >= 8:
                break
            vv = [x for x in ind["valori"].values() if x is not None]
            # differenze trascurabili in valore assoluto (es. 0,4% vs 0,2%) non sono "notizie"
            if "%" in ind["unita"] and vv and (max(vv) - min(vv)) < 1.0:
                continue
            if ind["verdetto"].startswith(("Lieve", "Forte")):
                v = ind["valori"]
                parts_txt = " vs ".join(f"{p} {v[p]:g}" for p in sorted(v, key=v.get, reverse=True))
                forza = "nettamente" if ind["verdetto"].startswith("Forte") else "leggermente"
                insights.append(f"**{ind['indicatore']}** — prevale {forza} {ind['prevale']} "
                                f"({parts_txt}{' ' + ind['unita'] if ind['unita'] else ''}; rapporto {ind['rapporto'] or '∞'}×).")
                insight_items.append({"indicatore": ind["indicatore"], "prevale": ind["prevale"], "forte": forza == "nettamente",
                                      "valori": dict(v), "unita": ind["unita"], "rapporto": ind["rapporto"]})
        return {"indicators": I, "reciprocity_index": reciprocity, "insights": insights, "insight_items": insight_items,
                "thresholds": {"mild": config.BALANCE_MILD_RATIO, "strong": config.BALANCE_STRONG_RATIO}}

    # ---------------------------------------------------------- bioritmi
    def heatmap(self, sender: Optional[str] = None) -> pd.DataFrame:
        df = self.df if sender is None else self.df[self.df["sender"] == sender]
        piv = df.groupby(["weekday", "hour"]).size().unstack(fill_value=0) if not df.empty else pd.DataFrame()
        piv = piv.reindex(index=range(7), columns=range(24), fill_value=0)
        piv.index = config.WEEKDAY_NAMES_IT
        return piv

    @_memo
    def vampire_index(self) -> pd.DataFrame:
        df = self.df
        if df.empty:
            return pd.DataFrame(columns=["sender", "night_messages", "messages", "vampire_index_pct"])
        g = df.groupby("sender").agg(night_messages=("is_night_message", "sum"), messages=("id", "count")).reset_index()
        g["vampire_index_pct"] = (g["night_messages"] / g["messages"] * 100).round(2)
        return g.sort_values("vampire_index_pct", ascending=False).reset_index(drop=True)

    @_memo
    def hostility_seasonality(self) -> Dict[str, Any]:
        sdf = self.sessions_df
        if sdf.empty:
            return {"mean_heatmap": pd.DataFrame(), "high_count_heatmap": pd.DataFrame(), "by_weekday": pd.DataFrame(),
                    "by_slot": pd.DataFrame(), "peak": {}}
        s = sdf.assign(weekday=sdf["start"].dt.day_of_week, hour=sdf["start"].dt.hour,
                       high=(sdf["conflict_score"] > self.high_conflict_threshold).astype(int))
        mean_hm = s.pivot_table(index="weekday", columns="hour", values="conflict_score", aggfunc="mean")
        mean_hm = mean_hm.reindex(index=range(7), columns=range(24)).round(1)
        mean_hm.index = config.WEEKDAY_NAMES_IT
        high_hm = s.pivot_table(index="weekday", columns="hour", values="high", aggfunc="sum")
        high_hm = high_hm.reindex(index=range(7), columns=range(24)).fillna(0).astype(int)
        high_hm.index = config.WEEKDAY_NAMES_IT
        by_wd = s.groupby("weekday").agg(sessions=("session_id", "count"), mean_conflict=("conflict_score", "mean"),
                                         high_conflict_sessions=("high", "sum")).reindex(range(7)).fillna(0).reset_index()
        by_wd["weekday"] = by_wd["weekday"].map(dict(enumerate(config.WEEKDAY_NAMES_IT)))
        by_wd["mean_conflict"] = by_wd["mean_conflict"].round(2)
        s["slot"] = (s["hour"] // 3) * 3
        by_slot = s.groupby("slot").agg(sessions=("session_id", "count"), mean_conflict=("conflict_score", "mean"),
                                        high_conflict_sessions=("high", "sum")).reset_index()
        by_slot["slot"] = by_slot["slot"].apply(lambda h: f"{int(h):02d}:00-{int(h) + 3:02d}:00")
        by_slot["mean_conflict"] = by_slot["mean_conflict"].round(2)
        peak = {}
        combo = s.groupby(["weekday", "slot"]).agg(n=("session_id", "count"), mean=("conflict_score", "mean"), high=("high", "sum"))
        combo = combo[combo["n"] >= 3]
        if not combo.empty:
            best = combo.sort_values(["high", "mean"], ascending=False).iloc[0]
            wd, sl = combo.sort_values(["high", "mean"], ascending=False).index[0]
            peak = {"weekday": config.WEEKDAY_NAMES_IT[int(wd)], "slot": f"{int(sl):02d}:00-{int(sl) + 3:02d}:00",
                    "sessions": int(best["n"]), "mean_conflict": round(float(best["mean"]), 2),
                    "high_conflict_sessions": int(best["high"])}
        return {"mean_heatmap": mean_hm, "high_count_heatmap": high_hm, "by_weekday": by_wd, "by_slot": by_slot, "peak": peak}

    def top_conflict_sessions(self, n: int = 20) -> pd.DataFrame:
        sdf = self.sessions_df
        if sdf.empty:
            return pd.DataFrame()
        cols = ["session_id", "start_time", "end_time", "message_count", "participants", "initiator", "conflict_score",
                "emotional_intensity_score", "wall_of_text_count", "cold_reply_count", "accusation_lexicon_hits"]
        return sdf.sort_values("conflict_score", ascending=False).head(n)[cols].reset_index(drop=True)

    def top_intensity_sessions(self, n: int = 20, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        sdf = self.sessions_df
        if sdf.empty:
            return pd.DataFrame()
        if start:
            sdf = sdf[sdf["start_time"] >= start]
        if end:
            sdf = sdf[sdf["start_time"] <= end + " 23:59:59"]
        return sdf.sort_values("emotional_intensity_score", ascending=False).head(n)[
            ["session_id", "start_time", "message_count", "participants", "conflict_score", "emotional_intensity_score",
             "affection_lexicon_hits"]].reset_index(drop=True)

    @_memo
    def conflict_timeline(self) -> pd.DataFrame:
        sdf = self.sessions_df
        if sdf.empty:
            return pd.DataFrame(columns=["month", "mean_conflict", "high_conflict_sessions", "sessions"])
        s = sdf.assign(month=sdf["start_time"].str[:7], high=(sdf["conflict_score"] > self.high_conflict_threshold).astype(int))
        out = s.groupby("month").agg(sessions=("session_id", "count"), mean_conflict=("conflict_score", "mean"),
                                     mean_intensity=("emotional_intensity_score", "mean"),
                                     high_conflict_sessions=("high", "sum")).reset_index()
        return out.round(2)

    # ------------------------------------------------------------- dossier
    def dossier(self) -> Dict[str, Any]:
        """Dossier psicometrico globale compatto (JSON) per l'agente."""
        out: Dict[str, Any] = {
            "filters": {"start_date": self.start_date, "end_date": self.end_date},
            "participants": self.participants,
            "is_group_chat": len(self.participants) > 2,
            "definitions": {
                "initiation": "primo messaggio dopo > 6 ore di silenzio",
                "triple_text_event": "catena di >=3 tentativi consecutivi dello stesso utente distanziati >=3 min senza risposta",
                "latency": "tempo di risposta quando cambia mittente, esclusi gap > 8 ore; 'writer'->'responder'",
                "stonewalling": "messaggio > 350 caratteri scritto dal mittente (no inoltri/link) seguito da risposta < 15 caratteri (o sticker/gif) o silenzio > 4h",
                "high_conflict_session": "conflict_score sopra la soglia calibrata sulla chat (top 2% delle sessioni, max 65): vedi conflict_threshold_calibration",
                "devaluation_index": "media variazioni % (volume, affetto, cuori, lunghezza, latenza) primi vs ultimi 60 giorni; >0 = raffreddamento",
                "we_index_pct": "% pronomi 'noi' su totale pronomi noi+io+tu",
                "conflict_score": "euristico 0-100 per sessione (lessico conflitto/accusa, risposte fredde, muri di testo, punteggiatura, maiuscole, densità)",
            },
        }

        def section(name: str, fn):
            try:
                out[name] = to_jsonable(fn())
            except Exception as exc:  # una sezione fallita non deve bloccare il dossier
                out[name] = {"error": f"{type(exc).__name__}: {exc}"}

        section("overview", self.overview_kpis)
        section("initiation_rate", lambda: self.initiation_rate()["per_user"])
        section("double_triple_texting", lambda: self.double_texting()["per_user"])
        section("effort_investment", lambda: self.effort_ratio()["per_user"])
        section("latency_per_responder", lambda: self.response_latency()["per_responder"])
        section("latency_pairs", lambda: self.response_latency()["pairs"].head(30))
        section("fading_detector", self.fading_detector)
        section("love_bombing_devaluation", lambda: {
            "windows": self.love_bombing_devaluation()["windows"],
            "per_user": self.love_bombing_devaluation()["per_user"],
        })

        def stonewall():
            st_ = self.stonewalling()
            ev = st_["events"].head(6)
            return {
                "group_filter_applied": st_.get("group_filter_applied", False),
                "excluded_group_noise": st_.get("excluded_group_noise", 0),
                "per_responder": st_["per_responder"],
                "recent_examples": [
                    {"long": f'[{r.timestamp}] {r.author}: "{r.long_text[:160]}…"',
                     "reply": f'[{r.reply_timestamp}] {r.responder}: "{r.reply_text[:80]}"',
                     "silence_min": r.silence_minutes, "type": r.type, "session_id": r.session_id}
                    for r in ev.itertuples()
                ],
            }

        section("stonewalling", stonewall)

        def pronouns():
            ps = self.pronoun_shift()
            tot = ps["monthly"][ps["monthly"]["sender"] == "Totale"]
            yearly = tot.assign(year=tot["month"].str[:4]).groupby("year")[["we", "i", "you", "words"]].sum()
            yearly["we_index_pct"] = (yearly["we"] / (yearly["we"] + yearly["i"] + yearly["you"]).replace(0, np.nan) * 100).round(2)
            return {"per_user": ps["per_user"][["sender", "we", "i", "you", "you_absolute", "we_index_pct", "you_index_pct",
                                                "you_absolute_per_1000_words"]],
                    "we_index_by_year": yearly["we_index_pct"].to_dict()}

        section("pronoun_shift_gottman", pronouns)

        def apology():
            ab = self.apology_vs_blame()
            return {"per_user": ab["per_user"], "apology_examples": ab["apology_examples"][:4],
                    "blame_examples": ab["blame_examples"][:4]}

        section("apology_vs_blame", apology)

        def recovery():
            cr = self.conflict_recovery()
            return {"per_user": cr["per_user"], "high_conflict_sessions": cr.get("high_conflict_sessions", 0),
                    "followed_by_silence": cr.get("followed_by_silence", 0)}

        section("conflict_recovery", recovery)
        section("tfidf_signature_words", lambda: {s: d["term"].head(15).tolist() for s, d in self.tfidf_signature().items()})

        def affection():
            at = self.affection_timeline()
            return {"summary_top": at["summary"].head(20) if not at["summary"].empty else [],
                    "detected_pet_names": at["detected_terms"]}

        section("affection_pet_names", affection)

        def emojis():
            ef = self.emoji_fingerprint()
            top = ef["top"]
            return {"ratio": ef["ratio"],
                    "top8_per_user": {s: top[top["sender"] == s].head(8)[["emoji", "count"]].values.tolist() for s in self.participants} if not top.empty else {},
                    "biggest_shifts": ef["shift"].head(10) if not ef["shift"].empty else []}

        section("emoji_fingerprint", emojis)
        section("markov_reply_edges", lambda: self.markov_matrix()["edges"].head(25))
        section("catalysts", lambda: self.catalysts_killers()["catalysts"].head(10))
        section("conversation_killers", lambda: self.catalysts_killers()["killers"].head(10))
        section("vampire_index", self.vampire_index)
        section("hostility_seasonality", lambda: {"peak": self.hostility_seasonality()["peak"],
                                                  "by_weekday": self.hostility_seasonality()["by_weekday"],
                                                  "by_slot": self.hostility_seasonality()["by_slot"]})
        section("top_conflict_sessions", lambda: self.top_conflict_sessions(10)[
            ["session_id", "start_time", "participants", "conflict_score", "emotional_intensity_score"]])

        def yearly_volume():
            v = self.df.assign(year=self.df["month"].str[:4]).groupby(["year", "sender"]).size().unstack(fill_value=0)
            return {str(y): row.to_dict() for y, row in v.iterrows()}

        section("yearly_volume", yearly_volume)

        def balance():
            b = self.balance_scorecard()
            return {"reciprocity_index_0_100": b["reciprocity_index"],
                    "indicators": [[i["indicatore"], i["valori"], i["unita"], i["verdetto"]] for i in b["indicators"]]}

        section("balance_scorecard", balance)

        def yearly():
            y = self.yearly_scorecard()
            keep = ["Anno", "Persona", "Quota messaggi %", "Avvii conversazione %", "Latenza mediana (min)",
                    "Affetto /1000 msg", "Conflitto /1000 msg", "Vocali %"]
            return y[keep] if not y.empty else []

        section("yearly_scorecard", yearly)
        section("conflict_threshold_calibration", self.conflict_calibration_info)
        section("conflict_by_month", lambda: self.conflict_timeline().tail(24))
        return out
