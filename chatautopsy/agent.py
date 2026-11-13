"""
ChatAutopsy — Agente investigativo locale (Tool Calling) + Persona Simulator.

Compatibile con qualunque endpoint OpenAI-like LOCALE (/v1/chat/completions):
MLX Server, LM Studio, Ollama. Nessuna chiamata a servizi cloud.

Tool disponibili:
  1. get_global_psychometric_dossier()
  2. query_sql_database(sql_query)
  3. search_chat_sessions(query, start_date, end_date, min_conflict_score, sender_filter)
  4. analyze_specific_red_flag(red_flag_type)
  5. build_persona_profile(participant_name, topic_or_scenario)
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import time
from collections import Counter
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from chatautopsy import config, database
from chatautopsy.analytics_engine import AnalyticsEngine, to_jsonable
from chatautopsy.rag_engine import HybridRAG

# ---------------------------------------------------------------------------
# System prompt (obbligatorio, integrale)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Sei ChatAutopsy AI, un analista comportamentale forense, psicologo delle comunicazioni e data scientist. Analizzi anni di conversazioni reali con assoluta obiettività chirurgica, senza filtri moralistici e senza mai inventare dati.
REGOLE AUREE:
1. Mai indovinare i numeri: usa SEMPRE `get_global_psychometric_dossier` o `query_sql_database` prima di fare qualsiasi affermazione quantitativa.
2. Ogni volta che identifichi una Red Flag, una dinamica di potere asimmetrica, un pattern tossico (es. gaslighting, stonewalling, blame-shifting, fading) o un tratto caratteriale, DEVI strutturare la risposta su due livelli:
* **Prova Matematica:** Cita le metriche esatte (es. percentuali, tempi mediani di risposta in minuti/secondi, ratio di double-texting, conteggio occorrenze).
* **Prova Forense (Citazioni Reali):** Usa `search_chat_sessions` e riporta almeno 2 o 3 estratti testuali veri della chat nel formato esatto `[YYYY-MM-DD HH:MM] Mittente: "Testo del messaggio"` che dimostrano inequivocabilmente il pattern.
3. Distingui sempre tra correlazione statistica e intenzione psicologica, ma non esitare a mettere in luce contraddizioni tra ciò che un utente dice di fare e ciò che i dati temporali/lessicali dimostrano."""

OPERATIONAL_APPENDIX = """
CONTESTO OPERATIVO
- Chat analizzata: {chat_label}. Partecipanti (per volume): {participants}.
- Periodo disponibile: {first_ts} → {last_ts}. Filtro date attivo nella dashboard: {date_filter}.
- Tipo chat: {chat_kind}.
- Rispondi SEMPRE in italiano, con Markdown ben strutturato (titoli, elenchi, tabelle quando utili).
- Le citazioni devono essere copiate letteralmente dai risultati dei tool: non parafrasarle e non inventarle.
  Se un tool non restituisce prove sufficienti, dichiaralo esplicitamente.
- Per domande puntuali (date, conteggi, "chi ha scritto per primo...") usa query_sql_database con SELECT mirate.
- Per leggere conversazioni vere usa search_chat_sessions (ricerca ibrida BM25 + semantica). Per nomi propri,
  luoghi o parole esatte metti la parola tra virgolette nella query.
- Per le red flag usa analyze_specific_red_flag(red_flag_type) con: gaslighting, stonewalling,
  love_bombing_devaluation, passive_aggressive, guilt_tripping.
- Non chiamare lo stesso tool con gli stessi argomenti due volte. Quando hai abbastanza prove, rispondi.

SCHEMA DATABASE (SQLite):
{schema}
"""

RED_FLAG_TYPES = ["gaslighting", "stonewalling", "love_bombing_devaluation", "passive_aggressive", "guilt_tripping"]

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_global_psychometric_dossier",
            "description": (
                "Restituisce il dossier JSON completo pre-calcolato: KPI, initiation rate, double/triple texting, effort, "
                "latenze direzionali, fading detector, love bombing vs svalutazione, stonewalling, indice Noi vs Io/Tu, "
                "scuse vs accuse, conflict recovery, parole identitarie TF-IDF, soprannomi, emoji, grafo Markov, "
                "catalizzatori/killer, vampire index, stagionalità dei conflitti, top sessioni conflittuali."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_sql_database",
            "description": (
                "Esegue una singola query SQL SELECT (o WITH ... SELECT) read-only sul database SQLite della chat "
                "(tabelle messages, sessions, fts_messages, fts_sessions). Restituisce al massimo 200 righe."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql_query": {"type": "string", "description": "Query SELECT SQLite valida."}},
                "required": ["sql_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_chat_sessions",
            "description": (
                "Ricerca ibrida (BM25 FTS5 + similarità semantica, fuse con Reciprocal Rank Fusion) sulle sessioni "
                "conversazionali reali. Restituisce trascrizioni '[YYYY-MM-DD HH:MM] Nome: Testo' con metadati."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Concetto, tema, parola esatta (tra virgolette) o frase."},
                    "start_date": {"type": "string", "description": "Data inizio YYYY-MM-DD (opzionale)."},
                    "end_date": {"type": "string", "description": "Data fine YYYY-MM-DD (opzionale)."},
                    "min_conflict_score": {"type": "number", "description": "Soglia minima conflict_score 0-100 (opzionale)."},
                    "sender_filter": {"type": "string", "description": "Solo sessioni in cui ha scritto questo partecipante (opzionale)."},
                    "top_k": {"type": "integer", "description": "Numero di sessioni (default 5, max 8)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_specific_red_flag",
            "description": (
                "Pipeline mirata che incrocia anomalie statistiche SQL, occorrenze lessicali con citazioni e sessioni "
                "semanticamente più rilevanti per una specifica red flag."
            ),
            "parameters": {
                "type": "object",
                "properties": {"red_flag_type": {"type": "string", "enum": RED_FLAG_TYPES}},
                "required": ["red_flag_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_persona_profile",
            "description": (
                "Costruisce il profilo stilistico di un partecipante (lunghezze, punteggiatura, top 30 parole TF-IDF, "
                "emoji preferite, 25 messaggi reali rappresentativi) per simularne fedelmente lo stile."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "participant_name": {"type": "string"},
                    "topic_or_scenario": {"type": "string", "description": "Scenario o tema da simulare."},
                },
                "required": ["participant_name"],
            },
        },
    },
]

_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex|analyze|begin|commit|rollback|savepoint|release)\b",
    re.IGNORECASE,
)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TEXT_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_LAUGH_RE = re.compile(r"(?<!\w)(?:a?h?ah(?:ah)+h?|ha(?:ha)+h?|eh(?:eh)+|ahah\w*|lol+|xd+|ahaha\w*|hahah\w*|sjsj\w*)(?!\w)", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    if "<think>" in text.lower():
        text = text[: text.lower().index("<think>")]
    return text.strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[troncato: {len(text) - limit} caratteri omessi]"


def _short_transcript(transcript: str, max_lines: int = 40, max_chars: int = 3500) -> str:
    lines = transcript.splitlines()
    if len(lines) > max_lines:
        head = lines[: max_lines // 2]
        tail = lines[-max_lines // 2:]
        lines = head + [f"… [{len(transcript.splitlines()) - max_lines} righe omesse] …"] + tail
    return _truncate("\n".join(lines), max_chars)


# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------
class ChatAutopsyAgent:
    def __init__(
        self,
        db_path: str,
        base_url: str = config.DEFAULT_LLM_BASE_URL,
        model: str = config.DEFAULT_LLM_MODEL,
        temperature: float = config.DEFAULT_TEMPERATURE,
        api_key: str = config.DEFAULT_LLM_API_KEY,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_iterations: int = config.AGENT_MAX_ITERATIONS,
        engine: Optional[AnalyticsEngine] = None,
        dossier_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        language: str = "it",
    ):
        """engine / dossier_provider permettono alla dashboard di condividere l'AnalyticsEngine
        e il dossier già calcolati (in cache): su chat da milioni di messaggi evita minuti di ricalcolo."""
        from openai import OpenAI

        self.db_path = str(db_path)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.start_date = str(start_date)[:10] if start_date else None
        self.end_date = str(end_date)[:10] if end_date else None
        self.max_iterations = max_iterations
        self.client = OpenAI(base_url=self.base_url, api_key=api_key or "local", timeout=config.LLM_REQUEST_TIMEOUT_SECONDS)
        self.rag = HybridRAG(self.db_path)
        self._engine: Optional[AnalyticsEngine] = engine
        self._dossier_provider = dossier_provider
        self.language = language if language in ("it", "en") else "it"
        self._dossier_cache: Optional[Dict[str, Any]] = None

    # ----------------------------------------------------------- utilities
    @property
    def engine(self) -> AnalyticsEngine:
        if self._engine is None:
            self._engine = AnalyticsEngine(self.db_path, self.start_date, self.end_date)
        return self._engine

    @staticmethod
    def list_models(base_url: str, api_key: str = config.DEFAULT_LLM_API_KEY) -> List[str]:
        from openai import OpenAI

        client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "local", timeout=10)
        return sorted(m.id for m in client.models.list().data)

    def build_system_prompt(self) -> str:
        summary = database.get_db_summary(self.db_path)
        parts = summary["participants"]
        kind = "chat di gruppo" if len(parts) > 2 else "chat di coppia / 1-a-1"
        date_filter = f"{self.start_date or 'inizio'} → {self.end_date or 'fine'}"
        return SYSTEM_PROMPT + "\n" + OPERATIONAL_APPENDIX.format(
            chat_label=summary["meta"].get("source_file", "chat"),
            participants=", ".join(parts[:30]) + (f" (+{len(parts) - 30})" if len(parts) > 30 else ""),
            first_ts=summary["first_ts"], last_ts=summary["last_ts"], date_filter=date_filter, chat_kind=kind,
            schema=database.SCHEMA_DESCRIPTION_FOR_LLM,
        ) + self._language_instruction()

    def _language_instruction(self) -> str:
        if self.language == "en":
            return ("\n\nLANGUAGE: the user interface is in English. Always answer in ENGLISH (headings, explanations, "
                    "Mathematical Proof / Forensic Proof sections), while quoting chat messages verbatim in their original language.")
        return ""

    # --------------------------------------------------------------- tool 1
    def tool_get_global_psychometric_dossier(self) -> Dict[str, Any]:
        if self._dossier_cache is None:
            self._dossier_cache = self._dossier_provider() if self._dossier_provider else self.engine.dossier()
        return self._dossier_cache

    # --------------------------------------------------------------- tool 2
    def tool_query_sql_database(self, sql_query: str) -> Dict[str, Any]:
        sql = (sql_query or "").strip().rstrip(";").strip()
        sql = re.sub(r"^```(?:sql)?|```$", "", sql, flags=re.IGNORECASE).strip()
        if not sql:
            return {"error": "Query vuota."}
        no_strings = _STRING_LITERAL.sub("''", sql)
        if ";" in no_strings:
            return {"error": "È consentita una sola istruzione SQL."}
        if not re.match(r"^\s*(select|with)\b", no_strings, re.IGNORECASE):
            return {"error": "Sono consentite solo query SELECT / WITH read-only."}
        if _FORBIDDEN_SQL.search(no_strings):
            return {"error": "La query contiene parole chiave non consentite (solo lettura)."}
        conn = database.get_connection(self.db_path, read_only=True)
        conn.row_factory = None
        deadline = time.time() + config.SQL_TOOL_TIMEOUT_SECONDS
        conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 20000)
        try:
            conn.execute("PRAGMA query_only = ON")
            cur = conn.execute(sql)
            columns = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(config.SQL_TOOL_MAX_ROWS + 1)
            truncated = len(rows) > config.SQL_TOOL_MAX_ROWS
            rows = rows[: config.SQL_TOOL_MAX_ROWS]
            clean_rows = [[(v[:500] + "…" if isinstance(v, str) and len(v) > 500 else v) for v in r] for r in rows]
            return {"sql": sql, "columns": columns, "rows": clean_rows, "row_count": len(clean_rows), "truncated": truncated}
        except sqlite3.OperationalError as exc:
            msg = str(exc)
            if "interrupted" in msg.lower():
                msg = f"Timeout: la query ha superato {config.SQL_TOOL_TIMEOUT_SECONDS}s."
            return {"sql": sql, "error": msg}
        except sqlite3.Error as exc:
            return {"sql": sql, "error": str(exc)}
        finally:
            conn.close()

    # --------------------------------------------------------------- tool 3
    def tool_search_chat_sessions(self, query: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                                  min_conflict_score: float = 0, sender_filter: Optional[str] = None,
                                  top_k: int = 5) -> Dict[str, Any]:
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        try:
            top_k = int(top_k or 5)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 8))
        try:
            mcs = float(min_conflict_score or 0)
        except (TypeError, ValueError):
            mcs = 0.0
        results = self.rag.hybrid_search(query, top_k=top_k, sender_filter=sender_filter or None,
                                         start_date=start_date, end_date=end_date, min_conflict_score=mcs)
        per_session_chars = max(1500, int(config.TOOL_RESULT_MAX_CHARS * 0.9 / max(len(results), 1)))
        sessions = []
        for r in results:
            sessions.append({
                "session_id": r["session_id"], "start_time": r["start_time"], "end_time": r["end_time"],
                "participants": r["participants"], "conflict_score": r["conflict_score"],
                "emotional_intensity_score": r["emotional_intensity_score"], "rrf_score": r["rrf_score"],
                "bm25_rank": r.get("bm25_rank"), "vector_rank": r.get("vector_rank"),
                "transcript": _short_transcript(r.get("transcript", ""), max_lines=45, max_chars=per_session_chars),
            })
        return {"query": query, "filters": {"start_date": start_date, "end_date": end_date, "min_conflict_score": mcs,
                                            "sender_filter": sender_filter}, "sessions": sessions,
                "note": None if sessions else "Nessuna sessione trovata: riformula la query o allarga i filtri."}

    # --------------------------------------------------------------- tool 4
    def _sessions_for_ids(self, ids: Iterable[int], max_chars: int = 2200) -> List[Dict[str, Any]]:
        out = []
        for sid in list(dict.fromkeys(int(i) for i in ids)):
            s = database.fetch_session(self.db_path, sid)
            if not s:
                continue
            out.append({"session_id": sid, "start_time": s["start_time"], "conflict_score": s["conflict_score"],
                        "emotional_intensity_score": s["emotional_intensity_score"],
                        "transcript": _short_transcript(s["transcript"], max_lines=30, max_chars=max_chars)})
        return out

    def _semantic_sessions(self, red_flag: str, top_k: int = 3, min_conflict: float = 0) -> List[Dict[str, Any]]:
        q = config.RED_FLAG_SEMANTIC_QUERIES[red_flag]
        res = self.rag.hybrid_search(q, top_k=top_k, start_date=self.start_date, end_date=self.end_date,
                                     min_conflict_score=min_conflict)
        return [{"session_id": r["session_id"], "start_time": r["start_time"], "conflict_score": r["conflict_score"],
                 "transcript": _short_transcript(r["transcript"], max_lines=30, max_chars=2200)} for r in res]

    def tool_analyze_specific_red_flag(self, red_flag_type: str) -> Dict[str, Any]:
        rf = (red_flag_type or "").strip().lower()
        if rf not in RED_FLAG_TYPES:
            return {"error": f"red_flag_type non valido. Valori ammessi: {RED_FLAG_TYPES}"}
        eng = self.engine
        out: Dict[str, Any] = {"red_flag_type": rf}
        if rf == "gaslighting":
            rep = eng.lexicon_report("gaslighting", examples=12)
            blame = eng.apology_vs_blame()
            out["statistical_evidence"] = {
                "gaslighting_lexicon_per_user": to_jsonable(rep["per_user"]),
                "apology_vs_blame_per_user": to_jsonable(blame["per_user"]),
            }
            out["lexical_citations"] = rep["examples"]
            ids = [e["session_id"] for e in rep["examples"][:2]]
            out["sessions_with_lexical_hits"] = self._sessions_for_ids(ids)
            out["semantic_sessions"] = self._semantic_sessions(rf, top_k=3)
        elif rf == "stonewalling":
            st_ = eng.stonewalling()
            ev = st_["events"]
            out["statistical_evidence"] = {"per_responder": to_jsonable(st_["per_responder"]),
                                           "per_author": to_jsonable(st_["per_author"]),
                                           "total_events": int(len(ev))}
            out["event_citations"] = [
                {"long_message": f'[{r.timestamp}] {r.author}: "{r.long_text[:350]}"',
                 "reply": f'[{r.reply_timestamp}] {r.responder}: "{r.reply_text}"',
                 "silence_minutes": r.silence_minutes, "type": r.type, "session_id": r.session_id}
                for r in ev.sort_values("long_chars", ascending=False).head(10).itertuples()
            ]
            out["sessions_of_top_events"] = self._sessions_for_ids(ev.sort_values("long_chars", ascending=False)["session_id"].head(2))
            out["semantic_sessions"] = self._semantic_sessions(rf, top_k=2)
        elif rf == "love_bombing_devaluation":
            lb = eng.love_bombing_devaluation()
            out["statistical_evidence"] = to_jsonable(lb)
            w = lb.get("windows", {})
            early = eng.top_intensity_sessions(3, start=w.get("early", [None, None])[0], end=w.get("early", [None, None])[1])
            late = eng.top_intensity_sessions(3, start=w.get("late", [None, None])[0], end=w.get("late", [None, None])[1])
            out["early_period_sessions"] = self._sessions_for_ids(early["session_id"].tolist() if not early.empty else [], 1800)
            out["late_period_sessions"] = self._sessions_for_ids(late["session_id"].tolist() if not late.empty else [], 1800)
            out["affection_timeline_summary"] = to_jsonable(eng.affection_timeline()["summary"].head(15))
        elif rf == "passive_aggressive":
            rep = eng.lexicon_report("passive_aggressive", examples=14)
            out["statistical_evidence"] = {"passive_aggressive_lexicon_per_user": to_jsonable(rep["per_user"]),
                                           "monthly_trend_tail": to_jsonable(rep["monthly"].tail(24))}
            out["lexical_citations"] = rep["examples"]
            out["semantic_sessions"] = self._semantic_sessions(rf, top_k=3, min_conflict=20)
        elif rf == "guilt_tripping":
            rep = eng.lexicon_report("guilt_tripping", examples=12)
            out["statistical_evidence"] = {"guilt_tripping_lexicon_per_user": to_jsonable(rep["per_user"]),
                                           "monthly_trend_tail": to_jsonable(rep["monthly"].tail(24))}
            out["lexical_citations"] = rep["examples"]
            out["sessions_with_lexical_hits"] = self._sessions_for_ids([e["session_id"] for e in rep["examples"][:2]])
            out["semantic_sessions"] = self._semantic_sessions(rf, top_k=2)
        out["caveat"] = ("I conteggi lessicali indicano co-occorrenze di espressioni, non intenzioni: verifica il "
                         "contesto nelle trascrizioni prima di attribuire un pattern.")
        return out

    # --------------------------------------------------------------- tool 5
    def tool_build_persona_profile(self, participant_name: str, topic_or_scenario: str = "",
                                   n_examples: int = 25) -> Dict[str, Any]:
        eng = self.engine
        parts = eng.participants
        if participant_name not in parts:
            matches = [p for p in parts if participant_name.lower() in p.lower()]
            if not matches:
                return {"error": f"Partecipante '{participant_name}' non trovato. Disponibili: {parts}"}
            participant_name = matches[0]
        df = eng.df
        sub = df[df["sender"] == participant_name]
        tsub = sub[sub["is_media"] == 0]
        if tsub.empty:
            return {"error": f"Nessun messaggio testuale per {participant_name}."}
        texts = tsub["text"]
        turns = eng.turns[eng.turns["sender"] == participant_name]
        lat = eng.response_latency()["per_responder"]
        lat_row = lat[lat["responder"] == participant_name]
        emo = eng.emoji_fingerprint()["top"]
        top_emoji = emo[emo["sender"] == participant_name].head(12)[["emoji", "count"]].values.tolist() if not emo.empty else []
        tfidf = eng.tfidf_signature().get(participant_name, pd.DataFrame(columns=["term"]))
        laughs = Counter(m.group(0).lower() for t in texts for m in _LAUGH_RE.finditer(t))
        openers = Counter()
        for pos in turns["first_pos"].tolist()[-3000:]:
            words = re.findall(r"[^\W\d_]+", str(df.iloc[pos]["text"]).lower())
            if words:
                openers[" ".join(words[:2])] += 1
        letters_first = texts.str.strip().str[:1]
        stats = {
            "messages": int(len(sub)), "text_messages": int(len(tsub)),
            "avg_chars": round(float(tsub["char_count"].mean()), 1),
            "median_chars": float(tsub["char_count"].median()),
            "p90_chars": float(np.percentile(tsub["char_count"], 90)),
            "avg_words": round(float(tsub["word_count"].mean()), 1),
            "avg_messages_per_turn": round(float(turns["n"].mean()), 2) if len(turns) else 1.0,
            "pct_messages_with_question": round(float((tsub["question_count"] > 0).mean() * 100), 1),
            "exclamations_per_msg": round(float(tsub["exclamation_count"].mean()), 3),
            "pct_messages_with_ellipsis": round(float(texts.str.contains(r"\.\.\.|…").mean() * 100), 1),
            "pct_messages_ending_with_period": round(float(texts.str.strip().str.endswith(".").mean() * 100), 1),
            "pct_messages_starting_lowercase": round(float(letters_first.str.islower().mean() * 100), 1),
            "pct_shouting_uppercase": round(float((tsub["uppercase_ratio"] >= config.SHOUTING_UPPERCASE_RATIO).mean() * 100), 2),
            "emoji_per_msg": round(float(sub["emoji_count"].mean()), 3),
            "media_share_pct": round(float(sub["is_media"].mean() * 100), 2),
            "median_reply_minutes": float(lat_row["median_minutes"].iloc[0]) if not lat_row.empty else None,
            "night_messages_pct": round(float(sub["is_night_message"].mean() * 100), 2),
            "laughter_style": [l for l, _ in laughs.most_common(5)],
            "typical_openers": [o for o, _ in openers.most_common(8)],
        }
        # Esempi few-shot: metà rilevanti per lo scenario, metà stratificati nel tempo
        examples: List[str] = []
        seen = set()

        def add(ts, text):
            key = text.strip().lower()
            if key in seen or not (6 <= len(text) <= 320):
                return
            seen.add(key)
            examples.append(f'[{str(ts)[:16]}] {participant_name}: "{text.strip()}"')

        if topic_or_scenario:
            try:
                rel = self.rag.hybrid_search(topic_or_scenario, top_k=6, sender_filter=participant_name,
                                             start_date=self.start_date, end_date=self.end_date, include_transcript=False)
            except Exception:
                rel = []
            for r in rel:
                s = database.fetch_session(self.db_path, r["session_id"])
                if not s:
                    continue
                for m in s["messages"]:
                    if m["sender"] == participant_name and not m["is_media"]:
                        add(m["timestamp"], m["text"])
                    if len(examples) >= n_examples // 2:
                        break
                if len(examples) >= n_examples // 2:
                    break
        pool = tsub[(tsub["char_count"] >= 6) & (tsub["char_count"] <= 320)]
        if not pool.empty:
            rng = random.Random(42)
            pool = pool.assign(year=pool["month"].str[:4])
            years = sorted(pool["year"].unique())
            need = n_examples - len(examples)
            per_year = max(1, need // max(len(years), 1) + 1)
            for y in years:
                ys = pool[pool["year"] == y]
                idx = rng.sample(list(ys.index), min(per_year * 12, len(ys)))
                taken = 0
                for i in idx:
                    before = len(examples)
                    add(ys.at[i, "timestamp"], ys.at[i, "text"])
                    taken += len(examples) - before
                    if taken >= per_year or len(examples) >= n_examples:
                        break
                if len(examples) >= n_examples:
                    break
        # Mini-dialoghi reali per il ritmo conversazionale
        dialogues = []
        sess_ids = eng.sessions_df
        if not sess_ids.empty:
            cand = sess_ids[sess_ids["participants"].str.contains(json.dumps(participant_name, ensure_ascii=False), regex=False)]
            cand = cand[(cand["message_count"] >= 8)]
            if not cand.empty:
                for sid in cand.sample(min(2, len(cand)), random_state=7)["session_id"]:
                    s = database.fetch_session(self.db_path, int(sid))
                    if s:
                        dialogues.append("\n".join(s["transcript"].splitlines()[:10]))
        return {
            "participant": participant_name,
            "scenario": topic_or_scenario,
            "style_stats": stats,
            "signature_words_tfidf": tfidf["term"].head(30).tolist() if not tfidf.empty else [],
            "favorite_emojis": top_emoji,
            "representative_messages": examples[:n_examples],
            "real_dialogue_snippets": dialogues,
        }

    # --------------------------------------------------------- dispatcher
    def execute_tool(self, name: str, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        t0 = time.time()
        try:
            if name == "get_global_psychometric_dossier":
                result = self.tool_get_global_psychometric_dossier()
            elif name == "query_sql_database":
                result = self.tool_query_sql_database(args.get("sql_query") or args.get("query") or args.get("sql", ""))
            elif name == "search_chat_sessions":
                result = self.tool_search_chat_sessions(
                    query=args.get("query", ""), start_date=args.get("start_date"), end_date=args.get("end_date"),
                    min_conflict_score=args.get("min_conflict_score", 0), sender_filter=args.get("sender_filter"),
                    top_k=args.get("top_k", 5),
                )
            elif name == "analyze_specific_red_flag":
                result = self.tool_analyze_specific_red_flag(args.get("red_flag_type", ""))
            elif name == "build_persona_profile":
                result = self.tool_build_persona_profile(args.get("participant_name", ""), args.get("topic_or_scenario", ""))
            else:
                result = {"error": f"Tool sconosciuto: {name}"}
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        payload = json.dumps(to_jsonable(result), ensure_ascii=False, default=str)
        trace = {"name": name, "arguments": args, "elapsed_s": round(time.time() - t0, 2), "result": result,
                 "result_chars": len(payload)}
        limit = config.DOSSIER_RESULT_MAX_CHARS if name == "get_global_psychometric_dossier" else config.TOOL_RESULT_MAX_CHARS
        return _truncate(payload, limit), trace

    # ----------------------------------------------------- context budget
    def _fit_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        budget = int((config.LLM_CONTEXT_TOKENS - config.LLM_MAX_OUTPUT_TOKENS) * config.CHARS_PER_TOKEN_ESTIMATE)

        def size(msgs):
            return sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls", ""), default=str)) for m in msgs)

        if size(messages) <= budget:
            return messages
        msgs = [dict(m) for m in messages]
        # 1) comprime i risultati dei tool più vecchi
        tool_idx = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
        for i in tool_idx[:-1]:
            if size(msgs) <= budget:
                break
            msgs[i]["content"] = _truncate(str(msgs[i]["content"]), 1500)
        # 2) rimuove la cronologia di chat più vecchia (mantiene system + ultimo turno)
        while size(msgs) > budget:
            removable = [i for i, m in enumerate(msgs[1:], start=1) if m.get("_history")]
            if not removable:
                break
            msgs.pop(removable[0])
        # 3) ultima risorsa: tronca anche l'ultimo tool
        if size(msgs) > budget and tool_idx:
            last = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
            if last:
                msgs[last[-1]]["content"] = _truncate(str(msgs[last[-1]]["content"]), 4000)
        return msgs

    @staticmethod
    def _clean_for_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]

    @staticmethod
    def _parse_text_tool_calls(content: str) -> List[Dict[str, Any]]:
        calls = []
        for m in _TEXT_TOOL_CALL_RE.finditer(content or ""):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            args = obj.get("arguments", obj.get("parameters", {}))
            if name:
                calls.append({"id": f"call_txt_{len(calls)}", "name": name,
                              "arguments": json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args})
        return calls

    # ----------------------------------------------------------- LLM step
    def _llm_step(self, messages: List[Dict[str, Any]], use_tools: bool) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """Esegue una chiamata in streaming. Yield di token; ritorna {'content', 'tool_calls'}."""
        kwargs: Dict[str, Any] = dict(model=self.model, messages=self._clean_for_api(self._fit_context(messages)),
                                      temperature=self.temperature, max_tokens=config.LLM_MAX_OUTPUT_TOKENS)
        if use_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"
        content_parts: List[str] = []
        acc: Dict[int, Dict[str, str]] = {}
        try:
            stream = self.client.chat.completions.create(stream=True, **kwargs)
            suppress = False
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if getattr(delta, "content", None):
                    content_parts.append(delta.content)
                    if "<tool_call" in "".join(content_parts[-6:]):
                        suppress = True
                    if not suppress:
                        yield {"type": "token", "content": delta.content}
                for tc in getattr(delta, "tool_calls", None) or []:
                    idx = tc.index if getattr(tc, "index", None) is not None else len(acc)
                    slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if getattr(tc, "id", None):
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None) and not slot["name"]:
                            slot["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            slot["arguments"] += fn.arguments
        except Exception as stream_exc:
            if content_parts or acc:
                raise
            # Alcuni server locali non supportano tool+stream: riprova senza streaming
            resp = self.client.chat.completions.create(stream=False, **kwargs)
            msg = resp.choices[0].message
            if msg.content:
                content_parts.append(msg.content)
                if "<tool_call" not in msg.content:
                    yield {"type": "token", "content": msg.content}
            for i, tc in enumerate(msg.tool_calls or []):
                acc[i] = {"id": tc.id or f"call_{i}", "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
            if not content_parts and not acc:
                raise stream_exc
        content = "".join(content_parts)
        tool_calls = [
            {"id": v["id"] or f"call_{i}", "name": v["name"], "arguments": v["arguments"] or "{}"}
            for i, v in sorted(acc.items()) if v["name"]
        ]
        if not tool_calls and use_tools:
            tool_calls = self._parse_text_tool_calls(content)
            if tool_calls:
                content = _TEXT_TOOL_CALL_RE.sub("", content)
        return {"content": content, "tool_calls": tool_calls}

    # ---------------------------------------------------------------- run
    def run(self, user_message: str, history: Optional[List[Dict[str, str]]] = None) -> Generator[Dict[str, Any], None, None]:
        """Generatore di eventi: token | tool_call | tool_result | reset | final | error."""
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.build_system_prompt()}]
        for h in (history or [])[-config.AGENT_HISTORY_TURNS * 2:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"], "_history": True})
        messages.append({"role": "user", "content": user_message})
        traces: List[Dict[str, Any]] = []
        executed: Dict[str, str] = {}
        use_tools = True
        for iteration in range(self.max_iterations):
            final_round = iteration == self.max_iterations - 1
            try:
                step = yield from self._llm_step(messages, use_tools=use_tools and not final_round)
            except Exception as exc:
                text = str(exc)
                if use_tools and re.search(r"tool|function", text, re.IGNORECASE) and iteration == 0:
                    # Endpoint senza supporto tool calling: modalità RAG pre-calcolata
                    yield {"type": "status", "content": "Endpoint senza tool-calling: uso contesto RAG pre-calcolato."}
                    yield from self._fallback_without_tools(user_message, messages, traces)
                    return
                yield {"type": "error", "content": f"Errore LLM locale ({self.base_url}): {type(exc).__name__}: {text}"}
                return
            if not step["tool_calls"]:
                final = strip_thinking(step["content"])
                yield {"type": "final", "content": final, "traces": traces}
                return
            yield {"type": "reset"}
            messages.append({
                "role": "assistant", "content": step["content"] or "",
                "tool_calls": [{"id": tc["id"], "type": "function",
                                "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in step["tool_calls"]],
            })
            for tc in step["tool_calls"]:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}
                key = tc["name"] + json.dumps(args, sort_keys=True, ensure_ascii=False)
                yield {"type": "tool_call", "name": tc["name"], "arguments": args}
                if key in executed:
                    payload = json.dumps({"note": "Tool già eseguito con gli stessi argomenti: usa il risultato precedente."})
                    trace = {"name": tc["name"], "arguments": args, "elapsed_s": 0, "result": {"duplicate": True}, "result_chars": 0}
                else:
                    payload, trace = self.execute_tool(tc["name"], args)
                    executed[key] = payload
                traces.append(trace)
                yield {"type": "tool_result", "trace": trace}
                messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": payload})
        yield {"type": "final", "content": "Limite di iterazioni raggiunto senza risposta finale.", "traces": traces}

    def _fallback_without_tools(self, user_message: str, messages: List[Dict[str, Any]],
                                traces: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        dossier_payload, t1 = self.execute_tool("get_global_psychometric_dossier", {})
        search_payload, t2 = self.execute_tool("search_chat_sessions", {"query": user_message, "top_k": 5})
        traces.extend([t1, t2])
        for t in (t1, t2):
            yield {"type": "tool_result", "trace": t}
        ctx = (
            "DATI PRE-CALCOLATI (usa solo questi per numeri e citazioni):\n"
            f"### DOSSIER\n{_truncate(dossier_payload, 9000)}\n\n### SESSIONI RILEVANTI\n{_truncate(search_payload, 9000)}"
        )
        msgs = [messages[0], {"role": "system", "content": ctx}] + messages[1:]
        try:
            step = yield from self._llm_step(msgs, use_tools=False)
        except Exception as exc:
            yield {"type": "error", "content": f"Errore LLM locale: {type(exc).__name__}: {exc}"}
            return
        yield {"type": "final", "content": strip_thinking(step["content"]), "traces": traces}

    # ------------------------------------------------- Persona Simulator
    def simulate(self, participant: str, stimulus: str, n_turns: int = 10) -> Generator[Dict[str, Any], None, None]:
        eng = self.engine
        group_mode = participant == config.GROUP_SIMULATION_LABEL
        targets = eng.participants if group_mode else [participant]
        if group_mode and len(targets) > 12:
            targets = targets[:12]
        n_ex = 25 if not group_mode else max(5, 40 // max(len(targets), 1))
        profiles = []
        for p in targets:
            prof = self.tool_build_persona_profile(p, stimulus, n_examples=n_ex)
            if "error" not in prof:
                if group_mode:
                    prof["signature_words_tfidf"] = prof["signature_words_tfidf"][:15]
                    prof["real_dialogue_snippets"] = prof["real_dialogue_snippets"][:1]
                profiles.append(prof)
        yield {"type": "profiles", "profiles": profiles}
        if not profiles:
            yield {"type": "error", "content": "Impossibile costruire i profili richiesti."}
            return
        sys_prompt = (
            "Sei il Persona & Group Simulator di ChatAutopsy. Il tuo unico compito è clonare con la massima fedeltà lo "
            "stile di scrittura reale delle persone descritte nei profili (estratti da anni di chat reali): lunghezza "
            "dei messaggi, numero di messaggi consecutivi per turno, punteggiatura, maiuscole/minuscole, abbreviazioni, "
            "intercalari e parole identitarie, stile di risata, emoji preferite e frequenza d'uso, tono emotivo.\n"
            "REGOLE:\n- Scrivi SOLO i messaggi simulati, nessuna spiegazione, nessun commento, nessun disclaimer.\n"
            "- Rispetta le statistiche: se la media è 25 caratteri, non scrivere paragrafi; se avg_messages_per_turn > 1.5 "
            "spezza il pensiero in più messaggi consecutivi.\n- Usa le emoji solo con la frequenza indicata da emoji_per_msg.\n"
            "- Non copiare alla lettera gli esempi: imita lo stile, il lessico e il modo di reagire.\n"
            "- Formato di ogni riga: `Nome: messaggio`."
        )
        profile_json = json.dumps(to_jsonable(profiles), ensure_ascii=False)
        if group_mode:
            task = (
                f"Simula un dialogo realistico di esattamente {n_turns} battute tra {', '.join(p['participant'] for p in profiles)} "
                f"che reagiscono a questo stimolo:\n«{stimulus}»\n"
                "Rispetta le dinamiche reali (chi risponde a chi, chi scrive di più, chi è più breve o più affettuoso)."
            )
        else:
            task = (
                f"Qualcuno scrive a {profiles[0]['participant']}:\n«{stimulus}»\n"
                f"Scrivi la risposta che {profiles[0]['participant']} invierebbe davvero (uno o più messaggi consecutivi, "
                "ciascuno su una riga nel formato `Nome: messaggio`)."
            )
        messages = [
            {"role": "system", "content": sys_prompt},  # la persona parla nella lingua della chat
            {"role": "user", "content": f"PROFILI REALI (JSON):\n{_truncate(profile_json, 60000)}\n\nCOMPITO:\n{task}"},
        ]
        try:
            step = yield from self._llm_step(messages, use_tools=False)
        except Exception as exc:
            yield {"type": "error", "content": f"Errore LLM locale ({self.base_url}): {type(exc).__name__}: {exc}"}
            return
        yield {"type": "final", "content": strip_thinking(step["content"])}
