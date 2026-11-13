"""
ChatAutopsy — Dashboard Streamlit (100% locale, italiano / inglese).

Avvio:  chatautopsy   (oppure: streamlit run chatautopsy/app.py)
"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from chatautopsy import config, database, i18n, llm_manager, rag_engine, ui
from chatautopsy.analytics_engine import AnalyticsEngine
from chatautopsy.i18n import t, tv
from chatautopsy.ui import callout, explain, fmt_minutes, fmt_num, kpi, section, short_name, show, table

st.set_page_config(page_title="ChatAutopsy", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()
st.session_state.setdefault("lang", i18n.detect_lang())


# ---------------------------------------------------------------------------
# Cache & helper
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=2)
def get_engine(db_path: str, signature: float, start: Optional[str], end: Optional[str]) -> AnalyticsEngine:
    """Un solo AnalyticsEngine per (DB, filtro date): DataFrame, lessici e turni calcolati una volta
    e condivisi da tutte le analisi e dall'agente (decisivo su chat da milioni di messaggi)."""
    return AnalyticsEngine(db_path, start, end)


def current_engine() -> AnalyticsEngine:
    return get_engine(st.session_state.db_path, database.db_signature(st.session_state.db_path),
                      st.session_state.get("start_date"), st.session_state.get("end_date"))


def clear_all_caches() -> None:
    st.cache_data.clear()
    get_engine.clear()


@st.cache_data(show_spinner=False, max_entries=512)
def run_analytic(db_path: str, signature: float, start: Optional[str], end: Optional[str], method: str,
                 kwargs_json: str = "{}") -> Any:
    eng = get_engine(db_path, signature, start, end)
    return getattr(eng, method)(**json.loads(kwargs_json))


def A(method: str, **kwargs) -> Any:
    """Shortcut per chiamare un metodo dell'AnalyticsEngine con cache e filtri globali."""
    return run_analytic(st.session_state.db_path, database.db_signature(st.session_state.db_path),
                        st.session_state.get("start_date"), st.session_state.get("end_date"), method,
                        json.dumps(kwargs, sort_keys=True, default=str))


@st.cache_data(show_spinner=False)
def db_summary(db_path: str, signature: float) -> Dict[str, Any]:
    return database.get_db_summary(db_path)


def html_escape(s: str) -> str:
    return html.escape(s, quote=False)


def show_session(session_id: int) -> None:
    s = database.fetch_session(st.session_state.db_path, int(session_id))
    if not s:
        st.warning(t("Sessione non trovata."))
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("Sessione"), f"#{s['session_id']}")
    c2.metric(t("Tensione (0-100)"), f"{s['conflict_score']:.1f}")
    c3.metric(t("Intensità emotiva"), f"{s['emotional_intensity_score']:.1f}")
    c4.metric(t("Messaggi"), s["message_count"])
    st.caption(t("{start} → {end} · Partecipanti: {who}", start=s["start_time"], end=s["end_time"], who=", ".join(s["participants"]))
               + (t(" · continuazione (overlap 5 msg)") if s["is_continuation"] else ""))
    st.markdown(f"<div class='transcript'>{html_escape(s['transcript'])}</div>", unsafe_allow_html=True)


def selected_row(event) -> Optional[int]:
    try:
        rows = event.selection.rows
        return rows[0] if rows else None
    except AttributeError:
        return None


def verdict_text(v: str) -> str:
    if not v:
        return ""
    if v.startswith("Equilibrato"):
        return "🟢 " + tv(v)
    if v.startswith("Lieve"):
        return "🟡 " + tv(v)
    if v.startswith("Forte"):
        return "🔴 " + tv(v)
    return "⚪ " + tv(v)


def rolling(df: pd.DataFrame, by: str, col: str, window: int) -> pd.DataFrame:
    """Media mobile per serie (window mesi); window 1 = nessuna smussatura."""
    if window <= 1 or df.empty:
        return df
    out = df.sort_values("month").copy()
    out[col] = out.groupby(by)[col].transform(lambda s: s.rolling(window, min_periods=1).mean())
    return out


def choose(widget, label: str, options: Dict[str, str], key: str, **kwargs) -> str:
    """Widget a scelta singola con etichette tradotte: mostra le etichette, restituisce la chiave interna.
    La chiave del widget include la lingua, così cambiare lingua non lascia uno stato con etichette dell'altra."""
    keys_ = list(options)
    labels_ = [options[k_] for k_ in keys_]
    picked = widget(label, labels_, key=f"{key}_{i18n.get_lang()}", **kwargs)
    return keys_[labels_.index(picked)] if picked in labels_ else keys_[0]


def choose_many(label: str, options: Dict[str, str], key: str, default: Optional[List[str]] = None) -> List[str]:
    """Multiselect con etichette tradotte che restituisce le chiavi interne."""
    keys_ = list(options)
    labels_ = [options[k_] for k_ in keys_]
    dflt = [options[k_] for k_ in (default if default is not None else keys_) if k_ in options]
    picked = st.multiselect(label, labels_, default=dflt, key=f"{key}_{i18n.get_lang()}")
    return [keys_[labels_.index(x)] for x in picked if x in labels_]


def insight_text(item: Dict[str, Any]) -> str:
    v = item["valori"]
    vals = " vs ".join(f"{short_name(p)} {fmt_num(v[p], 2)}" for p in sorted(v, key=v.get, reverse=True))
    unit = t(item["unita"]) if item.get("unita") else ""
    return t("**{ind}** — prevale {forza} {who} ({vals}{unit}; rapporto {r}×).",
             ind=t(item["indicatore"]), forza=t("nettamente") if item["forte"] else t("leggermente"),
             who=short_name(item["prevale"]), vals=vals, unit=(" " + unit) if unit else "",
             r=fmt_num(item["rapporto"], 2) if item.get("rapporto") else "∞")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "db_path": None, "start_date": None, "end_date": None,
    "llm_base_url": config.DEFAULT_LLM_BASE_URL, "llm_model": config.DEFAULT_LLM_MODEL,
    "llm_temperature": config.DEFAULT_TEMPERATURE, "llm_models_found": [], "chats": {}, "pending_prompt": None,
    "llm_custom": False, "_last_upload_sig": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# `chatautopsy analyze PERCORSO` apre direttamente la chat appena costruita
if not st.session_state.get("_open_db_applied"):
    _od = os.environ.get("CHATAUTOPSY_OPEN_DB")
    if _od and Path(_od).exists():
        st.session_state.db_path = _od
    st.session_state._open_db_applied = True

LLM = llm_manager.manager()


@st.cache_resource(show_spinner=False)
def _llm_autostart_once() -> bool:
    """All'avvio del processo riaccende in background il modello scelto l'ultima volta (se già scaricato)."""
    try:
        LLM.autostart()
    except Exception:
        pass
    return True


_llm_autostart_once()


def sync_llm_session() -> Dict[str, Any]:
    """Se il modello gestito è pronto, l'AI Detective lo usa automaticamente (salvo endpoint personalizzato)."""
    stt = LLM.status()
    if not st.session_state.llm_custom and stt.get("model") and stt.get("base_url"):
        st.session_state.llm_base_url = stt["base_url"]
        st.session_state.llm_model = stt["model"]
    return stt


SECTIONS = ["🧾 Pagella", "📊 Volumi & Potere", "⏱️ Reattività", "🚩 Conflitti & Red Flags",
            "🗣️ Lessico & Affetto", "🕰️ Ritmi & Abitudini", "🕵️ AI Detective", "🎭 Persona Simulator"]


def run_ingest(container, *, path: Optional[str] = None, uploads: Optional[list] = None, name: str = "",
               embed_model: Optional[str] = None, tg_chat: Optional[str] = None, demo_text: Optional[str] = None) -> None:
    """Costruisce tutto (parsing → SQL → FTS5 → punteggi → embedding) e apre la chat appena creata."""
    from chatautopsy.ingestion import run_ingestion

    target_name = name.strip() or "chat"
    db_path = database.db_path_for_name(target_name)
    with container:
        bar = st.progress(0.0, text=t("Avvio…"))
        status = st.status(t("Parsing → SQL → indice full-text → punteggi → embedding"), expanded=True)
        log_box = status.empty()
        log_lines: List[str] = []

        def progress(stage: str, frac: float, detail: str) -> None:
            bar.progress(frac, text=f"**{stage}** · {detail}")
            if not log_lines or not log_lines[-1].startswith(f"**{stage}**") or "Embeddings" not in stage:
                log_lines.append(f"**{stage}** — {detail}")
            else:
                log_lines[-1] = f"**{stage}** — {detail}"
            log_box.markdown("\n\n".join(log_lines[-10:]))

        try:
            kwargs: Dict[str, Any] = dict(progress_callback=progress, build_embeddings=embed_model is not None,
                                          embedding_model=embed_model, telegram_chat_name=tg_chat or None)
            if demo_text is not None:
                result = run_ingestion(None, db_path, file_bytes=demo_text.encode("utf-8"), source_name=f"{target_name}.txt", **kwargs)
            elif path:
                result = run_ingestion(Path(path).expanduser(), db_path, **kwargs)
            else:
                # upload: l'export resta solo in memoria, su disco finisce solo il .db
                result = run_ingestion(None, db_path, files=[(u.name, u.getvalue()) for u in uploads or []],
                                       source_name=uploads[0].name if len(uploads or []) == 1 else f"{len(uploads or [])} file",
                                       **kwargs)
            status.update(label=t("✅ Completato in {s}s", s=f"{result['elapsed_seconds']:.0f}"), state="complete", expanded=False)
            if result.get("embedding_error"):
                st.warning(t("Ricerca semantica non disponibile ({e}): verrà usata la ricerca per parole.", e=result["embedding_error"]))
            st.session_state.db_path = str(db_path)
            st.session_state.start_date = None
            st.session_state.end_date = None
            st.session_state.section = SECTIONS[0]
            clear_all_caches()
            st.session_state.ingest_message = t("✅ {m} messaggi · {s} sessioni · {e} embedding", m=fmt_num(result["messages"]),
                                                s=fmt_num(result["sessions"]), e=fmt_num(result["embedded_sessions"]))
            st.rerun()
        except Exception as exc:
            status.update(label=t("❌ Analisi non riuscita"), state="error")
            st.error(f"{type(exc).__name__}: {exc}")


def _name_from(src: str) -> str:
    stem = Path(src).stem if Path(src).suffix else Path(src).name
    for prefix in ("ChatExport_", "WhatsApp Chat - ", "WhatsApp Chat with ", "Chat WhatsApp con "):
        stem = stem.replace(prefix, "")
    return stem.strip() or "chat"


NO_EMBEDDINGS = "Nessuna (solo ricerca per parole, più veloce)"


def embedding_choice(key: str) -> Optional[str]:
    labels = list(config.EMBEDDING_PRESETS.keys()) + [NO_EMBEDDINGS]
    cached = {lab for lab, m in config.EMBEDDING_PRESETS.items() if rag_engine.is_model_cached(m)}
    lab = choose(st.selectbox, t("Ricerca semantica"),
                 {x: t(x) + (" ✅" if x in cached else (" ⬇️" if x in config.EMBEDDING_PRESETS else "")) for x in labels}, key,
                       help=t("✅ già in locale · ⬇️ scaricato una sola volta alla prima analisi. "
                              "La ricerca semantica serve all'AI Detective; le statistiche non ne hanno bisogno."))
    return config.EMBEDDING_PRESETS.get(lab)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧬 ChatAutopsy")
    st.caption(t("Analisi forense 100% locale · nessun dato lascia il computer"))
    _langs = list(i18n.LANGS)
    _lang_labels = [i18n.LANGS[x] for x in _langs]
    _lang_new = _langs[_lang_labels.index(st.selectbox("🌐", _lang_labels, index=_langs.index(st.session_state.lang),
                                                       label_visibility="collapsed", key="lang_pick"))]
    if _lang_new != st.session_state.lang:
        st.session_state.lang = _lang_new
        llm_manager.save_settings(lang=_lang_new)
        st.rerun()
    if st.session_state.get("ingest_message"):
        st.success(st.session_state.pop("ingest_message"))

    st.markdown(t("### 📥 Nuova analisi"))
    src_mode = choose(st.radio, t("Sorgente"), {"path": t("📂 Percorso"), "upload": t("⬆️ Carica file")}, "src_mode",
                      horizontal=True, label_visibility="collapsed")
    with st.expander(t("Opzioni"), expanded=False):
        embed_model = embedding_choice("emb_choice")
        tg_chat = st.text_input(t("Chat Telegram (solo export completo)"), value="",
                                help=t("Nome esatto della chat in un export Telegram completo. Vuoto = la più grande."))
        auto_ingest = st.checkbox(t("Analizza appena carico il file"), value=True, key="auto_ingest")
    ingest_box = st.container()
    if src_mode == "path":
        src_path = st.text_input(t("File o cartella di export"), key="src_path",
                                 placeholder="~/Downloads/ChatExport_2024-06-28",
                                 help=t("Cartella Telegram (messages*.html o result.json), cartella o .zip WhatsApp, "
                                        "file _chat.txt / result.json."))
        p_ok = bool(src_path.strip()) and Path(src_path.strip()).expanduser().exists()
        if src_path.strip() and not p_ok:
            st.caption(t("⚠️ Percorso non trovato"))
        name_p = st.text_input(t("Nome analisi"), value=_name_from(src_path.strip()) if p_ok else "", key="name_path")
        if st.button(t("🚀 Analizza"), type="primary", width="stretch", disabled=not p_ok, key="go_path"):
            run_ingest(ingest_box, path=src_path.strip(), name=name_p, embed_model=embed_model, tg_chat=tg_chat)
    else:
        uploads = list(st.file_uploader(
            t("Export WhatsApp (.txt/.zip) o Telegram (result.json, tutte le pagine messages*.html, o .zip)"),
            type=["txt", "json", "zip", "html", "htm"], accept_multiple_files=True, key="uploader") or [])
        html_count = sum(1 for u in uploads if u.name.lower().endswith((".html", ".htm")))
        default_name = ("chat_telegram" if html_count > 1 else _name_from(uploads[0].name)) if uploads else ""
        name_u = st.text_input(t("Nome analisi"), value=default_name, key="name_upload")
        sig = tuple((u.name, u.size) for u in uploads)
        go = st.button(t("🚀 Analizza"), type="primary", width="stretch", disabled=not uploads, key="go_upload")
        if uploads and (go or (auto_ingest and sig != st.session_state._last_upload_sig)):
            st.session_state._last_upload_sig = sig
            run_ingest(ingest_box, uploads=uploads, name=name_u or default_name, embed_model=embed_model, tg_chat=tg_chat)

    st.markdown(t("### 🗂️ Chat analizzate"))
    if st.session_state.get("wipe_message"):
        st.success(st.session_state.pop("wipe_message"))
    dbs = database.list_databases()
    if dbs:
        labels = [p.stem for p in dbs]
        current = Path(st.session_state.db_path).stem if st.session_state.db_path else labels[0]
        idx = labels.index(current) if current in labels else 0
        choice = st.selectbox(t("Seleziona chat"), labels, index=idx, label_visibility="collapsed")
        chosen = str(dbs[labels.index(choice)])
        if chosen != st.session_state.db_path:
            st.session_state.db_path = chosen
            st.session_state.start_date = None
            st.session_state.end_date = None
        # precarica in background il modello di embedding del DB (prima ricerca semantica senza attese)
        rag_engine.prewarm_embedder(rag_engine.stored_embedding_model(chosen))
        with st.expander(t("🗑️ Elimina questa chat")):
            confirm_wipe = st.checkbox(t("Confermo l'eliminazione definitiva di «{c}»", c=choice), key="confirm_wipe")
            if st.button(t("🗑️ Elimina Database Selezionato (Wipe Sicuro)"), type="primary", width="stretch",
                         disabled=not confirm_wipe, key="wipe_db"):
                target = chosen
                removed = database.secure_wipe_database(target)
                with rag_engine._MATRIX_LOCK:
                    rag_engine._MATRIX_CACHE.pop(target, None)
                clear_all_caches()
                st.session_state.chats.pop(target, None)
                for key in ("db_path", "start_date", "end_date", "pending_prompt"):
                    st.session_state[key] = None
                for key in ("confirm_wipe", "sw_table", "top_conf_table", "pron_who", "tfidf_who", "aff_terms", "hm_who",
                            "word_trend_q", "sim_target", "score_cats", "yearly_metric", "mm_type", "cal_year", "sw_types",
                            "clima_who"):
                    st.session_state.pop(key, None)
                st.session_state.wipe_message = t("Database «{c}» sovrascritto e cancellato ({f}).", c=choice, f=", ".join(removed))
                st.rerun()
    else:
        st.caption(t("Nessuna chat ancora."))

    st.markdown(t("### 🤖 Modello AI locale"))

    @st.fragment(run_every=2 if LLM.busy() else None)
    def llm_panel() -> None:
        stt = sync_llm_session()
        phase = stt.get("phase")
        prev = st.session_state.get("_llm_prev_phase")
        st.session_state._llm_prev_phase = phase
        if prev in ("installing", "downloading", "starting") and phase in ("ready", "error"):
            st.rerun(scope="app")  # aggiorna AI Detective e frequenza di refresh
        if st.session_state.llm_custom:
            alive = llm_manager.endpoint_alive(st.session_state.llm_base_url, 1.5)
            st.markdown(f"{'🟢' if alive else '🔴'} " + t("Endpoint personalizzato") + f" `{st.session_state.llm_base_url}`")
        elif phase == "ready":
            st.markdown(t("🟢 **Pronto**") + f" · `{stt['model']}`")
        elif phase in ("installing", "downloading", "starting"):
            names = {"installing": t("Installazione"), "downloading": t("Download"), "starting": t("Avvio")}
            st.markdown(f"🟡 **{names[phase]}** · `{stt['model']}`")
            st.progress(float(stt.get("progress") or 0), text=stt.get("message") or "")
        elif phase == "error":
            st.markdown(t("❌ **Errore**"))
            st.caption(stt.get("error") or "")
        else:
            st.markdown(t("🔴 **Spento** — scegli un modello e premi il pulsante"))

        backends = llm_manager.available_backends()
        if not backends:
            st.caption(t("Nessun backend: su Mac Apple Silicon serve MLX (automatico), altrimenti installa Ollama da ollama.com."))
            return
        cat = llm_manager.catalog()
        cat = [e for e in cat if e["backend"] in backends]
        rec = llm_manager.recommended_model(backends[0])
        saved = llm_manager.load_settings().get("llm_model")
        ids = [e["id"] for e in cat]
        default_id = stt.get("model") if stt.get("model") in ids else (saved if saved in ids else rec)
        by_id = {e["id"]: e for e in cat}

        def fmt(mid: str) -> str:
            e = by_id[mid]
            tag = "✅" if e["cached"] else f"⬇️ {e['size_gb']} GB"
            return f"{t(e['label'])} · {tag}" + (" ⭐" if mid == rec else "") + ("" if e["fits"] else " ⚠️ RAM")

        sel = choose(st.selectbox, t("Modello"), {i_: fmt(i_) for i_ in ids}, "llm_pick",
                     index=ids.index(default_id) if default_id in ids else 0, disabled=LLM.busy(),
                           help=t("⭐ consigliato per {r} GB di RAM · ✅ già scaricato · ⬇️ verrà scaricato una sola volta. "
                                  "Tutto resta sul tuo computer.", r=f"{llm_manager.ram_gb():.0f}"))
        c1, c2 = st.columns(2)
        is_running_sel = phase == "ready" and stt.get("model") == sel
        label = t("⏹️ Ferma") if is_running_sel else (t("▶️ Avvia") if by_id[sel]["cached"] else t("⬇️ Scarica e avvia"))
        if c1.button(label, type="primary", width="stretch", disabled=LLM.busy(), key="llm_go"):
            if is_running_sel:
                LLM.stop_server()
                llm_manager.save_settings(llm_autostart=False)
            else:
                st.session_state.llm_custom = False
                LLM.ensure_ready(sel, by_id[sel]["backend"])
            st.rerun(scope="app")
        c2.toggle(t("Endpoint mio"), key="llm_custom", help=t("Usa un server già avviato da te (LM Studio, Ollama, MLX…)"))

    llm_panel()

    if st.session_state.llm_custom:
        st.session_state.llm_base_url = st.text_input("Base URL", value=st.session_state.llm_base_url, key="llm_url_in")
        st.session_state.llm_model = st.text_input(t("Modello"), value=st.session_state.llm_model, key="llm_model_in")
    st.session_state.llm_temperature = st.slider(t("Creatività delle risposte"), 0.0, 1.5,
                                                 float(st.session_state.llm_temperature), 0.05)

    if st.session_state.db_path and Path(st.session_state.db_path).exists():
        summ = db_summary(st.session_state.db_path, database.db_signature(st.session_state.db_path))
        if summ["first_ts"]:
            st.markdown(t("### 📅 Periodo analizzato"))
            dmin = datetime.strptime(summ["first_ts"][:10], "%Y-%m-%d").date()
            dmax = datetime.strptime(summ["last_ts"][:10], "%Y-%m-%d").date()
            rng = st.date_input(t("Intervallo"), value=(dmin, dmax), min_value=dmin, max_value=dmax,
                                format="YYYY/MM/DD" if i18n.get_lang() == "en" else "DD/MM/YYYY")
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                s_, e_ = rng
                st.session_state.start_date = None if s_ == dmin else s_.isoformat()
                st.session_state.end_date = None if e_ == dmax else e_.isoformat()
            st.caption(t("{m} messaggi · {s} sessioni · {e} embedding", m=fmt_num(summ["messages"]),
                         s=fmt_num(summ["sessions"]), e=fmt_num(summ["embedded_sessions"])))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if not st.session_state.db_path or not Path(st.session_state.db_path).exists():
    st.title("🧬 ChatAutopsy")
    st.markdown(t("**Autopsia forense, statistica e psicometrica delle tue chat — interamente sul tuo computer.**"))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(t("#### 1 · Esporta la chat"))
        st.markdown(t("- **WhatsApp**: chat → ⋮ → *Esporta chat* → *Senza media* (ottieni uno .zip)\n"
                      "- **Telegram Desktop**: chat → ⋮ → *Export chat history* (HTML o JSON, senza foto)"))
    with c2:
        st.markdown(t("#### 2 · Indicala qui"))
        st.markdown(t("Nella barra a sinistra incolla il **percorso** della cartella/file oppure **trascina il file**: "
                      "l'analisi parte da sola. Da terminale: `chatautopsy analyze PERCORSO`."))
    with c3:
        st.markdown(t("#### 3 · Esplora"))
        st.markdown(t("Pagella della relazione, potere, reattività, conflitti, lessico, ritmi. "
                      "Per l'AI Detective scegli un modello nel pannello **🤖 Modello AI locale**."))
    st.divider()
    st.markdown(t("##### Vuoi solo provare?"))
    demo_box = st.container()
    if st.button(t("🎭 Apri una chat dimostrativa (inventata)"), type="primary"):
        from chatautopsy import demo as _demo

        if i18n.get_lang() == "en":
            run_ingest(demo_box, demo_text=_demo.couple_chat(**_demo.SHOWCASE_EN), name="demo_emma_jake", embed_model=None)
        else:
            run_ingest(demo_box, demo_text=_demo.couple_chat(**_demo.SHOWCASE), name="demo_giulia_marco", embed_model=None)
    st.stop()

summary = db_summary(st.session_state.db_path, database.db_signature(st.session_state.db_path))
participants: List[str] = A("participant_list")
if not participants:
    st.warning(t("Nessun messaggio nell'intervallo di date selezionato."))
    st.stop()
CMAP = ui.color_map(participants)
is_group = len(participants) > 2
is_couple = len(participants) == 2
P_A, P_B = (participants[0], participants[1]) if len(participants) >= 2 else (participants[0], participants[0])
TOTAL = t("Totale")

st.title(f"🧬 ChatAutopsy · {Path(st.session_state.db_path).stem}")
st.caption(t("{kind} · {n} partecipanti · periodo {a} → {b} · i numeri seguono il filtro date della sidebar",
             kind=t("Chat di gruppo") if is_group else t("Chat di coppia"), n=len(participants),
             a=st.session_state.start_date or summary["first_ts"][:10], b=st.session_state.end_date or summary["last_ts"][:10]))

# Navigazione a sezioni: viene calcolata solo la sezione aperta (st.tabs eseguirebbe tutto a ogni caricamento,
# minuti su chat da milioni di messaggi). Le altre sezioni vengono precalcolate in background dopo il primo render.
SEC_LABELS = [t(x) for x in SECTIONS]
_cur = st.session_state.get("section")
if _cur not in SEC_LABELS:  # prima apertura, cambio lingua o sezione impostata in italiano dal codice
    from chatautopsy.locales_en import EN as _EN

    _idx = next((v.index(_cur) for v in (SECTIONS, [_EN.get(x, x) for x in SECTIONS]) if _cur in v), 0)
    st.session_state.section = SEC_LABELS[_idx]
_label = st.segmented_control(t("Sezione"), SEC_LABELS, key="section", label_visibility="collapsed")
if _label is None:  # click sulla sezione già attiva: resta dove sei
    section_sel = st.session_state.get("_section_last", SECTIONS[0])
else:
    section_sel = SECTIONS[SEC_LABELS.index(_label)]
st.session_state["_section_last"] = section_sel
st.divider()

# =========================================================================== PAGELLA
if section_sel == SECTIONS[0]:
    k = A("overview_kpis")
    bs = A("balance_scorecard")
    da = A("daily_activity")
    section(t("🧾 Pagella della relazione"),
            t("Ogni aspetto misurabile della chat trasformato in un indicatore valutabile: il valore di ciascuno, "
              "chi prevale e di quanto. È una fotografia dei comportamenti di scrittura, non una diagnosi."))
    cols = st.columns(5)
    kpi(cols[0], t("Messaggi totali"), fmt_num(k["total_messages"]),
        t("{a} testo · {b} media", a=fmt_num(k["text_messages"]), b=fmt_num(k["media_messages"])))
    kpi(cols[1], t("Giorni coperti"), fmt_num(k["days_analyzed"]),
        t("attivi il {p}% dei giorni", p=fmt_num(da["streaks"].get("active_days_pct"), 1)))
    kpi(cols[2], t("Messaggi al giorno"), fmt_num(k["messages_per_day"]),
        t("{n} nei giorni attivi", n=fmt_num(k["messages_per_active_day"])))
    kpi(cols[3], t("Serie record"), t("{n} giorni", n=fmt_num(da["streaks"].get("longest_streak_days"))),
        f"{da['streaks'].get('longest_streak_start') or ''} → {da['streaks'].get('longest_streak_end') or ''}")
    sil = da["silences"]
    kpi(cols[4], t("Silenzio più lungo"), t("{n} giorni", n=fmt_num(sil.iloc[0]["durata_giorni"], 1)) if not sil.empty else "—",
        t("dal {d}", d=sil.iloc[0]["dal"][:10]) if not sil.empty else "")
    st.write("")

    c1, c2 = st.columns([1, 2])
    with c1:
        ri = bs.get("reciprocity_index")
        if ri is not None:
            st.plotly_chart(ui.gauge(ri, t("Indice di reciprocità")), width="stretch", config={"displayModeBar": False})
            kind = "good" if ri >= 80 else "warn" if ri >= 60 else "bad"
            msg = (t("Scambio sostanzialmente bilanciato.") if ri >= 80 else
                   t("Bilanciamento parziale: alcuni aspetti pesano più su una persona.") if ri >= 60 else
                   t("Scambio marcatamente asimmetrico."))
            callout(f"**{fmt_num(ri)}/100** — {msg}", kind)
        else:
            callout(t("L'indice di reciprocità si calcola solo per le chat a due. Nei gruppi usa la tabella qui sotto "
                      "per vedere chi prevale su ogni indicatore."), "info")
    with c2:
        st.markdown(t("##### 🔎 Cosa salta all'occhio"))
        items = bs.get("insight_items") or []
        if items:
            for it in items:
                st.markdown(f"- {insight_text(it)}")
        else:
            st.markdown(t("Nessuno squilibrio rilevante: gli indicatori sono tutti in equilibrio."))
    explain(
        t("Sette indicatori di reciprocità (quota messaggi, quota caratteri, avvio e chiusura delle conversazioni, "
          "tempo di risposta, domande, lessico affettivo) combinati in un unico numero da 0 a 100."),
        t("100 = perfetta parità su tutti e sette; ≥80 equilibrio; 60-80 bilanciamento parziale; <60 asimmetria marcata. "
          "Nelle barre qui sotto la linea bianca è la parità: più la barra si allunga verso un nome, più quell'aspetto pesa su di lui/lei. "
          "La fascia verde è la zona 'equilibrato' (rapporto tra i due < 1,25)."),
        t("Per ogni indicatore: squilibrio = (max − min) / (max + min); reciprocità = 100 × (1 − media degli squilibri). "
          "Verdetto: rapporto max/min < {a} equilibrato, < {b} lieve prevalenza, oltre forte prevalenza.",
          a=config.BALANCE_MILD_RATIO, b=config.BALANCE_STRONG_RATIO),
        t("Misura la forma dello scambio, non il suo contenuto: chi scrive meno può comunque essere molto presente in altri modi "
          "(chiamate, incontri dal vivo). Gli indicatori lessicali si basano su vocabolari italiani e inglesi: ironia e sarcasmo sfuggono."),
    )

    inds = bs.get("indicators", [])
    if inds:
        cats = list(dict.fromkeys(i["categoria"] for i in inds))
        if is_couple:
            st.markdown(t("#### ⚖️ Tiro alla fune: chi pesa di più su ogni aspetto"))
            fig = ui.tug_of_war(inds, P_A, P_B, CMAP)
            fig = ui.style(fig, height=fig.layout.height, legend=False)
            fig.update_layout(margin=dict(t=70, l=10, r=60, b=10))
            st.plotly_chart(fig, width="stretch", key="tug_all", config={"displaylogo": False})
            st.caption(t("Le percentuali sono la quota di {a} · {b} sulla somma dei due valori. "
                         "La fascia verde centrale è la zona 'equilibrato'. Passa il mouse sulle barre per i valori reali. "
                         "Attenzione al verso: per tempo di risposta, accuse, conflitto e stonewalling prevalere è un segnale negativo.",
                         a=short_name(P_A), b=short_name(P_B)))
        col_area, col_ind, col_unit, col_ver, col_ratio, col_mean = (t("Area"), t("Indicatore"), t("Unità"), t("Verdetto"),
                                                                      t("Rapporto"), t("Cosa significa"))
        rows = []
        for i in inds:
            r = {col_area: t(i["categoria"]), col_ind: t(i["indicatore"])}
            for p in participants:
                r[short_name(p) if is_couple else p] = i["valori"].get(p)
            r.update({col_unit: t(i["unita"]) if i["unita"] else "", col_ver: verdict_text(i.get("verdetto", "")),
                      col_ratio: i.get("rapporto"), col_mean: t(i["significato"])})
            rows.append(r)
        tdf = pd.DataFrame(rows)
        st.markdown(t("#### 📋 Tutti gli indicatori"))
        cat_sel = choose_many(t("Aree"), {c_: t(c_) for c_ in cats}, "score_cats")
        view = tdf[tdf[col_area].isin([t(c) for c in cat_sel])]
        st.dataframe(view, hide_index=True, width="stretch", height=min(60 + 35 * len(view), 980),
                     column_config={col_mean: st.column_config.TextColumn(width="large"),
                                    col_ratio: st.column_config.NumberColumn(format="%.2f×")})
        ui.csv_button(tdf, t("pagella_indicatori.csv"), key="dl_score")

    section(t("📅 Pagella anno per anno"),
            t("Gli stessi comportamenti misurati anno per anno: per capire cosa è cambiato e quando."))
    ys = A("yearly_scorecard")
    if not ys.empty:
        ys = ys.rename(columns={c: t(c) for c in ys.columns})
        c_year, c_person = t("Anno"), t("Persona")
        metrics = [c for c in ys.columns if c not in (c_year, c_person)]
        default_metric = t("Quota messaggi %")
        msel = st.selectbox(t("Indicatore"), metrics, index=metrics.index(default_metric) if default_metric in metrics else 0,
                            key="yearly_metric")
        fig = px.line(ys, x=c_year, y=msel, color=c_person, markers=True, color_discrete_map=CMAP)
        if msel in (t("Quota messaggi %"), t("Avvii conversazione %")) and is_couple:
            fig.add_hline(y=50, line_dash="dot", line_color="#888", annotation_text=t("parità"))
        show(fig, 340, title=t("{m} per anno", m=msel), key="yearly_line")
        piv = ys.pivot(index=c_person, columns=c_year, values=msel)
        st.caption(t("Stessa metrica in tabella (righe = persone, colonne = anni):"))
        st.dataframe(piv.round(2), width="stretch")
        with st.expander(t("Tabella completa anno × persona (tutti gli indicatori)")):
            st.dataframe(ys, hide_index=True, width="stretch")
            ui.csv_button(ys, t("pagella_annuale.csv"), key="dl_yearly")
        explain(t("Per ogni anno e persona: volume, iniziativa, velocità, lunghezza, domande, affetto, conflitto, scuse, "
                  "colpe, emoji, ore notturne, vocali e media."),
                t("Confronta l'andamento delle linee: un calo simultaneo di quota, avvii e affetto di una persona è un "
                  "segnale di disinvestimento; una crescita dei vocali può invece indicare un cambio di canale, non di interesse."),
                t("Lessici e affetto contano solo testo scritto di proprio pugno (esclusi media e inoltri). "
                  "Latenza: mediana dei tempi di risposta < 8h."),
                t("Il primo e l'ultimo anno possono essere parziali (pochi mesi): confrontali con cautela."))

# =========================================================================== VOLUMI & POTERE
if section_sel == SECTIONS[1]:
    section(t("📈 Volumi nel tempo"), t("Quanto si scrive e come si distribuisce il volume tra le persone."))
    freq = choose(st.radio, t("Granularità"), {"M": t("Mensile"), "W": t("Settimanale"), "D": t("Giornaliera")}, "vol_freq",
                  horizontal=True)
    vol = A("volume_timeseries", freq=freq)
    fig = px.area(vol, x="period", y="messages", color="sender", color_discrete_map=CMAP,
                  labels={"period": "", "messages": t("messaggi"), "sender": ""})
    show(fig, 360, title=t("Messaggi per periodo (aree sovrapposte = totale)"), key="vol_area")

    sh = A("share_timeseries")
    if not sh.empty:
        c1, c2 = st.columns([3, 1])
        with c2:
            which = choose(st.radio, t("Quota di"), {"msgs": t("messaggi"), "chars": t("caratteri")}, "share_which")
            smooth = st.slider(t("Media mobile (mesi)"), 1, 12, 3, key="share_smooth")
        col = "share_pct" if which == "msgs" else "chars_share_pct"
        shp = rolling(sh, "sender", col, smooth)
        with c1:
            fig = px.line(shp, x="month", y=col, color="sender", color_discrete_map=CMAP,
                          labels={"month": "", col: t("% del totale"), "sender": ""})
            if is_couple:
                fig.add_hline(y=50, line_dash="dot", line_color="#888", annotation_text=t("parità"))
            fig.update_yaxes(range=[0, 100])
            show(fig, 320, title=t("Quota mensile dei messaggi per persona") if which == "msgs"
                 else t("Quota mensile dei caratteri per persona"), key="share_line")
        explain(t("La percentuale dei messaggi (o dei caratteri) del mese scritta da ciascuno."),
                t("In una coppia, linee vicine al 50% indicano un volume bilanciato. Divergenze persistenti indicano chi "
                  "'porta' la conversazione. I caratteri sono più affidabili dei messaggi: chi spezza i pensieri in tanti "
                  "messaggi brevi gonfia il conteggio dei messaggi."),
                t("Media mobile per ridurre il rumore mese su mese."))

    section(t("🚪 Chi apre e chi chiude"), t("Iniziativa: chi rompe il silenzio e chi manda l'ultimo messaggio."))
    ini = A("initiation_rate")
    clo = A("conversation_closers")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        fig = px.pie(ini["per_user"], names="sender", values="initiations", hole=0.55, color="sender", color_discrete_map=CMAP)
        fig.update_traces(textinfo="percent", sort=False)
        show(fig, 300, title=t("Avvia la conversazione"), key="ini_pie")
    with c2:
        if not clo["per_user"].empty:
            fig = px.pie(clo["per_user"], names="sender", values="closings", hole=0.55, color="sender", color_discrete_map=CMAP)
            fig.update_traces(textinfo="percent", sort=False)
            show(fig, 300, title=t("Manda l'ultimo messaggio"), key="clo_pie")
    with c3:
        if not ini["monthly"].empty:
            ism = st.slider(t("Media mobile (mesi)"), 1, 12, 3, key="ini_smooth")
            fig = px.line(rolling(ini["monthly"], "sender", "pct", ism), x="month", y="pct", color="sender",
                          color_discrete_map=CMAP, labels={"month": "", "pct": t("% degli avvii"), "sender": ""})
            if is_couple:
                fig.add_hline(y=50, line_dash="dot", line_color="#888")
            fig.update_yaxes(range=[0, 100])
            show(fig, 300, title=t("Quota degli avvii nel tempo"), key="ini_line")
    if is_couple and not ini["per_user"].empty:
        top = ini["per_user"].iloc[0]
        kind = "warn" if top["pct"] >= 64 else "info"
        callout(t("**{who}** avvia il **{p}%** delle conversazioni ({n} volte su {tot}). ", who=top["sender"],
                  p=fmt_num(top["pct"], 1), n=fmt_num(top["initiations"]), tot=fmt_num(ini["per_user"]["initiations"].sum()))
                + (t("Squilibrio marcato: una persona cerca il contatto molto più dell'altra.") if top["pct"] >= 64 else
                   t("Iniziativa ragionevolmente condivisa.")), kind)
    explain(t("Avvio = primo messaggio dopo almeno 6 ore di silenzio. Chiusura = ultimo messaggio prima di una pausa di oltre 6 ore."),
            t("Chi avvia sempre è chi cerca il contatto (o chi si sveglia prima!). Chi chiude spesso può essere chi dà la "
              "buonanotte o chi resta senza risposta: guarda i due grafici insieme."),
            limits=t("I risvegli quotidiani contano come avvii: con orari diversi, chi si sveglia prima risulta 'iniziatore'."))

    section(t("🔁 Inseguimento (double / triple texting)"),
            t("Messaggi ripetuti senza ricevere risposta, distanziati di almeno 3 minuti."))
    dt = A("double_texting")
    c1, c2 = st.columns([1, 1])
    with c1:
        d = dt["per_user"].melt(id_vars="sender", value_vars=["double_ratio_pct", "triple_ratio_pct"], var_name="tipo", value_name="pct")
        d["tipo"] = d["tipo"].map({"double_ratio_pct": t("≥2 tentativi"), "triple_ratio_pct": t("≥3 tentativi")})
        fig = px.bar(d, x="tipo", y="pct", color="sender", barmode="group", color_discrete_map=CMAP, text_auto=".1f",
                     labels={"tipo": "", "pct": t("% dei turni"), "sender": ""})
        show(fig, 320, title=t("Quota di turni con tentativi ripetuti"), key="dt_bar")
    with c2:
        if not dt["monthly"].empty:
            fig = px.bar(dt["monthly"], x="month", y="events", color="sender", color_discrete_map=CMAP,
                         labels={"month": "", "events": t("catene ≥3 tentativi"), "sender": ""})
            show(fig, 320, title=t("Catene di ≥3 tentativi per mese"), key="dt_month")
    table(dt["per_user"], key="dt_table")
    explain(t("Una 'catena' è una sequenza di messaggi della stessa persona senza risposta; un nuovo tentativo è un "
              "messaggio inviato almeno 3 minuti dopo il precedente."),
            t("Valori alti = bisogno di risposta / ansia da attesa, oppure semplicemente l'abitudine di 'raccontare' a "
              "puntate. Confronta con il tempo di risposta dell'altra persona."))

    section(t("✍️ Sforzo e investimento"), t("Quanto si scrive, quanto si chiede, quanto è elaborato ciò che si scrive."))
    eff = A("effort_ratio")
    pu = eff["per_user"]
    if not pu.empty:
        long = pu.melt(id_vars="sender", value_vars=["chars_share_pct", "avg_chars", "median_chars", "questions_per_100_msgs"],
                       var_name="m", value_name="v")
        long["m"] = long["m"].map({"chars_share_pct": t("% dei caratteri totali"), "avg_chars": t("Caratteri medi per messaggio"),
                                   "median_chars": t("Caratteri mediani"), "questions_per_100_msgs": t("Domande ogni 100 messaggi")})
        fig = px.bar(long, x="sender", y="v", color="sender", facet_col="m", facet_col_wrap=4, color_discrete_map=CMAP,
                     text_auto=".1f", labels={"sender": "", "v": ""})
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_xaxes(showticklabels=False)
        show(fig, 320, key="eff_bars")
        table(pu, key="eff_table")
    if not eff["monthly"].empty:
        fig = px.line(rolling(eff["monthly"], "sender", "avg_chars", 3), x="month", y="avg_chars", color="sender",
                      color_discrete_map=CMAP, labels={"month": "", "avg_chars": t("caratteri medi"), "sender": ""})
        show(fig, 300, title=t("Lunghezza media dei messaggi nel tempo (media mobile 3 mesi)"), key="eff_line")

    ld = A("length_distribution")
    if not ld["buckets"].empty:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.bar(ld["buckets"], x="fascia", y="pct", color="sender", barmode="group", color_discrete_map=CMAP,
                         text_auto=".1f", labels={"fascia": t("caratteri per messaggio"), "pct": t("% dei messaggi"), "sender": ""})
            show(fig, 320, title=t("Distribuzione della lunghezza dei messaggi"), key="len_dist")
        with c2:
            st.markdown(t("**Statistiche di lunghezza (caratteri)**"))
            table(ld["stats"], key="len_stats",
                  rename={"messaggi": "Messaggi", "media": "Media", "mediana": "Mediana", "p25": "p25", "p75": "p75",
                          "p90": "p90", "p99": "p99", "max": "max"})
    explain(t("Caratteri, domande e quota di testo prodotto da ciascuno; distribuzione delle lunghezze."),
            t("Messaggi lunghi e domande frequenti indicano coinvolgimento e cura. Una distribuzione schiacciata su 1-15 "
              "caratteri indica uno stile telegrafico (ok, sì, ahah)."),
            limits=t("Solo testo scritto di proprio pugno: media e inoltri esclusi dalle lunghezze."))

    section(t("🖼️ Come si comunica: testo, vocali, foto, sticker"),
            t("Il mix dei canali: oltre al testo, quanto si usano vocali, foto, sticker e contenuti inoltrati."))
    mm = A("media_mix")
    if not mm["per_user"].empty:
        media_colors = {tv(k_): v_ for k_, v_ in ui.MEDIA_COLORS.items()}
        c1, c2 = st.columns([2, 3])
        with c1:
            pm = mm["per_user"][mm["per_user"]["tipo"] != "Testo"].copy()
            pm["lbl"] = pm["pct"].map(lambda v: f"{v:.1f}" if v >= 1 else "")
            pm["sender"] = pm["sender"].map(short_name)
            pm["tipo"] = pm["tipo"].map(tv)
            fig = px.bar(pm, x="sender", y="pct", color="tipo", text="lbl", color_discrete_map=media_colors,
                         labels={"sender": "", "pct": t("% dei messaggi della persona"), "tipo": t("tipo")})
            show(fig, 400, title=t("Media e inoltri (% dei messaggi)"), key="mm_bar")
        with c2:
            tipi = [x for x in mm["per_user"]["tipo"].unique().tolist() if x != "Testo"]
            default_t = "Vocali / audio" if "Vocali / audio" in tipi else (tipi[0] if tipi else None)
            if default_t:
                tsel = choose(st.selectbox, t("Andamento nel tempo di"), {x_: tv(x_) for x_ in tipi}, "mm_type",
                              index=tipi.index(default_t))
                mon = mm["monthly"][mm["monthly"]["tipo"] == tsel]
                fig = px.line(rolling(mon, "sender", "pct", 3), x="month", y="pct", color="sender", color_discrete_map=CMAP,
                              labels={"month": "", "pct": t("% dei messaggi del mese"), "sender": ""})
                show(fig, 270, title=t("{x}: quota mensile (media mobile 3 mesi)", x=tv(tsel)), key="mm_line")
        piv = mm["per_user"].assign(tipo=mm["per_user"]["tipo"].map(tv)).pivot(index="tipo", columns="sender", values="count")
        with st.expander(t("Conteggi per tipo di contenuto")):
            st.dataframe(piv.fillna(0).astype(int), width="stretch")

    section(t("💘 Love bombing vs svalutazione"),
            t("Confronto tra i primi e gli ultimi 60 giorni del periodo: volume, velocità, affetto, cuori, lunghezza."))
    lb = A("love_bombing_devaluation")
    if not lb["per_user"].empty:
        w = lb["windows"]
        pu = lb["per_user"].assign(interpretation=lambda d_: d_["interpretation"].map(tv))
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = px.bar(pu, x="sender", y="devaluation_index", color="sender", color_discrete_map=CMAP, text="interpretation",
                         labels={"sender": "", "devaluation_index": t("indice")})
            fig.add_hline(y=0, line_color="#888")
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_yaxes(range=[-100, 100])
            show(fig, 360, title=t("Indice di svalutazione"), legend=False, key="lb_bar")
        with c2:
            rows = []
            first_col, last_col = t("Primi {n} gg", n=w["window_days"]), t("Ultimi {n} gg", n=w["window_days"])
            for r in pu.to_dict("records"):
                for m, lab in ui.LB_LABELS.items():
                    rows.append({t("Persona"): r["sender"], t("Metrica"): t(lab), first_col: r.get(f"early_{m}"),
                                 last_col: r.get(f"late_{m}"), t("Variazione %"): r.get(f"change_pct_{m}")})
            st.caption(t("Inizio {a} → {b} · fine {c} → {d}", a=w["early"][0], b=w["early"][1], c=w["late"][0], d=w["late"][1]))
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=390)
        explain(t("Media delle variazioni % di 5 segnali tra la finestra iniziale e quella finale (meno messaggi, meno "
                  "affetto, meno cuori, messaggi più corti, risposte più lente = raffreddamento)."),
                t(">40 svalutazione marcata · 15-40 raffreddamento progressivo · −15/+15 stabile · <−15 intensificazione."),
                limits=t("Su relazioni lunghe anni è normale che i primi mesi siano più intensi: usa il filtro date per "
                         "confrontare periodi specifici (es. l'ultimo anno) invece dell'intera storia."))

# =========================================================================== REATTIVITÀ
if section_sel == SECTIONS[2]:
    lat = A("response_latency")
    ldist = A("latency_distribution")
    spd = ldist["speed"]
    section(t("⏱️ Tempi di risposta"), t("Quanto si fa aspettare l'altro. Le pause oltre 8 ore (notte, giornate piene) sono escluse."))
    pr = lat["per_responder"]
    if not pr.empty:
        cols = st.columns(min(len(pr), 4))
        for i, r in enumerate(pr.sort_values("responder").to_dict("records")[:4]):
            s_ = spd[spd["responder"] == r["responder"]]
            sub = t("media {m}", m=fmt_minutes(r["mean_minutes"]))
            if not s_.empty:
                sub += t(" · entro 1 min {p}%", p=f"{s_.iloc[0]['within_1m_pct']:.0f}")
            kpi(cols[i % len(cols)], t("Risponde {p}", p=short_name(r["responder"])), fmt_minutes(r["median_minutes"]), sub)
        st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        if not ldist["buckets"].empty:
            fig = px.bar(ldist["buckets"], x="fascia", y="pct", color="responder", barmode="group", color_discrete_map=CMAP,
                         text_auto=".1f", labels={"fascia": t("tempo di risposta"), "pct": t("% delle risposte"), "responder": ""})
            show(fig, 340, title=t("Distribuzione dei tempi di risposta"), key="lat_dist")
    with c2:
        st.markdown(t("**Velocità di risposta**"))
        table(spd, key="spd_table")
        st.markdown(t("**Coppie scrittore → risponditore**"))
        table(lat["pairs"], key="pairs_table")
    if not ldist["by_hour"].empty:
        fig = px.line(ldist["by_hour"], x="hour", y="median_minutes", color="responder", markers=True, color_discrete_map=CMAP,
                      labels={"hour": t("ora del giorno"), "median_minutes": t("mediana (min)"), "responder": ""})
        fig.update_xaxes(dtick=2, range=[-0.5, 23.5])
        show(fig, 300, title=t("Tempo di risposta mediano per ora del giorno"), key="lat_hour")
    explain(t("Tempo tra un messaggio e la risposta dell'altra persona (solo quando cambia il mittente)."),
            t("La mediana è il tempo 'tipico'; la distribuzione mostra quanto spesso si risponde subito o si fa aspettare. "
              "Il grafico per ora rivela quando una persona è meno disponibile (lavoro, sonno)."),
            t("Pause > 8 ore escluse per non confondere sonno/assenze con lentezza."),
            t("Una risposta veloce non implica una risposta di qualità; molte chat attive in parallelo rallentano tutti."))

    section(t("📉 Fading detector"), t("Le risposte stanno rallentando rispetto al solito?"))
    lm = A("latency_monthly")
    if not lm.empty:
        lsm = st.slider(t("Media mobile (mesi)"), 1, 12, 3, key="lat_smooth")
        lmm = rolling(lm.rename(columns={"responder": "who"}), "who", "median_minutes", lsm).rename(columns={"who": "responder"})
        fig = px.line(lmm, x="month", y="median_minutes", color="responder", color_discrete_map=CMAP,
                      labels={"month": "", "median_minutes": t("mediana (min)"), "responder": ""})
        show(fig, 320, title=t("Tempo di risposta mediano per mese"), key="lat_month")
    fading = A("fading_detector")
    fcols = st.columns(min(len(fading), 4) or 1)
    for i, f in enumerate(fading[:8]):
        col = fcols[i % len(fcols)]
        if f.get("months_available", 0) and "baseline_median_min" in f:
            col.metric(t("{p} · ultimi 3 mesi", p=f["responder"]), fmt_minutes(f["recent_median_min"]),
                       delta=t("{d}% vs abituale ({b})", d=f"{(f['ratio'] - 1) * 100:+.0f}", b=fmt_minutes(f["baseline_median_min"]))
                       if f.get("ratio") else None, delta_color="inverse")
            col.markdown(f"<span class='{'flag' if f['flagged'] else ''}'>{tv(f['status'])}</span> · "
                         + t("z robusto = {z}", z=f["robust_z"]), unsafe_allow_html=True)
        else:
            col.metric(f"{f['responder']}", "—")
            col.caption(t("Dati insufficienti ({n} mesi)", n=f.get("months_available", 0)))
    explain(t("Confronta la mediana degli ultimi 3 mesi con la mediana abituale (tutti i mesi precedenti)."),
            t("Segnalato come FADING se la latenza recente è almeno 1,5× l'abituale e anomala (z robusto ≥ 2). "
              "Su tempi di pochi secondi anche +80% significa pochi secondi in più: guarda il valore assoluto."),
            t("z robusto = (recente − mediana storica) / (MAD × 1,4826)."))

    if is_group:
        mat = lat["matrix_median"]
        if isinstance(mat, pd.DataFrame) and not mat.empty:
            fig = px.imshow(mat, text_auto=".1f", color_continuous_scale="Magma", aspect="auto",
                            labels=dict(x=t("Chi risponde"), y=t("A chi risponde"), color="min"))
            show(fig, 420, title=t("Matrice dei tempi di risposta (mediana, minuti)"), key="lat_matrix")

# =========================================================================== CONFLITTI
if section_sel == SECTIONS[3]:
    cal = A("conflict_calibration_info")
    thr = cal["threshold"]
    section(t("🌡️ Termometro della tensione"),
            t("Ogni conversazione riceve un punteggio di tensione 0-100 calcolato senza AI: lessico di conflitto e accusa, "
              "risposte glaciali, muri di testo, punteggiatura insistita, maiuscole, densità dei messaggi."))
    if cal["mode"] == "relativa":
        mode_txt = t("(soglia relativa: il {p}% delle conversazioni più tese; la soglia assoluta di {a} non viene mai superata "
                     "qui — massimo {m}). ", p=100 - cal["percentile"], a=cal["absolute"], m=fmt_num(cal["max"]))
    else:
        mode_txt = t("(soglia assoluta). ")
    callout(t("Soglia di 'sessione tesa' per questa chat: **{x}** ", x=fmt_num(thr)) + mode_txt
            + t("Mediana delle sessioni: {m} · 90° percentile: {p}.", m=fmt_num(cal["median"]), p=fmt_num(cal["p90"])), "info")
    cd = A("conflict_distribution")
    c1, c2 = st.columns([2, 3])
    with c1:
        if not cd.empty:
            fig = go.Figure(go.Bar(x=cd["bin_start"] + 1.25, y=cd["sessions"], marker_color="#7986cb", width=2.3,
                                   hovertemplate=t("tensione %{x:.0f}: %{y} sessioni") + "<extra></extra>"))
            fig.add_vline(x=thr, line_color=ui.BAD, line_dash="dash", annotation_text=t("soglia"), annotation_font_color=ui.BAD)
            fig.update_yaxes(type="log", title_text=t("sessioni (scala log)"))
            fig.update_xaxes(title_text=t("punteggio di tensione"), range=[0, max(70, (cal.get("max") or 60) + 5)])
            show(fig, 340, title=t("Distribuzione della tensione"), legend=False, key="conf_hist")
    with c2:
        ct = A("conflict_timeline")
        if not ct.empty:
            fig = go.Figure()
            fig.add_bar(x=ct["month"], y=ct["high_conflict_sessions"], name=t("Sessioni tese (> {x})", x=fmt_num(thr)),
                        marker_color=ui.BAD, opacity=0.6, yaxis="y2")
            fig.add_scatter(x=ct["month"], y=ct["mean_conflict"], name=t("Tensione media"), line=dict(color=ui.WARN))
            mx = int(ct["high_conflict_sessions"].max() or 0)
            fig.update_layout(yaxis=dict(title=t("tensione media"), rangemode="tozero"),
                              yaxis2=dict(title=t("sessioni tese"), overlaying="y", side="right", rangemode="tozero", showgrid=False,
                                          dtick=max(1, -(-mx // 5)), tickformat="d"))
            show(fig, 340, title=t("Tensione mese per mese"), key="conf_month")
    explain(t("Il punteggio di tensione di ogni sessione (blocchi di conversazione separati da 45 minuti di silenzio)."),
            t("Il grafico a sinistra mostra quante conversazioni cadono in ogni fascia: quasi tutte sono serene, la coda a "
              "destra sono le discussioni. A destra: tensione media mensile e numero di sessioni sopra soglia."),
            t("Punteggio euristico deterministico; le sessioni piene di risate/emoji positive sono smorzate (×0,4) per non "
              "scambiare l'ilarità per litigio. Gli articoli inoltrati non contano."),
            t("Non capisce ironia, sarcasmo e litigi 'freddi' fatti di poche parole: per quelli guarda lo stonewalling."))

    section(t("🌦️ Clima: affetto vs conflitto"), t("Per ogni mese: quanto affetto e quanto conflitto si esprime, ogni 1000 messaggi scritti."))
    cm = A("climate_monthly")
    if not cm.empty:
        c1, c2 = st.columns([3, 1])
        with c2:
            csm = st.slider(t("Media mobile (mesi)"), 1, 12, 3, key="clima_smooth")
            opts = {"__all__": t("Entrambi")} if not is_group else {}
            opts.update({p_: p_ for p_ in participants[:6]})
            cwho = choose(st.radio, t("Persone"), opts, "clima_who")
        long = cm.melt(id_vars=["month", "sender"], value_vars=["affection_per_1000", "conflict_per_1000", "apology_per_1000", "blame_per_1000"],
                       var_name="m", value_name="v")
        names_m = {"affection_per_1000": t("Affetto"), "conflict_per_1000": t("Conflitto"),
                   "apology_per_1000": t("Scuse"), "blame_per_1000": t("Accuse")}
        long["m"] = long["m"].map(names_m)
        if cwho != "__all__":
            long = long[long["sender"] == cwho]
        long["serie"] = long["sender"].map(short_name) + " · " + long["m"]
        long = rolling(long, "serie", "v", csm)
        with c1:
            long["sender"] = long["sender"].map(short_name)
            fig = px.line(long, x="month", y="v", color="m", line_dash="sender",
                          color_discrete_map={names_m["affection_per_1000"]: "#f06292", names_m["conflict_per_1000"]: "#ff7043",
                                              names_m["apology_per_1000"]: "#81c784", names_m["blame_per_1000"]: "#ffd54f"},
                          labels={"month": "", "v": t("per 1000 messaggi"), "m": t("metrica"), "sender": t("persona")})
            show(fig, 380, title=t("Clima della relazione nel tempo"), key="clima_line")
        explain(t("Frequenza di parole affettuose (+ cuori), di conflitto, di scuse e di accuse per 1000 messaggi scritti."),
                t("Il dato più eloquente è il rapporto: finché l'affetto resta ben sopra il conflitto il clima è positivo. "
                  "Linee tratteggiate diverse = persone diverse."),
                limits=t("Lessici a parole chiave: 'ti ammazzo 😂' conta come conflitto; frasi affettuose senza parole tipiche sfuggono."))

    section(t("🧩 Pronomi: \"noi\" contro \"io/tu\""),
            t("Secondo le ricerche di Gottman le coppie solide usano più 'noi'; un 'tu' crescente in contesti accusatori è un segnale di critica."))
    ps = A("pronoun_shift")
    if not ps["monthly"].empty:
        who = choose_many(t("Serie"), {"Totale": TOTAL, **{p_: p_ for p_ in participants}}, "pron_who", default=["Totale"])
        m = ps["monthly"][ps["monthly"]["sender"].isin(who)]
        long = m.melt(id_vars=["month", "sender"], value_vars=["we_index_pct", "you_index_pct", "i_index_pct"],
                      var_name="pronome", value_name="pct")
        pron = {"we_index_pct": t("Noi"), "you_index_pct": t("Tu/Voi"), "i_index_pct": t("Io")}
        long["pronome"] = long["pronome"].map(pron)
        long = rolling(long.assign(k=long["sender"] + long["pronome"]), "k", "pct", 3)
        long["sender"] = long["sender"].map(lambda x: TOTAL if x == "Totale" else short_name(x))
        fig = px.line(long, x="month", y="pct", color="pronome", line_dash="sender",
                      color_discrete_map={pron["we_index_pct"]: "#81c784", pron["you_index_pct"]: "#ff8a65", pron["i_index_pct"]: "#4fc3f7"},
                      labels={"month": "", "pct": t("% sui pronomi"), "pronome": t("pronome"), "sender": t("serie")})
        show(fig, 360, title=t("Quota dei pronomi (media mobile 3 mesi)"), key="pron_line")
        table(ps["per_user"][["sender", "words", "we", "i", "you", "you_absolute", "we_index_pct", "you_index_pct",
                              "you_absolute_per_1000_words"]], key="pron_table")
        explain(t("La quota di 'noi', 'io' e 'tu' sul totale dei tre pronomi; '\"Tu sempre/mai\"' conta le generalizzazioni accusatorie."),
                t("Più 'noi' = pensiero di coppia. 'Io' alto è normale nel racconto della giornata. '\"Tu sempre / tu mai\"' "
                  "è la forma tipica della critica distruttiva."),
                limits=t("In italiano il soggetto è spesso sottinteso ('vieni?'): i pronomi espliciti sono un campione, non il totale."))

    section(t("🧱 Stonewalling (muro di gomma)"),
            t("Quando una persona apre il cuore con un messaggio lungo e riceve una risposta glaciale (pochi caratteri, uno sticker) "
              "o un silenzio di ore."))
    sw = A("stonewalling")
    notes = []
    if sw.get("excluded_shared_content"):
        notes.append(t("{n} messaggi lunghi esclusi perché link/articoli o inoltri", n=fmt_num(sw["excluded_shared_content"])))
    if sw.get("group_filter_applied"):
        notes.append(t("{n} casi esclusi perché nel gruppo scrivevano altri membri", n=fmt_num(sw.get("excluded_group_noise", 0))))
    if notes:
        st.caption(t("Filtri di qualità: ") + " · ".join(notes) + ".")
    c1, c2 = st.columns([1, 2])
    with c1:
        if not sw["per_responder"].empty:
            prr = sw["per_responder"]
            long = prr.melt(id_vars="responder", value_vars=["cold_replies", "long_silences"], var_name="t", value_name="n")
            n_cold, n_sil = t("Risposta glaciale"), t("Silenzio > 4h")
            long["t"] = long["t"].map({"cold_replies": n_cold, "long_silences": n_sil})
            fig = px.bar(long, x="responder", y="n", color="t", barmode="stack",
                         color_discrete_map={n_cold: "#4fc3f7", n_sil: "#9575cd"},
                         labels={"responder": t("chi risponde"), "n": t("episodi"), "t": ""})
            fig.update_xaxes(tickvals=prr["responder"], ticktext=[short_name(x) for x in prr["responder"]])
            show(fig, 330, title=t("Episodi per chi risponde"), key="sw_bar")
            table(prr, key="sw_resp")
    with c2:
        ev = sw["events"]
        if ev.empty:
            st.info(t("Nessun episodio di stonewalling rilevato nel periodo."))
        else:
            types_ = sorted(ev["type"].unique())
            typ = choose_many(t("Tipo"), {x_: tv(x_) for x_ in types_}, "sw_types")
            evf = ev[ev["type"].isin(typ)].reset_index(drop=True)
            event = table(evf[["timestamp", "author", "long_chars", "long_text", "responder", "reply_text", "silence_minutes", "type", "session_id"]],
                          height=360, key="sw_table", select=True, wide_text=["long_text"])
            r = selected_row(event)
            if r is not None and r < len(evf):
                with st.expander(t("📜 Conversazione completa #{n}", n=evf.loc[r, "session_id"]), expanded=True):
                    show_session(int(evf.loc[r, "session_id"]))
            else:
                st.caption(t("Seleziona una riga per leggere l'intera conversazione."))
    explain(t("Messaggio di oltre {a} caratteri scritto di proprio pugno, seguito da una risposta "
              "sotto i {b} caratteri (o sticker/GIF) oppure da oltre 4 ore di silenzio.",
              a=config.LONG_MESSAGE_CHARS, b=config.COLD_REPLY_MAX_CHARS),
            t("'% episodi su ricevuti' è il dato da confrontare: normalizza per quanti messaggi lunghi ciascuno ha ricevuto. "
              "Leggi gli episodi: un 'ok' dopo un lungo racconto organizzativo non è un muro emotivo."),
            limits=t("Non distingue un 'ok' freddo da un 'ok, ti chiamo!' seguito da una telefonata."))

    section(t("⚖️ Scuse, accuse e riappacificazione"),
            t("Chi riconosce i propri torti, chi attribuisce colpe e chi fa il primo passo dopo una discussione."))
    ab = A("apology_vs_blame")
    cr = A("conflict_recovery")
    c1, c2 = st.columns(2)
    with c1:
        if not ab["per_user"].empty:
            pu = ab["per_user"]
            long = pu.melt(id_vars="sender", value_vars=["apologies_per_1000", "blame_per_1000"], var_name="t", value_name="v")
            long["t"] = long["t"].map({"apologies_per_1000": t("Scuse"), "blame_per_1000": t("Accuse / colpe")})
            fig = px.bar(long, x="t", y="v", color="sender", barmode="group", color_discrete_map=CMAP, text_auto=".2f",
                         labels={"t": "", "v": t("per 1000 messaggi"), "sender": ""})
            show(fig, 330, title=t("Scuse e accuse ogni 1000 messaggi"), key="ab_bar")
            table(pu, key="ab_table")
    with c2:
        if not cr["per_user"].empty and cr["per_user"]["repairs_initiated"].sum() > 0:
            fig = px.pie(cr["per_user"], names="sender", values="repairs_initiated", hole=0.5, color="sender", color_discrete_map=CMAP)
            fig.update_traces(textinfo="value+percent")
            show(fig, 330, title=t("Chi fa il primo passo dopo una sessione tesa seguita da > 4h di silenzio"), key="cr_pie")
            table(cr["per_user"], key="cr_table")
            with st.expander(t("Episodi di riappacificazione ({n})", n=len(cr["events"]))):
                table(cr["events"], key="cr_events", wide_text=["repair_text"])
        else:
            st.info(t("Nessuna sessione tesa seguita da oltre 4 ore di silenzio nel periodo selezionato."))
    with st.expander(t("Esempi reali di scuse e accuse")):
        e1, e2 = st.columns(2)
        e1.markdown(t("**Scuse**"))
        for e in ab.get("apology_examples", [])[:8]:
            e1.markdown(f"- `{e['citation']}`")
        e2.markdown(t("**Accuse / colpe**"))
        for e in ab.get("blame_examples", [])[:8]:
            e2.markdown(f"- `{e['citation']}`")

    section(t("🚩 Altri segnali d'allarme lessicali"),
            t("Frequenza di frasi tipiche di gaslighting, passivo-aggressività e colpevolizzazione."))
    rf_names = {"gaslighting": (t("Gaslighting"), t("Gaslighting (\"te lo sei inventato\", \"sei pazza\")")),
                "passive_aggressive": (t("Passivo-aggressività"), t("Passivo-aggressività (\"fai come vuoi\", \"tranquillo\")")),
                "guilt_tripping": (t("Colpevolizzazione"), t("Colpevolizzazione (\"dopo tutto quello che ho fatto\")"))}
    rf_rows, rf_examples = [], {}
    for key_, (short_lab, lab) in rf_names.items():
        rep = A("lexicon_report", name=key_, examples=8)
        for r in rep["per_user"].to_dict("records"):
            rf_rows.append({"segnale": short_lab, "sender": r["sender"], "per_1000": r["per_1000_msgs"], "hits": r["hits"]})
        rf_examples[lab] = rep["examples"]
    if rf_rows:
        rfd = pd.DataFrame(rf_rows)
        fig = px.bar(rfd, x="segnale", y="per_1000", color="sender", barmode="group", color_discrete_map=CMAP, text_auto=".2f",
                     labels={"segnale": "", "per_1000": t("per 1000 messaggi"), "sender": "", "hits": t("occorrenze")},
                     hover_data=["hits"])
        show(fig, 320, title=t("Segnali d'allarme ogni 1000 messaggi"), key="rf_bar")
        with st.expander(t("Esempi reali per segnale (verifica sempre il contesto)")):
            for lab, exs in rf_examples.items():
                st.markdown(f"**{lab}**")
                for e in exs[:6]:
                    st.markdown(f"- `{e['citation']}`")
        explain(t("Conteggio di espressioni tipiche da vocabolari curati in italiano e inglese."),
                t("Valori bassi e simili tra le persone sono normali in qualunque coppia. Conta la tendenza nel tempo e "
                  "soprattutto il contesto: leggi gli esempi prima di trarre conclusioni."),
                limits=t("'Fai come vuoi' può essere detto con affetto: i lessici rilevano frasi, non intenzioni."))

    section(t("🔥 Le 20 conversazioni più tese"), t("Clicca una riga per leggere la trascrizione completa."))
    top = A("top_conflict_sessions", n=20)
    if not top.empty:
        event = table(top, key="top_conf_table", select=True)
        r = selected_row(event)
        if r is not None and r < len(top):
            show_session(int(top.loc[r, "session_id"]))

    section(t("📆 Quando si litiga"), t("Giorni e orari in cui le conversazioni sono più tese."))
    hs = A("hostility_seasonality")
    if isinstance(hs["mean_heatmap"], pd.DataFrame) and not hs["mean_heatmap"].empty:
        c1, c2 = st.columns([3, 2])
        with c1:
            hm_ = hs["mean_heatmap"].copy()
            hm_.index = [tv(x) for x in hm_.index]
            fig = px.imshow(hm_, color_continuous_scale="Reds", aspect="auto",
                            labels=dict(x=t("ora d'inizio"), y="", color=t("tensione")))
            show(fig, 340, title=t("Tensione media per giorno × ora"), key="host_hm")
        with c2:
            bw = hs["by_weekday"].assign(weekday=lambda d_: d_["weekday"].map(tv))
            fig = px.bar(bw, x="weekday", y="mean_conflict", text="high_conflict_sessions",
                         labels={"weekday": "", "mean_conflict": t("tensione media")}, color_discrete_sequence=["#ef5350"])
            fig.update_traces(texttemplate="%{text} " + t("tese"), textposition="outside", cliponaxis=False)
            show(fig, 340, title=t("Tensione media per giorno della settimana"), legend=False, key="host_wd")
        if hs["peak"]:
            pk = hs["peak"]
            callout(t("Fascia più tesa: **{d} {s}** — {h} sessioni tese su {n}, tensione media {m}.", d=tv(pk["weekday"]),
                      s=pk["slot"], h=pk["high_conflict_sessions"], n=pk["sessions"], m=fmt_num(pk["mean_conflict"])),
                    "warn" if pk["high_conflict_sessions"] else "info")

# =========================================================================== LESSICO
if section_sel == SECTIONS[4]:
    section(t("🧬 Parole identitarie"),
            t("Le parole e le coppie di parole che ciascuno usa molto più dell'altro: il suo 'dialetto' personale, "
              "inclusi soprannomi, intercalari e abitudini ortografiche (perchè/perché, quì)."))
    tf = A("tfidf_signature")
    show_p = participants[:6]
    if is_group:
        show_p = st.multiselect(t("Partecipanti"), participants, default=participants[:4], key="tfidf_who")
    cols = st.columns(min(max(len(show_p), 1), 3))
    for i, p in enumerate(show_p):
        d = tf.get(p)
        if d is None or d.empty:
            continue
        fig = px.bar(d.head(30).iloc[::-1], x="score", y="term", orientation="h", hover_data=["count", "exclusivity_pct"],
                     color_discrete_sequence=[CMAP.get(p)], labels={"score": t("punteggio"), "term": "",
                                                                     "count": t("occorrenze"), "exclusivity_pct": t("esclusività %")})
        cols[i % len(cols)].plotly_chart(ui.style(fig, 680, title=p, legend=False), width="stretch", key=f"tfidf_{i}")
    explain(t("TF-IDF × esclusività: quanto una parola è frequente per una persona e quanto è 'sua' rispetto all'altra."),
            t("In alto ci sono le parole più caratteristiche. Passa il mouse per vedere le occorrenze e la quota di esclusività."),
            limits=t("Esclude parole vuote, intercalari (eh, boh, ok), risate e link."))

    section(t("🔎 Cerca parole nella storia"), t("Scrivi fino a 4 parole o soprannomi: vedi chi li usa e quando."))
    default_q = "love, haha*" if i18n.get_lang() == "en" else "amore, ahah*"
    q = st.text_input(t("Parole separate da virgola (usa * per le radici, es. litig*)"), value=default_q, key="word_trend_q")
    terms = [x.strip() for x in q.split(",") if x.strip()][:4]
    if terms:
        norm = st.toggle(t("Normalizza per 1000 messaggi"), value=False, key="wt_norm")
        wt = A("word_trend", terms=terms)
        if not wt.empty:
            ycol = "per_1000_msgs" if norm else "count"
            fig = px.line(wt, x="month", y=ycol, color="sender", facet_row="term" if len(terms) > 1 else None,
                          color_discrete_map=CMAP, labels={"month": "", ycol: t("per 1000 msg") if norm else t("occorrenze"), "sender": ""})
            fig.update_yaxes(matches=None)
            show(fig, 210 * max(len(terms), 1) + 110, key="wt_line")
            tot = wt.groupby(["term", "sender"])["count"].sum().unstack(fill_value=0)
            st.dataframe(tot, width="stretch")

    section(t("💞 Parole d'amore e soprannomi"), t("Nascita, picco e scomparsa delle parole affettuose."))
    at = A("affection_timeline")
    if not at["monthly"].empty:
        top_terms = at["summary"].groupby("term")["total"].sum().sort_values(ascending=False).head(10).index.tolist()
        sel = st.multiselect(t("Termini"), sorted(at["monthly"]["term"].unique()), default=top_terms[:8], key="aff_terms")
        mm_ = at["monthly"][at["monthly"]["term"].isin(sel)]
        if not mm_.empty:
            fig = px.scatter(mm_, x="month", y="term", size="count", color="sender", color_discrete_map=CMAP,
                             labels={"month": "", "term": "", "sender": "", "count": t("occorrenze")})
            show(fig, 140 + 48 * len(sel), title=t("Uso mensile (dimensione del punto = occorrenze)"), key="aff_scatter")
        if at["detected_terms"]:
            st.caption(t("Parole scoperte automaticamente perché compaiono molto più spesso accanto a cuori e parole d'amore: ")
                       + ", ".join(f"**{d['term']}** ({d['messages']} msg)" for d in at["detected_terms"]))
        table(at["summary"], key="aff_table")

    section(t("😀 Impronta emoji"), t("Quali emoji usa ciascuno, quanto spesso e come cambiano nel tempo."))
    ef = A("emoji_fingerprint")
    if not ef["top"].empty:
        c1, c2 = st.columns([3, 1])
        with c1:
            fig = px.bar(ef["top"], x="emoji", y="count", color="sender", barmode="group", color_discrete_map=CMAP,
                         labels={"emoji": "", "count": t("volte"), "sender": ""})
            fig.update_xaxes(tickfont=dict(size=20))
            show(fig, 380, title=t("Le {n} emoji preferite di ciascuno", n=config.EMOJI_TOP_N), key="emo_top")
        with c2:
            fig = px.bar(ef["ratio"], x="sender", y="emoji_per_msg", color="sender", color_discrete_map=CMAP, text_auto=".2f",
                         labels={"sender": "", "emoji_per_msg": t("emoji per messaggio")})
            fig.update_xaxes(tickvals=ef["ratio"]["sender"], ticktext=[short_name(x) for x in ef["ratio"]["sender"]])
            show(fig, 380, title=t("Emoji per messaggio"), legend=False, key="emo_ratio")
        if not ef["monthly_groups"].empty:
            g = ef["monthly_groups"].assign(group=lambda d_: d_["group"].map(tv))
            fig = px.line(rolling(g.assign(k=g["sender"] + g["group"]), "k", "per_100_msgs", 3), x="month", y="per_100_msgs",
                          color="sender", facet_col="group", facet_col_wrap=2, color_discrete_map=CMAP,
                          labels={"month": "", "per_100_msgs": t("per 100 msg"), "sender": ""}, facet_row_spacing=0.12)
            fig.update_yaxes(matches=None, showticklabels=True)
            show(fig, 560, title=t("Famiglie di emoji nel tempo (per 100 messaggi, media mobile 3 mesi)"), key="emo_groups")
        if not ef["shift"].empty:
            with st.expander(t("Le variazioni più forti: primo vs ultimo quarto del periodo")):
                st.dataframe(ef["shift"].head(30), hide_index=True, width="stretch")

# =========================================================================== RITMI
if section_sel == SECTIONS[5]:
    da = A("daily_activity")
    stx = da["streaks"]
    section(t("📅 Calendario della chat"), t("Ogni quadratino è un giorno: più è chiaro, più messaggi."))
    cols = st.columns(4)
    kpi(cols[0], t("Giorni con messaggi"), f"{fmt_num(stx.get('active_days_pct'), 1)}%",
        t("{n} giorni di silenzio totale", n=fmt_num(stx.get("silent_days"))))
    kpi(cols[1], t("Serie record"), t("{n} giorni", n=fmt_num(stx.get("longest_streak_days"))),
        f"{stx.get('longest_streak_start') or ''} → {stx.get('longest_streak_end') or ''}")
    kpi(cols[2], t("Serie attuale"), t("{n} giorni", n=fmt_num(stx.get("current_streak_days"))),
        t("giorni consecutivi fino all'ultimo messaggio"))
    bd = da.get("busiest_days") or []
    kpi(cols[3], t("Giorno più intenso"), bd[0]["date"] if bd else "—", t("{n} messaggi", n=fmt_num(bd[0]["messages"])) if bd else "")
    st.write("")
    daily = da["daily"]
    if not daily.empty:
        years = sorted({int(x[:4]) for x in daily["date"]}, reverse=True)
        days_per_year = pd.Series([int(x[:4]) for x in daily["date"]]).value_counts()
        # di default l'ultimo anno con almeno ~6 mesi di dati (l'anno in corso è spesso quasi vuoto)
        default_year = next((y for y in years if days_per_year.get(y, 0) >= 180), years[0])
        ysel = st.selectbox(t("Anno"), years, index=years.index(default_year), key="cal_year")
        show(ui.calendar_heatmap(daily, ysel), 230, legend=False, title=t("Messaggi al giorno · {y}", y=ysel), key="cal_hm")
    sil = da["silences"]
    if not sil.empty:
        st.markdown(t("##### 🤐 I silenzi più lunghi"))
        table(sil, key="sil_table", wide_text=["messaggio_di_ripresa"],
              rename={"dal": "Dal", "al": "Al", "durata_ore": "Ore", "durata_giorni": "Giorni",
                      "ultimo_a_scrivere": "Ultimo a scrivere", "ha_rotto_il_silenzio": "Ha rotto il silenzio",
                      "messaggio_di_ripresa": "Messaggio di ripresa"})
        breakers = sil["ha_rotto_il_silenzio"].value_counts()
        callout(t("Chi rompe i silenzi più lunghi: ") + " · ".join(t("**{p}** {n} volte", p=short_name(p), n=n)
                                                                    for p, n in breakers.items()), "info")

    section(t("🌙 Cronotipo"), t("In che ore e in che giorni scrive ciascuno."))
    ch = A("chronotype")
    if not ch["hourly"].empty:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.line(ch["hourly"], x="hour", y="pct", color="sender", markers=True, color_discrete_map=CMAP,
                          labels={"hour": t("ora"), "pct": t("% dei propri messaggi"), "sender": ""})
            fig.update_xaxes(dtick=2, range=[-0.5, 23.5])
            show(fig, 330, title=t("Profilo orario (quota dei propri messaggi per ora)"), key="chr_hour")
        with c2:
            band_order = ["Mattina (6-12)", "Pomeriggio (12-18)", "Sera (18-24)", "Notte (0-6)"]
            band_colors = {"Mattina (6-12)": "#ffd54f", "Pomeriggio (12-18)": "#4fc3f7", "Sera (18-24)": "#9575cd", "Notte (0-6)": "#37474f"}
            bands = ch["bands"].assign(fascia=lambda d_: d_["fascia"].astype(str).map(tv))
            fig = px.bar(bands, x="sender", y="pct", color="fascia", text_auto=".0f",
                         category_orders={"fascia": [tv(b) for b in band_order]},
                         color_discrete_map={tv(k_): v_ for k_, v_ in band_colors.items()},
                         labels={"sender": "", "pct": "%", "fascia": ""})
            fig.update_xaxes(tickvals=ch["bands"]["sender"].unique(), ticktext=[short_name(x) for x in ch["bands"]["sender"].unique()])
            show(fig, 330, title=t("Fasce della giornata"), key="chr_bands")
        wd = ch["weekday"].assign(giorno=lambda d_: d_["giorno"].map(tv))
        fig = px.bar(wd, x="giorno", y="pct", color="sender", barmode="group", color_discrete_map=CMAP,
                     category_orders={"giorno": [tv(x) for x in config.WEEKDAY_NAMES_IT]},
                     labels={"giorno": "", "pct": t("% dei propri messaggi"), "sender": ""})
        show(fig, 300, title=t("Giorni della settimana"), key="chr_wd")
        if not ch.get("peaks", pd.DataFrame()).empty:
            callout(t("Ora di punta: ") + " · ".join(t("**{p}** alle {h}:00", p=short_name(r["sender"]), h=f"{int(r['peak_hour']):02d}")
                                                       for r in ch["peaks"].to_dict("records")), "info")
    who = choose(st.selectbox, t("Mappa 7×24 per"), {"__all__": t("Tutti"), **{p_: p_ for p_ in participants}}, "hm_who")
    hm = A("heatmap", sender=None if who == "__all__" else who)
    hm.index = [tv(x) for x in hm.index]
    fig = px.imshow(hm, color_continuous_scale="Inferno", aspect="auto", labels=dict(x=t("ora"), y="", color=t("messaggi")))
    show(fig, 340, title=t("Messaggi per giorno × ora · {w}", w=t("Tutti") if who == "__all__" else who), key="hm_7x24")
    vi = A("vampire_index")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(vi, x="sender", y="vampire_index_pct", color="sender", color_discrete_map=CMAP, text_auto=".1f",
                     labels={"sender": "", "vampire_index_pct": t("% messaggi tra 01:00 e 05:30")})
        show(fig, 300, title=t("🧛 Indice vampiro (messaggi nelle ore piccole)"), legend=False, key="vamp")
    with c2:
        ck = A("catalysts_killers")
        st.markdown(t("**🔥 Chi accende le conversazioni**"))
        table(ck["catalysts"], key="cat_table")
        st.markdown(t("**🪦 Dopo chi cala il silenzio** (pausa > 4h di giorno)"))
        table(ck["killers"], key="kill_table")

    section(t("🕸️ Dinamica delle risposte"), t("Probabilità che l'altra persona risponda entro 3 minuti."))
    mk = A("markov_matrix")
    edges = mk["edges"]
    if edges.empty:
        st.info(t("Nessuna transizione entro 3 minuti rilevata."))
    elif is_couple:
        cols = st.columns(len(edges))
        for i, e in enumerate(edges.to_dict("records")):
            kpi(cols[i], f"{short_name(e['source'])} → {short_name(e['target'])}", f"{e['probability'] * 100:.0f}%",
                t("risposte entro 3 min · latenza mediana {s} s · {n} volte", s=f"{e['median_latency_s']:.0f}", n=fmt_num(e["count"])))
        st.write("")
    else:
        min_p = st.slider(t("Probabilità minima arco"), 0.0, 0.5, 0.02, 0.01, key="mk_minp")
        G = nx.DiGraph()
        for p in participants:
            G.add_node(p, turns=mk["turns"].get(p, 0))
        for e in edges.itertuples():
            if e.probability >= min_p:
                G.add_edge(e.source, e.target, weight=e.probability, lat=e.median_latency_s, count=e.count)
        pos = nx.spring_layout(G, weight="weight", seed=7, k=1.2)
        fig = go.Figure()
        max_turns = max(mk["turns"].values()) if mk["turns"] else 1
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            dx, dy = x1 - x0, y1 - y0
            ox, oy = -dy * 0.06, dx * 0.06
            fig.add_annotation(x=x1 * 0.88 + x0 * 0.12 + ox, y=y1 * 0.88 + y0 * 0.12 + oy,
                               ax=x0 * 0.88 + x1 * 0.12 + ox, ay=y0 * 0.88 + y1 * 0.12 + oy,
                               xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
                               arrowsize=1.1, arrowwidth=1 + 9 * d["weight"], arrowcolor=CMAP.get(u, "#aaa"), opacity=0.75)
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in G.nodes], y=[pos[n][1] for n in G.nodes], mode="markers+text", text=list(G.nodes),
            textposition="top center",
            marker=dict(size=[18 + 40 * (G.nodes[n]["turns"] / max_turns) for n in G.nodes],
                        color=[CMAP.get(n) for n in G.nodes], line=dict(width=2, color="#fff")),
            hovertext=[t("{n}: {k} turni", n=n, k=G.nodes[n]["turns"]) for n in G.nodes], hoverinfo="text", showlegend=False))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x")
        show(fig, 560, title=t("Chi risponde a chi (spessore = probabilità)"), legend=False, key="mk_graph")
        mat = mk["matrix"]
        if isinstance(mat, pd.DataFrame) and not mat.empty:
            fig = px.imshow(mat * 100, text_auto=".0f", color_continuous_scale="Viridis", aspect="auto",
                            labels=dict(x=t("Risponde"), y=t("Ha scritto"), color="%"))
            show(fig, 420, title=t("Matrice di transizione (%)"), key="mk_matrix")

# =========================================================================== AI DETECTIVE
if section_sel == SECTIONS[6]:
    st.subheader(t("🕵️ AI Detective — Chat investigativa Hybrid RAG"))
    st.caption(f"LLM: `{st.session_state.llm_model}` @ `{st.session_state.llm_base_url}` · "
               + t("Ogni affermazione è ancorata a SQL e citazioni reali."))
    if not llm_manager.endpoint_alive(st.session_state.llm_base_url, 1.5):
        stt_ = LLM.status()
        if LLM.busy():
            callout(t("Il modello si sta preparando ({m}). Puoi intanto esplorare le altre sezioni.", m=stt_.get("message", "")), "warn")
        else:
            rec_ = llm_manager.recommended_model()
            entry_ = llm_manager.catalog_entry(rec_) if rec_ else None
            callout(t("Il modello AI non è attivo. Avvialo dal pannello **🤖 Modello AI locale** nella barra laterale, "
                      "oppure con un clic qui sotto (modello consigliato per il tuo computer)."), "warn")
            if entry_ and st.button(t("▶️ Prepara {m}", m=t(entry_["label"]))
                                    + ("" if llm_manager.is_model_cached(rec_, entry_["backend"]) else f" (download {entry_['size_gb']} GB)"),
                                    key="llm_quick"):
                st.session_state.llm_custom = False
                LLM.ensure_ready(rec_, entry_["backend"])
                st.rerun()
    chat_key = st.session_state.db_path
    history: List[Dict[str, Any]] = st.session_state.chats.setdefault(chat_key, [])

    prompts = config.DEEP_DIVE_PROMPTS_EN if i18n.get_lang() == "en" else config.DEEP_DIVE_PROMPTS
    bcols = st.columns(len(prompts))
    for i, (label, prompt) in enumerate(prompts.items()):
        if bcols[i].button(label, width="stretch", key=f"dd_{i}"):
            st.session_state.pending_prompt = prompt
    if history and st.button(t("🧹 Nuova indagine"), key="clear_chat"):
        st.session_state.chats[chat_key] = []
        st.rerun()

    def render_traces(traces: List[Dict[str, Any]]) -> None:
        if not traces:
            return
        with st.expander(t("🔍 Tool & Fonti Consultate ({n})", n=len(traces))):
            for tr in traces:
                st.markdown(f"**`{tr['name']}`** · {tr.get('elapsed_s', 0)}s")
                args = tr.get("arguments") or {}
                res = tr.get("result") or {}
                if tr["name"] == "query_sql_database":
                    st.code(args.get("sql_query", ""), language="sql")
                    if isinstance(res, dict) and res.get("columns"):
                        st.dataframe(pd.DataFrame(res["rows"], columns=res["columns"]), hide_index=True, width="stretch")
                    elif isinstance(res, dict) and res.get("error"):
                        st.error(res["error"])
                elif tr["name"] == "search_chat_sessions" and isinstance(res, dict):
                    st.caption(t("Query: {q} · filtri: {f}", q=args.get("query"), f=res.get("filters")))
                    for s in res.get("sessions", []):
                        st.markdown(t("Sessione **#{n}** · {d} · conflict {c} · RRF {r} (BM25 #{b}, Vector #{v})",
                                      n=s["session_id"], d=s["start_time"], c=s["conflict_score"], r=s["rrf_score"],
                                      b=s.get("bm25_rank"), v=s.get("vector_rank")))
                        st.markdown(f"<div class='transcript'>{html_escape(s['transcript'])}</div>", unsafe_allow_html=True)
                else:
                    st.json(args)
                    st.json(res, expanded=False)

    for msg in history:
        with st.chat_message(msg["role"], avatar="🕵️" if msg["role"] == "assistant" else "🧑"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_traces(msg.get("traces", []))

    typed = st.chat_input(t("Chiedi qualsiasi cosa sulla chat (es. 'Chi ha scritto per primo a Capodanno 2024?')"))
    prompt = typed or st.session_state.pending_prompt
    if prompt:
        st.session_state.pending_prompt = None
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🕵️"):
            from chatautopsy.agent import ChatAutopsyAgent, strip_thinking

            status_box = st.status(t("Indagine in corso…"), expanded=False)
            placeholder = st.empty()
            buffer = ""
            final_text = ""
            traces: List[Dict[str, Any]] = []
            try:
                agent = ChatAutopsyAgent(st.session_state.db_path, st.session_state.llm_base_url, st.session_state.llm_model,
                                         st.session_state.llm_temperature, start_date=st.session_state.start_date,
                                         end_date=st.session_state.end_date, engine=current_engine(),
                                         dossier_provider=lambda: A("dossier"), language=i18n.get_lang())
                api_history = [{"role": m["role"], "content": m["content"]} for m in history]
                for ev in agent.run(prompt, api_history):
                    kind_ = ev["type"]
                    if kind_ == "token":
                        buffer += ev["content"]
                        placeholder.markdown(strip_thinking(buffer) + " ▌")
                    elif kind_ == "reset":
                        buffer = ""
                        placeholder.empty()
                    elif kind_ == "tool_call":
                        status_box.update(label=f"🔧 {ev['name']}…", state="running")
                        status_box.write(f"🔧 `{ev['name']}` {json.dumps(ev['arguments'], ensure_ascii=False)[:300]}")
                    elif kind_ == "tool_result":
                        traces.append(ev["trace"])
                        status_box.write(t("✅ `{n}` · {c} caratteri · {s}s", n=ev["trace"]["name"],
                                           c=fmt_num(ev["trace"]["result_chars"]), s=ev["trace"]["elapsed_s"]))
                    elif kind_ == "status":
                        status_box.write(ev["content"])
                    elif kind_ == "final":
                        final_text = ev["content"]
                    elif kind_ == "error":
                        final_text = f"⚠️ {ev['content']}\n\n" + t("Verifica che il server LLM locale sia avviato e che il nome del modello sia corretto.")
                status_box.update(label=t("Indagine completata · {n} tool eseguiti", n=len(traces)), state="complete")
            except Exception as exc:
                status_box.update(label=t("Errore"), state="error")
                final_text = f"⚠️ {type(exc).__name__}: {exc}"
            placeholder.markdown(final_text or t("_(nessuna risposta)_"))
            render_traces(traces)
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": final_text, "traces": traces})

# =========================================================================== PERSONA
if section_sel == SECTIONS[7]:
    st.subheader(t("🎭 Persona & Group Simulator — Clonazione dello stile"))
    st.caption(t("Profilo stilistico reale (lunghezze, punteggiatura, TF-IDF, emoji, few-shot reali) iniettato nel prompt del LLM locale."))
    c1, c2 = st.columns([1, 2])
    with c1:
        target = choose(st.selectbox, t("Chi simulare"),
                        {**{p_: p_ for p_ in participants}, config.GROUP_SIMULATION_LABEL: t(config.GROUP_SIMULATION_LABEL)},
                        "sim_target")
        n_turns = st.slider(t("Battute del dialogo (modalità gruppo)"), 4, 20, 10, disabled=target != config.GROUP_SIMULATION_LABEL)
    with c2:
        stimulus = st.text_area(t("Stimolo / scenario"), value=t("Scusami, stasera faccio tardi"), height=110,
                                placeholder=t("Propongo di andare in vacanza in campeggio senza bagni"))
    if st.button(t("✨ Simula"), type="primary", disabled=not stimulus.strip()):
        from chatautopsy.agent import ChatAutopsyAgent, strip_thinking

        agent = ChatAutopsyAgent(st.session_state.db_path, st.session_state.llm_base_url, st.session_state.llm_model,
                                 max(st.session_state.llm_temperature, 0.6), start_date=st.session_state.start_date,
                                 end_date=st.session_state.end_date, engine=current_engine(),
                                 dossier_provider=lambda: A("dossier"), language=i18n.get_lang())
        out_box = st.empty()
        buf = ""
        final = ""
        profiles: List[Dict[str, Any]] = []
        with st.spinner(t("Costruzione profili e generazione…")):
            for ev in agent.simulate(target, stimulus.strip(), n_turns=n_turns):
                if ev["type"] == "profiles":
                    profiles = ev["profiles"]
                elif ev["type"] == "token":
                    buf += ev["content"]
                    out_box.markdown(f"<div class='transcript'>{html_escape(strip_thinking(buf))} ▌</div>", unsafe_allow_html=True)
                elif ev["type"] == "final":
                    final = ev["content"]
                elif ev["type"] == "error":
                    final = f"⚠️ {ev['content']}"
        out_box.markdown(f"<div class='transcript'>{html_escape(final)}</div>", unsafe_allow_html=True)
        for prof in profiles:
            with st.expander(t("🧬 Profilo stilistico iniettato · {p}", p=prof["participant"])):
                pc1, pc2 = st.columns(2)
                pc1.json(prof["style_stats"])
                pc2.markdown(t("**Parole identitarie:** ") + ", ".join(prof["signature_words_tfidf"]))
                pc2.markdown(t("**Emoji preferite:** ") + " ".join(f"{e}×{c}" for e, c in prof["favorite_emojis"]))
                st.markdown(t("**Messaggi reali (few-shot):**"))
                st.markdown("\n".join(f"- `{m}`" for m in prof["representative_messages"]))


# ---------------------------------------------------------------------------
# Precalcolo in background delle altre sezioni (dopo il render della sezione corrente)
# ---------------------------------------------------------------------------
WARM_CALLS = [
    ("overview_kpis", {}), ("balance_scorecard", {}), ("daily_activity", {}), ("yearly_scorecard", {}),
    ("share_timeseries", {}), ("initiation_rate", {}), ("conversation_closers", {}), ("double_texting", {}),
    ("effort_ratio", {}), ("length_distribution", {}), ("media_mix", {}), ("love_bombing_devaluation", {}),
    ("response_latency", {}), ("latency_distribution", {}), ("latency_monthly", {}), ("fading_detector", {}),
    ("conflict_distribution", {}), ("conflict_timeline", {}), ("climate_monthly", {}), ("pronoun_shift", {}),
    ("stonewalling", {}), ("apology_vs_blame", {}), ("conflict_recovery", {}),
    ("lexicon_report", {"name": "gaslighting", "examples": 8}), ("lexicon_report", {"name": "passive_aggressive", "examples": 8}),
    ("lexicon_report", {"name": "guilt_tripping", "examples": 8}), ("top_conflict_sessions", {"n": 20}),
    ("hostility_seasonality", {}), ("chronotype", {}), ("vampire_index", {}), ("catalysts_killers", {}),
    ("markov_matrix", {}), ("emoji_fingerprint", {}), ("affection_timeline", {}), ("tfidf_signature", {}),
]


@st.cache_resource(show_spinner=False, max_entries=4)
def start_warmup(db_path: str, signature: float, start: Optional[str], end: Optional[str]) -> bool:
    """Una sola volta per (DB, filtro): un thread daemon calcola le analisi sull'engine condiviso
    (memoizzate), così aprire le altre sezioni è quasi istantaneo."""
    import threading

    eng = get_engine(db_path, signature, start, end)

    def _run() -> None:
        for method, kwargs in WARM_CALLS:
            try:
                getattr(eng, method)(**kwargs)
            except Exception:
                pass

    threading.Thread(target=_run, name="chatautopsy-warmup", daemon=True).start()
    return True


start_warmup(st.session_state.db_path, database.db_signature(st.session_state.db_path),
             st.session_state.get("start_date"), st.session_state.get("end_date"))
