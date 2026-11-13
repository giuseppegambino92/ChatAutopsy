"""Internazionalizzazione: ogni stringa dell'interfaccia ha la traduzione inglese e l'app funziona in inglese."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from chatautopsy import analytics_engine, config, i18n, ui
from chatautopsy.locales_en import EN

PKG = Path(__file__).resolve().parents[1] / "chatautopsy"


def _source_keys():
    keys = set()
    for f in ("app.py", "ui.py", "cli.py", "ingestion.py", "llm_manager.py"):
        for n in ast.walk(ast.parse((PKG / f).read_text(encoding="utf-8"))):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("t", "_t") and n.args
                    and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
                keys.add(n.args[0].value)
    return keys


def test_every_ui_string_has_an_english_translation():
    missing = sorted(k for k in _source_keys() if k.strip() and k not in EN)
    assert not missing, missing


def test_engine_labels_are_translated(couple_db):
    labels = set(ui.LABELS.values()) | set(ui.LB_LABELS.values()) | set(analytics_engine.MEDIA_LABELS_IT.values())
    labels |= set(config.EMBEDDING_PRESETS) | {e["label"] for e in config.LLM_CATALOG} | set(config.TRACKED_EMOJI_GROUPS)
    labels |= set(config.WEEKDAY_NAMES_IT)
    b = analytics_engine.AnalyticsEngine(couple_db).balance_scorecard()
    for ind in b["indicators"]:
        labels |= {ind["categoria"], ind["indicatore"], ind["significato"]} | ({ind["unita"]} if ind["unita"] else set())
    labels |= set(analytics_engine.AnalyticsEngine(couple_db).yearly_scorecard().columns)
    missing = sorted(x for x in labels if x not in EN)
    assert not missing, missing


def test_placeholders_match():
    bad = [(k, v) for k, v in EN.items() if set(re.findall(r"\{(\w+)", k)) != set(re.findall(r"\{(\w+)", v))]
    assert not bad, bad


def test_t_and_tv_switch_language():
    i18n.set_lang("en")
    try:
        assert i18n.t("{n} giorni", n=3) == "3 days"
        assert i18n.tv("Forte prevalenza: Giulia") == "Strong lead: Giulia"
        assert i18n.tv("un messaggio qualunque") == "un messaggio qualunque"
    finally:
        i18n.set_lang("it")
    assert i18n.t("{n} giorni", n=3) == "3 giorni"


def test_app_renders_in_english(couple_db):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(PKG / "app.py"), default_timeout=300)
    at.session_state["db_path"] = str(couple_db)
    at.session_state["lang"] = "en"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    sections = [EN[x] for x in ["🧾 Pagella", "📊 Volumi & Potere", "⏱️ Reattività", "🚩 Conflitti & Red Flags",
                                "🗣️ Lessico & Affetto", "🕰️ Ritmi & Abitudini", "🕵️ AI Detective", "🎭 Persona Simulator"]]
    italian = re.compile(r"\b(messaggi|giorni|della|degli|chi|conversazioni|risposta|tensione)\b", re.IGNORECASE)
    leaks = []
    for s in sections:
        at.session_state["section"] = s
        at.run()
        assert not at.exception, (s, [e.value for e in at.exception])
        for h in at.subheader:
            if italian.search(h.value):
                leaks.append(h.value)
    assert not leaks, leaks
