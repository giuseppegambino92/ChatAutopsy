"""La dashboard si apre e ogni sezione viene renderizzata senza eccezioni (Streamlit AppTest)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "chatautopsy" / "app.py")
SECTIONS = ["🧾 Pagella", "📊 Volumi & Potere", "⏱️ Reattività", "🚩 Conflitti & Red Flags",
            "🗣️ Lessico & Affetto", "🕰️ Ritmi & Abitudini", "🕵️ AI Detective", "🎭 Persona Simulator"]


def _render_all(db_path: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["db_path"] = db_path
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    charts = 0
    for s in SECTIONS:
        at.session_state["section"] = s
        at.run()
        assert not at.exception, (s, [e.value for e in at.exception])
        charts += len(at.get("plotly_chart"))
    assert charts >= 30
    return at


def test_landing_page_without_database(monkeypatch, tmp_path):
    from chatautopsy import database

    monkeypatch.setattr(database, "list_databases", lambda: [])
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["db_path"] = str(tmp_path / "missing.db")
    at.run()
    assert not at.exception
    assert any("demo" in b.label.lower() or "dimostrativa" in b.label.lower() for b in at.button)


def test_all_sections_couple(couple_db):
    _render_all(str(couple_db))


def test_all_sections_group(group_db):
    _render_all(str(group_db))
