"""Pipeline completa (senza embedding) e motore di analisi su chat sintetiche."""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3

import pytest

from chatautopsy import config, database, demo, ingestion
from chatautopsy.analytics_engine import AnalyticsEngine, to_jsonable

METHODS = ["overview_kpis", "volume_timeseries", "initiation_rate", "double_texting", "effort_ratio",
           "response_latency", "latency_monthly", "fading_detector", "love_bombing_devaluation", "stonewalling",
           "pronoun_shift", "apology_vs_blame", "conflict_recovery", "tfidf_signature", "affection_timeline",
           "emoji_fingerprint", "markov_matrix", "catalysts_killers", "heatmap", "vampire_index",
           "hostility_seasonality", "top_conflict_sessions", "conflict_timeline", "share_timeseries", "media_mix",
           "length_distribution", "latency_distribution", "daily_activity", "conversation_closers", "chronotype",
           "climate_monthly", "yearly_scorecard", "balance_scorecard", "conflict_distribution",
           "conflict_calibration_info", "dossier"]


def test_ingestion_creates_only_db_files(couple_db, data_dir):
    files = {p.name for p in data_dir.iterdir() if p.is_file()}
    app_files = {"settings.json", "llm_server.json"}  # impostazioni dell'app, non dati della chat
    assert all(f.endswith((".db", ".db-wal", ".db-shm")) or f in app_files for f in files), files
    meta = database.get_all_meta(couple_db)
    assert meta["source_file"] == "couple.txt"
    assert "is_forwarded" in database.table_columns(couple_db, "messages")


@pytest.mark.parametrize("start,end", [(None, None), ("2022-06-01", "2022-12-31")])
def test_all_analytics_run(couple_db, start, end):
    eng = AnalyticsEngine(couple_db, start, end)
    for m in METHODS:
        assert getattr(eng, m)() is not None, m
    d = eng.dossier()
    assert not [k for k, v in d.items() if isinstance(v, dict) and "error" in v]
    assert len(json.dumps(to_jsonable(d), default=str)) < config.DOSSIER_RESULT_MAX_CHARS


def test_shared_engine_equals_fresh_engines(couple_db):
    dump = lambda o: json.dumps(to_jsonable(o), sort_keys=True, default=str)  # noqa: E731
    fresh = {m: dump(getattr(AnalyticsEngine(couple_db), m)()) for m in METHODS}
    shared = AnalyticsEngine(couple_db)
    rev = {m: dump(getattr(shared, m)()) for m in reversed(METHODS)}
    again = {m: dump(getattr(shared, m)()) for m in METHODS}
    assert [m for m in METHODS if not (fresh[m] == rev[m] == again[m])] == []


def test_balance_scorecard_couple_vs_group(couple_db, group_db):
    b = AnalyticsEngine(couple_db).balance_scorecard()
    assert 0 <= b["reciprocity_index"] <= 100
    assert len(b["indicators"]) >= 25
    assert all(i["verdetto"] for i in b["indicators"])
    assert AnalyticsEngine(group_db).balance_scorecard()["reciprocity_index"] is None


def test_group_stonewalling_filter(group_db, couple_db):
    sw = AnalyticsEngine(group_db).stonewalling()
    assert sw["group_filter_applied"] is True
    assert sw["excluded_group_noise"] == 1
    assert len(sw["events"]) == 2
    swc = AnalyticsEngine(couple_db).stonewalling()
    assert swc["group_filter_applied"] is False and len(swc["events"]) > 0


def _score(lines):
    df = ingestion.extract_features([ingestion.RawMessage(dt.datetime(2024, 1, 1, 20, 0) + dt.timedelta(seconds=10 * i), s, t)
                                     for i, (s, t) in enumerate(lines)])
    return ingestion.score_session(df)


def test_laughter_damping_and_real_fight():
    laugh = [("A" if i % 2 else "B", t) for i, t in enumerate(
        ["AHAHAHAH NOOO!!!", "😂😂😂", "ahahah ma smettila!!", "🤣🤣", "MA DAI!!!???", "hahaha basta che muoio", "😂", "ahahaha no vabbè!!"] * 5)]
    fight = [("A" if i % 2 else "B", t) for i, t in enumerate(
        ["è colpa tua, esageri sempre!!", "non mi ascolti mai, sei sempre il solito", "basta, mi hai stufato?!", "ok",
         "fai come vuoi", "SEI INCREDIBILE", "te lo sei sognata", "vaffanculo"] * 5)]
    mixed = [(s, t.replace("ahah", "").replace("😂", "").replace("🤣", "").replace("hahaha", "")) for s, t in laugh]
    sl, sf, sm = _score(laugh), _score(fight), _score(mixed)
    assert sl["conflict_score"] < config.HIGH_CONFLICT_THRESHOLD / 2
    assert sf["conflict_score"] > 60
    assert sl["conflict_score"] <= sm["conflict_score"] * config.LAUGHTER_DAMPING_FACTOR + 0.5


def test_forwarded_text_excluded_from_scores():
    rows = [ingestion.RawMessage(dt.datetime(2024, 1, 1, 20, 0) + dt.timedelta(seconds=10 * i), "A" if i % 2 else "B",
                                 "guerra odio basta vaffanculo colpa tua!!", is_forwarded=True) for i in range(10)]
    assert ingestion.score_session(ingestion.extract_features(rows))["conflict_score"] < 15


def test_tfidf_ignores_fillers_and_laughter(data_dir):
    fill = ["ahahah amore ok bene dai", "boh già sì poi ora", "hahahahaha eh ah oh mah", "ahahahahah okok va lì qui cosa fatto detto"]
    lines = [(i, "Ugo" if i % 2 else "Ivo", fill[i % 4] + (" zuzzurellone" if i % 2 else " cippalippa")) for i in range(60)]
    db = data_dir / "tf.db"
    ingestion.run_ingestion(None, db, file_bytes=demo.whatsapp_lines(lines).encode(), source_name="tf.txt", build_embeddings=False)
    sig = AnalyticsEngine(db).tfidf_signature()
    words = {w for d in sig.values() for t in d["term"].head(5) for w in t.split()}
    assert words <= {"zuzzurellone", "cippalippa", "amore"}


def test_adaptive_conflict_threshold(couple_db):
    cal = AnalyticsEngine(couple_db).conflict_calibration_info()
    assert config.CONFLICT_RELATIVE_MIN <= cal["threshold"] <= config.HIGH_CONFLICT_THRESHOLD
    assert cal["sessions"] > 0


def test_wal_and_secure_wipe(data_dir):
    db = data_dir / "wipe_me.db"
    ingestion.run_ingestion(None, db, file_bytes=demo.couple_chat(days=3).encode(), source_name="w.txt", build_embeddings=False)
    with database.connect(db, read_only=True) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
    removed = database.secure_wipe_database(db)
    assert "wipe_me.db" in removed and not db.exists()
    assert not any(p.name.startswith("wipe_me") for p in data_dir.iterdir())


def test_old_schema_without_forwarded_column_still_works(data_dir, couple_db):
    old = data_dir / "old_schema.db"
    src = sqlite3.connect(couple_db)
    dst = sqlite3.connect(old)
    src.backup(dst)
    src.close()
    dst.execute("CREATE TABLE m2 AS SELECT id, session_id, timestamp, date, month, hour, weekday, sender, text, char_count, "
                "word_count, emoji_list, emoji_count, question_count, exclamation_count, uppercase_ratio, is_night_message, "
                "is_media, media_type, reply_time_seconds, gap_seconds FROM messages")
    dst.execute("DROP TABLE messages")
    dst.execute("ALTER TABLE m2 RENAME TO messages")
    dst.commit()
    dst.close()
    assert "is_forwarded" not in database.table_columns(old, "messages")
    assert AnalyticsEngine(old).balance_scorecard()["reciprocity_index"] is not None
    os.remove(old)
