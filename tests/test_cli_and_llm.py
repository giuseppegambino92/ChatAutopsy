"""CLI e gestore del modello locale (senza download né server reali)."""
from __future__ import annotations

from chatautopsy import cli, config, database, demo
from chatautopsy import llm_manager as lm


def test_cli_analyze_folder_without_opening(tmp_path, capsys):
    folder = tmp_path / "WhatsApp Chat - Prova"
    folder.mkdir()
    (folder / "_chat.txt").write_text(demo.couple_chat(days=20), encoding="utf-8")
    rc = cli.main(["analyze", str(folder), "--embeddings", "none", "--no-open"])
    assert rc == 0
    db = database.db_path_for_name("Prova")
    assert db.exists()
    out = capsys.readouterr().out
    assert "messaggi" in out and "Giulia" in out
    assert cli.main(["list"]) == 0
    assert cli.main(["wipe", "Prova", "--yes"]) == 0 and not db.exists()


def test_cli_missing_path():
    assert cli.main(["analyze", "/percorso/che/non/esiste", "--no-open"]) == 2


def test_cli_doctor_runs():
    assert cli.main(["doctor"]) == 0


def test_recommended_model_by_ram(monkeypatch):
    for ram, expected in [(4, "Qwen3.5-4B"), (8, "Qwen3.5-4B"), (16, "Qwen3.5-9B"), (36, "Qwen3.8-27B"), (128, "Qwen3.5-122B")]:
        monkeypatch.setattr(lm, "ram_gb", lambda r=ram: float(r))
        assert expected in lm.recommended_model("mlx"), ram
    monkeypatch.setattr(lm, "ram_gb", lambda: 16.0)
    assert lm.recommended_model("ollama") == "qwen3.5:9b"


def test_catalog_has_only_recent_models():
    from chatautopsy import config

    assert not [e["id"] for e in config.LLM_CATALOG if "2.5" in e["id"]]


def test_settings_roundtrip():
    lm.save_settings(llm_model="x", llm_backend="mlx")
    s = lm.load_settings()
    assert s["llm_model"] == "x" and s["llm_backend"] == "mlx"
    assert config.SETTINGS_FILE.parent == config.DATA_DIR


def test_endpoint_dead_port():
    assert lm.endpoint_alive("http://127.0.0.1:9/v1", timeout=0.5) is False


def test_catalog_flags():
    cat = lm.catalog()
    assert cat and all({"cached", "fits", "backend", "size_gb"} <= set(e) for e in cat)
