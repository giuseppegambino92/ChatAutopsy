"""Fixture comuni: ogni sessione di test usa una cartella dati temporanea (mai i dati reali)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Deve avvenire PRIMA di importare chatautopsy (config legge l'ambiente all'import)
_TMP_HOME = Path(tempfile.mkdtemp(prefix="chatautopsy-test-"))
os.environ["CHATAUTOPSY_HOME"] = str(_TMP_HOME)
os.environ["CHATAUTOPSY_DATA_DIR"] = str(_TMP_HOME / "data")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["CHATAUTOPSY_LANG"] = "it"  # i test in italiano usano le etichette originali; test_i18n copre l'inglese
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import pytest  # noqa: E402

from chatautopsy import config, demo  # noqa: E402
from chatautopsy.ingestion import run_ingestion  # noqa: E402


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return config.DATA_DIR


@pytest.fixture(scope="session")
def couple_db(data_dir: Path) -> Path:
    db = data_dir / "couple.db"
    run_ingestion(None, db, file_bytes=demo.couple_chat().encode("utf-8"), source_name="couple.txt", build_embeddings=False)
    return db


@pytest.fixture(scope="session")
def group_db(data_dir: Path) -> Path:
    db = data_dir / "group.db"
    run_ingestion(None, db, file_bytes=demo.group_chat().encode("utf-8"), source_name="group.txt", build_embeddings=False)
    return db
