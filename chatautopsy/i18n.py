"""
ChatAutopsy — internazionalizzazione (italiano / inglese).

Le stringhe dell'interfaccia sono scritte in italiano nel codice e passano da `t()`:
in inglese vengono sostituite con la traduzione di `locales_en.EN` (chiave = testo italiano).
I segnaposto usano la sintassi di str.format: t("{n} giorni", n=3).

Lingua attiva: per sessione Streamlit (st.session_state["lang"]) oppure globale (CLI, test).
Default: CHATAUTOPSY_LANG → impostazioni salvate → lingua del sistema → inglese.
"""
from __future__ import annotations

import json
import locale
import os
from typing import Any, Optional

LANGS = {"en": "🇬🇧 English", "it": "🇮🇹 Italiano"}
_GLOBAL = {"lang": None}


def detect_lang() -> str:
    env = (os.environ.get("CHATAUTOPSY_LANG") or "").lower()[:2]
    if env in LANGS:
        return env
    try:
        from chatautopsy import config

        saved = json.loads(config.SETTINGS_FILE.read_text(encoding="utf-8")).get("lang")
        if saved in LANGS:
            return saved
    except Exception:
        pass
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        v = os.environ.get(var, "")
        if v:
            return "it" if v.lower().startswith("it") else "en"
    try:
        loc = (locale.getlocale()[0] or "").lower()
        if loc.startswith("it"):
            return "it"
    except Exception:
        pass
    if os.uname().sysname == "Darwin":  # macOS: lingua dell'utente da AppleLanguages
        try:
            import subprocess

            out = subprocess.run(["defaults", "read", "-g", "AppleLanguages"], capture_output=True, text=True, timeout=2).stdout
            first = out.replace("(", "").replace('"', "").split(",")[0].strip().lower()
            if first.startswith("it"):
                return "it"
        except Exception:
            pass
    return "en"


def _session_lang() -> Optional[str]:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is None:
            return None
        import streamlit as st

        return st.session_state.get("lang")
    except Exception:
        return None


def get_lang() -> str:
    lang = _session_lang() or _GLOBAL["lang"]
    if lang not in LANGS:
        lang = _GLOBAL["lang"] = detect_lang()
    return lang


def set_lang(lang: str) -> None:
    if lang in LANGS:
        _GLOBAL["lang"] = lang


def t(text: str, **kwargs: Any) -> str:
    """Traduce una stringa dell'interfaccia (chiave = testo italiano) e applica i segnaposto."""
    if get_lang() == "en":
        from chatautopsy.locales_en import EN

        text = EN.get(text, text)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def tv(value: Any) -> Any:
    """Traduce un valore di dati (etichette prodotte dall'engine), lasciando invariato tutto il resto."""
    if not isinstance(value, str) or get_lang() != "en":
        return value
    from chatautopsy.locales_en import EN

    if value in EN:
        return EN[value]
    for prefix in ("Lieve prevalenza: ", "Forte prevalenza: "):
        if value.startswith(prefix):
            return EN[prefix + "{p}"].format(p=value[len(prefix):])
    return value


def decimal_sep() -> str:
    return "." if get_lang() == "en" else ","


def thousands_sep() -> str:
    return "," if get_lang() == "en" else "."
