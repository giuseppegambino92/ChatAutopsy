"""
ChatAutopsy — Ingestione.

Pipeline:
  1. Parser universale: WhatsApp .txt (iOS / Android, tutte le varianti regionali, 12h/24h),
     archivi .zip di export WhatsApp, Telegram result.json (chat singola o export completo),
     Telegram HTML (pagine messagesN.html, cartella ChatExport_… o .zip).
  2. Estrazione feature per messaggio (emoji, punteggiatura, maiuscole, notte, latenze).
  3. Smart Session Chunking (gap 45 min / blocchi da 45 con overlap 5) + trascrizione teatrale.
  4. Heuristic Conflict & Emotional Intensity Score (0-100) per sessione, senza LLM.
  5. Scrittura SQL, indicizzazione FTS5 e (opzionale) embedding vettoriali su MPS.
"""
from __future__ import annotations

import html
import io
import json
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import emoji
import numpy as np
import pandas as pd

from chatautopsy import config, database
from chatautopsy.i18n import t as _t

ProgressCallback = Callable[[str, float, str], None]

# ---------------------------------------------------------------------------
# Strutture
# ---------------------------------------------------------------------------
@dataclass
class RawMessage:
    timestamp: datetime
    sender: str
    text: str
    is_media: bool = False
    media_type: Optional[str] = None
    is_forwarded: bool = False  # inoltrato da altri (Telegram): non è testo scritto dal mittente


# ---------------------------------------------------------------------------
# Regex compilate
# ---------------------------------------------------------------------------
_INVISIBLE_CHARS = dict.fromkeys(
    map(ord, "‎‏‪‫‬‭‮﻿⁦⁧⁨⁩​"), None
)
_DATE = r"(?P<date>\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4})"
_TIME = r"(?P<time>\d{1,2}[:.]\d{2}(?:[:.]\d{2})?(?:\s?(?:[AaPp]\.?\s?[Mm]\.?))?)"
WA_IOS_RE = re.compile(rf"^\[{_DATE},?\s+{_TIME}\]\s?(?P<rest>.*)$")
WA_ANDROID_RE = re.compile(rf"^{_DATE},?\s+{_TIME}\s?[-–—]\s(?P<rest>.*)$")
WA_SENDER_RE = re.compile(r"^(?P<sender>[^:]{1,80}?):(?:\s(?P<text>.*))?$", re.DOTALL)

SYSTEM_RE = re.compile("|".join(f"(?:{p})" for p in config.SYSTEM_MESSAGE_PATTERNS), re.IGNORECASE)
SYSTEM_STRONG_RE = re.compile("|".join(f"(?:{p})" for p in config.SYSTEM_MESSAGE_PATTERNS_STRONG), re.IGNORECASE)
MEDIA_RES = [(re.compile(p, re.IGNORECASE), t) for p, t in config.MEDIA_PATTERNS]
EDITED_RE = re.compile(config.EDITED_SUFFIX_PATTERN, re.IGNORECASE)
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
MULTI_PUNCT_RE = re.compile(r"[?!]{2,}|\.{3,}|…")


def _lexicon_units(term: str) -> List[str]:
    """Spezza un termine in unità regex: caratteri escapati, spazi → \\s+."""
    units: List[str] = []
    for ch in term:
        if ch.isspace():
            if units and units[-1] == r"\s+":
                continue
            units.append(r"\s+")
        else:
            units.append(re.escape(ch))
    return units


def _trie_regex(sequences: List[List[str]]) -> str:
    """Alternanza fattorizzata a prefissi comuni (trie). Le continuazioni sono greedy:
    a parità di posizione vince il termine più lungo, come nell'alternanza ordinata per lunghezza."""
    trie: Dict = {}
    for seq in sequences:
        node = trie
        for u in seq:
            node = node.setdefault(u, {})
        node[""] = {}

    def depth(node: Dict) -> int:
        return max((1 + depth(child) for k, child in node.items() if k), default=0)

    def build(node: Dict) -> str:
        end = "" in node
        branches = []
        # rami più lunghi prima, lookaround (confine destro) per ultimo: vince il termine più lungo
        for unit in sorted((k for k in node if k), key=lambda k: (k.startswith("(?"), -depth(node[k]), k)):
            branches.append(unit + build(node[unit]))
        if not branches:
            return ""
        body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
        if end:
            return f"(?:{body})?" if len(branches) > 1 else (f"(?:{body})?" if len(body) > 1 else f"{body}?")
        return body

    return build(trie)


def compile_lexicon(terms: List[str]) -> re.Pattern:
    """Regex a confine di parola (unicode) per una lista di lemmi/frasi.

    I confini (?<!\\w)/(?!\\w) sono raccolti a fattor comune e i termini organizzati in trie:
    stesso risultato dell'alternanza piatta "frasi più lunghe prima", ma ~10x più veloce su
    milioni di messaggi (il motore re non prova più 100+ alternative a ogni carattere).
    Se il termine inizia/finisce con punteggiatura (es. "ok.", ":(") il relativo confine non si applica.
    """
    cleaned = sorted({t.strip().lower() for t in terms if t.strip()}, key=len, reverse=True)
    if not cleaned:
        return re.compile(r"(?!x)x")
    # Il confine destro diventa l'ultima unità del ramo, così "ok" e "ok." convivono nello stesso trie
    # e vince comunque il più lungo; il confine sinistro separa due trie (con/senza lookbehind).
    right_boundary = r"(?!\w)"
    left_boundary = r"(?<!\w)"
    groups: Dict[bool, List[List[str]]] = {True: [], False: []}
    for t in cleaned:
        units = _lexicon_units(t)
        if t[-1].isalnum():
            units.append(right_boundary)
        groups[t[0].isalnum()].append(units)
    parts = []
    if groups[True]:
        parts.append(left_boundary + "(?:" + _trie_regex(groups[True]) + ")")
    if groups[False]:
        parts.append("(?:" + _trie_regex(groups[False]) + ")")
    return re.compile("|".join(parts), re.IGNORECASE | re.UNICODE)


def _compile_lexicon_flat(terms: List[str]) -> re.Pattern:
    """Versione di riferimento (alternanza piatta): usata solo dai test di equivalenza."""
    cleaned = sorted({t.strip().lower() for t in terms if t.strip()}, key=len, reverse=True)
    parts = []
    for t in cleaned:
        esc = re.escape(t).replace(r"\ ", r"\s+")
        right = r"(?!\w)" if t[-1].isalnum() else ""
        left = r"(?<!\w)" if t[0].isalnum() else ""
        parts.append(f"{left}{esc}{right}")
    return re.compile("|".join(parts) if parts else r"(?!x)x", re.IGNORECASE | re.UNICODE)


CONFLICT_RE = compile_lexicon(config.CONFLICT_LEXICON)
ACCUSATION_RE = compile_lexicon(config.ACCUSATION_LEXICON)
NEGATION_RE = compile_lexicon(config.NEGATION_LEXICON)
AFFECTION_RE = compile_lexicon(config.AFFECTION_LEXICON)
LAUGHTER_RE = re.compile(config.LAUGHTER_PATTERN, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Lettura file
# ---------------------------------------------------------------------------
def read_input_bytes(filename: str, data: bytes) -> Tuple[str, object]:
    """Decodifica un export interamente in memoria. Ritorna ("whatsapp", testo) o ("telegram", dict json)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            html_names = [n for n in names if _TG_HTML_FILE_RE.search(n) and not n.startswith("__MACOSX")]
            if html_names:
                return "telegram_html", [(n, zf.read(n)) for n in html_names]
            json_names = [n for n in names if n.lower().endswith("result.json")]
            txt_names = [n for n in names if n.lower().endswith(".txt") and not n.startswith("__MACOSX")]
            if json_names:
                return "telegram", json.loads(zf.read(json_names[0]).decode("utf-8-sig"))
            if not txt_names:
                raise ValueError("L'archivio .zip non contiene né un file .txt WhatsApp né un result.json Telegram.")
            preferred = [n for n in txt_names if "chat" in n.lower()] or txt_names
            target = max(preferred, key=lambda n: zf.getinfo(n).file_size)
            return "whatsapp", _decode_bytes(zf.read(target))
    if suffix == ".json":
        return "telegram", json.loads(data.decode("utf-8-sig"))
    if suffix in (".html", ".htm"):
        if not is_telegram_html(filename, data):
            raise ValueError("Il file HTML non sembra una pagina di export Telegram (messagesN.html).")
        return "telegram_html", [(filename, data)]
    return "whatsapp", _decode_bytes(data)


def read_input_file(path: str | Path) -> Tuple[str, object]:
    """Variante da file (uso CLI). La dashboard usa read_input_bytes: nessun file raw su disco."""
    path = Path(path)
    if path.is_dir():
        # cartella di export Telegram HTML (ChatExport_…): legge solo le pagine messages*.html
        pages = [(p.name, p.read_bytes()) for p in path.iterdir() if _TG_HTML_FILE_RE.search(p.name)]
        if pages:
            return "telegram_html", pages
        if (path / "result.json").exists():
            return read_input_file(path / "result.json")
        # cartella WhatsApp scompattata (_chat.txt o "Chat WhatsApp con ….txt") o .zip di export al suo interno
        txts = [p for p in path.iterdir() if p.suffix.lower() == ".txt" and not p.name.startswith(".")]
        if txts:
            preferred = [p for p in txts if "chat" in p.name.lower()] or txts
            target = max(preferred, key=lambda p: p.stat().st_size)
            return "whatsapp", _decode_bytes(target.read_bytes())
        zips = sorted(path.glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
        if zips:
            return read_input_file(zips[0])
        raise ValueError(f"La cartella {path.name} non contiene un export riconoscibile "
                         "(messages*.html, result.json, un .txt WhatsApp o uno .zip di export).")
    with open(path, "rb") as fh:
        return read_input_bytes(path.name, fh.read())


def _decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and "\x00" in text[:200]:
                continue
            return text
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# WhatsApp
# ---------------------------------------------------------------------------
def _normalize_line(line: str) -> str:
    return line.translate(_INVISIBLE_CHARS).replace(" ", " ").replace(" ", " ").rstrip("\r")


def _infer_date_order(date_strings: List[str]) -> str:
    """Ritorna 'ymd', 'dmy' oppure 'mdy' analizzando tutte le date dell'export."""
    first_gt12 = second_gt12 = False
    ymd = 0
    for ds in date_strings:
        parts = re.split(r"[./\-]", ds)
        if len(parts) != 3:
            continue
        if len(parts[0]) == 4:
            ymd += 1
            continue
        a, b = int(parts[0]), int(parts[1])
        if a > 12:
            first_gt12 = True
        if b > 12:
            second_gt12 = True
    if ymd and ymd >= len(date_strings) / 2:
        return "ymd"
    if second_gt12 and not first_gt12:
        return "mdy"
    return "dmy"


def _parse_wa_datetime(date_str: str, time_str: str, order: str) -> Optional[datetime]:
    parts = [int(p) for p in re.split(r"[./\-]", date_str)]
    if len(parts) != 3:
        return None
    if order == "ymd":
        y, m, d = parts
    elif order == "mdy":
        m, d, y = parts
    else:
        d, m, y = parts
    if y < 100:
        y += 2000
    t = time_str.strip().lower().replace(" ", "").replace(".", ":")
    ampm = None
    for suffix, tag in (("a:m:", "am"), ("p:m:", "pm"), ("am", "am"), ("pm", "pm"), ("a:m", "am"), ("p:m", "pm")):
        if t.endswith(suffix):
            ampm = tag
            t = t[: -len(suffix)]
            break
    t = t.rstrip(":")
    tparts = [int(x) for x in t.split(":") if x != ""]
    if len(tparts) < 2:
        return None
    hh, mm = tparts[0], tparts[1]
    ss = tparts[2] if len(tparts) > 2 else 0
    if ampm == "pm" and hh < 12:
        hh += 12
    if ampm == "am" and hh == 12:
        hh = 0
    try:
        return datetime(y, m, d, hh, mm, ss)
    except ValueError:
        return None


def classify_media(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None
    for rx, mtype in MEDIA_RES:
        if rx.match(stripped):
            return mtype
    return None


def parse_whatsapp_text(content: str) -> List[RawMessage]:
    raw_lines = content.splitlines()
    lines = [_normalize_line(l) for l in raw_lines]
    # Primo passaggio: righe con timestamp e formato prevalente
    headers: List[Tuple[int, re.Match]] = []
    ios_hits = android_hits = 0
    for i, line in enumerate(lines):
        m = WA_IOS_RE.match(line)
        if m:
            ios_hits += 1
            headers.append((i, m))
            continue
        m = WA_ANDROID_RE.match(line)
        if m:
            android_hits += 1
            headers.append((i, m))
    if not headers:
        raise ValueError(
            "Formato WhatsApp non riconosciuto: nessuna riga con timestamp valido "
            "(attesi '[DD/MM/YY, HH:MM:SS] Nome: testo' oppure 'DD/MM/YY, HH:MM - Nome: testo')."
        )
    order = _infer_date_order([m.group("date") for _, m in headers[:5000]])
    header_index = {i: m for i, m in headers}

    messages: List[RawMessage] = []
    current: Optional[Dict] = None

    def flush() -> None:
        if current is None:
            return
        text = EDITED_RE.sub("", current["text"]).strip("\n").strip()
        sender = current["sender"].strip()
        if len(text) < 250 and SYSTEM_STRONG_RE.search(text):
            return
        if current["lrm"] and len(text) < 200 and SYSTEM_RE.search(text) and classify_media(text) is None:
            return
        media_type = classify_media(text)
        messages.append(
            RawMessage(
                timestamp=current["ts"],
                sender=sender,
                text=text,
                is_media=media_type is not None,
                media_type=media_type,
            )
        )

    for i, line in enumerate(lines):
        m = header_index.get(i)
        if m is not None:
            ts = _parse_wa_datetime(m.group("date"), m.group("time"), order)
            rest = m.group("rest")
            sm = WA_SENDER_RE.match(rest)
            if ts is None:
                if current is not None:
                    current["text"] += "\n" + line
                continue
            flush()
            if sm is None:
                # Riga di sistema senza mittente (es. "Mario ha aggiunto Luigi")
                current = None
                continue
            raw_rest = raw_lines[i].split(": ", 1)
            lrm = len(raw_rest) == 2 and raw_rest[1].lstrip(" ").startswith("\u200e")
            current = {"ts": ts, "sender": sm.group("sender"), "text": sm.group("text") or "", "lrm": lrm}
        else:
            if current is not None:
                current["text"] += "\n" + line
    flush()
    return messages


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
_TG_MEDIA_MAP = {
    "sticker": "sticker",
    "voice_message": "audio",
    "audio_file": "audio",
    "video_message": "video",
    "video_file": "video",
    "animation": "gif",
}


def _flatten_tg_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out = []
        for part in value:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                out.append(str(part.get("text", "")))
        return "".join(out)
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return str(value)


def _select_telegram_chat(data: dict, chat_name: Optional[str] = None) -> dict:
    if "messages" in data:
        return data
    chats = []
    if isinstance(data.get("chats"), dict):
        chats.extend(data["chats"].get("list", []))
    if isinstance(data.get("left_chats"), dict):
        chats.extend(data["left_chats"].get("list", []))
    chats = [c for c in chats if isinstance(c, dict) and c.get("messages")]
    if not chats:
        raise ValueError("Il JSON Telegram non contiene chat con messaggi.")
    if chat_name:
        for c in chats:
            if str(c.get("name", "")).strip().lower() == chat_name.strip().lower():
                return c
    return max(chats, key=lambda c: len(c.get("messages", [])))


def parse_telegram_json(data: dict, chat_name: Optional[str] = None) -> List[RawMessage]:
    chat = _select_telegram_chat(data, chat_name)
    out: List[RawMessage] = []
    for m in chat.get("messages", []):
        if m.get("type") != "message":
            continue
        date_raw = m.get("date")
        if not date_raw:
            continue
        try:
            ts = datetime.fromisoformat(str(date_raw).replace("Z", ""))
        except ValueError:
            continue
        sender = m.get("from") or m.get("actor") or m.get("from_id") or "Utente eliminato"
        text = _flatten_tg_text(m.get("text")).strip()
        media_type = None
        if "photo" in m:
            media_type = "image"
        elif m.get("media_type"):
            media_type = _TG_MEDIA_MAP.get(m["media_type"], "media")
        elif "file" in m:
            media_type = "document"
        elif "poll" in m:
            media_type = "poll"
            if not text:
                text = "Sondaggio: " + str(m["poll"].get("question", ""))
        elif "location_information" in m:
            media_type = "location"
        elif "contact_information" in m:
            media_type = "contact"
        is_media = media_type is not None and not text
        if not text and media_type is None:
            continue
        if is_media and not text:
            if media_type == "sticker" and m.get("sticker_emoji"):
                text = f"[sticker {m['sticker_emoji']}]"
            else:
                text = f"[{media_type}]"
        out.append(RawMessage(timestamp=ts.replace(tzinfo=None), sender=str(sender).strip(), text=text,
                              is_media=is_media, media_type=media_type if is_media else None,
                              is_forwarded=bool(m.get("forwarded_from") or m.get("forwarded_from_id"))))
    return out


# ---------------------------------------------------------------------------
# Telegram — export HTML (Telegram Desktop, "Formato: HTML")
# ---------------------------------------------------------------------------
_MONTHS = {
    # it
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5, "giugno": 6, "luglio": 7,
    "agosto": 8, "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
    # en
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    # es / pt / fr / de
    "enero": 1, "febrero": 2, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
    "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    "janeiro": 1, "fevereiro": 2, "março": 3, "maio": 5, "junho": 6, "julho": 7, "setembro": 9,
    "outubro": 10, "dezembro": 12, "janvier": 1, "février": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "août": 8, "septembre": 9, "octobre": 10, "décembre": 12,
    "januar": 1, "februar": 2, "märz": 3, "juni": 6, "juli": 7, "oktober": 10, "dezember": 12,
}
_TG_HTML_NUMERIC_DATE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?")
_TG_HTML_TEXT_DATE = re.compile(r"(\d{1,2})\s+([^\W\d_]+)\s+(\d{4}),?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?", re.UNICODE)
_TG_HTML_TEXT_DATE_MDY = re.compile(r"([^\W\d_]+)\s+(\d{1,2}),?\s+(\d{4}),?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?", re.UNICODE)
_TG_HTML_FILE_RE = re.compile(r"(?:^|/)messages(\d*)\.html$", re.IGNORECASE)


def _parse_tg_html_date(title: str) -> Optional[datetime]:
    m = _TG_HTML_NUMERIC_DATE.search(title)
    if m:
        d, mo, y, h, mi, s = m.groups()
        try:
            return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s or 0))
        except ValueError:
            return None
    candidates = []
    m = _TG_HTML_TEXT_DATE.search(title)
    if m:
        d, mon, y, h, mi, s = m.groups()
        candidates.append((d, mon, y, h, mi, s))
    m = _TG_HTML_TEXT_DATE_MDY.search(title)  # "March 1, 2024 10:07"
    if m:
        mon, d, y, h, mi, s = m.groups()
        candidates.append((d, mon, y, h, mi, s))
    for d, mon, y, h, mi, s in candidates:
        month = _MONTHS.get(mon.lower())
        if month:
            try:
                return datetime(int(y), month, int(d), int(h), int(mi), int(s or 0))
            except ValueError:
                continue
    return None


def _tg_html_media(media_class: str, title: str) -> str:
    t = title.strip().lower()
    if t.startswith("sticker"):
        return "sticker"
    if t in ("gif", "animation", "animazione"):
        return "gif"
    return {
        "media_photo": "image",
        "media_video": "video",
        "media_voice_message": "audio",
        "media_audio_file": "audio",
        "media_file": "document",
        "media_contact": "contact",
        "media_location": "location",
        "media_live_location": "location",
        "media_venue": "location",
        "media_poll": "poll",
    }.get(media_class, "media")


_TG_BLOCK_SPLIT = re.compile(r'<div class="message (?=[^"]*")')
_TG_CLASS = re.compile(r'^([^"]*)"')
_TG_DATE_TITLE = re.compile(r'class="pull_right date details" title="([^"]*)"')
_TG_FROM = re.compile(r'<div class="from_name">(.*?)</div>', re.S)
_TG_DETAILS_SPAN = re.compile(r'<span class="[^"]*details[^"]*">.*?</span>', re.S)
_TG_TEXT = re.compile(r'<div class="text">(.*?)</div>', re.S)
_TG_MEDIA = re.compile(r'class="(?:media clearfix pull_left[^"]*?\b(media_\w+)[^"]*|(media_poll))"')
_TG_MEDIA_TITLE = re.compile(r'<div class="title bold">(.*?)</div>', re.S)
_TG_MEDIA_WRAP = re.compile(r'<a class="(photo_wrap|video_file_wrap|animated_wrap|sticker_wrap)')
_TG_POLL_Q = re.compile(r'<div class="question bold">(.*?)</div>', re.S)
_TG_BR = re.compile(r"<br\s*/?>", re.I)
_TG_TAG = re.compile(r"<[^>]+>")
_TG_WRAP_MEDIA = {"photo_wrap": "media_photo", "video_file_wrap": "media_video",
                  "animated_wrap": "media_gif", "sticker_wrap": "media_sticker"}


def _tg_html_text(fragment: str) -> str:
    text = html.unescape(_TG_TAG.sub("", _TG_BR.sub("\n", fragment)))
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def _parse_tg_html_page(page: str, state: Dict[str, Optional[str]], out: List[RawMessage]) -> None:
    """Scansione a blocchi di una pagina messagesN.html (un blocco = un <div class="message …">).

    - messaggi "joined": il mittente è ereditato dal messaggio precedente (anche tra pagine diverse);
    - inoltri: il from_name dentro "forwarded body" non sostituisce il mittente reale;
    - messaggi di servizio (separatori di data, chiamate, pin) ignorati, come nel parser JSON.
    """
    for block in _TG_BLOCK_SPLIT.split(page)[1:]:
        cm = _TG_CLASS.match(block)
        if not cm:
            continue
        classes = cm.group(1).split()
        if "default" not in classes:
            continue
        fwd = block.find('class="forwarded body"')
        own = block if fwd < 0 else block[:fwd]
        state["blocks"] = (state.get("blocks") or 0) + 1
        dm = _TG_DATE_TITLE.search(own)
        ts = _parse_tg_html_date(html.unescape(dm.group(1))) if dm else None
        if ts is None:
            state["bad_dates"] = (state.get("bad_dates") or 0) + 1
            if dm and not state.get("bad_date_sample"):
                state["bad_date_sample"] = dm.group(1)[:60]
        fm = _TG_FROM.search(own)
        if fm:
            sender = " ".join(html.unescape(_TG_TAG.sub("", _TG_DETAILS_SPAN.sub("", fm.group(1)))).split())
            if sender:
                state["last_sender"] = sender
        sender = state.get("last_sender")
        if ts is None or not sender:
            continue
        tm = _TG_TEXT.search(block)
        text = _tg_html_text(tm.group(1)) if tm else ""
        media_type = None
        mm = _TG_MEDIA.search(block)
        title = ""
        if mm:
            tt = _TG_MEDIA_TITLE.search(block, mm.end())
            title = " ".join(html.unescape(_TG_TAG.sub("", tt.group(1))).split()) if tt else ""
            media_type = _tg_html_media(mm.group(1) or mm.group(2), title)
            if media_type == "poll" and not text:
                pq = _TG_POLL_Q.search(block)
                text = "Sondaggio: " + (_tg_html_text(pq.group(1)) if pq else "")
        else:
            wm = _TG_MEDIA_WRAP.search(block)
            if wm:
                media_type = {"media_photo": "image", "media_video": "video",
                              "media_gif": "gif", "media_sticker": "sticker"}[_TG_WRAP_MEDIA[wm.group(1)]]
        if not text and media_type is None:
            continue
        is_media = media_type is not None and not text
        if is_media:
            if media_type == "sticker" and title.lower().startswith("sticker") and title[7:].strip():
                text = f"[sticker {title[7:].strip()}]"
            else:
                text = f"[{media_type}]"
        forwarded = fwd >= 0 or 'class="forwarded_from' in block
        out.append(RawMessage(timestamp=ts, sender=sender, text=text,
                              is_media=is_media, media_type=media_type if is_media else None,
                              is_forwarded=forwarded))


def _tg_html_sort_key(name: str) -> Tuple[int, str]:
    m = _TG_HTML_FILE_RE.search(name.replace("\\", "/"))
    if not m:
        return (10**9, name)
    return (int(m.group(1) or 1), name)


def is_telegram_html(name: str, data: bytes) -> bool:
    if not name.lower().endswith((".html", ".htm")):
        return False
    head = data[:20000].decode("utf-8", errors="ignore")
    return 'class="history"' in head or 'class="message default' in head or "page_header" in head


def parse_telegram_html_pages(pages: List[Tuple[str, bytes]]) -> List[RawMessage]:
    """Parsa le pagine HTML di un export Telegram, in ordine numerico (messages, messages2, …, messages270)."""
    msgs: List[RawMessage] = []
    state: Dict[str, object] = {"last_sender": None}
    for name, data in sorted(pages, key=lambda p: _tg_html_sort_key(p[0])):
        _parse_tg_html_page(data.decode("utf-8", errors="replace"), state, msgs)  # type: ignore[arg-type]
    blocks, bad = int(state.get("blocks") or 0), int(state.get("bad_dates") or 0)
    if blocks and bad / blocks > 0.05:
        raise ValueError(
            f"Formato data non riconosciuto in {bad:,} messaggi su {blocks:,} "
            f"(es. «{state.get('bad_date_sample')}»): riesporta la chat da Telegram Desktop in formato JSON."
        )
    # gli id dei file garantiscono l'ordine, ma un ordinamento stabile protegge da upload parziali/disordinati
    msgs.sort(key=lambda r: r.timestamp)
    return msgs


def parse_payload(kind: str, payload: object, telegram_chat_name: Optional[str] = None) -> Tuple[str, List[RawMessage]]:
    if kind == "telegram":
        return "telegram", parse_telegram_json(payload, telegram_chat_name)  # type: ignore[arg-type]
    if kind == "telegram_html":
        return "telegram", parse_telegram_html_pages(payload)  # type: ignore[arg-type]
    return "whatsapp", parse_whatsapp_text(payload)  # type: ignore[arg-type]


def parse_many(files: List[Tuple[str, bytes]], telegram_chat_name: Optional[str] = None) -> Tuple[str, List[RawMessage]]:
    """Più file caricati insieme: tutte le pagine HTML Telegram vengono unite in un'unica chat."""
    html_pages = [(n, d) for n, d in files if is_telegram_html(n, d)]
    if html_pages:
        return "telegram", parse_telegram_html_pages(html_pages)
    if len(files) == 1:
        return parse_bytes(files[0][0], files[0][1], telegram_chat_name)
    raise ValueError("Più file caricati: è supportato solo l'insieme di pagine messages*.html di un export Telegram HTML. "
                     "Per WhatsApp o Telegram JSON carica un solo file.")


def parse_bytes(filename: str, data: bytes, telegram_chat_name: Optional[str] = None) -> Tuple[str, List[RawMessage]]:
    kind, payload = read_input_bytes(filename, data)
    return parse_payload(kind, payload, telegram_chat_name)


def parse_file(path: str | Path, telegram_chat_name: Optional[str] = None) -> Tuple[str, List[RawMessage]]:
    path = Path(path)
    if path.is_dir() and (path / "result.json").exists():
        path = path / "result.json"
    kind, payload = read_input_file(path)
    return parse_payload(kind, payload, telegram_chat_name)


# ---------------------------------------------------------------------------
# Feature per messaggio
# ---------------------------------------------------------------------------
def _uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 4:
        return 0.0
    return round(sum(1 for c in letters if c.isupper()) / len(letters), 4)


def extract_features(raws: List[RawMessage]) -> pd.DataFrame:
    if not raws:
        raise ValueError("Nessun messaggio valido trovato nel file.")
    order = sorted(range(len(raws)), key=lambda i: (raws[i].timestamp, i))
    rows = []
    for i in order:
        r = raws[i]
        text = r.text or ""
        if r.is_media:
            emojis: List[str] = []
            char_count = word_count = q = ex = 0
            upper = 0.0
        else:
            emojis = [e["emoji"] for e in emoji.emoji_list(text)]
            plain = emoji.replace_emoji(text, replace="")
            char_count = len(text)
            word_count = len(WORD_RE.findall(plain))
            q = text.count("?")
            ex = text.count("!")
            upper = _uppercase_ratio(plain)
        minutes = r.timestamp.hour * 60 + r.timestamp.minute
        rows.append(
            {
                "timestamp": r.timestamp,
                "sender": r.sender,
                "text": text,
                "char_count": char_count,
                "word_count": word_count,
                "emoji_list": json.dumps(emojis, ensure_ascii=False),
                "emoji_count": len(emojis),
                "question_count": q,
                "exclamation_count": ex,
                "uppercase_ratio": upper,
                "is_night_message": int(config.NIGHT_START_MINUTES <= minutes < config.NIGHT_END_MINUTES),
                "is_media": int(bool(r.is_media)),
                "media_type": r.media_type,
                "is_forwarded": int(bool(getattr(r, "is_forwarded", False))),
            }
        )
    df = pd.DataFrame(rows)
    df["id"] = np.arange(1, len(df) + 1)
    ts = pd.to_datetime(df["timestamp"])
    gap = ts.diff().dt.total_seconds()
    sender_changed = df["sender"].ne(df["sender"].shift())
    df["gap_seconds"] = gap
    df["reply_time_seconds"] = gap.where(sender_changed & gap.notna())
    df.loc[df.index[0], "reply_time_seconds"] = np.nan
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["month"] = ts.dt.strftime("%Y-%m")
    df["hour"] = ts.dt.hour.astype(int)
    df["weekday"] = ts.dt.day_of_week.astype(int)
    df["timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")

    # Feature ausiliarie per gli score di sessione (non persistite in messages)
    lower = df["text"].fillna("").str.lower()
    # testo inoltrato da terzi (articoli, catene): non è linguaggio del mittente, escluso dagli score
    text_mask = df["is_media"].eq(0) & df["is_forwarded"].eq(0)
    df["_conflict_hits"] = np.where(text_mask, lower.str.count(CONFLICT_RE), 0)
    df["_accusation_hits"] = np.where(text_mask, lower.str.count(ACCUSATION_RE), 0)
    df["_negation_hits"] = np.where(text_mask, lower.str.count(NEGATION_RE), 0)
    df["_affection_hits"] = np.where(text_mask, lower.str.count(AFFECTION_RE), 0)
    df["_multi_punct"] = np.where(text_mask, df["text"].fillna("").str.count(MULTI_PUNCT_RE), 0)
    letters = df["text"].fillna("").str.count(r"[^\W\d_]")
    df["_shouting"] = (
        text_mask & (df["uppercase_ratio"] >= config.SHOUTING_UPPERCASE_RATIO) & (letters >= config.SHOUTING_MIN_LETTERS)
    ).astype(int)
    prev_char = df["char_count"].shift()
    is_cold_media = df["media_type"].isin(config.COLD_MEDIA_TYPES)
    short_reply = (text_mask & (df["char_count"] < config.COLD_REPLY_MAX_CHARS)) | is_cold_media
    df["_cold_reply"] = (sender_changed & (prev_char > config.COLD_REPLY_TRIGGER_CHARS) & short_reply).astype(int)
    df["_wall"] = (text_mask & (df["char_count"] > config.LONG_MESSAGE_CHARS)).astype(int)
    positive = set(config.POSITIVE_LAUGH_EMOJIS) | {e.replace("\ufe0f", "") for e in config.POSITIVE_LAUGH_EMOJIS}
    has_pos_emoji = df["emoji_list"].map(lambda s: s != "[]" and any(e in positive for e in json.loads(s)))
    df["_laughter"] = (text_mask & (lower.str.contains(LAUGHTER_RE) | has_pos_emoji)).astype(int)
    return df


# ---------------------------------------------------------------------------
# Smart Session Chunking + Heuristic Scores
# ---------------------------------------------------------------------------
def _sat(x: float, k: float) -> float:
    """Funzione di saturazione morbida 0..1."""
    if x <= 0 or not np.isfinite(x):
        return 0.0
    return float(1.0 - np.exp(-x / k))


def _format_transcript_line(ts: str, sender: str, text: str, is_media: int, media_type: Optional[str],
                            is_forwarded: int = 0) -> str:
    if is_media:
        body = f"[{media_type or 'media'}]"
    else:
        body = text.replace("\r", "").replace("\n", " / ")
        if len(body) > config.TRANSCRIPT_MAX_CHARS_PER_MESSAGE:
            body = body[: config.TRANSCRIPT_MAX_CHARS_PER_MESSAGE] + "…"
    if is_forwarded:
        body = "[inoltrato] " + body
    return f"[{ts[:16]}] {sender}: {body}"


def chunk_sessions(df: pd.DataFrame) -> List[Dict]:
    """Ritorna la lista delle sessioni come dict con le posizioni (indici di riga) dei messaggi inclusi."""
    gaps = df["gap_seconds"].to_numpy()
    n = len(df)
    sessions: List[Dict] = []
    members: List[int] = []
    core_start = 0
    overlap = 0
    continuation = False
    for i in range(n):
        if not members:
            members = [i]
            core_start = i
            continue
        gap = gaps[i]
        if np.isfinite(gap) and gap > config.SESSION_GAP_SECONDS:
            sessions.append({"members": members, "overlap": overlap, "continuation": continuation, "core_start": core_start})
            members, overlap, continuation, core_start = [i], 0, False, i
        elif len(members) >= config.SESSION_MAX_MESSAGES:
            sessions.append({"members": members, "overlap": overlap, "continuation": continuation, "core_start": core_start})
            ov = members[-config.SESSION_OVERLAP_MESSAGES:] if config.SESSION_OVERLAP_MESSAGES > 0 else []
            members, overlap, continuation, core_start = ov + [i], len(ov), True, i
        else:
            members.append(i)
    if members:
        sessions.append({"members": members, "overlap": overlap, "continuation": continuation, "core_start": core_start})
    return sessions


def score_session(sub: pd.DataFrame) -> Dict[str, float]:
    n = len(sub)
    text_sub = sub[sub["is_media"] == 0]
    text_n = max(len(text_sub), 1)
    t0 = pd.Timestamp(sub["timestamp"].iloc[0])
    t1 = pd.Timestamp(sub["timestamp"].iloc[-1])
    duration_min = max((t1 - t0).total_seconds() / 60.0, 0.0)
    density = n / max(duration_min, 1.0)

    chars = text_sub["char_count"].to_numpy(dtype=float)
    cv = float(chars.std() / chars.mean()) if len(chars) > 1 and chars.mean() > 0 else 0.0
    walls = int(sub["_wall"].sum())
    cold = int(sub["_cold_reply"].iloc[1:].sum()) if n > 1 else 0
    punct = int(sub["_multi_punct"].sum())
    shout = int(sub["_shouting"].sum())
    conflict_hits = int(sub["_conflict_hits"].sum())
    acc_hits = int(sub["_accusation_hits"].sum())
    neg_hits = int(sub["_negation_hits"].sum())
    aff_hits = int(sub["_affection_hits"].sum())
    laughs = int(sub["_laughter"].sum())
    excl = float(sub["exclamation_count"].sum())
    quest = float(sub["question_count"].sum())
    emojis = float(sub["emoji_count"].sum())

    wall_ratio = walls / text_n
    cold_ratio = cold / text_n
    punct_rate = punct / text_n
    upper_rate = shout / text_n
    conflict_rate = conflict_hits / text_n
    acc_rate = acc_hits / text_n
    neg_rate = neg_hits / text_n
    aff_rate = aff_hits / text_n
    size_factor = min(1.0, (text_n / 8.0) ** 0.5)

    conflict_raw = (
        0.28 * _sat(conflict_rate, 0.15)
        + 0.17 * _sat(acc_rate, 0.08)
        + 0.08 * _sat(neg_rate, 0.6)
        + 0.13 * _sat(cold_ratio, 0.08)
        + 0.12 * _sat(wall_ratio, 0.12)
        + 0.08 * _sat(punct_rate, 0.2)
        + 0.06 * _sat(upper_rate, 0.1)
        + 0.04 * _sat(max(cv - 0.8, 0.0), 0.8)
        + 0.04 * _sat(density, 2.0)
    )
    damp = 1.0 - 0.25 * _sat(aff_rate, 0.2)
    laughter_rate = laughs / text_n
    # Raffiche di risate/emoji positive (alta densità, molti "!!", maiuscole) non sono litigi.
    laughter_damped = laughter_rate >= config.LAUGHTER_DAMPING_THRESHOLD
    if laughter_damped:
        damp *= config.LAUGHTER_DAMPING_FACTOR
    conflict = float(np.clip(100.0 * conflict_raw * damp * size_factor, 0, 100))

    intensity_raw = (
        0.17 * _sat(excl / text_n, 0.4)
        + 0.14 * _sat(punct_rate, 0.2)
        + 0.11 * _sat(upper_rate, 0.1)
        + 0.14 * _sat(emojis / text_n, 0.8)
        + 0.12 * _sat(wall_ratio, 0.12)
        + 0.10 * _sat(density, 2.0)
        + 0.14 * _sat(aff_rate + conflict_rate + acc_rate, 0.3)
        + 0.08 * _sat(quest / text_n, 0.6)
    )
    intensity = float(np.clip(100.0 * intensity_raw * size_factor, 0, 100))
    return {
        "duration_minutes": round(duration_min, 2),
        "messages_per_minute": round(density, 3),
        "wall_of_text_count": walls,
        "cold_reply_count": cold,
        "multi_punct_count": punct,
        "shouting_count": shout,
        "conflict_lexicon_hits": conflict_hits,
        "accusation_lexicon_hits": acc_hits,
        "affection_lexicon_hits": aff_hits,
        "laughter_ratio": round(laughter_rate, 3),
        "conflict_score": round(conflict, 2),
        "emotional_intensity_score": round(intensity, 2),
    }


def build_sessions(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    chunks = chunk_sessions(df)
    session_ids = np.zeros(len(df), dtype=np.int64)
    records: List[Dict] = []
    ts_arr = df["timestamp"].to_numpy()
    sender_arr = df["sender"].to_numpy()
    text_arr = df["text"].fillna("").to_numpy()
    media_arr = df["is_media"].to_numpy()
    mtype_arr = df["media_type"].to_numpy()
    fwd_arr = df["is_forwarded"].to_numpy() if "is_forwarded" in df else np.zeros(len(df), dtype=int)
    ids = df["id"].to_numpy()
    for sid, ch in enumerate(chunks, start=1):
        members = ch["members"]
        core = [m for m in members if m >= ch["core_start"]]
        session_ids[core] = sid
        sub = df.iloc[members]
        participants: List[str] = []
        for s in sub["sender"]:
            if s not in participants:
                participants.append(s)
        transcript = "\n".join(
            _format_transcript_line(ts_arr[m], sender_arr[m], text_arr[m], media_arr[m], mtype_arr[m], fwd_arr[m])
            for m in members
        )
        scores = score_session(sub)
        records.append(
            {
                "session_id": sid,
                "start_time": ts_arr[members[0]],
                "end_time": ts_arr[members[-1]],
                "first_message_id": int(ids[members[0]]),
                "last_message_id": int(ids[members[-1]]),
                "message_count": len(members),
                "overlap_count": int(ch["overlap"]),
                "is_continuation": int(ch["continuation"]),
                "participant_count": len(participants),
                "participants": json.dumps(participants, ensure_ascii=False),
                "initiator": sender_arr[ch["core_start"]],
                "transcript": transcript,
                **scores,
            }
        )
    df = df.copy()
    df["session_id"] = session_ids
    return df, records


# ---------------------------------------------------------------------------
# Persistenza
# ---------------------------------------------------------------------------
MESSAGE_COLUMNS = [
    "id", "session_id", "timestamp", "date", "month", "hour", "weekday", "sender", "text", "char_count",
    "word_count", "emoji_list", "emoji_count", "question_count", "exclamation_count", "uppercase_ratio",
    "is_night_message", "is_media", "media_type", "reply_time_seconds", "gap_seconds", "is_forwarded",
]
SESSION_COLUMNS = [
    "session_id", "start_time", "end_time", "first_message_id", "last_message_id", "message_count", "overlap_count",
    "is_continuation", "participant_count", "participants", "initiator", "duration_minutes", "messages_per_minute",
    "wall_of_text_count", "cold_reply_count", "multi_punct_count", "shouting_count", "conflict_lexicon_hits",
    "accusation_lexicon_hits", "affection_lexicon_hits", "laughter_ratio", "conflict_score", "emotional_intensity_score", "transcript",
]


def _py(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return int(v)
    return v


def write_to_database(db_path: str | Path, df: pd.DataFrame, sessions: List[Dict], meta: Dict[str, object]) -> None:
    database.init_db(db_path, reset=True)
    with database.connect(db_path) as conn:
        conn.executemany(
            f"INSERT INTO sessions ({', '.join(SESSION_COLUMNS)}) VALUES ({', '.join('?' * len(SESSION_COLUMNS))})",
            ([_py(s[c]) for c in SESSION_COLUMNS] for s in sessions),
        )
        msg_rows = df[MESSAGE_COLUMNS].itertuples(index=False, name=None)
        conn.executemany(
            f"INSERT INTO messages ({', '.join(MESSAGE_COLUMNS)}) VALUES ({', '.join('?' * len(MESSAGE_COLUMNS))})",
            ([_py(v) for v in row] for row in msg_rows),
        )
        for k, v in meta.items():
            database.set_meta(conn, k, v)


# ---------------------------------------------------------------------------
# Orchestrazione
# ---------------------------------------------------------------------------
def run_ingestion(
    file_path: Optional[str | Path],
    db_path: str | Path,
    progress_callback: Optional[ProgressCallback] = None,
    build_embeddings: bool = True,
    embedding_model: Optional[str] = None,
    telegram_chat_name: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    source_name: Optional[str] = None,
    files: Optional[List[Tuple[str, bytes]]] = None,
) -> Dict[str, object]:
    """Esegue l'intera pipeline. progress_callback(stage, fraction 0..1, detail).

    Passando file_bytes + source_name (o files = [(nome, bytes), …] per le pagine HTML Telegram)
    l'export viene processato solo in memoria (nessun file raw su disco).
    """

    def report(stage: str, frac: float, detail: str = "") -> None:
        if progress_callback:
            progress_callback(stage, float(min(max(frac, 0.0), 1.0)), detail)

    t_start = time.time()
    if file_bytes is None and file_path is None and not files:
        raise ValueError("Specificare file_path, file_bytes oppure files.")
    name = source_name or (Path(file_path).name if file_path is not None else
                           (files[0][0] if files else "chat.txt"))
    report("Parsing", 0.02, _t("Lettura in memoria di {name}…", name=name))
    if files:
        source, raws = parse_many(files, telegram_chat_name)
        files = None  # rilascia i riferimenti ai buffer raw
    elif file_bytes is not None:
        source, raws = parse_bytes(name, file_bytes, telegram_chat_name)
        file_bytes = None  # rilascia il riferimento al buffer raw
    else:
        source, raws = parse_file(file_path, telegram_chat_name)
    report("Parsing", 0.12, _t("{n} messaggi validi estratti ({src}).", n=f"{len(raws):,}", src=source))

    report("Feature", 0.15, _t("Estrazione feature per messaggio (emoji, punteggiatura, latenze)…"))
    df = extract_features(raws)

    report(_t("Calcolo Euristico"), 0.25, "Smart Session Chunking + Conflict/Intensity Score…")
    df, sessions = build_sessions(df)
    report(_t("Calcolo Euristico"), 0.35, _t("{n} sessioni conversazionali create.", n=f"{len(sessions):,}"))

    report("SQL", 0.40, _t("Scrittura su SQLite…"))
    meta = {
        "source_type": source,
        "source_file": name,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "message_count": int(len(df)),
        "session_count": int(len(sessions)),
        "participants": json.dumps(df["sender"].value_counts().index.tolist(), ensure_ascii=False),
    }
    write_to_database(db_path, df, sessions, meta)
    report("SQL", 0.48, _t("Tabelle messages/sessions scritte."))

    from chatautopsy import rag_engine  # import locale: evita di caricare torch se non serve

    rag = rag_engine.HybridRAG(db_path, model_name=embedding_model)
    report("FTS5", 0.50, _t("Indicizzazione full-text BM25 (FTS5)…"))
    fts_stats = rag.index_fts()
    report("FTS5", 0.55, _t("FTS5 pronto: {m} messaggi, {s} sessioni.", m=f"{fts_stats['messages']:,}", s=f"{fts_stats['sessions']:,}"))

    embedded = 0
    embed_error = None
    if build_embeddings:
        def emb_progress(done: int, total: int, device: str) -> None:
            frac = 0.56 + 0.43 * (done / max(total, 1))
            report("Vector Embeddings", frac, _t("{d}/{n} sessioni vettorizzate su {dev}", d=f"{done:,}", n=f"{total:,}", dev=device.upper()))

        try:
            report("Vector Embeddings", 0.56, _t("Caricamento modello di embedding locale…"))
            embedded = rag.build_vector_index(progress_callback=emb_progress)
        except Exception as exc:  # embedding opzionale: l'app funziona anche solo con BM25
            embed_error = f"{type(exc).__name__}: {exc}"
            report("Vector Embeddings", 0.99, _t("Embedding non riuscito ({e}). Ricerca solo BM25.", e=embed_error))
    elapsed = time.time() - t_start
    report(_t("Completato"), 1.0, _t("Ingestione completata in {s}s.", s=f"{elapsed:,.1f}"))
    return {
        "source_type": source,
        "messages": int(len(df)),
        "sessions": int(len(sessions)),
        "participants": df["sender"].value_counts().to_dict(),
        "embedded_sessions": int(embedded),
        "embedding_error": embed_error,
        "elapsed_seconds": round(elapsed, 2),
        "db_path": str(db_path),
    }
