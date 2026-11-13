"""Parser universale: WhatsApp (iOS/Android/USA), Telegram JSON e HTML, cartelle e archivi."""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from chatautopsy import demo, ingestion


def test_whatsapp_ios_android_same_volume():
    _, ios = ingestion.parse_bytes("chat.txt", demo.couple_chat(days=60).encode())
    _, android = ingestion.parse_bytes("chat.txt", demo.couple_chat(days=60, android=True).encode())
    assert len(ios) > 500
    assert len(ios) == len(android)  # stessa conversazione; l'avviso di cifratura iOS è filtrato
    assert sum(m.is_media for m in ios) == sum(m.is_media for m in android)  # "<Media omessi>" riconosciuto
    assert {m.sender for m in ios} == set(demo.COUPLE)
    assert any(m.is_media for m in ios) and any(m.is_media for m in android)


def test_whatsapp_us_12h_multiline_and_system_event():
    txt = ("[1/15/23, 9:05:01 PM] Tom: hi\n[1/15/23, 9:06:10 PM] Ann: x\nriga2\n"
           "[1/15/23, 9:07:00 PM] Tom added Luca")
    msgs = ingestion.parse_whatsapp_text(txt)
    assert len(msgs) == 2
    assert msgs[0].timestamp.hour == 21
    assert msgs[1].text == "x\nriga2"


def test_telegram_json_full_export_picks_largest_chat():
    small = {"name": "A", "type": "personal_chat", "messages": [
        {"id": 1, "type": "message", "date": "2024-01-01T10:00:00", "from": "A", "text": "ciao"}]}
    big = {"name": "B", "type": "personal_chat", "messages": [
        {"id": i, "type": "message", "date": f"2024-01-01T10:{i:02d}:00", "from": "B" if i % 2 else "C",
         "text": [{"type": "plain", "text": "ok "}, {"type": "bold", "text": f"n{i}"}]} for i in range(5)]
        + [{"id": 99, "type": "service", "date": "2024-01-01T11:00:00", "actor": "B", "action": "phone_call"}]}
    data = {"chats": {"list": [small, big]}}
    kind, msgs = ingestion.parse_bytes("result.json", json.dumps(data).encode())
    assert kind == "telegram" and len(msgs) == 5
    assert msgs[1].text == "ok n1"


def _edge_case_page() -> bytes:
    return '''<div class="page_header"><div class="content"><div class="name bold">Test</div></div></div><div class="history">
<div class="message service"><div class="body details">1 marzo 2024</div></div>
<div class="message default clearfix" id="message1"><div class="body"><div class="pull_right date details" title="01.03.2024 10:00:00 UTC+01:00">10:00</div>
<div class="from_name">Ugo <span class="details">via @bot</span></div><div class="text">ciao &amp; <strong>bella</strong><br>seconda riga</div></div></div>
<div class="message default clearfix joined" id="message2"><div class="body"><div class="pull_right date details" title="01.03.2024 10:00:30 UTC+01:00">10:00</div>
<div class="forwarded body"><div class="from_name">Pippo<span class="date details"> 01.01.2020</span></div><div class="text">testo inoltrato</div></div></div></div>
<div class="message default clearfix" id="message3"><div class="body"><div class="pull_right date details" title="1 marzo 2024, 10:05:00">10:05</div>
<div class="from_name">Ivo</div><div class="reply_to details">In reply to <a href="#go_to_message1">this message</a></div>
<div class="media_wrap clearfix"><div class="media clearfix pull_left media_photo"><div class="fill pull_left"></div><div class="body"><div class="title bold">Sticker 😂</div></div></div></div></div></div>
<div class="message default clearfix joined" id="message4"><div class="body"><div class="pull_right date details" title="1 marzo 2024, 10:06:00">10:06</div>
<div class="media_wrap clearfix"><div class="media clearfix pull_left block_link media_location"><div class="title bold">Location</div></div></div></div></div>
<div class="message default clearfix" id="message5"><div class="body"><div class="pull_right date details" title="March 1, 2024 10:07">x</div>
<div class="from_name">Ugo</div><div class="forwarded_from details">Forwarded from Zia</div><div class="text">notizia lunga</div></div></div>
<div class="message service"><div class="body details">Chiamata persa</div></div>
</div>'''.encode()


def test_telegram_html_edge_cases():
    kind, msgs = ingestion.parse_bytes("messages.html", _edge_case_page())
    got = [(m.timestamp.strftime("%H:%M:%S"), m.sender, m.text, m.media_type, m.is_forwarded) for m in msgs]
    assert got == [
        ("10:00:00", "Ugo", "ciao & bella\nseconda riga", None, False),
        ("10:00:30", "Ugo", "testo inoltrato", None, True),
        ("10:05:00", "Ivo", "[sticker 😂]", "sticker", False),
        ("10:06:00", "Ivo", "[location]", "location", False),
        ("10:07:00", "Ugo", "notizia lunga", None, True),
    ]


def test_telegram_html_multi_page_order_and_inputs(tmp_path):
    p1 = demo.telegram_html_page([("01.03.2024 10:00:00", "A", "uno"), ("01.03.2024 10:01:00", "B", "due")])
    p2 = demo.telegram_html_page([("01.03.2024 10:02:00", "B", "tre"), ("01.03.2024 10:03:00", "A", "quattro")])
    pages = [("messages2.html", p2.encode()), ("messages.html", p1.encode())]
    _, msgs = ingestion.parse_many(pages)
    assert [m.text for m in msgs] == ["uno", "due", "tre", "quattro"]
    # cartella
    folder = tmp_path / "ChatExport_test"
    folder.mkdir()
    for n, d in pages:
        (folder / n).write_bytes(d)
    _, from_dir = ingestion.parse_file(folder)
    assert len(from_dir) == 4
    # zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in pages:
            z.writestr(f"ChatExport_test/{n}", d)
        z.writestr("ChatExport_test/css/style.css", "body{}")
    kind, from_zip = ingestion.parse_bytes("export.zip", buf.getvalue())
    assert kind == "telegram" and len(from_zip) == 4


def test_whatsapp_unzipped_folder(tmp_path):
    (tmp_path / "_chat.txt").write_text(demo.couple_chat(days=5), encoding="utf-8")
    kind, msgs = ingestion.parse_file(tmp_path)
    assert kind == "whatsapp" and len(msgs) > 10


def test_unknown_html_and_unreadable_dates_are_rejected():
    with pytest.raises(ValueError):
        ingestion.parse_bytes("pagina.html", b"<html><body>ciao</body></html>")
    page = (b'<div class="history"><div class="message default clearfix" id="m1"><div class="body">'
            b'<div class="pull_right date details" title="2024/03/01 10:00">x</div><div class="from_name">A</div>'
            b'<div class="text">x</div></div></div></div>')
    with pytest.raises(ValueError, match="Formato data"):
        ingestion.parse_bytes("messages.html", page)


@pytest.mark.parametrize("title,expected", [
    ("18 settembre 2018, 18:43:06", "2018-09-18 18:43:06"),
    ("01.03.2024 10:00:00 UTC+01:00", "2024-03-01 10:00:00"),
    ("March 1, 2024 10:07", "2024-03-01 10:07:00"),
    ("5 août 2021, 09:01:02", "2021-08-05 09:01:02"),
    ("3 Dezember 2020, 23:59:59", "2020-12-03 23:59:59"),
])
def test_telegram_html_dates(title, expected):
    assert str(ingestion._parse_tg_html_date(title)) == expected
