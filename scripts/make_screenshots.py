"""
Regenerate the README cover and screenshots from the DEMO chat (made up) — never from real data.
Rigenera copertina e schermate del README dalla chat DIMOSTRATIVA (inventata), mai da dati reali.

    pip install -e ".[docs]"                      # Playwright + Pillow (uses the installed Chrome, or: playwright install chromium)
    python scripts/make_screenshots.py            # both languages -> docs/images/en, docs/images/it
    python scripts/make_screenshots.py --lang en  # one language only

Steps: demo chat in a temp folder → headless Streamlit dashboard in the chosen language → Playwright captures every
section (tall viewport, no scrolling) → crops by section heading → cover composed in HTML.
"""
from __future__ import annotations

import argparse
import base64
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCALE = 1.5          # pixel density (sharp on retina screens)
WIDTH = 1440
TALL = 9000          # tall viewport: the whole section renders without scrolling

DEMO = {"en": ("demo_emma_jake", "SHOWCASE_EN"), "it": ("demo_giulia_marco", "SHOWCASE")}

# (sezione, [(titolo iniziale, titolo finale o None, nome file)]) — chiavi italiane, tradotte con locales_en per --lang en
SECTIONS = {
    "🧾 Pagella": [("🧾 Pagella della relazione", "#### ⚖️ Tiro alla fune: chi pesa di più su ogni aspetto", "report_card"),
                  ("#### ⚖️ Tiro alla fune: chi pesa di più su ogni aspetto", "#### 📋 Tutti gli indicatori", "tug_of_war"),
                  ("📅 Pagella anno per anno", None, "year_by_year")],
    "📊 Volumi & Potere": [("📈 Volumi nel tempo", "🚪 Chi apre e chi chiude", "volume"),
                          ("🚪 Chi apre e chi chiude", "🔁 Inseguimento (double / triple texting)", "initiative"),
                          ("🖼️ Come si comunica: testo, vocali, foto, sticker", "💘 Love bombing vs svalutazione", "media_mix")],
    "⏱️ Reattività": [("⏱️ Tempi di risposta", "📉 Fading detector", "responsiveness")],
    "🚩 Conflitti & Red Flags": [("🌡️ Termometro della tensione", "🌦️ Clima: affetto vs conflitto", "tension"),
                                ("🌦️ Clima: affetto vs conflitto", "🧩 Pronomi: \"noi\" contro \"io/tu\"", "climate"),
                                ("🧱 Stonewalling (muro di gomma)", "⚖️ Scuse, accuse e riappacificazione", "stonewalling")],
    "🗣️ Lessico & Affetto": [("🧬 Parole identitarie", "🔎 Cerca parole nella storia", "signature_words"),
                             ("💞 Parole d'amore e soprannomi", "😀 Impronta emoji", "pet_names"),
                             ("😀 Impronta emoji", None, "emoji")],
    "🕰️ Ritmi & Abitudini": [("📅 Calendario della chat", "🌙 Cronotipo", "calendar"),
                             ("🌙 Cronotipo", "Mappa 7×24 per", "chronotype")],
}

COVER = {
    "en": ("Forensic, statistical and psychometric autopsy of your WhatsApp and Telegram chats.",
           "100% on your computer.", ["🔒 Zero cloud", "📊 50+ indicators", "🤖 Local AI · MLX · Ollama",
                                      "⚡ 1M+ messages", "🌐 English · Italiano"]),
    "it": ("Autopsia forense, statistica e psicometrica delle tue chat WhatsApp e Telegram.",
           "100% sul tuo computer.", ["🔒 Zero cloud", "📊 50+ indicatori", "🤖 AI locale · MLX · Ollama",
                                      "⚡ 1M+ messaggi", "🌐 English · Italiano"]),
}


def tr(text: str, lang: str) -> str:
    if lang == "en":
        sys.path.insert(0, str(ROOT))
        from chatautopsy.locales_en import EN

        text = EN.get(text, text)
    return text.lstrip("#").strip()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_demo(data_dir: Path, lang: str) -> Path:
    name, params = DEMO[lang]
    env = dict(os.environ, CHATAUTOPSY_DATA_DIR=str(data_dir), HF_HUB_OFFLINE="1", CHATAUTOPSY_LANG=lang)
    code = ("from chatautopsy import demo, database\nfrom chatautopsy.ingestion import run_ingestion\n"
            f"db=database.db_path_for_name('{name}')\n"
            f"run_ingestion(None, db, file_bytes=demo.couple_chat(**demo.{params}).encode(), "
            f"source_name='{name}.txt', build_embeddings=False)\nprint(db)")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return Path(out.stdout.strip().splitlines()[-1])


def start_app(data_dir: Path, db: Path, port: int, lang: str) -> subprocess.Popen:
    env = dict(os.environ, CHATAUTOPSY_DATA_DIR=str(data_dir), CHATAUTOPSY_OPEN_DB=str(db), HF_HUB_OFFLINE="1",
               CHATAUTOPSY_LANG=lang)
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "chatautopsy" / "app.py"), "--server.port", str(port),
           "--server.headless", "true", "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false",
           "--theme.base", "dark",
           "--theme.primaryColor", "#ff4b6e", "--theme.backgroundColor", "#0e1117",
           "--theme.secondaryBackgroundColor", "#171b26", "--theme.textColor", "#e6e6eb"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("Dashboard did not start")


HIDE_CHROME = """
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
header[data-testid="stHeader"] { display: none !important; }
"""


def wait_idle(page, timeout: float = 600) -> None:
    t0 = time.time()
    time.sleep(1.5)
    while time.time() - t0 < timeout:
        running = page.locator('[data-testid="stStatusWidget"]').count() > 0 or page.locator('[data-testid="stSpinner"]').count() > 0
        if not running:
            time.sleep(2.5)  # Plotly: wait for the first draw
            if page.locator('[data-testid="stStatusWidget"]').count() == 0:
                return
        time.sleep(0.8)


def headings(page):
    out = []
    for h in page.locator('[data-testid="stHeading"], h3, h4, h5, label').all():
        try:
            box = h.bounding_box()
            txt = h.inner_text().strip()
        except Exception:
            continue
        if box and txt:
            out.append((txt, box["y"]))
    out.sort(key=lambda x: x[1])
    return out


def crop(page, full_png: Path, start: str, end, out: Path, max_h: int = 1500, pad_top: int = 14) -> None:
    from PIL import Image

    hs = headings(page)
    y0 = next((y for txt, y in hs if start.lower() in txt.lower()), None)
    if y0 is None:
        raise RuntimeError(f"heading not found: {start!r} (found: {[x for x, _ in hs][:30]})")
    y1 = next((y for txt, y in hs if end and end.lower() in txt.lower() and y > y0), None) if end else None
    main = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
    img = Image.open(full_png)
    x0, x1 = main["x"] + 8, main["x"] + main["width"] - 8
    top = max(y0 - pad_top, 0)
    bottom = min((y1 - 10) if y1 else top + max_h, top + max_h)
    img.crop(tuple(int(v * SCALE) for v in (x0, top, x1, bottom))).save(out, optimize=True)


def launch(p):
    try:
        return p.chromium.launch(channel="chrome", headless=True)
    except Exception:
        return p.chromium.launch(headless=True)


def shoot(lang: str) -> None:
    from playwright.sync_api import sync_playwright

    out_dir = ROOT / "docs" / "images" / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="chatautopsy-shots-"))
    data_dir = tmp / "data"
    data_dir.mkdir()
    print(f"[{lang}] building the demo chat…")
    db = build_demo(data_dir, lang)
    port = free_port()
    app = start_app(data_dir, db, port, lang)
    try:
        with sync_playwright() as p:
            browser = launch(p)
            page = browser.new_page(viewport={"width": WIDTH, "height": 900}, device_scale_factor=SCALE)
            page.goto(f"http://127.0.0.1:{port}")
            page.wait_for_selector('[data-testid="stMainBlockContainer"]', timeout=300_000)
            page.add_style_tag(content=HIDE_CHROME)
            wait_idle(page)
            page.screenshot(path=str(out_dir / "overview.png"))
            page.screenshot(path=str(out_dir / "sidebar.png"), clip=page.locator('[data-testid="stSidebar"]').bounding_box())
            print("  ✓ overview.png, sidebar.png")
            page.set_viewport_size({"width": WIDTH, "height": TALL})
            for section, crops in SECTIONS.items():
                label = tr(section, lang).split(" ", 1)[1]
                page.locator("button", has_text=label).first.click()
                wait_idle(page)
                full = tmp / "full.png"
                page.screenshot(path=str(full), full_page=True)
                for start, end, name in crops:
                    try:
                        crop(page, full, tr(start, lang), tr(end, lang) if end else None, out_dir / f"{name}.png")
                        print(f"  ✓ {name}.png")
                    except Exception as exc:
                        print(f"  ✗ {name}: {exc}")
            browser.close()
    finally:
        app.terminate()
    make_cover(lang)


def make_cover(lang: str) -> None:
    from playwright.sync_api import sync_playwright

    out_dir = ROOT / "docs" / "images" / lang
    tagline, strong, chips = COVER[lang]

    def b64(name: str) -> str:
        return base64.b64encode((out_dir / name).read_bytes()).decode()

    chips_html = "".join(f'<div class="chip">{c}</div>' for c in chips)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
    * {{ box-sizing: border-box; margin: 0; }}
    body {{ width: 1280px; height: 640px; overflow: hidden; font-family: -apple-system, 'SF Pro Display', 'Inter', sans-serif;
           background: radial-gradient(900px 500px at 15% 10%, #26304a 0%, #0e1117 60%); color: #eef0f6; }}
    .wrap {{ display: flex; height: 100%; padding: 56px 0 56px 64px; gap: 40px; }}
    .left {{ width: 520px; display: flex; flex-direction: column; justify-content: center; }}
    h1 {{ font-size: 64px; font-weight: 800; letter-spacing: -1px; }}
    h1 span {{ background: linear-gradient(90deg, #ff4b6e, #ff8a65); -webkit-background-clip: text; color: transparent; }}
    p.tag {{ font-size: 23px; line-height: 1.4; color: #c7cbd8; margin: 18px 0 26px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .chip {{ border: 1px solid #39425c; background: #171d2b; border-radius: 999px; padding: 8px 14px; font-size: 16px; color: #dfe3ee; }}
    .right {{ flex: 1; position: relative; }}
    .shot {{ position: absolute; border-radius: 14px; box-shadow: 0 30px 80px rgba(0,0,0,.55); border: 1px solid #2b3347; }}
    .s1 {{ width: 700px; top: 10px; left: 40px; }}
    .s2 {{ width: 520px; top: 300px; left: 0px; }}
    </style></head><body><div class="wrap"><div class="left">
      <h1>🧬 Chat<span>Autopsy</span></h1>
      <p class="tag">{tagline}<br><b>{strong}</b></p>
      <div class="chips">{chips_html}</div></div>
      <div class="right"><img class="shot s1" src="data:image/png;base64,{b64('report_card.png')}">
      <img class="shot s2" src="data:image/png;base64,{b64('climate.png')}"></div></div></body></html>"""
    tmp = Path(tempfile.mkdtemp()) / "cover.html"
    tmp.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        b = launch(p)
        pg = b.new_page(viewport={"width": 1280, "height": 640}, device_scale_factor=2)
        pg.goto(tmp.as_uri())
        pg.wait_for_timeout(500)
        pg.screenshot(path=str(out_dir / "cover.png"))
        b.close()
    print("  ✓ cover.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "it", "all"], default="all")
    ap.add_argument("--cover-only", action="store_true")
    a = ap.parse_args()
    for lg in (["en", "it"] if a.lang == "all" else [a.lang]):
        make_cover(lg) if a.cover_only else shoot(lg)
