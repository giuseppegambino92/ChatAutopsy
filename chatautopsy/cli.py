"""
ChatAutopsy — interfaccia a riga di comando.

    chatautopsy                       apre la dashboard
    chatautopsy analyze PERCORSO      file/cartella/zip di export -> database + embedding -> dashboard
    chatautopsy demo                  chat inventata per provare l'app
    chatautopsy llm setup [MODELLO]   scarica e avvia il modello AI locale (consigliato in base alla RAM)
    chatautopsy llm status|list|stop
    chatautopsy list | wipe NOME | doctor
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from chatautopsy.i18n import get_lang, t

APP_PATH = Path(__file__).resolve().parent / "app.py"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _bar(frac: float, width: int = 28) -> str:
    n = int(max(0.0, min(1.0, frac)) * width)
    return "█" * n + "░" * (width - n)


def _progress_printer():
    last = {"line": ""}

    def cb(stage: str, frac: float, detail: str) -> None:
        line = f"\r{_bar(frac)} {frac * 100:5.1f}%  {stage}: {detail}"[:150]
        sys.stdout.write(line + " " * max(0, len(last["line"]) - len(line)))
        sys.stdout.flush()
        last["line"] = line

    return cb


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def launch_app(port: int = 8501, open_db: Optional[str] = None, open_browser: bool = True) -> int:
    env = dict(os.environ)
    if open_db:
        env["CHATAUTOPSY_OPEN_DB"] = str(open_db)
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH),
           "--server.port", str(port), "--server.headless", "false" if open_browser else "true",
           "--server.maxUploadSize", "2048", "--browser.gatherUsageStats", "false",
           "--theme.base", "dark", "--theme.primaryColor", "#ff4b6e", "--theme.backgroundColor", "#0e1117",
           "--theme.secondaryBackgroundColor", "#171b26", "--theme.textColor", "#e6e6eb"]
    print(t("🧬 ChatAutopsy → http://localhost:{p}   (Ctrl+C per chiudere)", p=port))
    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 0


# ---------------------------------------------------------------------------
# Comandi
# ---------------------------------------------------------------------------
def cmd_analyze(args: argparse.Namespace) -> int:
    from chatautopsy import config, database
    from chatautopsy.ingestion import run_ingestion

    src = Path(args.path).expanduser().resolve()
    if not src.exists():
        print(t("✗ Percorso non trovato: {p}", p=src))
        return 2
    name = args.name or _default_name(src)
    db_path = database.db_path_for_name(name)
    emb = {"fast": config.EMBEDDING_LIGHT_MODEL, "quality": config.EMBEDDING_QUALITY_MODEL}.get(args.embeddings)
    print(f"📥 {src}\n🗂️  database: {db_path}")
    t0 = time.time()
    res = run_ingestion(src, db_path, progress_callback=_progress_printer(), build_embeddings=emb is not None,
                        embedding_model=emb, telegram_chat_name=args.chat)
    print("\n" + t("✅ {m} messaggi · {s} sessioni · {e} embedding", m=f"{res['messages']:,}", s=f"{res['sessions']:,}",
                    e=f"{res['embedded_sessions']:,}") + " "
          f"· {time.time() - t0:.0f}s")
    if res.get("embedding_error"):
        print(t("⚠️  embedding non generati ({e}): la ricerca userà solo BM25.", e=res["embedding_error"]))
    for who, n in list(res["participants"].items())[:8]:
        print(f"   · {who}: {n:,}")
    if args.llm:
        _llm_setup(None, wait=False)
    if args.no_open:
        return 0
    return launch_app(args.port, open_db=str(db_path))


def _default_name(src: Path) -> str:
    stem = src.stem if src.is_file() else src.name
    for prefix in ("ChatExport_", "WhatsApp Chat - ", "WhatsApp Chat with ", "Chat WhatsApp con "):
        stem = stem.replace(prefix, "")
    return stem.strip() or "chat"


def cmd_demo(args: argparse.Namespace) -> int:
    from chatautopsy import database, demo
    from chatautopsy.ingestion import run_ingestion

    if get_lang() == "en":
        name, params, who = "demo_emma_jake", demo.SHOWCASE_EN, "Emma & Jake"
    else:
        name, params, who = "demo_giulia_marco", demo.SHOWCASE, "Giulia & Marco"
    db_path = database.db_path_for_name(name)
    print(t("🎭 Creo una chat di coppia inventata ({w}, ~900 giorni)…", w=who))
    run_ingestion(None, db_path, progress_callback=_progress_printer(), build_embeddings=False,
                  file_bytes=demo.couple_chat(**params).encode("utf-8"), source_name=f"{name}.txt")
    print()
    return 0 if args.no_open else launch_app(args.port, open_db=str(db_path))


def _llm_setup(model: Optional[str], wait: bool = True) -> int:
    from chatautopsy import llm_manager as lm

    backend = None
    if model:
        entry = lm.catalog_entry(model)
        backend = entry["backend"] if entry else ("ollama" if ":" in model else "mlx")
    else:
        backend = lm.default_backend()
        model = lm.recommended_model(backend)
    if not model or not backend:
        print(t("✗ Nessun backend disponibile: su Mac Apple Silicon usa MLX, altrimenti installa Ollama (ollama.com)."))
        return 2
    print(f"🤖 {model} ({backend}) · RAM {lm.ram_gb():.0f} GB")
    m = lm.manager()

    def show(st: dict) -> None:
        sys.stdout.write(f"\r{_bar(st.get('progress', 0))} {st.get('phase', '')}: {st.get('message', '')}"[:150].ljust(150))
        sys.stdout.flush()

    st = m.ensure_ready(model, backend, wait=wait, on_progress=show)
    print()
    if st["phase"] == "ready":
        print(t("✅ Pronto su {u}", u=st["base_url"]))
        return 0
    if st["phase"] == "error":
        print(f"✗ {st.get('error')}")
        return 1
    return 0


def cmd_llm(args: argparse.Namespace) -> int:
    from chatautopsy import llm_manager as lm

    if args.action == "list":
        rec = lm.recommended_model()
        print(t("RAM: {r} GB · backend disponibili: {b}", r=f"{lm.ram_gb():.0f}", b=", ".join(lm.available_backends()) or t("nessuno")))
        for e in lm.catalog():
            flags = ("✅ scaricato" if e["cached"] else f"⬇️ {e['size_gb']} GB") + ("" if e["fits"] else " ⚠️ RAM insufficiente")
            star = " ⭐ consigliato" if e["id"] == rec else ""
            print(f"  [{e['backend']:6s}] {e['id']:45s} {flags}{star}")
        return 0
    if args.action == "setup":
        rc = _llm_setup(args.model, wait=True)
        if rc == 0:
            print(t("Il server resta attivo finché non esegui `chatautopsy llm stop`."))
            try:
                while True:  # la CLI tiene vivo il server (processo figlio) finché l'utente non chiude
                    time.sleep(3600)
            except KeyboardInterrupt:
                lm.manager().stop_server()
        return rc
    if args.action == "stop":
        print(t("Fermato.") if lm.manager().stop_server() else t("Nessun server avviato da ChatAutopsy."))
        return 0
    s = lm.load_settings()
    model, backend = s.get("llm_model"), s.get("llm_backend")
    url = lm.base_url_for(backend) if backend else None
    alive = bool(url and lm.endpoint_alive(url))
    print(f"Modello scelto: {model or '—'} ({backend or '—'}) · server: {'🟢 attivo' if alive else '🔴 spento'}"
          + (f" su {url}" if alive else ""))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    from chatautopsy import database

    dbs = database.list_databases()
    if not dbs:
        print(t("Nessun database. Usa `chatautopsy analyze PERCORSO` o `chatautopsy demo`."))
    for p in dbs:
        s = database.get_db_summary(p)
        print(f"  {p.stem:35s} {s['messages']:>10,} msg  {s['first_ts'][:10] if s['first_ts'] else ''} → "
              f"{s['last_ts'][:10] if s['last_ts'] else ''}  {p.stat().st_size / 1e6:,.0f} MB")
    return 0


def cmd_wipe(args: argparse.Namespace) -> int:
    from chatautopsy import database

    p = database.db_path_for_name(args.name)
    if not p.exists():
        print(t("✗ {n} non esiste.", n=p.stem))
        return 2
    if not args.yes and input(t("Eliminare definitivamente «{n}»? [s/N] ", n=p.stem)).strip().lower() not in ("s", "si", "sì", "y", "yes"):
        return 1
    print(t("Rimossi:"), ", ".join(database.secure_wipe_database(p)))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    import importlib.util
    import platform

    from chatautopsy import config, rag_engine
    from chatautopsy import llm_manager as lm

    ok = lambda b: "✅" if b else "❌"  # noqa: E731
    print(f"Python {platform.python_version()} · {platform.system()} {platform.machine()} · RAM {lm.ram_gb():.0f} GB")
    print(t("Dati: {d}", d=config.DATA_DIR))
    for mod in ("streamlit", "pandas", "sklearn", "sqlite_vec", "torch", "sentence_transformers", "mlx_lm"):
        print(f"  {ok(importlib.util.find_spec(mod) is not None)} {mod}")
    try:
        import torch

        dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(t("  accelerazione embedding: {d}", d=dev))
    except Exception:
        print(t("  embedding: non installati (pip install 'chatautopsy[embeddings]') — la ricerca userà BM25"))
    for name, m in config.EMBEDDING_PRESETS.items():
        print(f"  {ok(rag_engine.is_model_cached(m))} embedding {name}")
    print(t("  backend LLM: {b} · consigliato: {r}", b=", ".join(lm.available_backends()) or t("nessuno"), r=lm.recommended_model() or "—"))
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chatautopsy", description=t("Analisi 100% locale di chat WhatsApp e Telegram."))
    p.add_argument("--port", type=int, default=8501, help=t("porta della dashboard (default 8501)"))
    p.add_argument("--lang", choices=["en", "it"], help="interface language / lingua dell'interfaccia")
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("analyze", help=t("analizza un export (file .txt/.json/.zip/.html o cartella) e apre la dashboard"))
    a.add_argument("path")
    a.add_argument("--name", help=t("nome dell'analisi (default: dal nome del file/cartella)"))
    a.add_argument("--embeddings", choices=["fast", "quality", "none"], default="fast",
                   help=t("ricerca semantica: fast=MiniLM (default), quality=bge-m3, none=solo parole chiave"))
    a.add_argument("--chat", help=t("export Telegram completo (result.json): nome della chat da analizzare"))
    a.add_argument("--llm", action="store_true", help=t("prepara anche il modello AI locale in background"))
    a.add_argument("--no-open", action="store_true", help=t("non aprire la dashboard al termine"))
    a.add_argument("--port", type=int, default=8501)
    a.set_defaults(func=cmd_analyze)

    d = sub.add_parser("demo", help=t("crea una chat inventata e apre la dashboard"))
    d.add_argument("--no-open", action="store_true")
    d.add_argument("--port", type=int, default=8501)
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("app", help=t("apre la dashboard"))
    s.add_argument("--port", type=int, default=8501)
    s.set_defaults(func=lambda ns: launch_app(ns.port))

    l_ = sub.add_parser("llm", help=t("gestione del modello AI locale"))
    l_.add_argument("action", choices=["setup", "status", "list", "stop"])
    l_.add_argument("model", nargs="?", help=t("id del modello (default: consigliato per la tua RAM)"))
    l_.set_defaults(func=cmd_llm)

    sub.add_parser("list", help=t("elenca le chat analizzate")).set_defaults(func=cmd_list)
    w = sub.add_parser("wipe", help=t("cancella in modo sicuro il database di una chat"))
    w.add_argument("name")
    w.add_argument("-y", "--yes", action="store_true")
    w.set_defaults(func=cmd_wipe)
    sub.add_parser("doctor", help=t("verifica dipendenze, accelerazione e modelli")).set_defaults(func=cmd_doctor)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--lang" in argv[:-1]:  # la lingua serve già per l'help, prima del parsing completo
        lang = argv[argv.index("--lang") + 1]
        from chatautopsy import i18n

        i18n.set_lang(lang)
        os.environ["CHATAUTOPSY_LANG"] = lang  # anche per la dashboard avviata dopo
    args = build_parser().parse_args(argv)
    if not getattr(args, "cmd", None):
        return launch_app(args.port)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
