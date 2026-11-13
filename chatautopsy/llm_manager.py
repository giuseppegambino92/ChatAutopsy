"""
ChatAutopsy — Gestore del modello linguistico locale.

Automatizza l'intera catena per l'AI Detective:
  1. sceglie un modello adatto alla RAM della macchina (catalogo in config.LLM_CATALOG);
  2. installa il backend se manca (mlx-lm su Apple Silicon) o usa Ollama se presente;
  3. scarica i pesi una sola volta (cache Hugging Face / Ollama) mostrando l'avanzamento;
  4. avvia il server OpenAI-compatible locale, ne verifica la salute e lo riavvia all'occorrenza;
  5. ricorda la scelta (settings.json) per l'avvio automatico alle sessioni successive.

Tutto resta su localhost: il server ascolta su 127.0.0.1, nessun dato della chat lascia la macchina.
Le uniche connessioni di rete sono il download una tantum dei pesi del modello scelto.
"""
from __future__ import annotations

import atexit
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from chatautopsy import config
from chatautopsy.i18n import t


# ---------------------------------------------------------------------------
# Sistema
# ---------------------------------------------------------------------------
def ram_gb() -> float:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout
            return int(out.strip()) / 1024 ** 3
        if hasattr(os, "sysconf"):
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except Exception:
        pass
    return 8.0


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mlx_installed() -> bool:
    return importlib.util.find_spec("mlx_lm") is not None


def ollama_binary() -> Optional[str]:
    return shutil.which("ollama") or next((p for p in ("/usr/local/bin/ollama", "/opt/homebrew/bin/ollama")
                                           if Path(p).exists()), None)


def available_backends() -> List[str]:
    out = []
    if is_apple_silicon():
        out.append("mlx")  # installabile automaticamente se manca
    if ollama_binary():
        out.append("ollama")
    return out


def default_backend() -> Optional[str]:
    b = available_backends()
    return b[0] if b else None


# ---------------------------------------------------------------------------
# Impostazioni persistenti
# ---------------------------------------------------------------------------
def load_settings() -> Dict[str, Any]:
    try:
        return json.loads(config.SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(**updates: Any) -> Dict[str, Any]:
    s = load_settings()
    s.update(updates)
    config.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(config.SETTINGS_FILE)
    return s


# ---------------------------------------------------------------------------
# Catalogo
# ---------------------------------------------------------------------------
def catalog_entry(model_id: str) -> Optional[Dict[str, Any]]:
    return next((dict(e) for e in config.LLM_CATALOG if e["id"] == model_id), None)


def catalog(backend: Optional[str] = None) -> List[Dict[str, Any]]:
    ram = ram_gb()
    out = []
    for e in config.LLM_CATALOG:
        if backend and e["backend"] != backend:
            continue
        item = dict(e)
        item["fits"] = ram >= e["min_ram_gb"]
        item["cached"] = is_model_cached(e["id"], e["backend"])
        out.append(item)
    return out


def recommended_model(backend: Optional[str] = None) -> Optional[str]:
    """Il modello ⭐ per la RAM di questa macchina: tra quelli marcati "recommend", il più grande che ci sta."""
    backend = backend or default_backend()
    if not backend:
        return None
    ram = ram_gb()
    cands = [e for e in config.LLM_CATALOG if e["backend"] == backend and e.get("recommend")]
    fits = [e for e in cands if e["min_ram_gb"] <= ram]
    if not fits:  # macchina sotto la soglia minima: il più piccolo disponibile
        fits = sorted(cands, key=lambda e: e["min_ram_gb"])[:1]
    return max(fits, key=lambda e: e["min_ram_gb"])["id"] if fits else None


def is_model_cached(model_id: str, backend: str) -> bool:
    if backend == "mlx":
        try:
            from huggingface_hub import scan_cache_dir

            for repo in scan_cache_dir().repos:
                if repo.repo_id == model_id and repo.repo_type == "model":
                    files = {f.file_name for rev in repo.revisions for f in rev.files}
                    has_template = "chat_template.jinja" in files or "tokenizer_config.json" in files
                    return any(f.endswith(".safetensors") for f in files) and "config.json" in files and has_template
        except Exception:
            return False
        return False
    if backend == "ollama":
        return model_id in ollama_models()
    return False


# ---------------------------------------------------------------------------
# HTTP util
# ---------------------------------------------------------------------------
def _get_json(url: str, timeout: float = 2.0) -> Optional[Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def endpoint_models(base_url: str, timeout: float = 2.0) -> Optional[List[str]]:
    """Modelli esposti da un endpoint OpenAI-compatible, None se non risponde."""
    data = _get_json(base_url.rstrip("/") + "/models", timeout)
    if not isinstance(data, dict):
        return None
    return [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]


def endpoint_alive(base_url: str, timeout: float = 2.0) -> bool:
    return endpoint_models(base_url, timeout) is not None


def ollama_models() -> List[str]:
    data = _get_json(config.OLLAMA_URL + "/api/tags", 2.0)
    if not isinstance(data, dict):
        return []
    return [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]


def base_url_for(backend: str) -> str:
    return f"http://127.0.0.1:{config.MLX_PORT}/v1" if backend == "mlx" else config.OLLAMA_URL + "/v1"


# ---------------------------------------------------------------------------
# Gestore (stato condiviso tra sessioni Streamlit e CLI nello stesso processo)
# ---------------------------------------------------------------------------
class LLMManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self.state: Dict[str, Any] = {"phase": "idle", "model": None, "backend": None, "progress": 0.0,
                                      "message": "", "base_url": None, "error": None, "updated": time.time()}
        atexit.register(self.shutdown)

    # ------------------------------------------------------------- stato
    def _set(self, **kw: Any) -> None:
        with self._lock:
            self.state.update(kw, updated=time.time())

    def status(self) -> Dict[str, Any]:
        with self._lock:
            st = dict(self.state)
        # verifica dal vivo: un server avviato altrove (o fuori da ChatAutopsy) conta come pronto
        if st["phase"] in ("idle", "ready", "error", "stopped") and st.get("base_url"):
            alive = endpoint_alive(st["base_url"], 1.5)
            if alive and st["phase"] != "ready":
                st["phase"] = "ready"
            elif not alive and st["phase"] == "ready":
                st["phase"] = "stopped"
                st["message"] = "Il server del modello non risponde più."
        return st

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------ avvio
    def ensure_ready(self, model_id: str, backend: Optional[str] = None, wait: bool = False,
                     on_progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """Installa backend, scarica il modello, avvia il server. Non bloccante (wait=False) per la UI."""
        entry = catalog_entry(model_id) or {"id": model_id, "backend": backend or default_backend(), "size_gb": None}
        backend = backend or entry["backend"]
        if backend not in ("mlx", "ollama"):
            raise ValueError(f"Backend non supportato: {backend}")
        with self._lock:
            if self.busy():
                if wait and self._thread is not None:
                    self._thread.join()
                return self.status()
            self.state.update(phase="starting", model=model_id, backend=backend, progress=0.0, error=None,
                              base_url=base_url_for(backend), message=t("Preparazione…"), updated=time.time())
            save_settings(llm_backend=backend, llm_model=model_id, llm_autostart=True)
            self._thread = threading.Thread(target=self._run, args=(entry, backend, on_progress),
                                            name="chatautopsy-llm", daemon=True)
            self._thread.start()
        if wait:
            self._thread.join()
        return self.status()

    def _run(self, entry: Dict[str, Any], backend: str, cb: Optional[Callable]) -> None:
        def report(**kw: Any) -> None:
            self._set(**kw)
            if cb:
                try:
                    cb(self.status())
                except Exception:
                    pass

        try:
            if backend == "mlx":
                self._run_mlx(entry, report)
            else:
                self._run_ollama(entry, report)
        except Exception as exc:  # qualunque errore diventa uno stato leggibile in UI
            report(phase="error", error=f"{type(exc).__name__}: {exc}", message=str(exc))

    # --------------------------------------------------------------- MLX
    def _run_mlx(self, entry: Dict[str, Any], report: Callable[..., None]) -> None:
        model_id = entry["id"]
        if not is_apple_silicon():
            raise RuntimeError("Il backend MLX richiede un Mac con Apple Silicon: usa Ollama.")
        if not mlx_installed():
            report(phase="installing", message=t("Installazione di mlx-lm (una tantum)…"), progress=0.02)
            r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "mlx-lm>=0.20"],
                               capture_output=True, text=True)
            if r.returncode != 0 or not mlx_installed():
                raise RuntimeError("Installazione di mlx-lm fallita: " + (r.stderr or r.stdout)[-400:])
        base_url = base_url_for("mlx")
        running = self._running_server_model()
        if running == model_id and endpoint_alive(base_url):
            report(phase="ready", progress=1.0, message=t("{m} già attivo.", m=model_id))
            return
        # Sempre: se il modello è completo in cache non scarica nulla, altrimenti completa i file mancanti
        # (es. chat_template.jinja). Senza rete si prosegue con la cache, se presente.
        try:
            self._download_hf(model_id, report)
        except Exception:
            if not is_model_cached(model_id, "mlx"):
                raise
        self.stop_server()
        report(phase="starting", progress=0.97, message=t("Avvio del server MLX…"))
        self._start_mlx_server(model_id)
        self._wait_alive(base_url, report, timeout=240)
        self._smoke_test(base_url, model_id)
        report(phase="ready", progress=1.0, message=t("{m} pronto su {u}", m=model_id, u=base_url))

    def _download_hf(self, model_id: str, report: Callable[..., None]) -> None:
        from huggingface_hub import HfApi, snapshot_download

        report(phase="downloading", progress=0.03, message=t("Recupero dimensione del modello…"))
        total = 0
        try:
            info = HfApi().model_info(model_id, files_metadata=True)
            total = sum((s.size or 0) for s in info.siblings or [])
        except Exception:
            entry = catalog_entry(model_id) or {}
            total = int((entry.get("size_gb") or 0) * 1024 ** 3)
        done = threading.Event()
        err: List[BaseException] = []
        counted = {"bytes": 0}

        from tqdm.auto import tqdm as _tqdm

        class _ByteCounter(_tqdm):  # intercetta l'avanzamento in byte dei singoli file (anche con backend xet)
            def update(self, n=1):  # type: ignore[override]
                if getattr(self, "unit", "") == "B" and n:
                    counted["bytes"] += int(n)
                return super().update(n)

        # *.jinja: i modelli recenti tengono il chat template in chat_template.jinja (senza, il server risponde 404)
        patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.py", "*.jinja"]

        def _dl() -> None:
            try:
                try:
                    snapshot_download(model_id, allow_patterns=patterns, tqdm_class=_ByteCounter)
                except TypeError:  # versioni di huggingface_hub senza tqdm_class
                    snapshot_download(model_id, allow_patterns=patterns)
            except BaseException as exc:  # noqa: BLE001
                err.append(exc)
            finally:
                done.set()

        threading.Thread(target=_dl, daemon=True).start()
        cache_dir = _hf_repo_dir(model_id)
        while not done.wait(1.0):
            got = max(_dir_size(cache_dir), counted["bytes"])
            frac = min(got / total, 0.999) if total else 0.0
            report(phase="downloading", progress=0.03 + 0.92 * frac,
                   message=f"Download {_gb(got)} / {_gb(total) if total else '?'}")
        if err:
            raise RuntimeError(f"Download di {model_id} fallito: {err[0]}")
        report(phase="downloading", progress=0.95, message=t("Download completato."))

    def _start_mlx_server(self, model_id: str) -> None:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = open(config.LOG_DIR / "llm_server.log", "ab")
        env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_OFFLINE="1")
        cmd = [sys.executable, "-m", "mlx_lm.server", "--model", model_id, "--host", "127.0.0.1",
               "--port", str(config.MLX_PORT)]
        self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        _write_pid(self._proc.pid, model_id, "mlx")

    # ------------------------------------------------------------ Ollama
    def _run_ollama(self, entry: Dict[str, Any], report: Callable[..., None]) -> None:
        model_id = entry["id"]
        binary = ollama_binary()
        if not binary:
            raise RuntimeError("Ollama non è installato: scaricalo da ollama.com oppure usa MLX su Apple Silicon.")
        if not endpoint_alive(config.OLLAMA_URL + "/v1"):
            report(phase="starting", progress=0.02, message=t("Avvio di Ollama…"))
            config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            log = open(config.LOG_DIR / "ollama.log", "ab")
            self._proc = subprocess.Popen([binary, "serve"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            _write_pid(self._proc.pid, "ollama", "ollama")
            self._wait_alive(config.OLLAMA_URL + "/v1", report, timeout=60)
        if model_id not in ollama_models():
            report(phase="downloading", progress=0.05, message=t("Download di {m} con Ollama…", m=model_id))
            req = urllib.request.Request(config.OLLAMA_URL + "/api/pull", data=json.dumps({"model": model_id}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                for line in r:
                    try:
                        ev = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    if ev.get("error"):
                        raise RuntimeError(ev["error"])
                    tot, comp = ev.get("total"), ev.get("completed")
                    frac = (comp / tot) if tot and comp else 0.0
                    report(phase="downloading", progress=0.05 + 0.9 * frac,
                           message=f"{ev.get('status', '')} {_gb(comp or 0)} / {_gb(tot or 0)}".strip())
        report(phase="ready", progress=1.0, message=t("{m} pronto su Ollama", m=model_id))

    # ------------------------------------------------------------ comuni
    def _wait_alive(self, base_url: str, report: Callable[..., None], timeout: float) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if endpoint_alive(base_url, 2.0):
                return
            if self._proc is not None and self._proc.poll() is not None:
                tail = _log_tail(config.LOG_DIR / "llm_server.log")
                raise RuntimeError(f"Il server si è chiuso subito (codice {self._proc.returncode}). Log: {tail}")
            report(message=t("Caricamento del modello in memoria… {s}s", s=int(time.time() - t0)))
            time.sleep(1.5)
        raise RuntimeError(f"Il server non risponde dopo {int(timeout)}s (vedi {config.LOG_DIR / 'llm_server.log'}).")

    def _smoke_test(self, base_url: str, model_id: str) -> None:
        """Una vera richiesta di chat (1 token): il server può rispondere a /models ma non saper generare."""
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps({"model": model_id, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            tail = _log_tail(config.LOG_DIR / "llm_server.log", 400)
            self.stop_server()
            raise RuntimeError(f"Il modello non riesce a generare (HTTP {exc.code}). Log: {tail}") from exc

    def _running_server_model(self) -> Optional[str]:
        info = _read_pid()
        if info and _pid_alive(info.get("pid")):
            return info.get("model")
        return None

    def stop_server(self) -> bool:
        stopped = False
        info = _read_pid()
        if info and info.get("backend") == "mlx" and _pid_alive(info.get("pid")):
            _terminate(int(info["pid"]))
            stopped = True
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            stopped = True
        self._proc = None
        _PID_FILE().unlink(missing_ok=True)
        with self._lock:
            if self.state.get("phase") == "ready":
                self.state.update(phase="stopped", message=t("Server fermato."))
        return stopped

    def shutdown(self) -> None:
        """All'uscita dell'app ferma solo il server avviato da questo processo (libera la RAM)."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(5)
            except Exception:
                pass
            _PID_FILE().unlink(missing_ok=True)

    def autostart(self) -> Optional[Dict[str, Any]]:
        """Riavvia in background il modello scelto l'ultima volta, se è già scaricato (nessun download implicito)."""
        s = load_settings()
        model, backend = s.get("llm_model"), s.get("llm_backend")
        if not (model and backend and s.get("llm_autostart", True)):
            return None
        if self.busy() or endpoint_alive(base_url_for(backend), 1.0):
            with self._lock:
                if self.state["phase"] == "idle":
                    self.state.update(model=model, backend=backend, base_url=base_url_for(backend))
            return self.status()
        if backend in available_backends() and is_model_cached(model, backend):
            return self.ensure_ready(model, backend)
        return None


# ---------------------------------------------------------------------------
# Util file/processi
# ---------------------------------------------------------------------------
def _PID_FILE() -> Path:  # noqa: N802
    return config.DATA_DIR / "llm_server.json"


def _write_pid(pid: int, model: str, backend: str) -> None:
    _PID_FILE().write_text(json.dumps({"pid": pid, "model": model, "backend": backend, "started": time.time()}))


def _read_pid() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_PID_FILE().read_text())
    except Exception:
        return None


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _terminate(pid: int) -> None:
    import signal

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return
    for _ in range(20):
        if not _pid_alive(pid):
            return
        time.sleep(0.25)
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def _hf_repo_dir(model_id: str) -> Path:
    try:
        from huggingface_hub import constants

        root = Path(constants.HF_HUB_CACHE)
    except Exception:
        root = Path.home() / ".cache" / "huggingface" / "hub"
    return root / ("models--" + model_id.replace("/", "--"))


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _gb(n: float) -> str:
    return f"{n / 1024 ** 3:.2f} GB"


def _log_tail(path: Path, n: int = 600) -> str:
    try:
        return path.read_text(errors="ignore")[-n:]
    except Exception:
        return ""


_MANAGER: Optional[LLMManager] = None
_MANAGER_LOCK = threading.Lock()


def manager() -> LLMManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = LLMManager()
        return _MANAGER
