"""Start CampusLens AI locally with one command: backend API + frontend.

Usage (from the repository root, with the virtual environment active):

    python scripts/run_dev.py              # backend :8000 + frontend :5500, opens the browser
    python scripts/run_dev.py --no-browser
    python scripts/run_dev.py --reload     # restart the backend when Python files change

What it does:

1. Creates ``.env`` from ``.env.example`` if ``.env`` does not exist yet.
2. Checks that ports 8000 and 5500 on 127.0.0.1 are free.
3. Warns (does not stop) if Ollama is not reachable or the model is not pulled.
4. Starts ``uvicorn backend.app.main:app`` on http://127.0.0.1:8000.
5. Serves the ``frontend`` folder on http://127.0.0.1:5500.

The ports are fixed on purpose: the frontend calls ``http://127.0.0.1:8000/api``
and the backend's CORS setting allows only ``http://127.0.0.1:5500``.
Press Ctrl+C to stop both servers.
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_PORT = 5500
STUDENT_URL = f"http://{HOST}:{FRONTEND_PORT}/index.html"
ADMIN_URL = f"http://{HOST}:{FRONTEND_PORT}/admin-login.html"


def log(message: str) -> None:
    print(f"[run_dev] {message}", flush=True)


def ensure_env_file() -> None:
    env_path = ROOT / ".env"
    example_path = ROOT / ".env.example"
    if env_path.exists():
        return
    if not example_path.exists():
        log(
            "UYARI: .env ve .env.example bulunamadı; varsayılan ayarlarla devam ediliyor."
        )
        return
    shutil.copyfile(example_path, env_path)
    log(".env bulunamadı; .env.example kopyalanarak oluşturuldu.")


def read_env_values() -> dict[str, str]:
    """Read simple KEY=VALUE lines from .env (only for the pre-flight checks)."""
    values: dict[str, str] = {}
    env_path = ROOT / ".env"
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((HOST, port)) != 0


def check_ollama(env: dict[str, str]) -> None:
    if env.get("LLM_PROVIDER", "ollama").lower() != "ollama":
        return
    base_url = env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = env.get("OLLAMA_MODEL", "")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as response:
            tags = json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        log(
            f"UYARI: Ollama'ya ulaşılamadı ({base_url}). Sorular yanıtlanamaz; Ollama'yı başlatın."
        )
        return
    names = {item.get("name", "") for item in tags.get("models", [])}
    if model and model not in names and f"{model}:latest" not in names:
        log(
            f"UYARI: '{model}' modeli Ollama'da yok. Şunu çalıştırın: ollama pull {model}"
        )


class FrontendHandler(SimpleHTTPRequestHandler):
    """Static file handler with correct MIME types and no caching.

    On Windows the registry can map .js to text/plain, which makes browsers
    refuse to load ES modules; the types are therefore set explicitly.
    """

    extensions_map: ClassVar[dict[str, str]] = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".svg": "image/svg+xml",
        ".json": "application/json",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return  # keep the console readable; backend logs are enough


def start_frontend_server() -> ThreadingHTTPServer:
    handler = functools.partial(FrontendHandler, directory=str(FRONTEND_DIR))
    server = ThreadingHTTPServer((HOST, FRONTEND_PORT), handler)
    threading.Thread(target=server.serve_forever, name="frontend", daemon=True).start()
    return server


def start_backend(reload: bool) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        HOST,
        "--port",
        str(BACKEND_PORT),
    ]
    if reload:
        command += ["--reload", "--reload-dir", "backend"]
    return subprocess.Popen(command, cwd=ROOT)


def wait_for_backend(process: subprocess.Popen[bytes], timeout_s: float = 120) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(
                f"http://{HOST}:{BACKEND_PORT}/api/health", timeout=2
            ) as response:
                health = json.load(response)
            if health.get("database") not in (None, "ok"):
                log(
                    "UYARI: Veritabanı hazır değil (DATABASE_URL ayarını kontrol edin)."
                )
            if health.get("qdrant") == "collection_missing":
                log(
                    "UYARI: Henüz indekslenmiş belge yok. Admin > Belgeler ekranından PDF yükleyin."
                )
            elif health.get("qdrant") not in (None, "ok"):
                log("UYARI: Qdrant hazır değil (QDRANT_URL ayarını kontrol edin).")
            return True
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(1)
    return False


def stop_backend(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        process.terminate()
    else:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _raise_keyboard_interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CampusLens AI: backend + frontend tek komutla"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="tarayıcıyı otomatik açma"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Python dosyaları değişince backend'i yeniden başlat",
    )
    args = parser.parse_args(argv)

    if not (ROOT / "backend" / "app" / "main.py").exists() or not FRONTEND_DIR.is_dir():
        log(
            "HATA: Bu komut proje kökündeki dosyalarla çalışır (backend/ ve frontend/ bulunamadı)."
        )
        return 1

    ensure_env_file()
    busy = [port for port in (BACKEND_PORT, FRONTEND_PORT) if not port_is_free(port)]
    if busy:
        ports = ", ".join(str(port) for port in busy)
        log(
            f"HATA: {ports} portu kullanımda. Açık kalan uvicorn/Live Server/http.server'ı kapatın."
        )
        return 1
    check_ollama(read_env_values())

    # Terminal kapatılırken (SIGTERM) de iki sunucu birlikte kapansın.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    frontend = start_frontend_server()
    log(f"Frontend: http://{HOST}:{FRONTEND_PORT}")
    backend = start_backend(args.reload)
    log(
        f"Backend başlatılıyor: http://{HOST}:{BACKEND_PORT} (ilk açılış modeller yüzünden uzun sürebilir)"
    )

    try:
        if not wait_for_backend(backend):
            log("HATA: Backend başlatılamadı. Yukarıdaki hata mesajına bakın.")
            return 1
        log("Hazır.")
        log(f"  Öğrenci ekranı: {STUDENT_URL}")
        log(f"  Admin girişi:   {ADMIN_URL}")
        log(f"  API belgeleri:  http://{HOST}:{BACKEND_PORT}/docs")
        log("Durdurmak için Ctrl+C.")
        if not args.no_browser:
            webbrowser.open(STUDENT_URL)
        return backend.wait()
    except KeyboardInterrupt:
        log("Durduruluyor…")
        return 0
    finally:
        stop_backend(backend)
        frontend.shutdown()
        frontend.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
