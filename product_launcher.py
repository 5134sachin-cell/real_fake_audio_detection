"""Portable Windows launcher for the packaged VoiceWorkbench app."""

from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser

from src.logger import get_logger
from src.runtime_paths import ensure_runtime_dirs, runtime_home, tts_cache_dir
from src.webapp import VOICE_CLONER, app

logger = get_logger("launcher")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _local_ip() -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        probe.close()


def _open_browser_later(url: str, delay_sec: float = 1.5) -> None:
    def _worker() -> None:
        time.sleep(delay_sec)
        try:
            webbrowser.open(url)
        except Exception:
            logger.warning("Could not open browser automatically for %s", url)

    threading.Thread(target=_worker, daemon=True).start()


def main() -> None:
    ensure_runtime_dirs()
    os.environ.setdefault("VOICE_CLONE_CACHE_DIR", str(tts_cache_dir()))

    host = os.environ.get("VOICE_WORKBENCH_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = int(os.environ.get("VOICE_WORKBENCH_PORT", "8000").strip() or "8000")
    preload_clone = _env_flag("VOICE_WORKBENCH_PRELOAD_CLONE", False)
    open_browser = _env_flag("VOICE_WORKBENCH_OPEN_BROWSER", True)

    logger.info("Runtime home: %s", runtime_home())
    logger.info("Serving VoiceWorkbench on %s:%s", host, port)
    logger.info("Local URL: http://127.0.0.1:%s", port)
    logger.info("LAN URL: http://%s:%s", _local_ip(), port)

    if preload_clone:
        try:
            VOICE_CLONER._ensure_model()
        except Exception as exc:
            logger.warning("Clone model preload failed. The app will still start: %s", exc)

    if open_browser:
        _open_browser_later(f"http://127.0.0.1:{port}")

    try:
        from waitress import serve

        serve(app, host=host, port=port, threads=8)
    except Exception as exc:
        logger.warning("Waitress unavailable or failed (%s). Falling back to Flask server.", exc)
        app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
