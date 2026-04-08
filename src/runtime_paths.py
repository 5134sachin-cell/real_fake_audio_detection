"""Runtime path helpers for source and packaged execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "VoiceWorkbench"


def bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def runtime_home() -> Path:
    override = os.environ.get("VOICE_WORKBENCH_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    if getattr(sys, "frozen", False):
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        if local_appdata:
            return Path(local_appdata).expanduser().resolve() / APP_NAME
        return (Path.home() / "AppData" / "Local" / APP_NAME).resolve()

    return bundle_root()


def templates_dir() -> Path:
    return bundle_root() / "templates"


def static_dir() -> Path:
    return bundle_root() / "static"


def models_dir() -> Path:
    return bundle_root() / "models"


def outputs_dir() -> Path:
    return runtime_home() / "outputs"


def uploads_dir() -> Path:
    return outputs_dir() / "uploads"


def logs_dir() -> Path:
    return runtime_home() / "logs"


def tts_cache_dir() -> Path:
    return runtime_home() / ".tts_cache"


def ensure_runtime_dirs() -> dict[str, Path]:
    dirs = {
        "runtime_home": runtime_home(),
        "outputs": outputs_dir(),
        "uploads": uploads_dir(),
        "logs": logs_dir(),
        "tts_cache": tts_cache_dir(),
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
