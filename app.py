"""Root entrypoint to run Flask API."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_path("app/app.py", run_name="__main__")
