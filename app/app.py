"""Compatibility entrypoint for Flask development mode."""

from __future__ import annotations

from src.webapp import app, run_dev_server


if __name__ == "__main__":
    run_dev_server()
