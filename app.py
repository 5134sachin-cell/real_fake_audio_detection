"""Root entrypoint to run the development Flask app."""

from __future__ import annotations

from src.webapp import run_dev_server


if __name__ == "__main__":
    run_dev_server()
