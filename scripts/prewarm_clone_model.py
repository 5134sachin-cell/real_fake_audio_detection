"""Download and verify the configured voice cloning model ahead of packaging."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.voice_cloner import VoiceCloner


def main() -> None:
    cloner = VoiceCloner()
    cloner._ensure_model()
    print(cloner.status())


if __name__ == "__main__":
    main()
