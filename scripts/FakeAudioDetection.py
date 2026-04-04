"""Backward-compatible CLI entrypoint for detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector import analyze_audio, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect if an audio file is human or AI generated.")
    parser.add_argument("audio_path", help="Path to input wav file")
    args = parser.parse_args()
    model, config = load_model()
    label, confidence = analyze_audio(args.audio_path, model, config)
    print(f"Result: {label} ({confidence}%)")


if __name__ == "__main__":
    main()
