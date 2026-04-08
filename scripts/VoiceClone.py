"""Backward-compatible CLI entrypoint for voice cloning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.voice_cloner import VoiceCloner


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone a voice from speaker wav.")
    parser.add_argument("--speaker_wav", required=True, help="Path to speaker wav")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed [0.7, 1.3]")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    args = parser.parse_args()

    cloner = VoiceCloner()
    result = cloner.clone(
        speaker_wav=args.speaker_wav,
        text=args.text,
        speed=args.speed,
        language=args.language,
        output_dir="outputs",
    )
    print(f"Saved cloned audio: {result.output_path}")
    print(
        "Output details: "
        f"speaker={result.speaker_duration_sec:.2f}s, "
        f"clone={result.output_duration_sec:.2f}s, "
        f"sample_rate={result.sample_rate}Hz"
    )


if __name__ == "__main__":
    main()
