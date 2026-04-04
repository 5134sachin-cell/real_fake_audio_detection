"""Quick inference sanity test."""

import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector import analyze_audio, load_model

model, config = load_model()

print("=" * 50)
print("  CNN Inference Test")
print("=" * 50)

for f in glob.glob("data/real/*.wav")[:3]:
    label, conf = analyze_audio(f, model, config)
    print(f"  REAL  -> {os.path.basename(f):20s} {label} ({conf}%)")

for f in glob.glob("data/fake/*.wav")[:3]:
    label, conf = analyze_audio(f, model, config)
    print(f"  FAKE  -> {os.path.basename(f):20s} {label} ({conf}%)")

print("=" * 50)
