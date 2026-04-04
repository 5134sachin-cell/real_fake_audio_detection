"""Detailed validation report over available real/fake datasets."""

import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector import analyze_audio, find_model_path, load_model

model, config = load_model()
model_path = find_model_path()

real_files = glob.glob("data/real/*.wav") + glob.glob("real_audios/*.wav")
fake_files = glob.glob("data/fake/*.wav") + glob.glob("fake_audios/*.wav")

print("=" * 60)
print("  CNN Inference Verification")
print("=" * 60)
print("\nREAL FILES (expected: HUMAN VOICE)")
for f in real_files[:10]:
    label, conf = analyze_audio(f, model, config)
    mark = "PASS" if "HUMAN" in label else "FAIL"
    print(f"  [{mark}] {os.path.basename(f):25} -> {label} ({conf}%)")

print("\nFAKE FILES (expected: AI GENERATED)")
for f in fake_files[:10]:
    label, conf = analyze_audio(f, model, config)
    mark = "PASS" if "AI" in label else "FAIL"
    print(f"  [{mark}] {os.path.basename(f):25} -> {label} ({conf}%)")

print(f"\nModel file: {model_path} ({os.path.getsize(model_path):,} bytes)")
print("=" * 60)
