"""Detailed validation report over available real/fake datasets."""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector import analyze_audio, find_model_path, load_model


def _sample_files(patterns: list[str], sample_limit: int | None, seed: int) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    if sample_limit is None or sample_limit >= len(files):
        return files

    rng = random.Random(seed)
    sampled = files[:]
    rng.shuffle(sampled)
    return sorted(sampled[:sample_limit])


def _evaluate_group(
    files: list[str],
    *,
    expected_label_fragment: str,
    label: str,
    model,
    config,
    preview_count: int,
) -> tuple[int, int]:
    print(f"\n{label} FILES (expected: {expected_label_fragment})")
    correct = 0
    for index, path in enumerate(files, start=1):
        pred_label, conf = analyze_audio(path, model, config)
        is_pass = expected_label_fragment in pred_label
        correct += int(is_pass)
        if index <= preview_count or not is_pass:
            mark = "PASS" if is_pass else "FAIL"
            print(f"  [{mark}] {os.path.basename(path):25} -> {pred_label} ({conf}%)")
    return correct, len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the detector on real/fake WAV files.")
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=100,
        help="Max files per class to evaluate. Omit or use a large value to test all files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--preview_count",
        type=int,
        default=10,
        help="How many predictions per class to print before switching to only failures.",
    )
    args = parser.parse_args()

    model, config = load_model()
    model_path = find_model_path()

    real_files = _sample_files(["data/real/*.wav", "real_audios/*.wav"], args.sample_limit, args.seed)
    fake_files = _sample_files(["data/fake/*.wav", "fake_audios/*.wav"], args.sample_limit, args.seed + 1)

    print("=" * 60)
    print("  CNN Inference Verification")
    print("=" * 60)
    print(f"Model file: {model_path} ({os.path.getsize(model_path):,} bytes)")
    print(
        f"Evaluating real={len(real_files)} fake={len(fake_files)} "
        f"(sample_limit={args.sample_limit}, seed={args.seed})"
    )

    real_correct, real_total = _evaluate_group(
        real_files,
        expected_label_fragment="HUMAN",
        label="REAL",
        model=model,
        config=config,
        preview_count=args.preview_count,
    )
    fake_correct, fake_total = _evaluate_group(
        fake_files,
        expected_label_fragment="AI",
        label="FAKE",
        model=model,
        config=config,
        preview_count=args.preview_count,
    )

    overall_correct = real_correct + fake_correct
    overall_total = real_total + fake_total
    real_acc = 100.0 * real_correct / max(1, real_total)
    fake_acc = 100.0 * fake_correct / max(1, fake_total)
    overall_acc = 100.0 * overall_correct / max(1, overall_total)

    print("\nSummary")
    print(f"  REAL accuracy : {real_correct}/{real_total} = {real_acc:.2f}%")
    print(f"  FAKE accuracy : {fake_correct}/{fake_total} = {fake_acc:.2f}%")
    print(f"  OVERALL       : {overall_correct}/{overall_total} = {overall_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
