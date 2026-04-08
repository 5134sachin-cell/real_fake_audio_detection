"""Evaluate an arbitrary WAV folder or labeled manifest with the detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector import load_model
from src.inference.evaluation import (
    collect_eval_entries_from_dir,
    evaluate_entries,
    load_eval_entries_from_manifest,
    summarize_rows,
    write_rows_to_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a custom WAV folder with the detector.")
    parser.add_argument(
        "--dir",
        dest="directory",
        default=None,
        help="Folder containing WAV files to evaluate.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional CSV with columns `path,label` for labeled evaluation.",
    )
    parser.add_argument(
        "--expected_label",
        choices=("human", "ai"),
        default=None,
        help="Expected label for every file in --dir.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan the folder for WAV files.",
    )
    parser.add_argument(
        "--csv_out",
        default=None,
        help="Optional CSV output path for per-file predictions.",
    )
    parser.add_argument(
        "--preview_count",
        type=int,
        default=20,
        help="How many file predictions to print before switching to summary only.",
    )
    args = parser.parse_args()

    if bool(args.directory) == bool(args.manifest):
        raise SystemExit("Use exactly one of --dir or --manifest.")

    model, config = load_model()
    if args.manifest:
        entries = load_eval_entries_from_manifest(args.manifest)
    else:
        entries = collect_eval_entries_from_dir(
            args.directory,
            expected_label=args.expected_label,
            recursive=args.recursive,
        )

    rows = evaluate_entries(entries, model=model, config_dict=config)
    summary = summarize_rows(rows)

    print("=" * 68)
    print("  External Detector Evaluation")
    print("=" * 68)
    print(f"Files evaluated: {summary['total']}")
    if summary["labeled_total"]:
        print(
            "Labeled accuracy: "
            f"overall={summary['labeled_accuracy']:.2f}% "
            f"human={summary['expected_human_accuracy']:.2f}% "
            f"ai={summary['expected_ai_accuracy']:.2f}%"
        )
    print(
        "Prediction split: "
        f"HUMAN={summary['predicted_human']} "
        f"AI={summary['predicted_ai']}"
    )

    printed = 0
    for row in rows:
        should_print = printed < args.preview_count
        if row["expected_label"] and row["correct"] is False:
            should_print = True
        if not should_print:
            continue

        expected = row["expected_label"] or "unlabeled"
        mark = ""
        if row["expected_label"]:
            mark = "PASS" if row["correct"] else "FAIL"
            mark = f"[{mark}] "
        print(
            f"{mark}{row['filename']:25s} -> {row['predicted_label']:19s} "
            f"({float(row['confidence']):6.2f}%) expected={expected}"
        )
        printed += 1

    if args.csv_out:
        output_path = write_rows_to_csv(rows, args.csv_out)
        print(f"CSV report: {output_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
