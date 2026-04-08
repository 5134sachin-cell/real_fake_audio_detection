"""Helpers for evaluating arbitrary WAV folders or labeled manifests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import soundfile as sf

from src.inference.detector import analyze_audio

HUMAN_LABEL = "HUMAN VOICE"
AI_LABEL = "AI GENERATED VOICE"


@dataclass(frozen=True)
class EvalEntry:
    path: str
    expected_label: str | None = None


def normalize_expected_label(label: str | None) -> str | None:
    if label is None:
        return None

    normalized = str(label).strip().lower()
    if not normalized:
        return None
    if normalized in {"human", "real", "human voice", "0"}:
        return HUMAN_LABEL
    if normalized in {"ai", "fake", "synthetic", "generated", "ai generated voice", "1"}:
        return AI_LABEL
    raise ValueError(f"Unsupported label: {label!r}")


def collect_eval_entries_from_dir(
    folder: str,
    *,
    expected_label: str | None = None,
    recursive: bool = False,
) -> List[EvalEntry]:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Evaluation folder not found: {folder}")

    pattern = "**/*.wav" if recursive else "*.wav"
    normalized_label = normalize_expected_label(expected_label)
    return [
        EvalEntry(str(path), normalized_label)
        for path in sorted(folder_path.glob(pattern))
        if path.is_file()
    ]


def load_eval_entries_from_manifest(csv_path: str) -> List[EvalEntry]:
    manifest_path = Path(csv_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {csv_path}")

    entries: List[EvalEntry] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"path", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest {csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )

        for index, row in enumerate(reader, start=2):
            raw_path = (row.get("path") or "").strip()
            if not raw_path:
                continue

            normalized_label = normalize_expected_label(row.get("label"))
            if normalized_label is None:
                raise ValueError(
                    f"Manifest {csv_path} has an empty/invalid label on line {index}."
                )
            entries.append(EvalEntry(raw_path, normalized_label))
    return entries


def describe_entries(entries: Sequence[EvalEntry]) -> Dict[str, int]:
    counts = {"total": len(entries), "labeled": 0, "human": 0, "ai": 0, "unlabeled": 0}
    for entry in entries:
        if entry.expected_label is None:
            counts["unlabeled"] += 1
            continue
        counts["labeled"] += 1
        if entry.expected_label == HUMAN_LABEL:
            counts["human"] += 1
        elif entry.expected_label == AI_LABEL:
            counts["ai"] += 1
    return counts


def _read_audio_metadata(path: str) -> tuple[int, float]:
    data, sample_rate = sf.read(path, always_2d=False)
    duration_sec = len(data) / float(sample_rate) if sample_rate else 0.0
    return int(sample_rate), float(duration_sec)


def evaluate_entries(
    entries: Sequence[EvalEntry],
    *,
    model,
    config_dict,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for entry in entries:
        sample_rate, duration_sec = _read_audio_metadata(entry.path)
        predicted_label, confidence = analyze_audio(entry.path, model, config_dict)
        expected_label = entry.expected_label
        correct = (
            expected_label is not None and predicted_label == expected_label
        )
        rows.append(
            {
                "path": entry.path,
                "filename": Path(entry.path).name,
                "sample_rate": sample_rate,
                "duration_sec": round(duration_sec, 3),
                "expected_label": expected_label or "",
                "predicted_label": predicted_label,
                "confidence": round(float(confidence), 2),
                "correct": "" if expected_label is None else bool(correct),
            }
        )
    return rows


def summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total": len(rows),
        "predicted_human": 0,
        "predicted_ai": 0,
        "labeled_total": 0,
        "labeled_correct": 0,
        "expected_human_total": 0,
        "expected_human_correct": 0,
        "expected_ai_total": 0,
        "expected_ai_correct": 0,
    }

    for row in rows:
        predicted = row["predicted_label"]
        expected = row["expected_label"]
        correct = row["correct"]

        if predicted == HUMAN_LABEL:
            summary["predicted_human"] += 1
        elif predicted == AI_LABEL:
            summary["predicted_ai"] += 1

        if expected:
            summary["labeled_total"] += 1
            if bool(correct):
                summary["labeled_correct"] += 1
            if expected == HUMAN_LABEL:
                summary["expected_human_total"] += 1
                if bool(correct):
                    summary["expected_human_correct"] += 1
            elif expected == AI_LABEL:
                summary["expected_ai_total"] += 1
                if bool(correct):
                    summary["expected_ai_correct"] += 1

    labeled_total = int(summary["labeled_total"])
    labeled_correct = int(summary["labeled_correct"])
    summary["labeled_accuracy"] = (
        round((100.0 * labeled_correct / labeled_total), 2) if labeled_total else None
    )

    expected_human_total = int(summary["expected_human_total"])
    expected_ai_total = int(summary["expected_ai_total"])
    summary["expected_human_accuracy"] = (
        round((100.0 * int(summary["expected_human_correct"]) / expected_human_total), 2)
        if expected_human_total
        else None
    )
    summary["expected_ai_accuracy"] = (
        round((100.0 * int(summary["expected_ai_correct"]) / expected_ai_total), 2)
        if expected_ai_total
        else None
    )
    return summary


def write_rows_to_csv(rows: Iterable[Dict[str, object]], output_path: str) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "filename",
        "sample_rate",
        "duration_sec",
        "expected_label",
        "predicted_label",
        "confidence",
        "correct",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(output)
