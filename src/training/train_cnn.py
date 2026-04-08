"""Train DeepfakeCNN for fake-audio detection.

Key improvements vs the previous version:
- Do NOT preload thousands of WAVs into memory (faster startup; avoids "stuck" feeling).
- Use on-the-fly audio loading + augmentation inside the Dataset.
- Provide CLI args to limit dataset size for quick training/debugging.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.inference.evaluation import (
    AI_LABEL,
    HUMAN_LABEL,
    collect_eval_entries_from_dir,
    describe_entries,
    evaluate_entries,
    load_eval_entries_from_manifest,
    summarize_rows,
    write_rows_to_csv,
)
from src.logger import get_logger
from src.models.cnn import DeepfakeCNN
from src.preprocessing.audio_features import (
    AudioConfig,
    augment_audio,
    collect_wav_files,
    load_audio,
    resolve_data_dirs,
    to_mel_spec,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_PATH = Path("models/fake_audio_cnn.pth")
CHECKPOINT_DIR = Path("models/checkpoints")
DEFAULT_BATCH_SIZE = 16
# Higher defaults so `python -m src.training.train_cnn` uses more data without extra flags.
# Use `--max_real_files N --max_fake_files N` for quick/debug runs (e.g. 500).
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-4
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_SEED = 42
DEFAULT_AUGMENT_PROB = 0.7
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_MIN_DELTA = 0.1
DEFAULT_INFERENCE_SEGMENT_COUNT = 5
DEFAULT_BALANCE_STRATEGY = "truncate"
DEFAULT_SUPPLEMENTAL_REPEAT = 20
# None = no cap (balanced min(real, fake) after shuffle). Set ints to limit file count.
DEFAULT_MAX_REAL_FILES: int | None = None
DEFAULT_MAX_FAKE_FILES: int | None = None

logger = get_logger("train_cnn")


class AudioPathDataset(Dataset):
    """Dataset backed by file paths (avoids expensive preloading)."""

    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        config: AudioConfig,
        *,
        training: bool,
        augment_prob: float,
    ):
        self.samples = list(samples)
        self.config = config
        self.training = training
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        crop_mode = "random" if self.training else "center"
        y = load_audio(path, self.config, crop_mode=crop_mode)
        if self.training:
            # Augment both classes to improve robustness and reduce distribution shift.
            if random.random() < self.augment_prob:
                y = augment_audio(y, self.config)
        spec = to_mel_spec(y, self.config)
        return torch.tensor(spec).unsqueeze(0), torch.tensor(label, dtype=torch.long)


def _sample_balanced_files(
    real_dir: str,
    fake_dir: str,
    *,
    max_real_files: int | None,
    max_fake_files: int | None,
    balance_strategy: str,
    seed: int,
) -> Tuple[List[str], List[str]]:
    real_files = collect_wav_files(real_dir)
    fake_files = collect_wav_files(fake_dir)
    logger.info("Using real folder: %s", real_dir)
    logger.info("Using fake folder: %s", fake_dir)
    logger.info("Found real=%d fake=%d wav files", len(real_files), len(fake_files))

    if not real_files or not fake_files:
        raise RuntimeError(
            "Missing dataset files. Put WAVs into `data/real` and `data/fake` (or legacy folders)."
        )

    rng = random.Random(seed)
    rng.shuffle(real_files)
    rng.shuffle(fake_files)

    if max_real_files is not None:
        real_files = real_files[:max_real_files]
    if max_fake_files is not None:
        fake_files = fake_files[:max_fake_files]

    if balance_strategy == "all":
        if len(real_files) <= 1 or len(fake_files) <= 1:
            raise RuntimeError(
                f"Not enough data after limiting. real={len(real_files)} fake={len(fake_files)}."
            )
        return real_files, fake_files

    n = min(len(real_files), len(fake_files))
    if n <= 1:
        raise RuntimeError(f"Not enough data after limiting. Balanced count would be {n}.")
    return real_files[:n], fake_files[:n]


def _checkpoint_score(path: Path) -> float:
    if not path.exists():
        return -1.0
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        logger.warning("Could not read existing checkpoint score from %s: %s", path, exc)
        return -1.0
    return float(checkpoint.get("best_val_acc", -1.0))


def _make_checkpoint_payload(
    model: DeepfakeCNN,
    *,
    config: AudioConfig,
    best_val_acc: float,
    class_names: Sequence[str],
    flip_labels: bool,
    augment_prob: float,
    inference_segment_count: int,
    metadata: Dict[str, object],
) -> Dict[str, object]:
    return {
        "model_state": model.state_dict(),
        **asdict(config),
        "best_val_acc": best_val_acc,
        "class_names": list(class_names),
        "flip_labels": flip_labels,
        "augment_prob": augment_prob,
        "inference_segment_count": inference_segment_count,
        **metadata,
    }


def _slugify_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip().lower())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "external_eval"


def _maybe_prepare_external_eval_entries(
    *,
    external_eval_dir: str | None,
    external_eval_manifest: str | None,
    external_eval_expected_label: str | None,
    external_eval_recursive: bool,
) -> list:
    if external_eval_dir and external_eval_manifest:
        raise ValueError(
            "Use either --external_eval_dir or --external_eval_manifest, not both."
        )
    if not external_eval_dir and not external_eval_manifest:
        return []

    if external_eval_manifest:
        entries = load_eval_entries_from_manifest(external_eval_manifest)
    else:
        entries = collect_eval_entries_from_dir(
            external_eval_dir,
            expected_label=external_eval_expected_label,
            recursive=external_eval_recursive,
        )

    if not entries:
        raise RuntimeError("No WAV files found for external evaluation.")

    stats = describe_entries(entries)
    logger.info(
        "External evaluation loaded: total=%d labeled=%d human=%d ai=%d unlabeled=%d",
        stats["total"],
        stats["labeled"],
        stats["human"],
        stats["ai"],
        stats["unlabeled"],
    )
    return entries


def _run_external_eval(
    *,
    model: DeepfakeCNN,
    config: AudioConfig,
    class_names: Sequence[str],
    inference_segment_count: int,
    external_eval_entries,
    external_eval_name: str,
) -> None:
    if not external_eval_entries:
        return

    eval_config = {
        **asdict(config),
        "class_names": list(class_names),
        "inference_segment_count": inference_segment_count,
    }
    rows = evaluate_entries(external_eval_entries, model=model, config_dict=eval_config)
    summary = summarize_rows(rows)

    os.makedirs("outputs", exist_ok=True)
    report_path = write_rows_to_csv(
        rows,
        os.path.join("outputs", f"{_slugify_name(external_eval_name)}_predictions.csv"),
    )
    logger.info(
        "External eval `%s`: predicted_human=%d predicted_ai=%d report=%s",
        external_eval_name,
        int(summary["predicted_human"]),
        int(summary["predicted_ai"]),
        report_path,
    )

    if summary["labeled_total"]:
        logger.info(
            "External eval `%s` labeled accuracy: overall=%.2f%% human=%.2f%% ai=%.2f%% "
            "(correct=%d/%d)",
            external_eval_name,
            float(summary["labeled_accuracy"] or 0.0),
            float(summary["expected_human_accuracy"] or 0.0),
            float(summary["expected_ai_accuracy"] or 0.0),
            int(summary["labeled_correct"]),
            int(summary["labeled_total"]),
        )

    uncertain_rows = sorted(rows, key=lambda row: float(row["confidence"]))[:5]
    for row in uncertain_rows:
        expected = row["expected_label"] or "unlabeled"
        logger.info(
            "External eval sample: %s -> %s (%.2f%%, expected=%s)",
            row["filename"],
            row["predicted_label"],
            float(row["confidence"]),
            expected,
        )


def _load_supplemental_train_paths(
    *,
    supplemental_manifest: str | None,
    supplemental_repeat: int,
    real_label_idx: int,
    fake_label_idx: int,
) -> tuple[list[tuple[str, int]], Dict[str, int]]:
    if not supplemental_manifest:
        return [], {"entries": 0, "human": 0, "ai": 0, "repeat": 0, "added_samples": 0}

    entries = load_eval_entries_from_manifest(supplemental_manifest)
    repeat_count = max(1, int(supplemental_repeat))
    train_paths: list[tuple[str, int]] = []
    human_count = 0
    ai_count = 0
    for entry in entries:
        if entry.expected_label == HUMAN_LABEL:
            label_idx = real_label_idx
            human_count += 1
        elif entry.expected_label == AI_LABEL:
            label_idx = fake_label_idx
            ai_count += 1
        else:
            continue

        for _ in range(repeat_count):
            train_paths.append((entry.path, label_idx))

    stats = {
        "entries": len(entries),
        "human": human_count,
        "ai": ai_count,
        "repeat": repeat_count,
        "added_samples": len(train_paths),
    }
    logger.info(
        "Supplemental manifest loaded: entries=%d human=%d ai=%d repeat=%d added_train_samples=%d",
        stats["entries"],
        stats["human"],
        stats["ai"],
        stats["repeat"],
        stats["added_samples"],
    )
    return train_paths, stats


def _maybe_load_initial_checkpoint(model: DeepfakeCNN, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return

    init_path = Path(checkpoint_path)
    if not init_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(init_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    logger.info("Warm-started model weights from %s", init_path)


def train(
    *,
    max_real_files: int | None = None,
    max_fake_files: int | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    val_split: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
    flip_labels: bool = False,
    augment_prob: float = DEFAULT_AUGMENT_PROB,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    inference_segment_count: int = DEFAULT_INFERENCE_SEGMENT_COUNT,
    balance_strategy: str = DEFAULT_BALANCE_STRATEGY,
    promote_shared: bool = False,
    external_eval_dir: str | None = None,
    external_eval_manifest: str | None = None,
    external_eval_expected_label: str | None = None,
    external_eval_recursive: bool = False,
    external_eval_name: str = "external_eval",
    supplemental_manifest: str | None = None,
    supplemental_repeat: int = DEFAULT_SUPPLEMENTAL_REPEAT,
    init_from_checkpoint: str | None = None,
) -> str:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    external_eval_entries = _maybe_prepare_external_eval_entries(
        external_eval_dir=external_eval_dir,
        external_eval_manifest=external_eval_manifest,
        external_eval_expected_label=external_eval_expected_label,
        external_eval_recursive=external_eval_recursive,
    )

    real_dir, fake_dir = resolve_data_dirs()
    real_files, fake_files = _sample_balanced_files(
        real_dir,
        fake_dir,
        max_real_files=max_real_files,
        max_fake_files=max_fake_files,
        balance_strategy=balance_strategy,
        seed=seed,
    )

    # Label mapping (index -> class):
    # Without flipping: index 0=HUMAN, index 1=AI
    # With flipping:    index 0=AI,    index 1=HUMAN
    real_label_idx = 1 if flip_labels else 0
    fake_label_idx = 0 if flip_labels else 1
    class_names = ["AI GENERATED VOICE", "HUMAN VOICE"] if flip_labels else [
        "HUMAN VOICE",
        "AI GENERATED VOICE",
    ]

    # Create stratified (train/val) split over file paths (no audio preloading).
    paths = [(p, real_label_idx) for p in real_files] + [(p, fake_label_idx) for p in fake_files]
    labels = [lbl for _, lbl in paths]
    train_paths, val_paths = train_test_split(
        paths, test_size=val_split, random_state=seed, stratify=labels
    )
    supplemental_train_paths, supplemental_stats = _load_supplemental_train_paths(
        supplemental_manifest=supplemental_manifest,
        supplemental_repeat=supplemental_repeat,
        real_label_idx=real_label_idx,
        fake_label_idx=fake_label_idx,
    )
    train_paths = train_paths + supplemental_train_paths

    config = AudioConfig()
    train_ds = AudioPathDataset(train_paths, config, training=True, augment_prob=augment_prob)
    train_eval_ds = AudioPathDataset(train_paths, config, training=False, augment_prob=0.0)
    val_ds = AudioPathDataset(val_paths, config, training=False, augment_prob=0.0)

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Device: %s | train_batches~%d val_batches~%d | train_samples=%d val_samples=%d",
        device,
        len(train_loader),
        len(val_loader),
        len(train_paths),
        len(val_paths),
    )

    model = DeepfakeCNN().to(device)
    _maybe_load_initial_checkpoint(model, init_from_checkpoint)
    class_counts = torch.tensor(
        [
            sum(1 for _, label in train_paths if label == 0),
            sum(1 for _, label in train_paths if label == 1),
        ],
        dtype=torch.float32,
    )
    class_weights = class_counts.sum() / (class_counts.numel() * class_counts.clamp_min(1.0))
    logger.info(
        "Train class counts: idx0=%d idx1=%d | loss_weights=[%.3f, %.3f] | balance=%s",
        int(class_counts[0].item()),
        int(class_counts[1].item()),
        float(class_weights[0].item()),
        float(class_weights[1].item()),
        balance_strategy,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    shared_best_acc = _checkpoint_score(MODEL_PATH)
    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run_checkpoint_path = CHECKPOINT_DIR / (
        f"fake_audio_cnn_{run_stamp}_ep{epochs}_real{len(real_files)}_fake{len(fake_files)}.pth"
    )
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for specs, lbls in train_loader:
            specs, lbls = specs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for specs, lbls in train_eval_loader:
                specs, lbls = specs.to(device), lbls.to(device)
                logits = model(specs)
                preds = logits.argmax(dim=1)
                train_correct += (preds == lbls).sum().item()
                train_total += lbls.size(0)

        val_loss = 0.0
        correct = 0
        total = 0
        val_class_correct = [0, 0]
        val_class_total = [0, 0]
        with torch.no_grad():
            for specs, lbls in val_loader:
                specs, lbls = specs.to(device), lbls.to(device)
                logits = model(specs)
                val_loss += criterion(logits, lbls).item()
                preds = logits.argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
                for idx in range(2):
                    mask = lbls == idx
                    count = int(mask.sum().item())
                    if count:
                        val_class_total[idx] += count
                        val_class_correct[idx] += int((preds[mask] == lbls[mask]).sum().item())

        epoch_train_loss = running_loss / max(1, len(train_loader))
        epoch_val_loss = val_loss / max(1, len(val_loader))
        train_acc = 100.0 * train_correct / max(1, train_total)
        val_acc = 100.0 * correct / max(1, total)
        val_class_acc = [
            100.0 * val_class_correct[idx] / max(1, val_class_total[idx]) for idx in range(2)
        ]
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d/%d | lr=%.6f train_loss=%.4f val_loss=%.4f train_acc=%.2f%% val_acc=%.2f%% "
            "val_class_acc=[%s=%.2f%%, %s=%.2f%%]",
            epoch,
            epochs,
            current_lr,
            epoch_train_loss,
            epoch_val_loss,
            train_acc,
            val_acc,
            class_names[0],
            val_class_acc[0],
            class_names[1],
            val_class_acc[1],
        )
        scheduler.step(val_acc)

        if best_epoch == 0 or val_acc > (best_val_acc + min_delta):
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            metadata = {
                "train_samples": len(train_paths),
                "val_samples": len(val_paths),
                "real_files_used": len(real_files),
                "fake_files_used": len(fake_files),
                "seed": seed,
                "balance_strategy": balance_strategy,
                "best_epoch": best_epoch,
                "supplemental_manifest": supplemental_manifest or "",
                "supplemental_entries": supplemental_stats["entries"],
                "supplemental_human_entries": supplemental_stats["human"],
                "supplemental_ai_entries": supplemental_stats["ai"],
                "supplemental_repeat": supplemental_stats["repeat"],
                "supplemental_added_train_samples": supplemental_stats["added_samples"],
                "init_from_checkpoint": init_from_checkpoint or "",
            }
            checkpoint_payload = _make_checkpoint_payload(
                model,
                config=config,
                best_val_acc=best_val_acc,
                class_names=class_names,
                flip_labels=flip_labels,
                augment_prob=augment_prob,
                inference_segment_count=inference_segment_count,
                metadata=metadata,
            )
            torch.save(checkpoint_payload, run_checkpoint_path)
            logger.info("Saved run checkpoint to %s", run_checkpoint_path)

            should_promote_shared = (
                promote_shared
                or not MODEL_PATH.exists()
                or (
                    max_real_files is None
                    and max_fake_files is None
                    and best_val_acc >= (shared_best_acc + min_delta)
                )
            )
            if should_promote_shared:
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint_payload, MODEL_PATH)
                shared_best_acc = best_val_acc
                logger.info("Promoted checkpoint to shared model at %s", MODEL_PATH)
            else:
                logger.info(
                    "Preserved existing shared model (best_val_acc=%.2f%%). "
                    "Use `--promote_shared` if you want this run to replace %s.",
                    max(shared_best_acc, 0.0),
                    MODEL_PATH,
                )
            _run_external_eval(
                model=model,
                config=config,
                class_names=class_names,
                inference_segment_count=inference_segment_count,
                external_eval_entries=external_eval_entries,
                external_eval_name=external_eval_name,
            )
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            logger.info(
                "Early stopping at epoch %d after %d epochs without improvement. "
                "Best epoch=%d best_val_acc=%.2f%%",
                epoch,
                epochs_without_improvement,
                best_epoch,
                best_val_acc,
            )
            break

    os.makedirs("outputs", exist_ok=True)
    epochs_x = range(1, len(val_accs) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_x, train_losses, label="train_loss")
    plt.plot(epochs_x, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Training and validation loss")
    plt.subplot(1, 2, 2)
    plt.plot(epochs_x, train_accs, label="train_acc")
    plt.plot(epochs_x, val_accs, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Training and validation accuracy")
    plt.tight_layout()
    plt.savefig("outputs/training_metrics.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_x, train_accs, marker="o", markersize=3, label="Training accuracy")
    plt.plot(epochs_x, val_accs, marker="o", markersize=3, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Model accuracy vs epoch (fake vs human voice detection)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/accuracy_curve.png", dpi=150)
    plt.close()

    logger.info(
        "Training done. Best validation accuracy: %.2f%% at epoch %d | run_checkpoint=%s | shared_model=%s",
        best_val_acc,
        best_epoch,
        run_checkpoint_path,
        MODEL_PATH,
    )
    return str(run_checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepfakeCNN (fake vs human voice).")
    parser.add_argument(
        "--max_real_files",
        type=int,
        default=DEFAULT_MAX_REAL_FILES,
        help="Max real WAV files to use after shuffle (omit for no limit / full balanced set).",
    )
    parser.add_argument(
        "--max_fake_files",
        type=int,
        default=DEFAULT_MAX_FAKE_FILES,
        help="Max fake WAV files to use after shuffle (omit for no limit / full balanced set).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--val_split", type=float, default=DEFAULT_VAL_SPLIT, help="Validation split.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--flip_labels",
        action="store_true",
        help="If your dataset folders are inverted, flip the label indices during training.",
    )
    parser.add_argument(
        "--augment_prob",
        type=float,
        default=DEFAULT_AUGMENT_PROB,
        help="Probability of applying audio augmentation to a training sample.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Stop after this many non-improving epochs (0 disables early stopping).",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=DEFAULT_MIN_DELTA,
        help="Minimum validation-accuracy improvement needed to count as a new best checkpoint.",
    )
    parser.add_argument(
        "--inference_segment_count",
        type=int,
        default=DEFAULT_INFERENCE_SEGMENT_COUNT,
        help="How many segments inference should average for this checkpoint.",
    )
    parser.add_argument(
        "--balance_strategy",
        choices=("truncate", "all"),
        default=DEFAULT_BALANCE_STRATEGY,
        help="`truncate` keeps classes equally sized by dropping extras. `all` keeps every file and uses class-weighted loss.",
    )
    parser.add_argument(
        "--promote_shared",
        action="store_true",
        help="Force this run to replace models/fake_audio_cnn.pth even if it is a limited/debug run.",
    )
    parser.add_argument(
        "--external_eval_dir",
        type=str,
        default=None,
        help=(
            "Optional WAV folder to inspect whenever a new best checkpoint is found. "
            "Useful for held-out folders such as D:\\Final Year Project\\voice_sample."
        ),
    )
    parser.add_argument(
        "--external_eval_manifest",
        type=str,
        default=None,
        help="Optional CSV with columns `path,label` for labeled external evaluation.",
    )
    parser.add_argument(
        "--external_eval_expected_label",
        choices=("human", "ai"),
        default=None,
        help="Expected label for every file in --external_eval_dir. Omit for unlabeled inspection.",
    )
    parser.add_argument(
        "--external_eval_recursive",
        action="store_true",
        help="Recursively scan WAV files under --external_eval_dir.",
    )
    parser.add_argument(
        "--external_eval_name",
        type=str,
        default="external_eval",
        help="Short output name for outputs/<name>_predictions.csv.",
    )
    parser.add_argument(
        "--supplemental_manifest",
        type=str,
        default=None,
        help=(
            "Optional labeled CSV with columns `path,label` to inject custom hard examples "
            "directly into the training set."
        ),
    )
    parser.add_argument(
        "--supplemental_repeat",
        type=int,
        default=DEFAULT_SUPPLEMENTAL_REPEAT,
        help="How many times each supplemental manifest sample is repeated in the training set.",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path used to warm-start training instead of starting from random weights.",
    )
    args = parser.parse_args()

    train(
        max_real_files=args.max_real_files,
        max_fake_files=args.max_fake_files,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        flip_labels=args.flip_labels,
        augment_prob=args.augment_prob,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        inference_segment_count=args.inference_segment_count,
        balance_strategy=args.balance_strategy,
        promote_shared=args.promote_shared,
        external_eval_dir=args.external_eval_dir,
        external_eval_manifest=args.external_eval_manifest,
        external_eval_expected_label=args.external_eval_expected_label,
        external_eval_recursive=args.external_eval_recursive,
        external_eval_name=args.external_eval_name,
        supplemental_manifest=args.supplemental_manifest,
        supplemental_repeat=args.supplemental_repeat,
        init_from_checkpoint=args.init_from_checkpoint,
    )


if __name__ == "__main__":
    main()
