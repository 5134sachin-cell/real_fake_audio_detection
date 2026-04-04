"""Train DeepfakeCNN for fake-audio detection.

Key improvements vs the previous version:
- Do NOT preload thousands of WAVs into memory (faster startup; avoids "stuck" feeling).
- Use on-the-fly audio loading + augmentation inside the Dataset.
- Provide CLI args to limit dataset size for quick training/debugging.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.models.cnn import DeepfakeCNN
from src.preprocessing.audio_features import (
    AudioConfig,
    augment_audio,
    collect_wav_files,
    load_audio,
    resolve_data_dirs,
    to_mel_spec,
)
from src.logger import get_logger

MODEL_PATH = "models/fake_audio_cnn.pth"
DEFAULT_BATCH_SIZE = 16
# Higher defaults so `python -m src.training.train_cnn` uses more data without extra flags.
# Use `--max_real_files N --max_fake_files N` for quick/debug runs (e.g. 500).
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-4
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_SEED = 42
DEFAULT_AUGMENT_PROB = 0.7
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

    n = min(len(real_files), len(fake_files))
    if n <= 1:
        raise RuntimeError(f"Not enough data after limiting. Balanced count would be {n}.")

    return real_files[:n], fake_files[:n]


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
) -> str:
    random.seed(seed)
    torch.manual_seed(seed)

    real_dir, fake_dir = resolve_data_dirs()
    real_files, fake_files = _sample_balanced_files(
        real_dir,
        fake_dir,
        max_real_files=max_real_files,
        max_fake_files=max_fake_files,
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

    config = AudioConfig()
    train_ds = AudioPathDataset(train_paths, config, training=True, augment_prob=augment_prob)
    val_ds = AudioPathDataset(val_paths, config, training=False, augment_prob=augment_prob)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | train_batches~%d val_batches~%d", device, len(train_loader), len(val_loader))

    model = DeepfakeCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []

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
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for specs, lbls in val_loader:
                specs, lbls = specs.to(device), lbls.to(device)
                logits = model(specs)
                val_loss += criterion(logits, lbls).item()
                preds = logits.argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)

        epoch_train_loss = running_loss / max(1, len(train_loader))
        epoch_val_loss = val_loss / max(1, len(val_loader))
        val_acc = 100.0 * correct / max(1, total)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accs.append(val_acc)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f val_acc=%.2f%%",
            epoch,
            epochs,
            epoch_train_loss,
            epoch_val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    **asdict(config),
                    "best_val_acc": best_val_acc,
                    "class_names": class_names,  # index -> label
                    "flip_labels": flip_labels,
                    "augment_prob": augment_prob,
                },
                MODEL_PATH,
            )
            logger.info("Saved new best model to %s", MODEL_PATH)

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label="val_acc")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.tight_layout()
    plt.savefig("outputs/training_metrics.png")
    plt.close()

    logger.info("Training done. Best validation accuracy: %.2f%%", best_val_acc)
    return MODEL_PATH


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
    )


if __name__ == "__main__":
    main()
