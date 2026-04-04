"""
Debug helper: inspect the latest uploaded WAV in `outputs/uploads/`.

Prints:
- duration, peak, RMS
- per-window (multi-segment) predictions + probabilities
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.getcwd())

from src.inference.detector import load_model
from src.preprocessing.audio_features import AudioConfig, load_audio_resampled, to_mel_spec


def main() -> None:
    model, cfg = load_model()
    device = next(model.parameters()).device

    files = glob.glob("outputs/uploads/*.wav")
    if not files:
        raise SystemExit("No WAVs found in outputs/uploads/. Record something first.")

    files = sorted(files, key=lambda f: os.path.getmtime(f), reverse=True)
    wav_path = files[0]

    y, sr = sf.read(wav_path, always_2d=False)
    y = np.asarray(y, dtype=np.float32)

    duration_s = len(y) / float(sr) if sr else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0

    print(f"Latest upload: {wav_path}")
    print(f"orig_sr: {sr} | duration_s: {duration_s:.3f} | peak: {peak:.6f} | rms: {rms:.6f}")

    config = AudioConfig(
        sample_rate=cfg.get("sample_rate", 22050),
        duration=cfg.get("duration", 3.0),
        n_mels=cfg.get("n_mels", 64),
        hop_length=cfg.get("hop_length", 512),
        n_fft=cfg.get("n_fft", 1024),
        target_shape=tuple(cfg.get("target_shape", (64, 130))),
    )

    y2 = load_audio_resampled(wav_path, config)
    target_len = int(config.sample_rate * config.duration)

    max_start = max(0, len(y2) - target_len)
    starts = np.linspace(0, max_start, num=3, dtype=np.int64).tolist()
    starts = sorted(set(starts))

    class_names = cfg.get("class_names", ["HUMAN VOICE", "AI GENERATED VOICE"])
    if len(class_names) != 2:
        class_names = ["HUMAN VOICE", "AI GENERATED VOICE"]

    probs_all: list[np.ndarray] = []
    for s in starts:
        seg = y2[s : s + target_len]
        if len(seg) < target_len:
            seg = np.pad(seg, (0, target_len - len(seg)), mode="constant")

        spec = to_mel_spec(seg.astype(np.float32), config)
        t = torch.tensor(spec).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(t)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        probs_np = probs.detach().cpu().numpy()
        pred_idx = int(np.argmax(probs_np))
        probs_all.append(probs_np)

        print(
            f" window_start={int(s)} -> pred={class_names[pred_idx]} "
            f"probs=[{probs_np[0]:.4f}, {probs_np[1]:.4f}]"
        )

    probs_mean = np.mean(np.stack(probs_all, axis=0), axis=0)
    pred_idx = int(np.argmax(probs_mean))
    confidence = float(probs_mean[pred_idx]) * 100.0

    print(f"AVG prediction -> {class_names[pred_idx]} ({confidence:.2f}%)")


if __name__ == "__main__":
    main()

