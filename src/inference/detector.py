"""Inference utilities for fake-audio detection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from src.logger import get_logger
from src.models.cnn import DeepfakeCNN
from src.preprocessing.audio_features import AudioConfig, load_audio_resampled, to_mel_spec
from src.runtime_paths import bundle_root, models_dir, runtime_home

logger = get_logger("detector")
MODEL_CANDIDATES = (
    runtime_home() / "models" / "fake_audio_cnn.pth",
    models_dir() / "fake_audio_cnn.pth",
    bundle_root() / "fake_audio_cnn.pth",
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _segment_signal_score(segment: np.ndarray) -> float:
    if segment.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(segment**2)))
    peak = float(np.max(np.abs(segment)))
    return max(0.0, rms + (0.1 * peak))


def _candidate_starts(y: np.ndarray, *, target_len: int, segment_count: int) -> list[int]:
    if len(y) <= target_len:
        return [0]

    max_start = len(y) - target_len
    if segment_count <= 1:
        return [max_start // 2]

    starts: set[int] = set()

    even_starts = np.linspace(0, max_start, num=segment_count, dtype=np.int64).tolist()
    starts.update(int(start) for start in even_starts)

    dense_count = max(segment_count * 3, segment_count + 2)
    dense_starts = np.linspace(0, max_start, num=dense_count, dtype=np.int64).tolist()
    scored = []
    for start in dense_starts:
        start = int(start)
        seg = y[start : start + target_len]
        scored.append((_segment_signal_score(seg), start))

    top_k = min(segment_count, len(scored))
    starts.update(start for _, start in sorted(scored, reverse=True)[:top_k])
    return sorted(starts)


def find_model_path() -> str:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return str(path)
    return str(MODEL_CANDIDATES[0])


def load_model() -> Tuple[DeepfakeCNN, Dict]:
    model_path = find_model_path()
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = DeepfakeCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    logger.info("Loaded model from %s", model_path)
    return model, checkpoint


def analyze_audio(
    audio_path: str,
    model: DeepfakeCNN,
    config_dict: Dict,
    *,
    segment_count: int = 5,
) -> Tuple[str, float]:
    config = AudioConfig(
        sample_rate=config_dict.get("sample_rate", 22050),
        duration=config_dict.get("duration", 3.0),
        n_mels=config_dict.get("n_mels", 64),
        hop_length=config_dict.get("hop_length", 512),
        n_fft=config_dict.get("n_fft", 1024),
        target_shape=tuple(config_dict.get("target_shape", (64, 130))),
    )

    segment_count = int(config_dict.get("inference_segment_count", segment_count))
    segment_count = max(1, segment_count)

    # Load the full (resampled) waveform so we can evaluate multiple windows.
    # This improves accuracy when speech is not centered in the uploaded/recorded clip.
    y = load_audio_resampled(audio_path, config)
    target_len = int(config.sample_rate * config.duration)

    if y.size == 0:
        raise ValueError("Empty audio provided.")

    starts = _candidate_starts(y, target_len=target_len, segment_count=segment_count)

    probs_list = []
    weights = []
    with torch.no_grad():
        for start in starts:
            seg = y[start : start + target_len]
            if len(seg) < target_len:
                seg = np.pad(seg, (0, target_len - len(seg)), mode="constant")

            spec = to_mel_spec(seg.astype(np.float32), config)
            tensor = torch.tensor(spec).unsqueeze(0).unsqueeze(0).to(DEVICE)
            probs = torch.softmax(model(tensor), dim=1).squeeze(0)  # shape: (2,)
            probs_list.append(probs)
            weights.append(_segment_signal_score(seg))

    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    if float(weight_tensor.sum().item()) <= 1e-8:
        weight_tensor = torch.ones_like(weight_tensor)
    weight_tensor = weight_tensor / weight_tensor.sum()
    probs = (torch.stack(probs_list, dim=0) * weight_tensor.unsqueeze(1)).sum(dim=0)

    # Allow flexible label mapping (e.g. if user flips real/fake labels during training).
    class_names = config_dict.get(
        "class_names",
        ["HUMAN VOICE", "AI GENERATED VOICE"],  # index -> label
    )
    class_names = list(class_names)
    if len(class_names) != 2:
        class_names = ["HUMAN VOICE", "AI GENERATED VOICE"]

    pred_idx = int(probs.argmax().item())
    label = class_names[pred_idx]
    confidence = round(float(probs[pred_idx].item()) * 100.0, 2)
    return label, confidence
