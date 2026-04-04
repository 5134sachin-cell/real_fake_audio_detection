"""Audio preprocessing and feature extraction utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    duration: float = 3.0
    n_mels: int = 64
    hop_length: int = 512
    n_fft: int = 1024
    target_shape: Tuple[int, int] = (64, 130)


BASE_DIR = Path(__file__).resolve().parents[2]


def _fit_audio_length(y: np.ndarray, config: AudioConfig, *, crop_mode: str) -> np.ndarray:
    if crop_mode not in {"random", "center", "start"}:
        raise ValueError(f"Unsupported crop_mode: {crop_mode}")

    target_len = int(config.sample_rate * config.duration)
    if len(y) >= target_len:
        if crop_mode == "random":
            start = random.randint(0, len(y) - target_len)
        elif crop_mode == "center":
            start = (len(y) - target_len) // 2
        else:
            start = 0
        y = y[start : start + target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return y.astype(np.float32)


def _mix_to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.float32)
    return y.mean(axis=1).astype(np.float32)


def _resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y.astype(np.float32)
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return signal.resample_poly(y, up, down).astype(np.float32)


def load_audio(path: str, config: AudioConfig, *, crop_mode: str = "random") -> np.ndarray:
    y, orig_sr = sf.read(path, always_2d=False)
    y = _mix_to_mono(np.asarray(y))
    y = _resample_audio(y, int(orig_sr), config.sample_rate)
    return _fit_audio_length(y, config, crop_mode=crop_mode)


def load_audio_resampled(path: str, config: AudioConfig) -> np.ndarray:
    """
    Load audio and resample to `config.sample_rate` but do NOT crop/pad.

    This is useful for inference strategies that evaluate multiple time windows
    (e.g., average predictions across left/center/right segments).
    """
    y, orig_sr = sf.read(path, always_2d=False)
    y = _mix_to_mono(np.asarray(y))
    y = _resample_audio(y, int(orig_sr), config.sample_rate)
    return y.astype(np.float32)


def augment_audio(y: np.ndarray, config: AudioConfig) -> np.ndarray:
    aug = y.astype(np.float32).copy()
    target_len = int(config.sample_rate * config.duration)

    speed_rate = random.uniform(0.9, 1.1)
    new_len = max(1, int(len(aug) * speed_rate))
    aug = np.interp(
        np.linspace(0, len(aug) - 1, new_len),
        np.arange(len(aug)),
        aug,
    ).astype(np.float32)

    gain = random.uniform(0.7, 1.3)
    aug = (aug * gain).astype(np.float32)

    noise_level = random.uniform(0.001, 0.004)
    aug = (aug + np.random.normal(0, noise_level, aug.shape)).astype(np.float32)

    if len(aug) >= target_len:
        start = random.randint(0, len(aug) - target_len)
        aug = aug[start : start + target_len]
    else:
        aug = np.pad(aug, (0, target_len - len(aug)), mode="constant").astype(np.float32)
    return aug


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def _mel_filter_bank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    freq_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(np.array([0.0], dtype=np.float32))[0]
    mel_max = _hz_to_mel(np.array([sample_rate / 2.0], dtype=np.float32))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float32)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, freq_bins - 1)

    filter_bank = np.zeros((n_mels, freq_bins), dtype=np.float32)
    for idx in range(1, n_mels + 1):
        left = bins[idx - 1]
        center = bins[idx]
        right = bins[idx + 1]

        if center <= left:
            center = min(freq_bins - 1, left + 1)
        if right <= center:
            right = min(freq_bins, center + 1)

        for freq in range(left, center):
            filter_bank[idx - 1, freq] = (freq - left) / max(1, center - left)
        for freq in range(center, right):
            filter_bank[idx - 1, freq] = (right - freq) / max(1, right - center)

    return filter_bank


def to_mel_spec(y: np.ndarray, config: AudioConfig) -> np.ndarray:
    _, _, stft = signal.stft(
        y,
        fs=config.sample_rate,
        window="hann",
        nperseg=config.n_fft,
        noverlap=config.n_fft - config.hop_length,
        nfft=config.n_fft,
        boundary=None,
        padded=False,
    )
    power_spec = np.abs(stft) ** 2
    mel_basis = _mel_filter_bank(config.sample_rate, config.n_fft, config.n_mels)
    mel = np.matmul(mel_basis, power_spec)
    log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))

    if log_mel.shape[1] < config.target_shape[1]:
        log_mel = np.pad(
            log_mel,
            ((0, 0), (0, config.target_shape[1] - log_mel.shape[1])),
        )
    else:
        log_mel = log_mel[:, : config.target_shape[1]]

    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel.astype(np.float32)


def collect_wav_files(folder: str) -> List[str]:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return []
    return [str(path) for path in sorted(folder_path.glob("*.wav"))]


def resolve_data_dirs() -> Tuple[str, str]:
    candidates: Sequence[Tuple[Path, Path]] = (
        (BASE_DIR / "data" / "real", BASE_DIR / "data" / "fake"),
        (BASE_DIR / "real_audios", BASE_DIR / "fake_audios"),
        (BASE_DIR / "voice models", BASE_DIR / "cloned_outputs"),
    )
    for real_dir, fake_dir in candidates:
        if not (real_dir.is_dir() and fake_dir.is_dir()):
            continue
        if collect_wav_files(str(real_dir)) and collect_wav_files(str(fake_dir)):
            return str(real_dir), str(fake_dir)
    # Fallback to first existing pair even if empty for clearer error messages upstream.
    for real_dir, fake_dir in candidates:
        if real_dir.is_dir() and fake_dir.is_dir():
            return str(real_dir), str(fake_dir)
    return str(BASE_DIR / "data" / "real"), str(BASE_DIR / "data" / "fake")
