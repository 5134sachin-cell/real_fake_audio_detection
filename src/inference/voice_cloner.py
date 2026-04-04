"""Voice cloning service wrapper with safe fallback behavior."""

from __future__ import annotations

import datetime as dt
import importlib.util
import math
import os

import numpy as np
import soundfile as sf
from scipy import signal

from src.logger import get_logger

logger = get_logger("voice_cloner")


def _read_audio(path: str, *, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        gcd = math.gcd(int(sr), int(target_sr))
        y = signal.resample_poly(y, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return y.astype(np.float32), int(sr)


def _trim_silence(y: np.ndarray, *, top_db: float = 25.0) -> np.ndarray:
    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y)))
    if peak <= 1e-8:
        return y
    threshold = peak * (10.0 ** (-top_db / 20.0))
    active = np.flatnonzero(np.abs(y) >= threshold)
    if active.size == 0:
        return y
    return y[active[0] : active[-1] + 1]


def _speed_adjust(input_wav: str, output_wav: str, speed: float) -> None:
    y, sr = _read_audio(input_wav)
    if abs(speed - 1.0) > 1e-6:
        new_len = max(1, int(len(y) / float(speed)))
        src_idx = np.arange(len(y), dtype=np.float32)
        dst_idx = np.linspace(0, max(0, len(y) - 1), new_len, dtype=np.float32)
        y = np.interp(dst_idx, src_idx, y).astype(np.float32)
    sf.write(output_wav, y, sr)


class VoiceCloner:
    def __init__(self) -> None:
        self._tts = None

    def status(self) -> dict:
        if self._tts is not None:
            return {
                "ready": True,
                "message": "Voice cloning model is loaded.",
            }
        if importlib.util.find_spec("TTS") is None:
            return {
                "ready": False,
                "message": "Coqui TTS is not installed. Install dependencies to enable voice cloning.",
            }
        return {
            "ready": True,
            "message": "Voice cloning package detected. The model will load on the first request.",
        }

    def _ensure_model(self):
        if self._tts is not None:
            return
        try:
            from TTS.api import TTS  # optional heavy dependency
        except Exception as exc:
            raise RuntimeError("Coqui TTS is not installed. Install with `pip install TTS`.") from exc
        try:
            self._tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", gpu=False)
        except Exception as exc:
            raise RuntimeError(f"Unable to load the voice cloning model: {exc}") from exc
        logger.info("Loaded voice cloning model.")

    def clone(
        self,
        speaker_wav: str,
        text: str,
        speed: float = 1.0,
        language: str = "en",
        output_dir: str = "outputs",
    ) -> str:
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"Speaker audio not found: {speaker_wav}")
        if not text.strip():
            raise ValueError("Text must not be empty.")
        if speed < 0.7 or speed > 1.3:
            raise ValueError("Speed must be in [0.7, 1.3].")

        os.makedirs(output_dir, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(output_dir, f"raw_{stamp}.wav")
        output_path = os.path.join(output_dir, f"clone_{stamp}.wav")
        # Preprocess speaker wav: resample, mono, silence trim.
        # Many TTS cloning models are sensitive to input format/duration.
        speaker_processed_path = os.path.join(output_dir, f"speaker_{stamp}.wav")
        y, sr = _read_audio(speaker_wav, target_sr=22050)
        y = _trim_silence(y, top_db=25)
        if len(y) < int(sr * 1.5):
            raise ValueError("Speaker wav is too short after trimming. Use at least ~1.5 seconds of speech.")
        sf.write(speaker_processed_path, y, sr)

        self._ensure_model()
        self._tts.tts_to_file(
            text=text,
            speaker_wav=speaker_processed_path,
            language=language,
            file_path=raw_path,
        )
        _speed_adjust(raw_path, output_path, speed)
        return output_path
