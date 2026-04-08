"""Voice cloning service wrapper with safe fallback behavior."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from src.logger import get_logger
from src.runtime_paths import tts_cache_dir

logger = get_logger("voice_cloner")
DEFAULT_MODEL_CANDIDATES = (
    "tts_models/multilingual/multi-dataset/xtts_v2",
    "tts_models/multilingual/multi-dataset/your_tts",
)
_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}
_SUPPORTED_TTS_TEXT_RE = re.compile(r"[^A-Za-z\u00C0-\u024F\u0400-\u052F!'(),\-.:;? ]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _installed_tts_registry() -> set[str]:
    spec = importlib.util.find_spec("TTS")
    if spec is None or not spec.origin:
        return set()

    models_path = Path(spec.origin).resolve().parent / ".models.json"
    if not models_path.is_file():
        return set()

    with models_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    names: set[str] = set()
    for model_type, lang_map in data.items():
        if not isinstance(lang_map, dict):
            continue
        for lang, dataset_map in lang_map.items():
            if not isinstance(dataset_map, dict):
                continue
            for dataset, model_map in dataset_map.items():
                if not isinstance(model_map, dict):
                    continue
                for model_name in model_map:
                    names.add(f"{model_type}/{lang}/{dataset}/{model_name}")
    return names


def _configured_model_candidates() -> list[str]:
    configured = os.environ.get("VOICE_CLONE_MODEL", "").strip()
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    return list(DEFAULT_MODEL_CANDIDATES)


def _select_supported_candidates(candidates: list[str]) -> list[str]:
    installed = _installed_tts_registry()
    if not installed:
        return candidates

    supported = [candidate for candidate in candidates if candidate in installed]
    return supported or candidates


def _ensure_local_tts_cache(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TTS_HOME", str(cache_root))
    os.environ.setdefault("XDG_DATA_HOME", str(cache_root))

    try:
        import TTS.utils.generic_utils as generic_utils  # pylint: disable=import-outside-toplevel
        import TTS.utils.manage as manage  # pylint: disable=import-outside-toplevel
    except Exception:
        return

    def _patched_get_user_data_dir(appname: str) -> Path:
        return cache_root / appname

    generic_utils.get_user_data_dir = _patched_get_user_data_dir
    manage.get_user_data_dir = _patched_get_user_data_dir


def _model_cache_dir(cache_root: Path, model_name: str) -> Path:
    return cache_root / "tts" / model_name.replace("/", "--")


def _purge_broken_model_cache(cache_root: Path, model_name: str) -> bool:
    model_dir = _model_cache_dir(cache_root, model_name)
    if not model_dir.exists():
        return False

    try:
        has_files = any(child.is_file() for child in model_dir.rglob("*"))
    except Exception:
        has_files = False

    if has_files:
        return False

    shutil.rmtree(model_dir, ignore_errors=True)
    logger.warning("Removed broken empty voice model cache at %s", model_dir)
    return True


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


def _peak_normalize(y: np.ndarray, *, target_peak: float = 0.92) -> np.ndarray:
    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y)))
    if peak <= 1e-8:
        return y
    gain = min(target_peak / peak, 8.0)
    return (y * gain).astype(np.float32)


def _simple_denoise(y: np.ndarray, sr: int) -> np.ndarray:
    if y.size == 0:
        return y
    # Keep speech-focused band to reduce low rumble and very high hiss.
    b, a = signal.butter(4, [80.0, 7600.0], btype="bandpass", fs=float(sr))
    return signal.filtfilt(b, a, y).astype(np.float32)


def _duration_seconds(y: np.ndarray, sr: int) -> float:
    if sr <= 0:
        return 0.0
    return float(len(y) / float(sr))


def _energy_score(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    return rms + (0.15 * peak)


def _select_reference_segments(
    y: np.ndarray,
    sr: int,
    *,
    segment_duration_sec: float = 4.0,
    max_segments: int = 3,
) -> list[np.ndarray]:
    if y.size == 0 or sr <= 0:
        return [y]

    segment_len = max(1, int(sr * segment_duration_sec))
    if len(y) <= int(segment_len * 1.35):
        return [y]

    hop = max(1, segment_len // 2)
    max_start = len(y) - segment_len
    candidate_starts = list(range(0, max_start + 1, hop))
    if candidate_starts[-1] != max_start:
        candidate_starts.append(max_start)

    ranked: list[tuple[float, int]] = []
    for start in candidate_starts:
        seg = y[start : start + segment_len]
        ranked.append((_energy_score(seg), start))
    ranked.sort(reverse=True)

    chosen: list[tuple[float, int]] = []
    min_spacing = max(1, int(segment_len * 0.8))
    for score, start in ranked:
        if all(abs(start - existing_start) >= min_spacing for _, existing_start in chosen):
            chosen.append((score, start))
            if len(chosen) >= max_segments:
                break

    if not chosen:
        return [y[:segment_len]]

    chosen.sort(key=lambda item: item[1])
    return [y[start : start + segment_len] for _, start in chosen]


def _speaker_reference_sample_rate(tts_obj) -> int:
    speaker_manager = getattr(getattr(tts_obj.synthesizer, "tts_model", None), "speaker_manager", None)
    encoder_ap = getattr(speaker_manager, "encoder_ap", None)
    encoder_sr = getattr(encoder_ap, "sample_rate", None)
    if isinstance(encoder_sr, (int, float)) and encoder_sr > 0:
        return int(encoder_sr)

    tts_config = getattr(tts_obj.synthesizer, "tts_config", None)
    audio_config = getattr(tts_config, "audio", None)
    if isinstance(audio_config, dict):
        tts_sr = audio_config.get("sample_rate")
        if isinstance(tts_sr, (int, float)) and tts_sr > 0:
            return int(tts_sr)

    return 16000


def _speaker_reference_segment_seconds(tts_obj) -> float:
    speaker_manager = getattr(getattr(tts_obj.synthesizer, "tts_model", None), "speaker_manager", None)
    encoder_config = getattr(speaker_manager, "encoder_config", None)
    voice_len = getattr(encoder_config, "voice_len", None)
    if voice_len is None and isinstance(encoder_config, dict):
        voice_len = encoder_config.get("voice_len")
    if isinstance(voice_len, (int, float)) and voice_len > 0:
        return float(max(3.0, min(6.0, voice_len * 2.0)))
    return 4.0


def _sanitize_tts_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    replacements = {
        "/": " or ",
        "\\": " ",
        "&": " and ",
        "@": " at ",
        "%": " percent ",
        "#": " number ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    rebuilt: list[str] = []
    for char in text:
        if char.isdigit():
            rebuilt.append(f" {_DIGIT_WORDS[char]} ")
        else:
            rebuilt.append(char)
    text = "".join(rebuilt)
    text = _SUPPORTED_TTS_TEXT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


@dataclass
class CloneResult:
    output_path: str
    speaker_duration_sec: float
    output_duration_sec: float
    sample_rate: int
    reference_clip_count: int
    reference_sample_rate: int


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
        self._last_load_error: str | None = None
        self._cache_root = Path(
            os.environ.get("VOICE_CLONE_CACHE_DIR", str(tts_cache_dir()))
        )
        self._requested_candidates = _configured_model_candidates()
        self._model_candidates = _select_supported_candidates(self._requested_candidates)
        self._model_name = self._model_candidates[0]

    def status(self) -> dict:
        if self._tts is not None:
            return {
                "ready": True,
                "verified": True,
                "message": f"Voice cloning model is loaded ({self._model_name}).",
            }
        if importlib.util.find_spec("TTS") is None:
            return {
                "ready": False,
                "verified": False,
                "message": "Coqui TTS is not installed. Install dependencies to enable voice cloning.",
            }
        if self._requested_candidates != self._model_candidates:
            fallback_note = (
                f" Installed package does not expose {self._requested_candidates[0]!r}; "
                f"using compatible fallback {self._model_name!r} instead."
            )
        else:
            fallback_note = ""
        if self._last_load_error:
            return {
                "ready": True,
                "verified": False,
                "message": (
                    f"Voice cloning package is installed, but the last model load failed "
                    f"for {self._model_name}: {self._last_load_error}.{fallback_note}"
                ),
            }
        return {
            "ready": True,
            "verified": False,
            "message": (
                f"Voice cloning package detected, but the model is not verified yet "
                f"({self._model_name}). The first clone request may still fail if the "
                f"cached model or compatible TTS version is missing.{fallback_note}"
            ),
        }

    def _ensure_model(self):
        if self._tts is not None:
            return
        _ensure_local_tts_cache(self._cache_root)
        try:
            from TTS.api import TTS  # optional heavy dependency
        except Exception as exc:
            self._last_load_error = str(exc)
            raise RuntimeError("Coqui TTS is not installed. Install with `pip install TTS`.") from exc

        errors: list[str] = []
        for model_name in self._model_candidates:
            for attempt in range(2):
                try:
                    self._tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
                    self._model_name = model_name
                    self._last_load_error = None
                    logger.info(
                        "Loaded voice cloning model: %s | cache=%s",
                        self._model_name,
                        self._cache_root,
                    )
                    return
                except Exception as exc:
                    message = str(exc)
                    should_retry = (
                        attempt == 0
                        and "Model file not found in the output path" in message
                        and _purge_broken_model_cache(self._cache_root, model_name)
                    )
                    if should_retry:
                        logger.warning(
                            "Retrying voice cloning model download after purging broken cache for %s",
                            model_name,
                        )
                        continue
                    errors.append(f"{model_name}: {exc}")
                    logger.warning("Failed loading voice cloning model %s: %s", model_name, exc)
                    break

        self._last_load_error = " | ".join(errors)
        raise RuntimeError(f"Unable to load the voice cloning model: {self._last_load_error}")

    def clone(
        self,
        speaker_wav: str,
        text: str,
        speed: float = 1.0,
        language: str = "en",
        output_dir: str = "outputs",
    ) -> CloneResult:
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"Speaker audio not found: {speaker_wav}")
        if not text.strip():
            raise ValueError("Text must not be empty.")
        if speed < 0.7 or speed > 1.3:
            raise ValueError("Speed must be in [0.7, 1.3].")

        self._ensure_model()
        os.makedirs(output_dir, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(output_dir, f"raw_{stamp}.wav")
        output_path = os.path.join(output_dir, f"clone_{stamp}.wav")
        clean_text = _sanitize_tts_text(text)
        if not clean_text:
            raise ValueError("Text becomes empty after cleaning unsupported characters. Use plain words and punctuation.")
        if clean_text != text.strip():
            logger.info(
                "Normalized clone text for TTS: original_chars=%d cleaned_chars=%d",
                len(text.strip()),
                len(clean_text),
            )

        # Align reference preprocessing to the speaker encoder instead of forcing 22.05 kHz.
        reference_sr = _speaker_reference_sample_rate(self._tts)
        reference_segment_sec = float(
            os.environ.get(
                "VOICE_CLONE_REF_SEGMENT_SEC",
                f"{_speaker_reference_segment_seconds(self._tts):.2f}",
            )
        )
        max_reference_clips = max(1, int(os.environ.get("VOICE_CLONE_MAX_REF_SEGMENTS", "3")))

        y, sr = _read_audio(speaker_wav, target_sr=reference_sr)
        y = _simple_denoise(y, sr)
        y = _trim_silence(y, top_db=25)
        y = _peak_normalize(y, target_peak=0.9)
        if len(y) < int(sr * 1.5):
            raise ValueError("Speaker wav is too short after trimming. Use at least ~1.5 seconds of speech.")

        reference_segments = _select_reference_segments(
            y,
            sr,
            segment_duration_sec=reference_segment_sec,
            max_segments=max_reference_clips,
        )
        speaker_inputs: list[str] = []
        if len(reference_segments) == 1:
            speaker_processed_path = os.path.join(output_dir, f"speaker_{stamp}.wav")
            sf.write(speaker_processed_path, reference_segments[0], sr)
            speaker_inputs = [speaker_processed_path]
        else:
            for idx, segment_y in enumerate(reference_segments, start=1):
                segment_path = os.path.join(output_dir, f"speaker_{stamp}_{idx:02d}.wav")
                sf.write(segment_path, segment_y, sr)
                speaker_inputs.append(segment_path)
        speaker_duration_sec = sum(_duration_seconds(segment_y, sr) for segment_y in reference_segments)
        logger.info(
            "Using %d speaker reference clip(s) at %d Hz for clone generation.",
            len(speaker_inputs),
            sr,
        )

        self._tts.tts_to_file(
            text=clean_text,
            speaker_wav=speaker_inputs if len(speaker_inputs) > 1 else speaker_inputs[0],
            language=language,
            file_path=raw_path,
        )
        _speed_adjust(raw_path, output_path, speed)
        out_y, out_sr = _read_audio(output_path)
        out_y = _trim_silence(out_y, top_db=42)
        out_y = _peak_normalize(out_y, target_peak=0.95)
        sf.write(output_path, out_y, out_sr)
        return CloneResult(
            output_path=output_path,
            speaker_duration_sec=speaker_duration_sec,
            output_duration_sec=_duration_seconds(out_y, out_sr),
            sample_rate=out_sr,
            reference_clip_count=len(speaker_inputs),
            reference_sample_rate=sr,
        )
