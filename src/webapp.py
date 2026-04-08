"""Flask app exposing a single-page UI plus detection/cloning APIs."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from src.inference.detector import analyze_audio, load_model
from src.inference.voice_cloner import VoiceCloner
from src.logger import get_logger
from src.runtime_paths import ensure_runtime_dirs, outputs_dir, static_dir, templates_dir, uploads_dir

logger = get_logger("api")

ensure_runtime_dirs()
TEMPLATE_DIR = templates_dir()
STATIC_DIR = static_dir()
OUTPUT_DIR = outputs_dir()
UPLOAD_DIR = uploads_dir()
ALLOWED_AUDIO_EXTENSIONS = {".wav"}

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

try:
    DETECTION_MODEL, DETECTION_CONFIG = load_model()
except Exception as exc:
    DETECTION_MODEL, DETECTION_CONFIG = None, None
    logger.warning("Detector model unavailable at startup: %s", exc)

VOICE_CLONER = VoiceCloner()


def build_status() -> dict:
    detector_message = (
        "Model loaded and ready for analysis."
        if DETECTION_MODEL is not None
        else "Detection model missing. Run training to enable analysis."
    )
    return {
        "detector": {
            "ready": DETECTION_MODEL is not None,
            "message": detector_message,
        },
        "clone": VOICE_CLONER.status(),
    }


def template_context(active_page: str, **extra) -> dict:
    return {
        "active_page": active_page,
        "system_status": build_status(),
        **extra,
    }


def error_response(message: str, status_code: int):
    return jsonify({"error": message}), status_code


def get_uploaded_file(*field_names: str):
    for field_name in field_names:
        file_storage = request.files.get(field_name)
        if file_storage is not None:
            return file_storage
    return None


def save_upload(file_storage, prefix: str) -> Path:
    filename = secure_filename(file_storage.filename or "")
    if not filename:
        raise ValueError("Select an audio file before submitting.")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio file. Use one of: {supported}.")

    saved_name = f"{prefix}_{uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_name
    file_storage.save(saved_path)
    return saved_path


@app.get("/health")
def health():
    return jsonify({"status": "ok", **build_status()})


@app.get("/")
def index():
    return render_template("index.html", **template_context("home"))


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return redirect(f"{url_for('index')}#detect-tool")

    if DETECTION_MODEL is None:
        return error_response("Detection model not loaded. Run training first.", 503)

    audio_file = get_uploaded_file("file", "audio")
    if audio_file is None:
        return error_response("Upload an audio file using the `file` field.", 400)

    try:
        audio_path = save_upload(audio_file, "detect")
    except ValueError as exc:
        return error_response(str(exc), 400)

    try:
        label, confidence = analyze_audio(str(audio_path), DETECTION_MODEL, DETECTION_CONFIG)
    except Exception as exc:
        logger.exception("Detection failed")
        return error_response(str(exc), 500)

    return jsonify(
        {
            "label": label,
            "confidence": confidence,
            "filename": audio_path.name,
        }
    )


@app.route("/clone", methods=["GET", "POST"])
def clone():
    if request.method == "GET":
        return redirect(f"{url_for('index')}#clone-tool")

    speaker_file = get_uploaded_file("speaker_audio", "speaker_wav")
    if speaker_file is None:
        return error_response("Upload a speaker sample using the `speaker_audio` field.", 400)

    text = request.form.get("text", "").strip()
    language = request.form.get("language", "en").strip() or "en"
    speed_raw = request.form.get("speed", "1.0").strip() or "1.0"

    if not text:
        return error_response("Enter the text you want to synthesize.", 400)

    try:
        speed = float(speed_raw)
    except ValueError:
        return error_response("Speed must be a numeric value between 0.7 and 1.3.", 400)

    try:
        speaker_path = save_upload(speaker_file, "speaker")
    except ValueError as exc:
        return error_response(str(exc), 400)

    try:
        clone_result = VOICE_CLONER.clone(
            speaker_wav=str(speaker_path),
            text=text,
            speed=speed,
            language=language,
            output_dir=str(OUTPUT_DIR),
        )
        output_path = Path(clone_result.output_path)
    except Exception as exc:
        logger.exception("Voice cloning failed")
        return error_response(str(exc), 500)

    response = send_file(
        output_path,
        mimetype="audio/wav",
        as_attachment=True,
        download_name=output_path.name,
        max_age=0,
    )
    response.headers["X-Clone-Speaker-Duration"] = f"{clone_result.speaker_duration_sec:.2f}"
    response.headers["X-Clone-Output-Duration"] = f"{clone_result.output_duration_sec:.2f}"
    response.headers["X-Clone-Sample-Rate"] = str(clone_result.sample_rate)
    response.headers["X-Clone-Reference-Clips"] = str(clone_result.reference_clip_count)
    response.headers["X-Clone-Reference-Sample-Rate"] = str(clone_result.reference_sample_rate)
    return response


def run_dev_server() -> None:
    app.run(debug=True)
