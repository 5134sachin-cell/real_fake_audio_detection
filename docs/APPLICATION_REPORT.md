# VoiceWorkbench Application Report

## 1. Application Summary

VoiceWorkbench is a two-part audio application:

- a detector that classifies speech as human or AI-generated
- a consent-based voice cloning tool that synthesizes text using a reference speaker sample

The project now includes:

- a browser-based Flask interface
- a CNN-based detector training and inference pipeline
- a Coqui TTS cloning pipeline
- a Windows portable product build
- documentation aligned to the current codebase

## 2. High-Level Architecture

### Frontend

- `templates/base.html`
- `templates/index.html`
- `static/style.css`

The frontend is a single-page UI with two tools:

- detection
- cloning

### Backend

- `src/webapp.py`
- `app/app.py`
- `app.py`

The backend exposes:

- `GET /`
- `GET /health`
- `POST /detect`
- `POST /clone`

### Detection Stack

- `src/inference/detector.py`
- `src/models/cnn.py`
- `src/preprocessing/audio_features.py`
- `src/training/train_cnn.py`
- `models/fake_audio_cnn.pth`

Workflow:

1. load audio
2. resample and convert to mel features
3. evaluate multiple windows
4. aggregate probabilities
5. return label and confidence

### Cloning Stack

- `src/inference/voice_cloner.py`
- bundled Coqui TTS model cache under `.tts_cache/`

Workflow:

1. validate reference WAV and text
2. normalize text for supported TTS characters
3. resample reference audio to the model-aligned speaker encoder rate
4. select short strong speaker segments
5. synthesize output with Coqui TTS
6. return downloadable WAV

### Product Build Stack

- `product_launcher.py`
- `src/runtime_paths.py`
- `scripts/build_windows.ps1`
- `packaging/VoiceWorkbenchLauncher.cs`

The Windows product build packages:

- the app code
- the Python runtime
- model files
- TTS cache

This allows the app to run on another Windows PC without VS Code or a separate Python install.

## 3. Repository Analysis

### Core Files To Keep

- `src/`
- `app/`
- `templates/`
- `static/`
- `models/`
- `configs/`
- `scripts/build_windows.ps1`
- `scripts/prewarm_clone_model.py`
- `scripts/FakeAudioDetection.py`
- `scripts/VoiceClone.py`
- `scripts/verify_predictions.py`
- `scripts/evaluate_external_samples.py`
- `scripts/test_endpoints.py`
- `product_launcher.py`
- `requirements.txt`
- `requirements-build.txt`
- `Dockerfile`
- `VoiceWorkbench.spec`

### Generated Or Local-Only Folders

These folders are valid during local use, but they should not be treated as source files:

- `.venv/`
- `.tts_cache/`
- `data/`
- `outputs/`
- `logs/`
- `dist/`
- `build/`
- `__pycache__/`

## 4. Cleanup Performed

### Removed

- `PROJECT_REPORT.md`
  Reason: stale and no longer matched the real repository state

- `scripts/verify_cnn.py`
  Reason: redundant with `scripts/verify_predictions.py`

- `scripts/inspect_upload.py`
  Reason: debug-only helper, not part of the supported workflow

- generated `outputs/`
  Reason: runtime artifacts only

- generated `logs/`
  Reason: runtime artifacts only

- all `__pycache__/`
  Reason: Python cache artifacts

### Kept Intentionally

- `scripts/FakeAudioDetection.py`
  Reason: useful lightweight CLI entrypoint for the detector

- `scripts/VoiceClone.py`
  Reason: useful lightweight CLI entrypoint for cloning

- `scripts/test_endpoints.py`
  Reason: still useful as a smoke test for the running web app

- `dist/VoiceWorkbench/`
  Reason: portable product output for end-user distribution

## 5. Application Workflow

### Detection Flow

1. the user uploads or records a WAV file
2. the backend saves it temporarily
3. the detector loads the trained checkpoint
4. the audio is split into useful windows
5. the CNN predicts real vs fake
6. the UI shows the label and confidence

### Cloning Flow

1. the user uploads a speaker sample
2. the user enters a short sentence
3. the backend cleans the text
4. the reference audio is converted into short speaker-focused segments
5. the TTS model synthesizes speech
6. the generated WAV is returned for playback/download

### Training Flow

1. real and fake WAV files are collected from dataset folders
2. the dataset is balanced
3. the audio is augmented and converted to mel features
4. the CNN is trained and validated
5. the best checkpoint is promoted to `models/fake_audio_cnn.pth`
6. optional external evaluation can monitor custom voice samples during training

## 6. Product Workflow

The recommended product mode is the Windows portable runtime build.

Build:

```powershell
.\scripts\build_windows.ps1
```

Deliver:

- copy `dist\VoiceWorkbench` to another Windows PC

Run:

- double-click `VoiceWorkbench.exe`

Result:

- the server starts locally
- the browser opens automatically
- runtime outputs are written inside `dist\VoiceWorkbench\runtime`

## 7. Important Notes

- The detector model is part of the source/runtime package and must remain available.
- The TTS cache is needed for offline clone demos.
- `dist/` is a release artifact, not source code.
- `data/` is intentionally ignored because datasets are local and large.

## 8. Final Repo State

After cleanup, the repository is now centered around:

- one application architecture
- one active detector training pipeline
- one active cloning workflow
- one clear Windows product build path
- one current README
- one current application report

This makes the project easier to understand, easier to demonstrate, and safer to maintain.
