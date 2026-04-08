# VoiceWorkbench

VoiceWorkbench is a combined speech application with two main features:

- fake-vs-human voice detection using a CNN classifier
- consent-based voice cloning using Coqui TTS

This repository now reflects the current working application, training pipeline, and Windows product build flow. Stale debug files and generated artifacts have been cleaned out.

## What The Application Does

### 1. Voice Detection

The detector accepts a `.wav` file and predicts either:

- `HUMAN VOICE`
- `AI GENERATED VOICE`

It uses a mel-spectrogram CNN model stored in `models/fake_audio_cnn.pth`.

### 2. Voice Cloning

The cloning tool accepts:

- a speaker reference `.wav`
- text to synthesize

It uses Coqui TTS with a safe fallback to `your_tts` when `xtts_v2` is not available in the installed package. The clone flow now preprocesses reference audio using model-aligned short `16 kHz` speaker segments for better voice matching.

## Current Architecture

### Web Layer

- `src/webapp.py`: main Flask application and routes
- `app/app.py`: compatibility Flask entrypoint
- `app.py`: root development launcher
- `templates/`: HTML templates
- `static/`: CSS assets

### Inference Layer

- `src/inference/detector.py`: detector model loading and multi-window inference
- `src/inference/voice_cloner.py`: cloning wrapper, text normalization, speaker preprocessing, model loading
- `src/inference/evaluation.py`: external folder and manifest evaluation helpers

### Training Layer

- `src/training/train_cnn.py`: detector training, checkpointing, external evaluation, supplemental adaptation
- `src/models/cnn.py`: CNN model definition
- `src/preprocessing/audio_features.py`: audio loading, augmentation, mel-spectrogram generation

### Product And Runtime Layer

- `product_launcher.py`: production launcher using Waitress with browser auto-open
- `src/runtime_paths.py`: runtime-safe path handling for source mode and packaged mode
- `scripts/build_windows.ps1`: Windows portable product build
- `packaging/VoiceWorkbenchLauncher.cs`: native Windows `.exe` launcher source
- `VoiceWorkbench.spec`: optional PyInstaller build spec

## Repository Layout

### Keep

- `app/`
- `configs/`
- `models/`
- `packaging/`
- `scripts/`
- `src/`
- `static/`
- `templates/`
- `app.py`
- `product_launcher.py`
- `requirements.txt`
- `requirements-build.txt`
- `Dockerfile`
- `.dockerignore`
- `.env.example`

### Generated Or Local-Only

These are created locally or during use and are intentionally ignored by Git:

- `.venv/`
- `.tts_cache/`
- `data/`
- `dist/`
- `logs/`
- `outputs/`
- `build/`
- `__pycache__/`

## Cleanup Done

The repo cleanup removed items that were stale, redundant, or purely debug-oriented:

- removed stale `PROJECT_REPORT.md`
- removed redundant `scripts/verify_cnn.py`
- removed debug-only `scripts/inspect_upload.py`
- removed generated `logs/`
- removed generated `outputs/`
- removed Python cache folders

The main CLI helpers were kept because they are still useful:

- `scripts/FakeAudioDetection.py`
- `scripts/VoiceClone.py`
- `scripts/verify_predictions.py`
- `scripts/evaluate_external_samples.py`
- `scripts/test_endpoints.py`
- `scripts/prewarm_clone_model.py`

## Safety Notes

- Only use cloning with speaker consent.
- Clearly label generated speech as synthetic.
- Do not use the cloning workflow for impersonation, fraud, or deceptive deployment.
- Delete uploaded speaker samples and generated outputs after demos when they are no longer needed.

## Local Development Setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run The App Locally

```powershell
python -m flask --app app/app.py run --host 0.0.0.0 --port 8000
```

Health check:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health | Select-Object -Expand Content
```

## Train The Detector

Quick balanced training:

```powershell
python -m src.training.train_cnn --epochs 10 --max_real_files 5000 --max_fake_files 5000 --balance_strategy truncate --promote_shared
```

Training with custom external monitoring:

```powershell
python -m src.training.train_cnn `
  --epochs 10 `
  --max_real_files 5000 `
  --max_fake_files 5000 `
  --balance_strategy truncate `
  --promote_shared `
  --external_eval_dir "D:\Final Year Project\voice_sample" `
  --external_eval_name voice_sample
```

## Evaluate Samples

Evaluate an unlabeled folder:

```powershell
python scripts\evaluate_external_samples.py `
  --dir "D:\Final Year Project\voice_sample" `
  --csv_out outputs\voice_sample_eval.csv
```

Evaluate a labeled manifest:

```powershell
python scripts\evaluate_external_samples.py `
  --manifest configs\external_eval_manifest.example.csv `
  --csv_out outputs\manifest_eval.csv
```

Quick detector validation:

```powershell
python scripts\verify_predictions.py --sample_limit 200 --preview_count 20
```

## Voice Cloning Notes

- `/health` reports package-ready vs model-verified clone status separately.
- The clone pipeline falls back to `tts_models/multilingual/multi-dataset/your_tts` when needed.
- Reference audio is now resampled to the speaker encoder rate and split into strong short clips for better matching.
- Very long technical prompts with symbols or numbered lists reduce quality; use short natural sentences for better results.

Example environment variables:

```powershell
$env:VOICE_CLONE_MODEL="tts_models/multilingual/multi-dataset/your_tts"
$env:VOICE_CLONE_CACHE_DIR=".tts_cache"
python app.py
```

## Windows Product Build

Build the portable Windows product:

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\build_windows.ps1
```

Output folder:

```text
dist\VoiceWorkbench\
```

Copy the whole `dist\VoiceWorkbench` folder to another Windows PC.

Run on the target PC by double-clicking:

- `VoiceWorkbench.exe`

Fallback launchers are also included:

- `Run VoiceWorkbench.bat`
- `Run VoiceWorkbench Hidden.vbs`

The portable build:

- does not require VS Code
- does not require a separate Python install
- stores runtime data under `dist\VoiceWorkbench\runtime`
- uses the bundled `.tts_cache` for offline clone model loading

Advanced option:

```powershell
.\scripts\build_windows.ps1 -BuildMode PyInstaller
```

Use the default `PortableRuntime` mode unless you specifically need a PyInstaller build.

## Docker

Build:

```powershell
docker build -t voiceworkbench .
```

Run:

```powershell
docker run --rm -p 8000:8000 voiceworkbench
```

Open from another machine using `http://<HOST-IP>:8000`.

## Project Report

Detailed application and cleanup documentation is available in:

- `docs/APPLICATION_REPORT.md`
