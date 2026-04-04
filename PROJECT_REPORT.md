# Merged Voice Project Report

Generated from the current codebase state on April 4, 2026.

## 1. Project Overview

### Project Title
Merged Voice Project: AI Voice Detection and Voice Cloning Web Application

### Project Summary
This project is a Flask-based application that combines two major speech-AI capabilities in one interface:

1. AI voice detection:
   The system classifies an input WAV audio file as either `HUMAN VOICE` or `AI GENERATED VOICE`.
2. Voice cloning:
   The system generates a cloned speech output from a speaker reference sample and input text.

The project includes:

- a single-page web interface
- backend APIs for detection and cloning
- a custom CNN-based audio classifier
- audio preprocessing and augmentation utilities
- model training scripts
- verification and endpoint test scripts
- support for local inference and generated output download

The codebase is modular and organized so that data processing, model definition, inference logic, web routes, and validation scripts are separated into clear folders.

## 2. Objectives

The main objectives of this project are:

- detect whether an uploaded speech sample is human or AI-generated
- generate cloned speech using a speaker-conditioned TTS pipeline
- provide a simple user-friendly browser interface for both tasks
- keep inference logic reusable from both the web app and CLI scripts
- support training and retraining of the detection model from local datasets

## 3. Core Features

### 3.1 Web Application Features

- Single-page dashboard UI
- Detector status and voice clone status display
- Audio upload for detection
- Microphone recording for detection
- Live waveform visualization during recording
- Audio preview before submission
- Inline error reporting
- Confidence meter for detection results
- Speaker audio upload for cloning
- Text input for synthesized speech
- Language selection for cloning
- Speed control for cloned output
- Downloadable generated WAV output

### 3.2 Backend/API Features

- `GET /` renders the main UI
- `GET /health` returns system readiness information
- `GET /detect` redirects to the detection section on the main page
- `POST /detect` performs fake-audio detection
- `GET /clone` redirects to the cloning section on the main page
- `POST /clone` performs voice cloning and returns a generated WAV file

### 3.3 Model and Training Features

- Custom CNN model for binary audio classification
- Balanced sampling of real and fake audio files
- Audio augmentation during training
- Mel-spectrogram-based features
- Multi-window averaging during inference for more stable prediction
- Automatic saving of the best validation model
- Training metrics plot generation

### 3.4 Utility and Validation Features

- CLI detection script
- CLI voice cloning script
- CNN sanity verification script
- detailed prediction verification script
- endpoint testing script
- uploaded audio inspection script

## 4. Repository Structure

```text
Merged_Voice_Project/
|-- app.py
|-- requirements.txt
|-- .gitignore
|-- PROJECT_REPORT.md
|-- app/
|   `-- app.py
|-- src/
|   |-- __init__.py
|   |-- logger.py
|   |-- inference/
|   |   |-- __init__.py
|   |   |-- detector.py
|   |   `-- voice_cloner.py
|   |-- models/
|   |   |-- __init__.py
|   |   `-- cnn.py
|   |-- preprocessing/
|   |   |-- __init__.py
|   |   `-- audio_features.py
|   `-- training/
|       |-- __init__.py
|       `-- train_cnn.py
|-- templates/
|   |-- base.html
|   `-- index.html
|-- static/
|   `-- style.css
|-- scripts/
|   |-- FakeAudioDetection.py
|   |-- VoiceClone.py
|   |-- verify_cnn.py
|   |-- verify_predictions.py
|   |-- test_endpoints.py
|   `-- inspect_upload.py
|-- models/
|   `-- fake_audio_cnn.pth
|-- outputs/
|   |-- training_metrics.png
|   |-- clone_*.wav
|   |-- raw_*.wav
|   |-- speaker_*.wav
|   `-- uploads/
`-- data/
    |-- real/
    `-- fake/
```

## 5. Detailed File and Module Analysis

## 5.1 Root Files

### `app.py`

This is the root launcher file. It simply delegates execution to `app/app.py` using `runpy`. Its role is to provide a short project entrypoint:

- keeps the main run command simple: `python app.py`
- avoids exposing implementation details at the root level

### `requirements.txt`

This file lists the project dependencies:

- `flask`
- `librosa`
- `matplotlib`
- `numpy`
- `requests`
- `scipy`
- `scikit-learn`
- `soundfile`
- `torch`
- `TTS`

These dependencies cover web serving, audio processing, visualization, machine learning, and TTS-based voice cloning.

### `.gitignore`

The `.gitignore` added for this project excludes:

- `.venv/`
- `data/`
- `outputs/`
- Python cache files
- environment files
- editor-specific folders

This is important because the dataset and generated outputs are large and should not normally be pushed to GitHub.

## 5.2 Web App Layer

### `app/app.py`

This is the main Flask application file. It is the backend integration point for the entire system.

#### Responsibilities

- initializes the Flask app
- sets template and static directories
- ensures output and upload directories exist
- loads the detection model at startup
- initializes the voice cloning service
- validates uploaded audio files
- exposes all web routes and API endpoints

#### Important Design Choices

- only `.wav` files are accepted
- maximum request size is limited to 20 MB
- detector and clone readiness are exposed through status cards in the UI
- errors are returned in structured JSON for frontend handling

#### Main Routes

##### `GET /health`

Returns a JSON object describing:

- overall app status
- detector readiness
- cloning readiness

##### `GET /`

Renders the main UI page.

##### `POST /detect`

Receives an uploaded audio file and:

- validates the file
- saves it in `outputs/uploads/`
- sends it to the detector
- returns a JSON response with:
  - label
  - confidence
  - saved filename

##### `POST /clone`

Receives:

- speaker reference WAV
- text to synthesize
- language code
- speed value

Then it:

- validates the input
- preprocesses the speaker file
- performs voice cloning
- returns the generated WAV file as a downloadable response

## 5.3 Frontend/UI Layer

### `templates/base.html`

This is the common base template. It provides:

- HTML skeleton
- title and description
- stylesheet import
- header and navigation
- footer

It gives the app a shared page structure and ensures consistency.

### `templates/index.html`

This is the main UI page and contains:

- project hero section
- system status cards
- detection tool section
- cloning tool section
- inline JavaScript for frontend interactivity

#### Detection UI Features

- WAV file upload
- file metadata display
- local audio preview
- microphone recording
- waveform canvas
- detection result label
- confidence percentage
- result meter

#### Voice Clone UI Features

- speaker sample upload
- speaker preview
- text input
- language input
- speed slider
- inline clone result preview
- downloadable WAV output

#### JavaScript Logic in `index.html`

The inline script handles:

- form submission using `fetch`
- JSON error handling
- dynamic UI feedback
- microphone capture
- Float32 to WAV conversion in the browser
- waveform drawing on a canvas
- automatic preview generation
- download URL creation for generated audio

### `static/style.css`

This file contains the full UI styling.

#### Styling Characteristics

- warm gradient background
- glass-like panels using blur
- responsive layout
- mobile-friendly sections
- separate visual styling for detection and clone result areas

The CSS was also cleaned to remove unused UI sections and development-style clutter from the visible page.

## 5.4 Inference Layer

### `src/inference/detector.py`

This file contains the fake-audio detection inference logic.

#### Main Responsibilities

- locate the trained model file
- load checkpoint and restore model weights
- build runtime audio configuration from checkpoint metadata
- analyze uploaded audio
- average predictions across multiple time windows

#### Key Method

##### `analyze_audio(...)`

This method:

1. loads and resamples audio
2. creates fixed-length windows
3. converts each window into a mel-spectrogram
4. runs the CNN on each segment
5. averages the output probabilities
6. returns the predicted class and confidence

#### Important Strength

Instead of analyzing only one central crop, the system can inspect multiple segments and average them. This makes detection more robust when speech is not centered in the clip.

### `src/inference/voice_cloner.py`

This file contains the voice cloning service wrapper.

#### Main Responsibilities

- speaker WAV preprocessing
- silence trimming
- resampling and mono conversion
- model lazy-loading for TTS
- cloned speech generation
- speed adjustment after synthesis

#### Voice Cloning Workflow

1. validate speaker file, text, speed, and output directory
2. read and normalize the speaker audio
3. trim silence from the speaker sample
4. ensure enough speech remains after trimming
5. load the TTS model if not already loaded
6. synthesize raw speech with the cloned voice
7. apply speed adjustment
8. return the final WAV path

#### Model Used

The code uses Coqui TTS with:

`tts_models/multilingual/multi-dataset/your_tts`

This is a speaker-conditioned multilingual voice cloning model.

## 5.5 Model Layer

### `src/models/cnn.py`

This file defines the `DeepfakeCNN` model.

#### Architecture Summary

- 3 convolution blocks
- BatchNorm after each convolution
- ReLU activation
- MaxPool after each block
- flattened dense classifier
- dropout regularization
- 2-output final layer for binary classification

#### Interpretation

The model is lightweight and suitable for mel-spectrogram classification. It is simple enough for a final-year project while still demonstrating a complete training and inference pipeline.

## 5.6 Preprocessing Layer

### `src/preprocessing/audio_features.py`

This is a central module in the project and contains most of the signal-processing logic.

#### Main Capabilities

- audio loading
- stereo to mono conversion
- resampling
- padding and cropping
- random augmentation
- mel filter-bank creation
- STFT-based mel-spectrogram generation
- dataset directory resolution

#### Training Preprocessing

For training, audio is:

- loaded from file
- converted to mono
- resampled
- cropped or padded to target duration
- optionally augmented
- converted to normalized log-mel spectrograms

#### Inference Preprocessing

For inference, the file can be loaded without immediate crop/pad so multiple segments can be evaluated.

#### Data Directory Resolution

The function `resolve_data_dirs()` checks multiple candidate folder pairs, but in the current repository the main active dataset is:

- `data/real`
- `data/fake`

## 5.7 Training Layer

### `src/training/train_cnn.py`

This file implements the training pipeline for the detection model.

#### Main Features

- balanced real/fake sampling
- stratified train/validation split
- on-the-fly audio loading
- on-the-fly augmentation
- checkpoint saving
- validation accuracy tracking
- training plot generation

#### Training Method

1. locate dataset folders
2. collect WAV paths from real and fake classes
3. shuffle files
4. balance both classes using the smaller class size
5. split into train and validation sets
6. load data lazily through a `Dataset`
7. train CNN using cross-entropy loss
8. validate after each epoch
9. save the best model
10. save a training metrics plot to `outputs/training_metrics.png`

#### Default Hyperparameters

- batch size: 16
- epochs: 20
- learning rate: `1e-4`
- validation split: `0.2`
- seed: `42`
- augmentation probability: `0.7`

#### Important Practical Observation

The full dataset is large and CPU training can take a long time. A smaller debug run was successfully executed with:

- `--epochs 3`
- `--max_real_files 1000`
- `--max_fake_files 1000`

## 5.8 Utility Layer

### `src/logger.py`

Provides a shared logger with a consistent timestamped log format for the project.

This keeps logs readable during:

- model loading
- training
- detection
- cloning

## 5.9 Scripts Layer

### `scripts/FakeAudioDetection.py`

CLI wrapper around the detector. Allows command-line classification of a WAV file.

### `scripts/VoiceClone.py`

CLI wrapper around the voice cloner. Allows generating cloned speech from the terminal.

### `scripts/verify_cnn.py`

Quick sanity check script:

- loads the model
- runs a few real samples
- runs a few fake samples
- prints labels and confidences

### `scripts/verify_predictions.py`

More detailed verification script:

- checks more real/fake files
- prints PASS/FAIL per sample
- prints model file size

### `scripts/test_endpoints.py`

API-level integration test:

- sends one file to `/detect`
- sends one file to `/clone`
- checks HTTP status and response behavior

This script was updated so it now fails gracefully if the Flask server is not running.

### `scripts/inspect_upload.py`

Debug helper for the latest uploaded WAV:

- reads duration
- computes peak and RMS
- runs multi-window prediction details
- prints average prediction

This is useful when debugging microphone or upload behavior.

## 6. Methodology

## 6.1 Fake Audio Detection Method

The detection pipeline uses supervised binary classification.

### Input

- WAV audio sample

### Preprocessing

- read audio
- convert to mono
- resample to target sample rate
- crop or pad to target duration
- convert to log-mel spectrogram
- normalize values

### Model

- custom CNN classifier

### Inference Strategy

- evaluate one or more time segments
- average class probabilities
- return top class with confidence

### Output Classes

- `HUMAN VOICE`
- `AI GENERATED VOICE`

## 6.2 Voice Cloning Method

The cloning pipeline uses a pretrained multilingual TTS model with speaker conditioning.

### Input

- speaker reference WAV
- synthesis text
- language code
- speed

### Preprocessing

- load speaker audio
- convert to mono
- resample to 22050 Hz
- trim silence
- verify minimum speech length

### Synthesis

- clone voice with Coqui TTS `your_tts`

### Post-processing

- apply speed adjustment
- save raw and final outputs

### Output

- downloadable WAV file

## 6.3 Training Method

The classifier training process uses:

- balanced class sampling
- random shuffling
- data augmentation
- train/validation split
- Adam optimizer
- cross-entropy loss
- best-checkpoint saving

This is a standard and valid deep-learning workflow for an undergraduate ML/audio project.

## 7. End-to-End Workflow

## 7.1 Detection Flow

1. user opens the web app
2. user uploads a WAV file or records audio
3. frontend previews the audio
4. file is sent to `POST /detect`
5. backend validates and stores the upload
6. detector preprocesses the signal
7. CNN predicts the class
8. UI displays label and confidence

## 7.2 Voice Cloning Flow

1. user uploads a speaker reference WAV
2. user enters text
3. user selects language and speed
4. form is sent to `POST /clone`
5. backend preprocesses speaker audio
6. TTS model generates speech
7. speed adjustment is applied
8. generated WAV is returned
9. UI previews and allows download

## 7.3 Training Flow

1. training script reads dataset directories
2. files are balanced and split
3. CNN is trained on spectrograms
4. best checkpoint is saved
5. metrics plot is generated
6. saved model is later loaded by the detector

## 8. Observed Validation Results

The following results were observed from actual runs on April 4, 2026.

## 8.1 CNN Sanity Verification

Sample results from `python scripts\verify_cnn.py`:

| Class Type | Sample File | Result | Confidence |
|---|---|---|---:|
| Real | `A_000_0_A.wav` | HUMAN VOICE | 92.60% |
| Real | `A_001_0_A.wav` | HUMAN VOICE | 99.62% |
| Real | `A_002_0_B.wav` | HUMAN VOICE | 97.87% |
| Fake | `A_10000_5_C.wav` | AI GENERATED VOICE | 100.00% |
| Fake | `A_10001_20_C.wav` | AI GENERATED VOICE | 99.54% |
| Fake | `A_10002_05_C.wav` | AI GENERATED VOICE | 100.00% |

These sample checks indicate that the detector pipeline was functioning correctly during validation.

## 8.2 Detailed Prediction Verification

Sample results from `python scripts\verify_predictions.py`:

- first 10 real files shown in the validation log: all passed
- first 10 fake files shown in the validation log: all passed
- model file size observed: `4,299,889 bytes`

This provides strong evidence that the end-to-end classifier path was working correctly on representative examples.

## 8.3 Endpoint Testing

Observed API results from `python scripts\test_endpoints.py` while the Flask app was running:

- `/detect` returned HTTP `200`
- `/clone` returned HTTP `200`
- cloned audio generation succeeded

This confirms that both APIs were working correctly through the web backend.

## 8.4 Training Experiment

A quick CPU training experiment was also run successfully:

- command:
  `python -m src.training.train_cnn --epochs 3 --max_real_files 1000 --max_fake_files 1000`
- dataset subset:
  `1000 real + 1000 fake`
- best validation accuracy:
  `76.25%`

This confirms that the training pipeline is operational, although this quick experiment should be treated as a debug or smaller-scale training run rather than the final best model benchmark.

## 9. Current Strengths

- clear modular code structure
- successful integration of detection and cloning in one app
- reusable backend logic from both UI and CLI
- audio preprocessing pipeline implemented from scratch
- real model training pipeline included
- test scripts available for validation
- practical user features such as waveform preview and download support
- cleaned single-page UI suitable for demos and submission

## 10. Current Limitations

- only WAV input is supported
- the web app uses Flask development server, not a production deployment server
- the frontend JavaScript is embedded inside `index.html` instead of a separate JS file
- there is no database or user management
- model versioning is manual, so retraining can overwrite the existing model file
- no formal unit-test framework such as `pytest` is included yet
- TTS model loading can be slow on the first cloning request
- large local datasets are required for meaningful retraining

## 11. Risks and Important Notes

- The detector model file path is fixed to `models/fake_audio_cnn.pth`. If retraining is performed, the previous model can be overwritten.
- Because the voice cloning model is heavy, the first clone request takes longer than later requests.
- This project is appropriate for local demonstration and academic submission, but production deployment would need stronger security, logging, scaling, and model management.

## 12. Suggested Improvements

### Technical Improvements

- separate frontend JavaScript into `static/app.js`
- add `README.md` for GitHub documentation
- pin exact dependency versions in `requirements.txt`
- add model version naming instead of overwriting a single `.pth` file
- add more structured evaluation metrics such as confusion matrix, precision, recall, and F1-score
- add test automation using `pytest`
- add configuration file support for paths and hyperparameters

### UI/UX Improvements

- add success notifications after detection and cloning
- add drag-and-drop audio upload
- add model/version badges in the UI
- show timing information for detection and cloning
- allow users to keep past results in a small history panel

### Research Improvements

- compare the custom CNN with stronger architectures
- evaluate on a formally split benchmark set
- add more diverse accents, noise conditions, and speaking styles
- study robustness against short clips and noisy recordings
- explore speaker verification or anti-spoofing baselines for comparison

## 13. How to Run the Project

### Install Dependencies

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the Web App

```powershell
python app.py
```

Then open:

`http://127.0.0.1:5000`

### Run Verification Scripts

```powershell
python scripts\verify_cnn.py
python scripts\verify_predictions.py
python scripts\test_endpoints.py
```

### Run Training

Full training:

```powershell
python -m src.training.train_cnn --epochs 20
```

Smaller debug training:

```powershell
python -m src.training.train_cnn --epochs 3 --max_real_files 1000 --max_fake_files 1000
```

## 14. Git and Upload Readiness

This project is now partially prepared for GitHub upload because:

- `.gitignore` has been added
- unnecessary local environment files are excluded
- dataset and outputs are ignored

Before uploading, it is recommended to also add:

- `README.md`
- screenshots of the UI
- example output images or short demo clips

## 15. Conclusion

This project is a complete applied AI audio system that combines:

- deepfake voice detection
- voice cloning
- web UI integration
- preprocessing and model training
- verification utilities

From a final-year project perspective, it is a strong submission because it demonstrates:

- machine learning model development
- signal processing
- inference pipeline design
- full-stack integration
- practical testing and validation

The repository is modular, functional, and already validated through successful detection, cloning, and endpoint tests. With a polished `README`, screenshots, and presentation material, it is suitable for GitHub publication, project demonstration, and academic reporting.
