"""Root entrypoint to run Flask API."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_path("app/app.py", run_name="__main__")

"""
specific training:
.\.venv\Scripts\Activate.ps1
python -m src.training.train_cnn --epochs 3 --max_real_files 1000 --max_fake_files 1000

full training:
.\.venv\Scripts\Activate.ps1
python -m src.training.train_cnn --epochs 20

medium training:
.\.venv\Scripts\Activate.ps1
python -m src.training.train_cnn --epochs 10 --max_real_files 3000 --max_fake_files 3000


.\.venv\Scripts\Activate.ps1
python scripts\verify_cnn.py
python scripts\verify_predictions.py
python scripts\test_endpoints.py


.\.venv\Scripts\Activate.ps1
python app.py


"""
