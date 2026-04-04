import requests
import glob
import os
import sys

BASE_URL = "http://127.0.0.1:5000"

print("--- Testing /detect endpoint ---")
real_files = glob.glob("data/real/*.wav")
if not real_files:
    print("No real audio files found in data/real/.")
    sys.exit(1)

test_file = real_files[0]
print(f"Uploading {test_file} for detection...")

try:
    with open(test_file, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/detect", files=files, timeout=60)
except requests.exceptions.RequestException as exc:
    print(f"FAILED: Could not connect to {BASE_URL}. Start the Flask app first.")
    print(f"Reason: {exc}")
    sys.exit(1)

print(f"Status: {response.status_code}")
if response.ok and "HUMAN VOICE" in response.text:
    print("SUCCESS: Human voice detected.")
elif response.ok and "AI GENERATED" in response.text:
    print("FAILED: Expected Human voice, got AI.")
else:
    print("FAILED:", response.text[:200])

print("\n--- Testing /clone endpoint ---")
print(f"Uploading {test_file} as speaker for cloning...")
with open(test_file, 'rb') as f:
    files = {'speaker_audio': f}
    data = {'text': 'Testing the voice cloning system.', 'language': 'en', 'speed': '1.0'}
    response = requests.post(f"{BASE_URL}/clone", files=files, data=data, timeout=120)

print(f"Status: {response.status_code}")
if response.status_code == 200 and (
    "attachment" in response.headers.get("Content-Disposition", "")
    or "audio" in response.headers.get("Content-Type", "")
):
    print("SUCCESS: Cloned audio generated and downloaded.")
else:
    print("FAILED: Response:", response.text[:200])
