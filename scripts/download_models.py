"""
scripts/download_models.py
==========================
Download the trained model files from Hugging Face Hub.

USAGE:
    python scripts/download_models.py

WHAT IT DOES:
    Downloads two files into the models/ folder:
    - efficientnet_model.keras  (~109MB) — Main model used for predictions
    - model_config.json         (<1KB)   — Already included, but downloads fresh copy

WHY MODELS AREN'T IN THE REPO:
    GitHub has a 100MB file size limit. The trained EfficientNetB3 model is 109MB.
    We host it on Hugging Face Model Hub instead, which is purpose-built for this.
"""

import os
import sys

# Make sure the script can find the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Check that huggingface_hub is installed ────────────────────────────────────
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print()
    print("ERROR: The 'huggingface_hub' package is not installed.")
    print()
    print("Fix: Make sure your virtual environment is activated, then run:")
    print("     pip install -r requirements.txt")
    print()
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────
HF_REPO   = "Vicky25july2003/ai-image-classifier-model"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
FILES     = [
    ("efficientnet_model.keras", "Primary EfficientNetB3 model (~109MB)"),
    ("model_config.json",        "Model configuration (thresholds, layer names)"),
]

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print()
    print("=" * 55)
    print("  Downloading AI Model Files")
    print("=" * 55)
    print(f"  Source: huggingface.co/{HF_REPO}")
    print(f"  Saving to: {MODEL_DIR}")
    print()

    all_ok = True

    for fname, description in FILES:
        dest = os.path.join(MODEL_DIR, fname)

        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  ✅ {fname} already exists ({size_mb:.1f} MB) — skipping")
            continue

        print(f"  ⬇  Downloading: {fname}")
        print(f"     {description}")

        try:
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=fname,
                local_dir=MODEL_DIR,
            )
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✅ Done! ({size_mb:.1f} MB saved)")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            print()
            print("  Try downloading manually from:")
            print(f"  https://huggingface.co/{HF_REPO}")
            print(f"  Save the file to: {MODEL_DIR}/")
            all_ok = False

        print()

    print("=" * 55)
    if all_ok:
        print("  ✅ All files ready!")
        print()
        print("  Next step: python scripts/verify_setup.py")
    else:
        print("  ⚠️  Some downloads failed — see messages above.")
    print("=" * 55)
    print()

if __name__ == "__main__":
    main()
