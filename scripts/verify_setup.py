"""
scripts/verify_setup.py
========================
Check that everything is installed and ready before running the app.

USAGE:
    python scripts/verify_setup.py

WHAT IT CHECKS:
    - Python version (3.10+ required)
    - All required packages are installed
    - Model files exist
    - Runtime folders exist
    - .env file exists (optional but recommended)
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PASS  = "✅"
FAIL  = "❌"
WARN  = "⚠️ "
ARROW = "   →"

errors   = []
warnings = []

def check(label, condition, fail_msg=None, warn=False):
    if condition:
        print(f"  {PASS} {label}")
    elif warn:
        print(f"  {WARN} {label}")
        if fail_msg:
            print(f"{ARROW} {fail_msg}")
        warnings.append(label)
    else:
        print(f"  {FAIL} {label}")
        if fail_msg:
            print(f"{ARROW} {fail_msg}")
        errors.append(label)

print()
print("=" * 55)
print("  AI Image Classifier — Setup Check")
print("=" * 55)

# ── Python version ─────────────────────────────────────────────────────────────
print()
print("Python version:")
v = sys.version_info
check(
    f"Python {v.major}.{v.minor}.{v.micro}",
    v.major == 3 and v.minor >= 10,
    "Need Python 3.10+. Download from: python.org/downloads"
)

# ── Required packages ──────────────────────────────────────────────────────────
print()
print("Required packages:")

packages = [
    ("flask",           "Flask",            "pip install flask==3.0.0"),
    ("tensorflow",      "TensorFlow",       "pip install tensorflow==2.16.1"),
    ("numpy",           "NumPy",            "pip install numpy==1.26.4"),
    ("cv2",             "OpenCV",           "pip install opencv-python-headless==4.8.1.78"),
    ("PIL",             "Pillow",           "pip install pillow==10.1.0"),
    ("matplotlib",      "Matplotlib",       "pip install matplotlib==3.8.2"),
    ("sklearn",         "scikit-learn",     "pip install scikit-learn==1.3.2"),
    ("huggingface_hub", "Hugging Face Hub", "pip install huggingface_hub"),
    ("gunicorn",        "Gunicorn",         "pip install gunicorn==21.2.0"),
    ("flask_mail",      "Flask-Mail",       "pip install flask-mail==0.10.0"),
]

for module, display, install_cmd in packages:
    try:
        mod = __import__(module)
        ver = getattr(mod, "__version__", "?")
        check(f"{display} ({ver})", True)
    except ImportError:
        check(
            f"{display} NOT installed",
            False,
            f"Run: pip install -r requirements.txt   (or: {install_cmd})"
        )

# ── Model files ────────────────────────────────────────────────────────────────
print()
print("Model files:")

model_dir = os.path.join(PROJECT_ROOT, "models")

enb3_path   = os.path.join(model_dir, "efficientnet_model.keras")
cnn_path    = os.path.join(model_dir, "cnn_model.keras")
config_path = os.path.join(model_dir, "model_config.json")

if os.path.exists(enb3_path):
    size = os.path.getsize(enb3_path) / (1024 * 1024)
    check(f"efficientnet_model.keras ({size:.1f} MB)", size > 50,
          "File exists but seems too small — may be corrupted. Re-run download_models.py")
else:
    check("efficientnet_model.keras — NOT FOUND", False,
          "Run: python scripts/download_models.py")

if os.path.exists(cnn_path):
    size = os.path.getsize(cnn_path) / (1024 * 1024)
    check(f"cnn_model.keras ({size:.1f} MB)", True)
else:
    check("cnn_model.keras not found (optional fallback model)", True, warn=True)

check(
    "model_config.json",
    os.path.exists(config_path),
    "File missing — re-run: python scripts/download_models.py"
)

# ── Runtime folders ────────────────────────────────────────────────────────────
print()
print("Runtime folders:")

for folder in ["static/uploads", "static/gradcam"]:
    full_path = os.path.join(PROJECT_ROOT, folder)
    if os.path.isdir(full_path):
        check(folder, True)
    else:
        os.makedirs(full_path, exist_ok=True)
        check(f"{folder} (created)", True)

# ── .env file ──────────────────────────────────────────────────────────────────
print()
print("Configuration:")

env_path = os.path.join(PROJECT_ROOT, ".env")
check(
    ".env file found",
    os.path.exists(env_path),
    "Optional but recommended. Copy: copy .env.example .env (Windows) OR cp .env.example .env (Mac/Linux)",
    warn=True
)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 55)

if not errors:
    print(f"  {PASS} All required checks passed!")
    if warnings:
        print()
        print(f"  {WARN} {len(warnings)} optional item(s) missing:")
        for w in warnings:
            print(f"     - {w}")
    print()
    print("  Ready! Run the app:")
    print("  python app.py")
    print()
    print("  Then open: http://localhost:2222")
else:
    print(f"  {FAIL} {len(errors)} problem(s) found. Fix before running:")
    print()
    for e in errors:
        print(f"     ❌ {e}")
    if warnings:
        print()
        print(f"  {WARN} Also {len(warnings)} warning(s):")
        for w in warnings:
            print(f"     ⚠️  {w}")
    sys.exit(1)

print("=" * 55)
print()
