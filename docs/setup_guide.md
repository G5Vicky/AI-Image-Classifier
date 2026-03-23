# Environment Setup Guide

Complete step-by-step setup for local development. Follow in order.

---

## Prerequisites Checklist

Before starting, verify each:

```bash
python --version      # Must be 3.10+
git --version         # Any version
git lfs version       # Must be installed
```

If any fail, see installation steps below.

---

## 1. Install Python 3.10+

### Windows
1. Download from https://python.org/downloads
2. Run installer
3. ✅ CHECK "Add Python to PATH" (critical — don't skip)
4. Click "Install Now"
5. Verify: open CMD → `python --version`

### Mac
```bash
brew install python@3.10
```
Or download from python.org/downloads

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

---

## 2. Install Git + Git LFS

### Windows
1. Download Git from https://git-scm.com/downloads
2. Install with all defaults
3. Open CMD and run:
```bash
git lfs install
```

### Mac
```bash
brew install git git-lfs
git lfs install
```

### Linux
```bash
sudo apt install git git-lfs
git lfs install
```

---

## 3. Clone the Repository

```bash
git clone https://huggingface.co/spaces/Vicky25july2003/ai-image-classifier
cd ai-image-classifier
```

---

## 4. Create Virtual Environment

```bash
# Create (one time only)
python -m venv venv

# Activate (every time you work on the project)
venv\Scripts\activate          # Windows CMD
.\venv\Scripts\Activate.ps1    # Windows PowerShell
source venv/bin/activate       # Mac/Linux

# You should see (venv) in your terminal prompt
```

---

## 5. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- TensorFlow 2.16.1 (~500MB — takes time)
- Flask 3.0.0
- OpenCV headless
- Matplotlib, NumPy, Pillow, scikit-learn
- Gunicorn, Flask-Mail
- huggingface_hub

---

## 6. Download Model

The model auto-downloads on first run. To pre-download manually:

```bash
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('model', exist_ok=True)

print('Downloading EfficientNetB3 model (~109MB)...')
hf_hub_download(
    repo_id='Vicky25july2003/ai-image-classifier-model',
    filename='efficientnet_model.keras',
    local_dir='model'
)

print('Downloading model config...')
hf_hub_download(
    repo_id='Vicky25july2003/ai-image-classifier-model',
    filename='model_config.json',
    local_dir='model'
)
print('Done.')
"
```

---

## 7. Run the Application

```bash
python app.py
```

Open http://localhost:2222 in your browser.

---

## Verifying Everything Works

After startup, you should see in terminal:
```
✅ Loaded model_config.json
✅ Loaded: EfficientNetB3
   threshold=0.45
   Grad-CAM target "block3a_expand_activation": Activation ✅
```

Check status API:
```
http://localhost:2222/api/status
```

Expected JSON:
```json
{
  "model": "EfficientNetB3",
  "model_loaded": true,
  "is_enb3": true,
  "threshold": 0.45
}
```

---

## Common Setup Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `python not found` | PATH not set | Reinstall Python, check Add to PATH |
| `pip not found` | Python install issue | Use `python -m pip install ...` |
| `No module named tensorflow` | venv not activated | Run `venv\Scripts\activate` first |
| `OSError: model not found` | Model not downloaded | Run manual download above |
| `Address already in use` | Port 2222 taken | Change `port=2222` to `port=3333` in app.py |
| `DLL load failed` (Windows) | Missing Visual C++ | Install Visual C++ Redistributable |
