# Deployment Guide — Free Hosting on Hugging Face Spaces

Full walkthrough from zero to live URL at $0/month.

---

## Why Hugging Face Spaces?

| Platform | ML Support | Free Model Storage | Sleep | Docker | Verdict |
|----------|-----------|-------------------|-------|--------|---------|
| HF Spaces | ✅ Native | ✅ Yes (HF Hub) | 48hr | ✅ Yes | ✅ Best |
| Render | ⚠️ Manual | ❌ No | 15min | ✅ Yes | ⚠️ OK |
| Railway | ⚠️ Manual | ❌ No | Paid only | ✅ Yes | ❌ Paid |
| Streamlit Cloud | ✅ Yes | ❌ No | ✅ No sleep | ❌ No | ⚠️ Streamlit only |
| Vercel | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No |

HF Spaces wins because: native ML ecosystem, model hub for large files, Docker support, no sleep on free tier (48hr threshold only).

---

## Prerequisites

- Hugging Face account: https://huggingface.co/join
- GitHub account: https://github.com/join
- Git + Git LFS installed locally
- Project code working locally

---

## Phase 1 — Upload Model to HF Model Hub

The model file (109MB) is too large for a code repo. Host it separately.

### Step 1 — Create model repo on HF

```python
from huggingface_hub import create_repo, login

login(token='YOUR_HF_TOKEN')  # from huggingface.co/settings/tokens (Write role)

create_repo(
    'YOUR_USERNAME/ai-image-classifier-model',
    repo_type='model',
    private=False
)
```

### Step 2 — Upload model files

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload EfficientNetB3 model
api.upload_file(
    path_or_fileobj='model/efficientnet_model.keras',
    path_in_repo='efficientnet_model.keras',
    repo_id='YOUR_USERNAME/ai-image-classifier-model',
    repo_type='model',
    token='YOUR_HF_TOKEN'
)

# Upload config
api.upload_file(
    path_or_fileobj='model/model_config.json',
    path_in_repo='model_config.json',
    repo_id='YOUR_USERNAME/ai-image-classifier-model',
    repo_type='model',
    token='YOUR_HF_TOKEN'
)
```

### Step 3 — Update app.py with your username

In `app.py`, find line:
```python
_HF_REPO = "Vicky25july2003/ai-image-classifier-model"
```
Replace with:
```python
_HF_REPO = "YOUR_USERNAME/ai-image-classifier-model"
```

---

## Phase 2 — Create HF Space

### Step 1 — Go to https://huggingface.co/new-space

Fill in:
- **Owner:** your username
- **Space name:** `ai-image-classifier`
- **SDK:** Docker ← important
- **Visibility:** Public
- Click **Create Space**

### Step 2 — Verify README.md has correct metadata

The first lines of `README.md` MUST be:
```yaml
---
title: Ai Image Classifier
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```

Without this, HF shows "No application file" even with a valid Dockerfile.

### Step 3 — Verify Dockerfile

```dockerfile
FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

RUN mkdir -p /app/static/uploads /app/static/gradcam /app/model

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300"]
```

Key points:
- Port must be **7860** (HF Spaces requirement)
- Timeout 300s (model download + TF load takes time)
- `useradd` user 1000 (HF Spaces security requirement)

---

## Phase 3 — Push Code

### Step 1 — Initialize repo with LFS

```bash
cd your-project-folder
git init
git lfs install
git lfs track "*.jpg" "*.jpeg" "*.png" "*.gif"
git lfs track "*.woff" "*.woff2" "*.keras" "*.map"
git add .gitattributes
```

### Step 2 — Commit everything

```bash
git add .
git commit -m "Initial deployment"
```

### Step 3 — Add HF Space as remote

```bash
git remote add space https://YOUR_USERNAME:YOUR_HF_TOKEN@huggingface.co/spaces/YOUR_USERNAME/ai-image-classifier
```

### Step 4 — Push to HF (main branch, not master)

```bash
git push space master:main
```

> ⚠️ HF Spaces uses `main` branch. Always push to `master:main`.

---

## Phase 4 — Monitor Build

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/ai-image-classifier`
2. Click **Logs** tab → **Build** subtab
3. Watch for TensorFlow install (~5-8 min)
4. Then **Container** subtab shows app startup logs

### Successful build output:
```
===== Build Succeeded =====
===== Application Startup =====
[INFO] Starting gunicorn 21.2.0
[INFO] Listening at: http://0.0.0.0:7860
✅ Loaded model_config.json
✅ Loaded: EfficientNetB3
```

---

## Redeployment (Code Updates)

After making changes:

```bash
git add .
git commit -m "describe changes"
git push space master:main
```

Build triggers automatically. Takes 2-5 min if TF is cached, 10 min on first build.

---

## Troubleshooting Deployment

| Symptom | Cause | Fix |
|---------|-------|-----|
| "No application file" | Wrong branch or missing README metadata | `git push space master:main --force` |
| Build stuck at TF install | Normal — TF is 500MB | Wait 10 minutes |
| Container crashes on start | Model download failed or OOM | Check Container logs |
| Push rejected: binary files | LFS not set up | `git lfs track "*.jpg"` etc., recommit |
| App loads but no Grad-CAM | Graph disconnection issue | Check Container logs for traceback |
| 502 Bad Gateway | Gunicorn worker crashed | Check Container logs, likely OOM |

---

## Free Tier Behavior

| Event | Behavior |
|-------|----------|
| First visit after deploy | App is running — fast |
| Visit after 48hr no traffic | 60-90 second cold start |
| Model file on cold start | Re-downloads from HF Hub (109MB) |
| Static files (CSS/JS) | Always available via Git LFS |
| Uploaded images after restart | Lost — not persisted |

### For Portfolio Use
Add this note next to your demo link:
> *"Free-tier hosted — may take ~60 seconds to wake up on first visit. This is expected behavior."*

Interviewers and recruiters understand this completely.
