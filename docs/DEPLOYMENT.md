# Deployment Guide — Put This on the Internet for Free

This guide deploys your app to **Hugging Face Spaces** so anyone in the world can access it via a permanent URL, at zero cost.

---

## Before You Start

You need:
- A free Hugging Face account: https://huggingface.co/join
- Git installed on your computer
- Git LFS installed: run `git lfs install` after installing Git
- The code working locally first

---

## Step 1 — Upload Your Model to Hugging Face Hub

The model file (109MB) can't go in the code repo. We host it separately.

**Create a model repository on Hugging Face:**
```python
python -c "
from huggingface_hub import create_repo, login
login()  # paste your HF token when prompted
create_repo('YOUR_USERNAME/ai-image-classifier-model', repo_type='model')
"
```

**Upload the model:**
```python
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='models/efficientnet_model.keras',
    path_in_repo='efficientnet_model.keras',
    repo_id='YOUR_USERNAME/ai-image-classifier-model',
    repo_type='model'
)
api.upload_file(
    path_or_fileobj='models/model_config.json',
    path_in_repo='model_config.json',
    repo_id='YOUR_USERNAME/ai-image-classifier-model',
    repo_type='model'
)
print('Done!')
"
```

---

## Step 2 — Update app.py With Your Username

Open `app.py` and find this line near the top:
```python
_HF_REPO = "Vicky25july2003/ai-image-classifier-model"
```

Change it to:
```python
_HF_REPO = "YOUR_USERNAME/ai-image-classifier-model"
```

---

## Step 3 — Create a Hugging Face Space

1. Go to: https://huggingface.co/new-space
2. Fill in:
   - **Space name:** `ai-image-classifier` (or any name you like)
   - **SDK:** Select **Docker**
   - **Visibility:** Public
3. Click **Create Space**

---

## Step 4 — Set Up Your Local Git Repo

```bash
# Initialize Git (if not done already)
git init

# Set up Git LFS for binary files
git lfs install
git lfs track "*.jpg" "*.jpeg" "*.png" "*.gif"
git lfs track "*.woff" "*.woff2" "*.keras" "*.map"
git add .gitattributes

# Add all files
git add .
git commit -m "Initial deployment commit"
```

---

## Step 5 — Push to Hugging Face Spaces

```bash
# Add your HF Space as a remote
# Replace YOUR_USERNAME and YOUR_HF_TOKEN
git remote add space https://YOUR_USERNAME:YOUR_HF_TOKEN@huggingface.co/spaces/YOUR_USERNAME/ai-image-classifier

# Push — IMPORTANT: HF uses 'main' branch, not 'master'
git push space master:main
```

> ⚠️ **Critical:** Always use `master:main` not just `master`. HF Spaces reads the `main` branch. Pushing to `master` alone results in "No application file" error.

---

## Step 6 — Watch the Build

1. Go to: `https://huggingface.co/spaces/YOUR_USERNAME/ai-image-classifier`
2. Click the **Logs** tab → **Build** subtab
3. Watch the build progress

**First build takes ~10 minutes** because it installs TensorFlow (589MB).  
Subsequent builds take ~3 minutes (Docker caches the dependency layer).

---

## Step 7 — Your App is Live!

When build succeeds, your app is accessible at:
```
https://huggingface.co/spaces/YOUR_USERNAME/ai-image-classifier
```

Share this URL with anyone. No sign-in needed to use it.

---

## Updating After Code Changes

Every time you change code and want to update the live app:

```bash
git add .
git commit -m "describe what you changed"
git push space master:main
```

Build triggers automatically. Takes 3–5 minutes.

---

## Free Tier Limitations

| Thing | Value |
|-------|-------|
| Cost | $0/month |
| Sleep after no visitors | 48 hours |
| Wake-up time after sleep | 60–90 seconds |
| CPU | 2 vCPU (shared) |
| RAM | 16 GB |
| GPU | None (CPU inference only) |
| Storage | Ephemeral (resets on restart) |

**For a portfolio project, this is perfect.** The URL is permanent. Visitors just need to wait ~60 seconds on first visit after a long gap.

---

## Troubleshooting Deployment

| Problem | Cause | Fix |
|---------|-------|-----|
| "No application file" | Pushed to wrong branch | `git push space master:main --force` |
| Build fails halfway | Dependency conflict | Check Build logs for exact error |
| App shows but model doesn't load | Model download failed | Check Container logs |
| Push rejected: binary files | LFS not set up | Run `git lfs track "*.jpg"` etc., then recommit |
| "No application file" after README change | Missing HF metadata header | See README.md top section — must start with `---` block |
