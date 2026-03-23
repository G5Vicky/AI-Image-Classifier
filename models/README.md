# models/ — AI Model Files

This folder holds the trained neural network files.

---

## ⚠️ Models Are NOT Included in the Repo

The trained model files are too large for GitHub (109MB limit is 100MB).  
They are hosted on **Hugging Face Model Hub** instead.

---

## How to Download

**Automatic (recommended):**
```bash
python scripts/download_models.py
```

**Manual download:**
1. Go to: https://huggingface.co/Vicky25july2003/ai-image-classifier-model
2. Download `efficientnet_model.keras`
3. Place it in this `models/` folder

---

## What Should Be in This Folder After Download

```
models/
├── efficientnet_model.keras    ← Primary model (109MB) — DOWNLOAD REQUIRED
├── cnn_model.keras             ← Fallback model (29MB)  — optional
└── model_config.json           ← Already included ✅
```

---

## What These Files Do

| File | Purpose |
|------|---------|
| `efficientnet_model.keras` | The main AI model. Used for all predictions. 93% accuracy. |
| `cnn_model.keras` | Backup model. Used only if EfficientNetB3 is missing. 88% accuracy. |
| `model_config.json` | Settings: decision thresholds, Grad-CAM layer names, preprocessing flags. |

---

## model_config.json Explained

```json
{
  "efficientnet": {
    "threshold": 0.45,              ← Score ≥ 0.45 means Real. Score < 0.45 means Fake.
    "gradcam_layer": "block3a_expand_activation",
    "img_size": [32, 32],
    "rescale": false                ← EfficientNetB3 normalizes internally
  },
  "cnn": {
    "threshold": 0.44,
    "gradcam_layer": "last_conv2d",
    "img_size": [32, 32],
    "rescale": true                 ← CNN needs pixels in [0,1]
  }
}
```

The threshold is **0.45** (not 0.5). This is because during training, a class weight was
applied that biased the model's scores upward. We calibrated the threshold post-training
to compensate.
