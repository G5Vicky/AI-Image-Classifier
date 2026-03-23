# Project Workflow — How the System Works Internally

This document traces every step from "user uploads image" to "result displayed on screen."

---

## High-Level Flow

```
User uploads image
      │
      ▼
Flask /submit route receives file
      │
      ├── Validate file (extension + size)
      ├── Save to static/uploads/<uuid>.<ext>
      │
      ▼
Preprocess image
      │
      ├── Load with Keras load_img (resize to 32×32)
      ├── Convert to numpy array (float32)
      └── Normalize:
          ├── EfficientNetB3: raw [0,255] (ENB3 normalizes internally)
          └── CNN: divide by 255.0 → [0,1]
      │
      ▼
Model prediction
      │
      ├── model.predict(img_array)
      ├── Returns sigmoid score [0.0 → 1.0]
      └── Apply threshold:
          ├── score >= 0.45 → "Real Image"
          └── score <  0.45 → "AI-Generated Image"
      │
      ▼
Grad-CAM pipeline
      │
      ├── Get EfficientNetB3 sub-model
      ├── Find target layer: block3a_expand_activation
      ├── Build grad_model inside sub-model graph
      ├── Forward pass under GradientTape
      ├── Compute gradients of score w.r.t. conv_outputs
      ├── Pool gradients → importance weights
      ├── Weighted sum of feature maps → raw heatmap (8×8)
      ├── ReLU → normalize to [0,1]
      └── Returns heatmap array
      │
      ▼
Overlay rendering (save_overlay)
      │
      ├── Read original image (cv2)
      ├── Resize heatmap to original image size
      ├── Apply jet colormap
      ├── Blend with original (alpha=0.45)
      └── Save 3-panel figure:
          [Original | Attention Map | Grad-CAM Overlay]
      │
      ▼
Return output.html
      │
      └── Template receives:
          ├── label (Real/AI-Generated)
          ├── confidence (%)
          ├── uploaded_image (filename)
          ├── gradcam_image (filename)
          ├── raw_score (sigmoid float)
          └── threshold (0.45)
```

---

## Application Startup Sequence

When `python app.py` or `gunicorn app:app` starts:

```
1. Import modules (TF, Flask, cv2, matplotlib, etc.)

2. HF Hub download check
   ├── For each file in [efficientnet_model.keras, model_config.json]
   ├── If not in model/ directory → download from HF Hub
   └── If already exists → skip

3. Load model_config.json
   ├── Read calibrated thresholds (0.45 for ENB3, 0.44 for CNN)
   └── Read Grad-CAM layer name

4. Model loading loop
   ├── Try efficientnet_model.keras first
   ├── Load with compile=False, custom_objects={'WarmUpCosineDecay': ...}
   ├── Warmup: run dummy input through model (CRITICAL for Grad-CAM graph tracing)
   ├── Find efficientnetb3 sub-model layer
   ├── Verify block3a_expand_activation exists in sub-model
   └── If ENB3 fails → try cnn_model.keras as fallback

5. Flask app starts
   └── Gunicorn binds to 0.0.0.0:7860 (HF Spaces) or 0.0.0.0:2222 (local)
```

---

## File Storage

| File Type | Location | Lifetime |
|-----------|----------|----------|
| User uploads | `static/uploads/<uuid>.<ext>` | Persists until container restart |
| Grad-CAM outputs | `static/gradcam/gc_<uuid>.jpg` | Persists until container restart |
| Model files | `model/*.keras` | Persists in local dev; re-downloaded on HF cold start |
| Contact messages | `contact_messages.txt` | Only when email fails |

---

## Thread Safety

Multiple users can submit simultaneously. The prediction uses a threading lock:

```python
_predict_lock = threading.Lock()

with _predict_lock:
    raw_pred = model.predict(img_arr, verbose=0)
```

This prevents concurrent TensorFlow graph executions which can corrupt results. Only one prediction runs at a time — acceptable for free-tier single-worker deployment.

---

## Demo Mode

If no model file is found (neither `.keras` file exists):
- `model` global variable stays `None`
- `/submit` route detects `model is None`
- Returns a random prediction (55-88% confidence) with `is_demo=True` flag
- `output.html` displays a "Demo Mode" banner

This ensures the app is always usable even without the model.
