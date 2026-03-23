# How the System Works

This document explains the complete technical flow — from you uploading an image to seeing the result.

---

## The Big Picture

```
You upload an image
       │
       ▼
Flask receives the file
       │
       ├── Checks: is it a valid image? Is it under 16MB?
       ├── Saves it with a random filename (so two users don't conflict)
       │
       ▼
Image preprocessing
       │
       ├── Resize to 32×32 pixels
       └── Normalize pixel values (different for each model — see below)
       │
       ▼
EfficientNetB3 model runs
       │
       ├── Produces a number between 0.0 and 1.0 (called sigmoid score)
       ├── Score ≥ 0.45 → "Real Image"
       └── Score < 0.45 → "AI-Generated Image"
       │
       ▼
Grad-CAM runs (explainability)
       │
       ├── Figures out WHICH parts of the image caused the decision
       ├── Generates an 8×8 "importance map"
       └── Overlays it on the original image with colors
       │
       ▼
You see the result page
       │
       └── Label + Confidence % + 3-panel Grad-CAM image
```

---

## The Two Models

### EfficientNetB3 (main model)
- Developed by Google in 2019
- Pretrained on 1.28 million ImageNet images
- Then fine-tuned on our 140,000 CIFAKE images
- ~93% accuracy

### Custom CNN (backup model)
- Built from scratch (no pretrained weights)
- 4 layers of convolution + pooling
- ~88% accuracy
- Only used if EfficientNetB3 file is missing

The app always tries EfficientNetB3 first.

---

## Why Two Different Preprocessing Pipelines?

This is important and often confuses people.

**EfficientNetB3** expects raw pixel values (0 to 255).  
→ Its internal code divides by 127.5 and subtracts 1, giving values from -1 to +1.  
→ If you divide the pixels by 255 BEFORE giving them to EfficientNetB3, it gets inputs from -1 to -0.998 (wrong!), and accuracy drops to ~50%.

**Custom CNN** expects rescaled pixel values (0 to 1).  
→ It was trained with pixels divided by 255.  
→ If you give it raw values (0 to 255), it gets inputs way outside its training range, and accuracy drops.

**Summary:**
```
EfficientNetB3 input: raw [0, 255]   → model divides internally
Custom CNN input:     rescaled [0,1] → you must divide by 255 before passing
```

The `models/model_config.json` file tells the app which pipeline to use.

---

## Why Threshold 0.45 Instead of 0.5?

During training, we used a class weight to penalize missing fake images more than missing real images. This caused the model to produce slightly higher scores than it should — borderline inputs that should score ~0.50 scored ~0.81 instead.

We measured this effect on a validation set and found that 0.45 is the correct cutoff point (maximizes F1-score). This value is stored in `model_config.json` and loaded when the app starts.

---

## What Does "Confidence %" Mean?

If the model says "Real Image" with score 0.80:
```
Confidence = 0.80 × 100 = 80%
```

If the model says "AI-Generated Image" with score 0.20:
```
Confidence = (1 - 0.20) × 100 = 80%
```

Confidence always represents "how sure is the model about this specific label."

---

## Application Startup

When you run `python app.py`, this happens in order:

1. Load `.env` file (if it exists)
2. Check if model files exist in `models/` — if not, download from Hugging Face
3. Load `model_config.json` (thresholds, layer names)
4. Load `efficientnet_model.keras` into memory
5. Run one "warmup" prediction (a blank image) to prepare TensorFlow's computation graph
6. Start the Flask web server on port 2222

Steps 4 and 5 are why startup takes ~20-30 seconds — loading a 109MB neural network takes time.

---

## Request Flow (Per Image Upload)

1. User submits image via HTML form
2. Flask receives the file at `POST /submit`
3. Validate extension (only jpg, png, bmp, webp allowed)
4. Save file as `static/uploads/<random-12-char-id>.jpg`
5. Preprocess image (resize + normalize)
6. Run `model.predict()` — get sigmoid score
7. Apply threshold → get label + confidence
8. Run Grad-CAM (see GRADCAM.md for details)
9. Save heatmap to `static/gradcam/gc_<same-id>.jpg`
10. Render `output.html` with all results

---

## Thread Safety

If two people upload images at the exact same moment, TensorFlow can get confused if both run `model.predict()` at the same time. The app uses a **threading lock** to prevent this:

```python
with _predict_lock:
    raw_pred = model.predict(img_arr, verbose=0)
```

Only one prediction runs at a time. The other waits. This is safe and correct.
