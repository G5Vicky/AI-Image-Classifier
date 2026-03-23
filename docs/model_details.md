# Model Details — Architecture, Training & Configuration

---

## Overview

Two models are trained and used:

| Model | File | Size | Role |
|-------|------|------|------|
| EfficientNetB3 | `efficientnet_model.keras` | ~109MB | Primary — used for all predictions |
| Custom CNN | `cnn_model.keras` | ~29MB | Fallback — loaded only if ENB3 missing |

---

## EfficientNetB3 Architecture

```
Input: (None, 32, 32, 3)  raw pixel values [0-255]
│
├── EfficientNetB3 Base (pretrained on ImageNet)
│   ├── Stem: Conv2D 3×3, stride 2
│   ├── MBConv blocks 1-7 (mobile inverted bottleneck)
│   │   └── block3a_expand_activation  ← Grad-CAM target (8×8 feature map)
│   └── Head Conv: 1×1 Conv + Activation
│
├── GlobalAveragePooling2D
│   └── Output: (None, 1536)
│
├── Dense(512, activation='relu')
├── BatchNormalization
├── Dropout(0.3)
│
├── Dense(256, activation='relu')
├── BatchNormalization
├── Dropout(0.3)
│
└── Dense(1, activation='sigmoid')
    └── Output: scalar in [0, 1]

Decision:
    score >= 0.45  →  label=1  →  "Real Image"
    score <  0.45  →  label=0  →  "AI-Generated Image"
```

---

## Custom CNN Architecture

```
Input: (None, 32, 32, 3)  pixel values [0,1] (rescaled)
│
├── Block 1:
│   ├── Conv2D(32, 3×3, padding='same', activation='relu')
│   ├── BatchNormalization
│   └── MaxPooling2D(2×2)
│
├── Block 2:
│   ├── Conv2D(64, 3×3, padding='same', activation='relu')
│   ├── BatchNormalization
│   └── MaxPooling2D(2×2)
│
├── Block 3:
│   ├── Conv2D(128, 3×3, padding='same', activation='relu')
│   ├── BatchNormalization
│   └── MaxPooling2D(2×2)
│
├── Block 4:
│   ├── Conv2D(256, 3×3, padding='same', activation='relu')  ← Grad-CAM target
│   ├── BatchNormalization
│   └── MaxPooling2D(2×2)
│
├── Flatten
├── Dense(512, relu) → Dropout(0.5)
└── Dense(1, sigmoid)
```

---

## Training Configuration

### Phase 1 — Feature Extraction (EfficientNetB3)

| Parameter | Value |
|-----------|-------|
| Base model frozen | Yes |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Epochs | 10 |
| Batch size | 32 |
| Loss | binary_crossentropy |

### Phase 2 — Fine-tuning (EfficientNetB3)

| Parameter | Value |
|-----------|-------|
| Base model frozen | No (top layers unfrozen) |
| Optimizer | Adam with WarmUpCosineDecay |
| Base learning rate | 1e-5 |
| Warmup steps | 2112 |
| Total steps | 10560 |
| Min learning rate | 1e-9 |
| Epochs | 5 |
| Batch size | 32 |

### WarmUpCosineDecay Schedule

```python
def __call__(self, step):
    # Linear warmup from 0 to base_lr
    warmup_lr = base_lr * (step / warmup_steps)
    
    # Cosine decay from base_lr to min_lr
    cosine_decay = 0.5 * (1 + cos(π * (step - warmup_steps) / decay_steps))
    cosine_lr = min_lr + (base_lr - min_lr) * cosine_decay
    
    return warmup_lr if step < warmup_steps else cosine_lr
```

This prevents catastrophic forgetting of ImageNet features during fine-tuning. The warmup gradually increases the learning rate, avoiding large gradient updates early in training.

---

## Preprocessing

### EfficientNetB3
```python
img = load_img(path, target_size=(32, 32))
arr = img_to_array(img).astype(np.float32)
# NO rescaling — pass raw [0, 255]
# EfficientNetB3 base applies (x / 127.5) - 1 internally
arr = np.expand_dims(arr, axis=0)  # shape: (1, 32, 32, 3)
```

### Custom CNN
```python
img = load_img(path, target_size=(32, 32))
arr = img_to_array(img).astype(np.float32)
arr = arr / 255.0  # rescale to [0, 1]
arr = np.expand_dims(arr, axis=0)
```

**Critical:** Using wrong preprocessing inverts the feature space and produces ~50% accuracy (random guessing).

---

## Calibrated Thresholds

The default sigmoid threshold of 0.5 was biased due to `class_weight={0: 1.5, 1: 1.0}` used during Phase 2 training. This penalized FAKE misses 1.5x harder, pushing sigmoid outputs upward.

Calibration was done by:
1. Running model on validation set
2. Computing ROC curve
3. Finding threshold that maximizes F1-score
4. Saving to `model_config.json`

Result:
- EfficientNetB3: `0.45` (not 0.5)
- Custom CNN: `0.44`

---

## model_config.json

```json
{
  "class_indices": {"FAKE": 0, "REAL": 1},
  "label_map": {"0": "AI-Generated Image", "1": "Real Image"},
  "sigmoid_rule": "score >= threshold → Real Image",
  "efficientnet": {
    "file": "efficientnet_model.keras",
    "threshold": 0.45,
    "img_size": [32, 32],
    "rescale": false,
    "gradcam_layer": "block3a_expand_activation"
  },
  "cnn": {
    "file": "cnn_model.keras",
    "threshold": 0.44,
    "img_size": [32, 32],
    "rescale": true,
    "gradcam_layer": "last_conv2d"
  }
}
```

---

## Performance Results

### EfficientNetB3

| Metric | Value |
|--------|-------|
| Test Accuracy | ~93% |
| AUC-ROC | ~98% |
| F1-Score | ~91% |
| Precision | ~93% |
| Recall | ~91% |

### Custom CNN

| Metric | Value |
|--------|-------|
| Test Accuracy | ~88% |
| AUC-ROC | ~97% |
| F1-Score | ~87% |
| True Positive Rate | ~76% |
| False Negative Rate | ~24% |

The CNN's 24% false negative rate on REAL images (classifying real as AI-generated) is why EfficientNetB3 is used as primary.
