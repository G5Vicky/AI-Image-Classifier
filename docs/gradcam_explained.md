# Grad-CAM — Complete Explanation

**Gradient-weighted Class Activation Mapping** — making neural network decisions interpretable.

---

## Why Explainability Matters

A model that says "this is AI-generated" with 94% confidence is useful.  
A model that says the same AND shows you *which parts of the image it detected as synthetic artifacts* is **trustworthy**.

Grad-CAM converts the model from a black box into an auditable system.

---

## Intuition (No Math)

Imagine you're deciding if a photo is real or fake. Your eyes focus on certain regions — maybe the hands look wrong, or the background lighting is inconsistent. Grad-CAM does exactly this for the neural network.

It asks: **"If I slightly change the activations at this spatial location, how much does the prediction change?"**

High sensitivity = that region is important.  
Low sensitivity = that region was ignored.

The result is a heatmap overlaid on the image showing WHERE the network focused.

---

## How It Works (Technical)

### Step 1 — Choose a Target Layer

We use `block3a_expand_activation` inside EfficientNetB3.

```
Why this layer?
- Located at the middle of the backbone
- Produces 8×8 spatial feature maps at 32×32 input
- Deep enough to have semantic meaning
- Shallow enough to retain spatial resolution
- Earlier layers (16×16, 32×32) → too noisy
- Later layers (2×2, 1×1) → too spatially coarse
```

### Step 2 — Build the Gradient Model

```python
base_model = outer_model.get_layer('efficientnetb3')

grad_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[
        target_layer.output,   # shape: (1, 8, 8, C) — feature maps
        base_model.output      # shape: (1, 1, 1, 1536) — base output
    ]
)
```

**Critical:** Must build inside the sub-model, not the outer model.  
See "The Bug" section below for why.

### Step 3 — Forward Pass Under GradientTape

```python
with tf.GradientTape() as tape:
    conv_outputs, base_features = grad_model(img_tensor, training=False)
    tape.watch(conv_outputs)
    
    # Replay the custom head layers
    x = tf.reshape(base_features, [batch_size, -1])
    for layer in outer_model.layers:
        if isinstance(layer, (Dense, BatchNormalization, Activation)):
            x = layer(x, training=False)
    
    score = x[:, 0]  # final sigmoid score
```

### Step 4 — Compute Gradients

```python
grads = tape.gradient(score, conv_outputs)
# grads shape: (1, 8, 8, C)
# grads[i,j,k] = how much does score change if conv_outputs[i,j,k] changes?
```

### Step 5 — Pool Gradients (Importance Weights)

```python
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
# pooled_grads shape: (C,)
# One importance weight per channel
```

### Step 6 — Weighted Sum of Feature Maps

```python
heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
# Dot product: (8, 8, C) × (C, 1) = (8, 8, 1)
heatmap = tf.squeeze(heatmap)  # (8, 8)
```

Each pixel in the 8×8 heatmap = weighted sum of channel activations at that spatial location.

### Step 7 — ReLU + Normalize

```python
heatmap = tf.maximum(heatmap, 0.0)  # ReLU — keep only positive contributions
heatmap = heatmap / tf.reduce_max(heatmap)  # normalize to [0, 1]
```

**Why ReLU?** We only care about features that PUSH TOWARD the predicted class (positive gradient). Negative gradients mean "evidence against this class" — not useful for visualization.

### Step 8 — Resize and Overlay

```python
hm = cv2.resize(heatmap.numpy(), (original_width, original_height))
hm_colored = cm.get_cmap('jet')(hm)[:, :, :3]  # apply jet colormap
hm_rgb = np.uint8(hm_colored * 255)
blended = cv2.addWeighted(original_img, 0.55, hm_rgb, 0.45, 0)
```

---

## The Bug That Plagued Previous Versions

All versions before this one had Grad-CAM silently returning `None`. Here's exactly why:

### The Architecture Problem

EfficientNetB3 is a nested model:
```
outer_model (EfficientNetB3_Classifier)
└── efficientnetb3 (sub-model) ← separate Keras graph
    └── block3a_expand_activation ← target layer
└── GlobalAveragePooling2D
└── Dense layers
```

After `load_model()`, the outer model and sub-model have **separate computation graphs**. The outer model's `.inputs` tensor is NOT connected to the sub-model's internal layers.

### The Wrong Approach (Previous Code)
```python
# ❌ This fails — graph disconnected
grad_model = tf.keras.Model(
    inputs=outer_model.inputs,        # outer model's input tensor
    outputs=[target_layer.output,     # lives in SUB-model graph!
             outer_model.output]      # different graph
)
# → Keras raises ValueError: Graph disconnected
# → Caught by try/except → returns None → "Grad-CAM unavailable"
```

### The Correct Approach (Current Code)
```python
# ✅ Stay inside the sub-model's own graph
base_model = outer_model.get_layer('efficientnetb3')
grad_model = tf.keras.Model(
    inputs=base_model.input,          # sub-model's own input
    outputs=[target_layer.output,     # target is INSIDE sub-model ✓
             base_model.output]       # output is INSIDE sub-model ✓
)
# Then replay the outer head layers manually
```

### The 5D Shape Bug (Windows-specific)
On some Keras versions (Windows), `base_model.output` inside `grad_model` returns a 5D tensor `(1, 1, 1, 1, 1536)` instead of expected 4D `(1, 1, 1, 1536)`.

**Wrong fix:**
```python
x = tf.reduce_mean(base_features, axis=[1, 2])  # breaks on 5D
```

**Correct fix:**
```python
x = tf.reshape(base_features, [tf.shape(base_features)[0], -1])
# Collapses ALL intermediate dims safely, regardless of rank
```

---

## Visualization: 3-Panel Output

The `save_overlay()` function produces a `14×5 inch` matplotlib figure:

```
┌──────────────┬──────────────┬──────────────────┐
│              │              │                  │
│   Original   │ Attention    │   Grad-CAM       │
│   Image      │ Map          │   Overlay        │
│              │ (heatmap)    │ (blended 45%)    │
│              │              │                  │
└──────────────┴──────────────┴──────────────────┘
     Dark background (#07080d), white titles, purple suptitle
```

Colors:
- 🔴 **Red/Yellow** → high attention (model focused here strongly)
- 🟢 **Green** → medium attention
- 🔵 **Blue** → low/no attention (model ignored this region)

---

## Interpreting Results

### For a Real Image:
- Heatmap typically focuses on **natural textures, consistent lighting, organic edges**
- Regions: faces, fur, foliage, sky gradients

### For an AI-Generated Image:
- Heatmap focuses on **artifacts specific to generative models**
- Typical artifacts: hands, background inconsistencies, text, symmetric patterns, smooth gradients that are "too perfect"

### All-Zero Heatmap:
If the model is extremely confident (sigmoid near 0.0 or 1.0), gradients can vanish, producing a near-zero heatmap. In this case, a uniform 0.3 fallback is returned — still visible but not meaningful. This is normal behavior for very confident predictions.
