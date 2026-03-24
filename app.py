# ══════════════════════════════════════════════════════════════════════════════
#  AI vs REAL Image Classifier — Flask Backend  app_final.py
#  KITSW Batch 11 | Major Project 2024-25
# ══════════════════════════════════════════════════════════════════════════════

import os, uuid, traceback, math, threading, json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_mail import Mail, Message
from huggingface_hub import hf_hub_download

# ── Auto-download model from HF Hub at startup ────────────────────────────────
_HF_REPO   = "Vicky25july2003/ai-image-classifier-model"
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(_MODEL_DIR, exist_ok=True)

for _hf_file in ["efficientnet_model.keras", "model_config.json"]:
    _dest = os.path.join(_MODEL_DIR, _hf_file)
    if not os.path.exists(_dest):
        print(f"Downloading {_hf_file} from HF Hub...")
        hf_hub_download(repo_id=_HF_REPO, filename=_hf_file, local_dir=_MODEL_DIR)
        print(f"✅ {_hf_file} ready")
    else:
        print(f"✅ {_hf_file} already exists")
# ─────────────────────────────────────────────────────────────────────────────

# ── App & Config ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY']          = 'aivsr3al2025kitsw'
app.config['MAIL_SERVER']         = 'smtp.gmail.com'
app.config['MAIL_PORT']           = 587
app.config['MAIL_USE_TLS']        = True
app.config['MAIL_USERNAME']       = 'b21it092@kitsw.ac.in'
app.config['MAIL_PASSWORD']       = os.environ.get('MAIL_PASSWORD', '')
app.config['MAIL_DEFAULT_SENDER'] = 'b21it092@kitsw.ac.in'
mail = Mail(app)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, 'static', 'uploads')
GRADCAM_FOLDER = os.path.join(BASE_DIR, 'static', 'gradcam')
MODEL_DIR      = os.path.join(BASE_DIR, 'model')
ALLOWED_EXT    = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

_predict_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-VERSION Keras serializable decorator
# ══════════════════════════════════════════════════════════════════════════════
def _get_register_fn():
    try:
        import keras
        if hasattr(keras, 'saving') and hasattr(keras.saving, 'register_keras_serializable'):
            return keras.saving.register_keras_serializable
    except ImportError:
        pass
    try:
        if hasattr(tf.keras, 'saving') and hasattr(tf.keras.saving, 'register_keras_serializable'):
            return tf.keras.saving.register_keras_serializable
    except Exception:
        pass
    try:
        if hasattr(tf.keras.utils, 'register_keras_serializable'):
            return tf.keras.utils.register_keras_serializable
    except Exception:
        pass
    def _noop(**kwargs):
        def decorator(cls): return cls
        return decorator
    return _noop

_register_serializable = _get_register_fn()


@_register_serializable(package='custom', name='WarmUpCosineDecay')
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr=1e-5, warmup_steps=2112.0,
                 total_steps=10560.0, min_lr=1e-9, **kwargs):
        super().__init__(**kwargs)
        self.base_lr      = float(base_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)
        self.min_lr       = float(min_lr)

    def __call__(self, step):
        step         = tf.cast(step, tf.float32)
        warmup_lr    = self.base_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        decay_steps  = tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(
            math.pi * (step - self.warmup_steps) / decay_steps))
        cosine_lr    = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'base_lr':      self.base_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps':  self.total_steps,
            'min_lr':       self.min_lr,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL CONFIG
# ══════════════════════════════════════════════════════════════════════════════
_CONFIG_PATH = os.path.join(MODEL_DIR, 'model_config.json')
_model_config = {}
if os.path.exists(_CONFIG_PATH):
    with open(_CONFIG_PATH) as f:
        _model_config = json.load(f)
    print(f'✅ Loaded model_config.json')
    print(f'   ENB3 threshold : {_model_config.get("efficientnet", {}).get("threshold", 0.5)}')
    print(f'   CNN  threshold : {_model_config.get("cnn", {}).get("threshold", 0.5)}')
else:
    print('ℹ️  model_config.json not found — using default threshold 0.5')


def _get_threshold(model_type):
    if _model_config:
        return float(_model_config.get(model_type, {}).get('threshold', 0.5))
    return 0.5


def _get_gradcam_layer(model_type):
    if _model_config:
        return _model_config.get(model_type, {}).get('gradcam_layer', 'block3a_expand_activation')
    return 'block3a_expand_activation'


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
model              = None
model_name         = 'Demo Mode'
is_enb3            = False
img_size           = (32, 32)
decision_threshold = 0.5
gradcam_layer_name = 'block3a_expand_activation'
enb3_submodel_name = None

_CUSTOM_OBJECTS = {'WarmUpCosineDecay': WarmUpCosineDecay}

for _fname, _name, _enb3_flag, _mtype in [
    ('efficientnet_model.keras', 'EfficientNetB3', True,  'efficientnet'),
    ('cnn_model.keras',          'Custom CNN',     False, 'cnn'),
]:
    _path = os.path.join(MODEL_DIR, _fname)
    if not os.path.exists(_path):
        print(f'ℹ️  {_fname} not found — skipping')
        continue
    try:
        _m = tf.keras.models.load_model(
            _path,
            custom_objects=_CUSTOM_OBJECTS,
            compile=False
        )

        _dummy = np.zeros((1, 32, 32, 3), dtype=np.float32)
        if not _enb3_flag:
            _dummy = _dummy / 255.0
        _m(_dummy, training=False)
        del _dummy

        model              = _m
        model_name         = _name
        is_enb3            = _enb3_flag
        img_size           = (32, 32)
        decision_threshold = _get_threshold(_mtype)
        gradcam_layer_name = _get_gradcam_layer(_mtype)

        print(f'✅ Loaded: {_name}')
        print(f'   input={model.input_shape}  output={model.output_shape}')
        print(f'   threshold={decision_threshold}  gradcam_layer={gradcam_layer_name}')

        if _enb3_flag:
            for _lyr in _m.layers:
                if hasattr(_lyr, 'layers') and 'efficientnet' in _lyr.name.lower():
                    enb3_submodel_name = _lyr.name
                    print(f'   ENB3 sub-model: "{enb3_submodel_name}" ({len(_lyr.layers)} layers)')
                    try:
                        _target = _lyr.get_layer(gradcam_layer_name)
                        print(f'   Grad-CAM target "{gradcam_layer_name}": {type(_target).__name__} ✅')
                    except ValueError:
                        print(f'   ⚠️  "{gradcam_layer_name}" not found — will use last Conv2D fallback')
                    break
            if enb3_submodel_name is None:
                print('   ⚠️  Could not find EfficientNet sub-model — Grad-CAM will use outer model')
        break

    except Exception as _e:
        print(f'⚠️  Failed to load {_fname}: {_e}')
        traceback.print_exc()

if model is None:
    print('⚠️  No model loaded — running in Demo Mode')


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(path):
    img = load_img(path, target_size=img_size)
    arr = img_to_array(img).astype(np.float32)
    if is_enb3:
        pass
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
def _make_gradcam_enb3(img_array, mdl, submodel_name, target_layer_name):
    base_model = mdl.get_layer(submodel_name)

    try:
        target_layer = base_model.get_layer(target_layer_name)
        print(f'   GradCAM: "{target_layer_name}" [{type(target_layer).__name__}]')
    except ValueError:
        target_layer = None
        for lyr in reversed(base_model.layers):
            if isinstance(lyr, tf.keras.layers.Conv2D):
                target_layer = lyr
                print(f'   GradCAM fallback Conv2D: "{lyr.name}"')
                break
        if target_layer is None:
            raise RuntimeError('No Grad-CAM target layer found in EfficientNetB3')

    base_grad_model = tf.keras.Model(
        inputs  = base_model.input,
        outputs = [target_layer.output, base_model.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        _out          = base_grad_model(img_tensor, training=False)
        conv_outputs  = tf.cast(_out[0], tf.float32)
        base_features = tf.cast(_out[1], tf.float32)
        tape.watch(conv_outputs)

        x = tf.reshape(base_features, [tf.shape(base_features)[0], -1])

        for lyr in mdl.layers:
            if lyr.name == submodel_name:
                continue
            if isinstance(lyr, (tf.keras.layers.InputLayer,
                                 tf.keras.layers.GlobalAveragePooling2D,
                                 tf.keras.layers.Dropout)):
                continue
            if isinstance(lyr, (tf.keras.layers.Dense,
                                 tf.keras.layers.BatchNormalization,
                                 tf.keras.layers.Activation)):
                x = lyr(x, training=False)

        score = x[:, 0]

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError('tape.gradient returned None — computation graph broken')

    return conv_outputs, grads, score


def _make_gradcam_cnn(img_array, mdl):
    target_layer = None
    for lyr in reversed(mdl.layers):
        if isinstance(lyr, tf.keras.layers.Conv2D):
            target_layer = lyr
            print(f'   GradCAM CNN: using last Conv2D "{lyr.name}"')
            break

    if target_layer is None:
        raise RuntimeError('No Conv2D layer found in CNN model')

    grad_model = tf.keras.Model(
        inputs  = mdl.inputs,
        outputs = [target_layer.output, mdl.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        outputs      = grad_model(img_tensor, training=False)
        conv_outputs = tf.cast(outputs[0], tf.float32)
        preds        = tf.cast(outputs[1], tf.float32)
        tape.watch(conv_outputs)
        score = preds[:, 0]

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError('tape.gradient returned None for CNN')

    return conv_outputs, grads, score


def _heatmap_from_grads(conv_outputs, grads):
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    if heatmap.shape.rank < 2:
        raise ValueError(f'Degenerate heatmap shape: {heatmap.shape}')

    heatmap = tf.maximum(heatmap, 0.0)
    mx      = tf.reduce_max(heatmap)

    if float(mx) <= 0:
        print('   GradCAM: all-zero heatmap — returning uniform fallback')
        shape = heatmap.shape.as_list()
        return np.ones(shape, dtype=np.float32) * 0.3

    return (heatmap / mx).numpy()


def make_gradcam(img_array, mdl):
    try:
        if is_enb3 and enb3_submodel_name is not None:
            conv_outputs, grads, _ = _make_gradcam_enb3(
                img_array, mdl, enb3_submodel_name, gradcam_layer_name)
        else:
            conv_outputs, grads, _ = _make_gradcam_cnn(img_array, mdl)

        heatmap = _heatmap_from_grads(conv_outputs, grads)
        print(f'   GradCAM: heatmap shape={heatmap.shape} '
              f'range=[{heatmap.min():.3f}, {heatmap.max():.3f}]')
        return heatmap

    except Exception as e:
        print(f'Grad-CAM error: {e}')
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  OVERLAY RENDERING
# ══════════════════════════════════════════════════════════════════════════════
def save_overlay(orig_path, heatmap, out_path, alpha=0.45):
    fig = None
    try:
        img = cv2.imread(orig_path)
        if img is None:
            raise IOError(f'cv2.imread returned None for {orig_path}')
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        hm      = cv2.resize(heatmap.astype(np.float32), (w, h))
        hm_rgba = cm_module.get_cmap('jet')(hm)
        hm_rgb  = np.uint8(hm_rgba[:, :, :3] * 255)
        blended = cv2.addWeighted(img, 1.0 - alpha, hm_rgb, alpha, 0)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.patch.set_facecolor('#07080d')
        for ax in axes:
            ax.axis('off')
            ax.set_facecolor('#07080d')

        axes[0].imshow(img);            axes[0].set_title('Original',         color='white', fontsize=11, pad=8)
        axes[1].imshow(hm, cmap='jet'); axes[1].set_title('Attention Map',    color='white', fontsize=11, pad=8)
        axes[2].imshow(blended);        axes[2].set_title('Grad-CAM Overlay', color='white', fontsize=11, pad=8)

        plt.suptitle('Gradient-weighted Class Activation Mapping',
                     color='#a78bfa', fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout(pad=1.5)
        plt.savefig(out_path, dpi=130, bbox_inches='tight',
                    facecolor='#07080d', edgecolor='none')
        return True

    except Exception as e:
        print(f'save_overlay error: {e}')
        traceback.print_exc()
        return False
    finally:
        if fig is not None:
            plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def interpret_prediction(score, threshold):
    is_real    = score >= threshold
    label      = 'Real Image' if is_real else 'AI-Generated Image'
    confidence = round((score if is_real else 1.0 - score) * 100.0, 1)
    return label, confidence


def _base_ctx(error=None, label=None, confidence=None,
              uploaded_image=None, gradcam_image=None,
              raw_score=None, is_demo=False):
    return dict(
        error          = error,
        label          = label,
        confidence     = confidence,
        model_used     = model_name,
        uploaded_image = uploaded_image,
        gradcam_image  = gradcam_image,
        raw_score      = str(round(raw_score, 5)) if raw_score is not None else None,
        threshold      = str(round(decision_threshold, 3)),
        is_demo        = is_demo,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def inner_page():
    return render_template('inner_page.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/gallery')
def gallery():
    return redirect(url_for('home'))

@app.route('/gallery_single')
def gallery_single():
    return redirect(url_for('home'))


@app.route('/api/status')
def status():
    return jsonify({
        'status':        'running',
        'model':         model_name,
        'model_loaded':  model is not None,
        'is_enb3':       is_enb3,
        'threshold':     decision_threshold,
        'gradcam_layer': gradcam_layer_name,
        'enb3_submodel': enb3_submodel_name,
        'config_loaded': bool(_model_config),
        'version':       'final',
    })


@app.route('/contact-send', methods=['POST'])
def contact_send():
    try:
        name    = request.form.get('name',    '').strip()
        email   = request.form.get('email',   '').strip()
        subject = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()

        if not name or not email or not message:
            return jsonify({'status': 'error',
                            'message': 'Name, email, and message are required.'}), 400

        body = (f"New message from your AI Image Classifier website:\n\n"
                f"From   : {name}\nEmail  : {email}\n"
                f"Subject: {subject}\n\nMessage:\n{message}\n")
        try:
            msg = Message(
                subject    = f'[AI Classifier] {subject or "Contact Form"} — from {name}',
                recipients = ['b21it092@kitsw.ac.in'],
                body       = body,
                reply_to   = email,
            )
            mail.send(msg)
        except Exception as mail_err:
            print(f'Mail send failed (logging): {mail_err}')
            with open(os.path.join(BASE_DIR, 'contact_messages.txt'), 'a', encoding='utf-8') as lf:
                lf.write(f'\n---\n{body}\n')

        return jsonify({'status': 'success', 'message': 'Message received!'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': 'Something went wrong.'}), 500


@app.route('/submit', methods=['POST'])
def submit():
    try:
        if 'file' not in request.files:
            return render_template('output.html', **_base_ctx(error='No file uploaded.'))

        f = request.files['file']
        if not f or not f.filename:
            return render_template('output.html', **_base_ctx(error='No file selected.'))

        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            return render_template('output.html', **_base_ctx(
                error=f'Format "{ext}" not supported. Upload JPG, PNG, BMP, or WebP.'))

        uid     = uuid.uuid4().hex[:12]
        fname   = f'{uid}{ext}'
        up_path = os.path.join(UPLOAD_FOLDER, fname)
        f.save(up_path)

        if model is None:
            import random
            return render_template('output.html', **_base_ctx(
                label='AI-Generated Image',
                confidence=round(random.uniform(55.0, 88.0), 1),
                uploaded_image=fname, is_demo=True))

        img_arr = preprocess(up_path)
        with _predict_lock:
            raw_pred = model.predict(img_arr, verbose=0)
        score = float(raw_pred[0][0])

        label, confidence = interpret_prediction(score, decision_threshold)

        print(f'   Prediction: sigmoid={score:.5f}  threshold={decision_threshold}'
              f'  → {label} ({confidence}%)')

        gc_fname = None
        try:
            heatmap = make_gradcam(img_arr, model)
            if heatmap is not None:
                gc_fname = f'gc_{uid}.jpg'
                gc_path  = os.path.join(GRADCAM_FOLDER, gc_fname)
                if not save_overlay(up_path, heatmap, gc_path):
                    print('   save_overlay returned False — gradcam_image will be None')
                    gc_fname = None
                else:
                    print(f'   Grad-CAM saved: {gc_fname}')
        except Exception as gc_err:
            print(f'Grad-CAM pipeline exception: {gc_err}')
            traceback.print_exc()
            gc_fname = None

        return render_template('output.html', **_base_ctx(
            label=label, confidence=confidence,
            uploaded_image=fname, gradcam_image=gc_fname,
            raw_score=score))

    except Exception as e:
        traceback.print_exc()
        return render_template('output.html',
                               **_base_ctx(error=f'Processing error: {str(e)}'))


if __name__ == '__main__':
    app.run(debug=True, port=2222, host='0.0.0.0', use_reloader=False)
