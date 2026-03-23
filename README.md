<div align="center">

# 🔍 AI Image Classifier
### Can your eyes tell the difference? This system can.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Try%20It%20Now-FF6B35?style=for-the-badge)](https://huggingface.co/spaces/Vicky25july2003/ai-image-classifier)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

Upload any image. Get an instant answer: **Real Photo** or **AI-Generated**.  
Plus a heatmap showing *exactly which parts of the image* gave it away.

<br/>

> 🎓 Major Project Phase-II · B.Tech Information Technology  
> Kakatiya Institute of Technology & Science, Warangal · 2024–2025

</div>

---

## 📌 Table of Contents

1. [What Does This Do?](#-what-does-this-do)
2. [Live Demo](#-live-demo)
3. [How It Works (Simple)](#-how-it-works-simple)
4. [Tech Stack](#-tech-stack)
5. [Results](#-results)
6. [Quick Start (I Know Python)](#-quick-start)
7. [Complete Installation Guide (I'm a Beginner)](#-complete-installation-guide)
8. [Project Structure](#-project-structure)
9. [Running the App](#-running-the-app)
10. [Deploying to the Internet (Free)](#-deploying-for-free)
11. [Common Problems & Fixes](#-common-problems--fixes)
12. [Team](#-team)

---

## 🤔 What Does This Do?

AI tools like DALL·E, Midjourney, and Stable Diffusion can now create fake photos that look completely real. This is a serious problem for:

- 📰 **News** — fake photos used as "evidence" in news stories
- ⚖️ **Courts** — AI-generated images submitted as real evidence
- 🔐 **Security** — fake faces unlocking face-recognition systems
- 🎨 **Art** — AI art falsely claimed as human-made

This project builds a system that can **automatically detect** whether an image is a real photograph or AI-generated — with **93% accuracy** — and explains *why* it made that decision using a color heatmap.

---

## 🚀 Live Demo

**[👉 Click here to try it — no account needed](https://huggingface.co/spaces/Vicky25july2003/ai-image-classifier)**

> ⏱️ **First visit may take 30–60 seconds to load** (the server wakes up from sleep on free hosting). This is normal. Refresh if needed.

---

## 🧠 How It Works (Simple)

```
You upload an image
        ↓
The system resizes it to 32×32 pixels
        ↓
A neural network (EfficientNetB3) analyzes it
        ↓
Output: "Real Image" or "AI-Generated Image" + confidence %
        ↓
Grad-CAM generates a heatmap showing WHERE the AI looked
        ↓
You see: original photo + heatmap + blended overlay
```

**The heatmap uses colors:**
- 🔴 **Red/Yellow** = the AI looked here a lot (suspicious region)
- 🔵 **Blue** = the AI mostly ignored this area

---

## 🛠️ Tech Stack

| What | Tool | Why |
|------|------|-----|
| **Main model** | EfficientNetB3 (Google, 2019) | Best accuracy-per-parameter ratio |
| **Backup model** | Custom CNN (built from scratch) | Fallback + learning exercise |
| **Explainability** | Grad-CAM | Shows which pixels influenced the decision |
| **Backend** | Flask 3.0 (Python) | Lightweight web server |
| **ML Framework** | TensorFlow 2.16 / Keras | Industry standard for deep learning |
| **Image processing** | OpenCV + Pillow | Reading and manipulating images |
| **Frontend** | HTML + Bootstrap 5 | Responsive dark-theme UI |
| **Deployment** | Hugging Face Spaces (Docker) | Free, permanent public URL |

---

## 📊 Results

| Metric | Custom CNN | **EfficientNetB3** |
|--------|:----------:|:------------------:|
| Accuracy | ~88% | **~93%** |
| AUC-ROC | ~97% | **~98%** |
| F1-Score | ~87% | **~91%** |

Trained on **140,000 images** (CIFAKE dataset: 70,000 real + 70,000 AI-generated).

---

## ⚡ Quick Start

*If you already know Python and Git, these 5 commands get you running:*

```bash
git clone https://github.com/G5Vicky/AI-Image-Classifier.git
cd AI-Image-Classifier
python -m venv venv && venv\Scripts\activate    # Windows
# OR: source venv/bin/activate                  # Mac/Linux
pip install -r requirements.txt
python scripts/download_models.py
python app.py
```

Then open **http://localhost:2222** in your browser.

**Never used Python before?** → Read the [Complete Installation Guide](#-complete-installation-guide) below.

---

## 📖 Complete Installation Guide

> This guide assumes **zero prior knowledge**. Follow every step in order.  
> For even more detail, see **[INSTALLATION.md](INSTALLATION.md)**.

### Step 1 — Install Python

1. Go to **https://python.org/downloads**
2. Click the big yellow "Download Python 3.10" (or higher) button
3. Run the installer
4. ✅ **IMPORTANT:** Check the box **"Add Python to PATH"** before clicking Install
5. Click "Install Now"

**Verify it worked** — open Command Prompt (`Win+R` → type `cmd` → Enter):
```
python --version
```
You should see `Python 3.10.x` or higher. If not, reinstall and check the PATH box.

---

### Step 2 — Install Git

1. Go to **https://git-scm.com/downloads**
2. Download for your operating system
3. Install with all default settings

---

### Step 3 — Download This Project

Open Command Prompt and run:
```bash
git clone https://github.com/G5Vicky/AI-Image-Classifier.git
cd AI-Image-Classifier
```

This downloads all the code into a folder called `AI-Image-Classifier`.

---

### Step 4 — Create a Virtual Environment

*A virtual environment is a clean, isolated Python workspace for this project.*

```bash
python -m venv venv
```

**Activate it:**
```bash
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac or Linux
```

✅ You'll see `(venv)` appear at the start of your command line. This means it's active.

> **You must activate the venv every time you open a new Command Prompt window.**

---

### Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs everything the project needs (Flask, TensorFlow, etc.).  
⏱️ **This takes 5–10 minutes** — TensorFlow is ~500MB. Be patient.

---

### Step 6 — Download the AI Models

The trained model files are too large to include in the GitHub repo (109MB). Download them automatically:

```bash
python scripts/download_models.py
```

This downloads `efficientnet_model.keras` into the `models/` folder.  
⏱️ Takes 1–3 minutes depending on your internet speed.

---

### Step 7 — Verify Everything Is Ready

```bash
python scripts/verify_setup.py
```

All items should show ✅. If anything shows ❌, read the error message and fix it before continuing.

---

### Step 8 — Run the App

```bash
python app.py
```

You'll see:
```
✅ Loaded model_config.json
✅ Loaded: EfficientNetB3
 * Running on http://0.0.0.0:2222
```

Open your browser and go to: **http://localhost:2222**

---

### Step 9 — Use It!

1. Click **INPUT** in the navigation bar
2. Drag an image onto the upload area (or click to browse)
3. Click **Submit**
4. Wait 2–5 seconds
5. See your result: label, confidence, and Grad-CAM heatmap

---

## 📁 Project Structure

```
AI-Image-Classifier/
│
├── app.py                  ← The entire backend (Flask + model + Grad-CAM)
├── requirements.txt        ← Python packages list
├── Dockerfile              ← For Docker/HF Spaces deployment
├── Procfile                ← Gunicorn start command
├── .gitattributes          ← Git LFS config (handles large binary files)
├── .gitignore              ← Files excluded from Git
├── .env.example            ← Environment variables template
│
├── models/
│   ├── model_config.json   ← Model thresholds and settings (included)
│   ├── README.md           ← How to download the .keras model files
│   └── *.keras             ← NOT in repo — download via scripts/download_models.py
│
├── templates/              ← HTML pages (Jinja2 templates for Flask)
│   ├── index.html          ← Home page
│   ├── inner_page.html     ← Upload page
│   ├── output.html         ← Results + Grad-CAM display
│   ├── about.html
│   ├── contact.html
│   └── services.html
│
├── static/                 ← CSS, JavaScript, images (served directly)
│   ├── assets/             ← Bootstrap, AOS, Swiper vendor libraries
│   ├── uploads/            ← Where uploaded images go (created at runtime)
│   └── gradcam/            ← Where Grad-CAM outputs go (created at runtime)
│
├── docs/
│   ├── HOW_IT_WORKS.md     ← Technical architecture explained simply
│   ├── GRADCAM.md          ← How Grad-CAM works (with diagrams)
│   └── DEPLOYMENT.md       ← Step-by-step free deployment guide
│
└── scripts/
    ├── download_models.py  ← Downloads model files from Hugging Face
    └── verify_setup.py     ← Checks everything is installed correctly
```

---

## 🖥️ Running the App

### Standard run (development)
```bash
python app.py
```
Open: http://localhost:2222

### Production run (Gunicorn)
```bash
gunicorn app:app --bind 0.0.0.0:2222 --workers 1 --timeout 300
```

### Check if everything is working
```bash
# Visit this URL in your browser after starting:
http://localhost:2222/api/status
```
You should see JSON with `"model_loaded": true`.

---

## 🌐 Deploying for Free

Want to put this on the internet so anyone can use it?  
Full instructions: **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**

**Short version:**
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Upload your model: `python scripts/upload_models.py`
3. Create a new Space (Docker type)
4. Push the code: `git push space master:main`
5. Your app is live at `huggingface.co/spaces/YOUR_USERNAME/AI-Image-Classifier`

---

## 🔧 Common Problems & Fixes

| Problem | What it means | Fix |
|---------|--------------|-----|
| `python not recognized` | Python not in PATH | Reinstall Python, check "Add to PATH" box |
| `pip not recognized` | Same issue | Use `python -m pip install ...` instead |
| `No module named flask` | Virtual env not activated | Run `venv\Scripts\activate` first |
| App says "Demo Mode" | Model not downloaded | Run `python scripts/download_models.py` |
| Port 2222 already in use | Another app is using it | Change `port=2222` to `port=3333` in last line of app.py |
| Page loads but upload fails | Missing static folders | They're created automatically — restart the app |
| Grad-CAM not showing | Model loading issue | Check terminal output for error messages |
| `tensorflow` install fails | Needs Visual C++ (Windows) | Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |

Still stuck? Open an issue on GitHub or check [INSTALLATION.md](INSTALLATION.md) for more detail.

---

## 👥 Team

| Name | Roll No | Contribution |
|------|---------|-------------|
| **G. Vigneshwar** | B21IT092 | Flask backend, Grad-CAM implementation, deployment |
| **Ch. Rohini** | B21IT096 | Model training, evaluation, dataset pipeline |
| **B. Sumeet** | B21IT106 | Data preprocessing, training experiments |
| **Shaik Thohid** | B21IT118 | Frontend UI, integration testing |

**Guide:** Sri. K. Goutham Raju, Assistant Professor, Dept. of IT, KITSW

---

## 📄 License

MIT License — free to use for educational and non-commercial purposes. See [LICENSE](LICENSE).

---

<div align="center">

**⭐ If this project helped you, please star the repo!**

Made with ❤️ at KITSW, Warangal | 2024–2025

</div>
