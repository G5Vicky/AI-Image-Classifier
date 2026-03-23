# 📥 Complete Installation Guide

> **Who is this for?** Anyone who has never used Python before.  
> Every single step is explained. Nothing is assumed.  
> Follow this exactly and you **will** get the app running.

---

## Before You Start — What You'll Install

| Tool | What it is | Why you need it |
|------|-----------|-----------------|
| Python | The programming language | The app is written in Python |
| Git | Version control tool | To download the project code |
| pip | Python package installer | To install the libraries the app needs |

---

## 🪟 Windows Installation

### 1. Install Python

**a)** Go to https://python.org/downloads  
**b)** Click the big yellow **"Download Python 3.10.x"** button  
**c)** Open the downloaded file  
**d)** On the first screen, ✅ **CHECK THE BOX: "Add Python to PATH"** (very important!)  
**e)** Click **"Install Now"**  
**f)** Wait for it to finish, then click Close  

**Test it worked:**  
Press `Win + R`, type `cmd`, press Enter. In the black window, type:
```
python --version
```
You should see something like `Python 3.10.11`. If you see an error, you missed the PATH checkbox — reinstall.

---

### 2. Install Git

**a)** Go to https://git-scm.com/downloads  
**b)** Click **"Download for Windows"**  
**c)** Run the installer  
**d)** Click **Next** on every screen (all defaults are fine)  
**e)** Click **Install**, then **Finish**  

**Test it worked:**  
In Command Prompt, type:
```
git --version
```
You should see `git version 2.x.x`.

---

### 3. Download the Project

In Command Prompt, type these two commands (press Enter after each):
```
git clone https://github.com/G5Vicky/AI-Image-Classifier.git
cd AI-Image-Classifier
```

You now have a folder called `AI-Image-Classifier` with all the code.

---

### 4. Create a Virtual Environment

```
python -m venv venv
```

This creates an isolated Python workspace. Now activate it:
```
venv\Scripts\activate
```

✅ You'll see `(venv)` appear at the start of the line:
```
(venv) C:\Users\YourName\AI-Image-Classifier>
```

> **Important:** Every time you open a new Command Prompt window, you must run `venv\Scripts\activate` again before running the app.

---

### 5. Install All Dependencies

```
pip install -r requirements.txt
```

**What this does:** Installs Flask, TensorFlow, OpenCV, and all other libraries the app needs.

⏱️ **This takes 5–15 minutes.** TensorFlow alone is ~500MB. Let it run — don't close the window.

When finished, you'll see your command prompt again.

---

### 6. Download the AI Models

The trained model files are too large for GitHub. Download them:
```
python scripts/download_models.py
```

This downloads `efficientnet_model.keras` (~109MB) from Hugging Face.  
⏱️ Takes 1–5 minutes depending on your internet speed.

---

### 7. Check Everything Is Ready

```
python scripts/verify_setup.py
```

Every item should show ✅. If something shows ❌, read the message and follow the fix suggestion.

---

### 8. Run the App

```
python app.py
```

Leave this window open. Open your browser and go to:
```
http://localhost:2222
```

You should see the AI Image Classifier homepage. 🎉

---

### 9. Stop the App

When you want to stop: go back to the Command Prompt window and press `Ctrl + C`.

---

## 🍎 Mac Installation

### 1. Install Python

**Option A (Recommended):** Install via Homebrew  
Open Terminal (`Cmd + Space` → type "Terminal" → Enter):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.10
```

**Option B:** Download from https://python.org/downloads → click macOS installer.

**Test:**
```bash
python3 --version
```

---

### 2. Install Git

```bash
brew install git
```
Or download from https://git-scm.com/downloads.

---

### 3. Download and Run

```bash
git clone https://github.com/G5Vicky/AI-Image-Classifier.git
cd AI-Image-Classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
python scripts/verify_setup.py
python app.py
```

Open: **http://localhost:2222**

---

## 🐧 Linux (Ubuntu/Debian) Installation

```bash
# Install Python and Git
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip git -y

# Clone the project
git clone https://github.com/G5Vicky/AI-Image-Classifier.git
cd AI-Image-Classifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py

# Verify
python scripts/verify_setup.py

# Run
python app.py
```

---

## ❓ Troubleshooting

### "python is not recognized"
You installed Python without checking "Add to PATH".  
Fix: Uninstall Python from Control Panel → reinstall → **check the PATH box**.

### "pip is not recognized"
Use `python -m pip` instead of just `pip`:
```
python -m pip install -r requirements.txt
```

### The app opens but shows "Demo Mode"
The model file wasn't downloaded. Run:
```
python scripts/download_models.py
```

### TensorFlow install fails with a long error
Install Microsoft Visual C++ Redistributable first:  
Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe  
Install it, then restart your computer, then try pip install again.

### "Port already in use" error
Another program is using port 2222. Fix: open `app.py`, go to the last line, change `port=2222` to `port=3333`. Then open http://localhost:3333 instead.

### "venv\Scripts\activate is not recognized" (Windows PowerShell)
Run this first:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activate again.

### The page loads but image upload does nothing
Make sure the `static/uploads/` and `static/gradcam/` folders exist. The app creates them automatically, but if there was a permissions error, create them manually:
```
mkdir static\uploads
mkdir static\gradcam
```

---

## ✅ Checklist Before Asking for Help

If something isn't working, check all of these:

- [ ] Python 3.10+ installed with PATH checkbox checked
- [ ] Virtual environment activated (you see `(venv)` in terminal)
- [ ] `pip install -r requirements.txt` completed without errors
- [ ] `python scripts/download_models.py` completed successfully
- [ ] `python scripts/verify_setup.py` shows all ✅
- [ ] You are in the `AI-Image-Classifier` folder when running commands

If all of these are checked and it still doesn't work, open a GitHub Issue with the exact error message from your terminal.
