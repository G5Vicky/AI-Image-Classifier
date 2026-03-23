# ============================================================
# AI Image Classifier — Dockerfile
# Used for Hugging Face Spaces deployment
#
# Build locally:  docker build -t ai-classifier .
# Run locally:    docker run -p 7860:7860 ai-classifier
# ============================================================

FROM python:3.10-slim

# HF Spaces requires user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies first (cached Docker layer)
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY --chown=user . /app

# Create folders that the app writes to at runtime
RUN mkdir -p /app/static/uploads \
             /app/static/gradcam  \
             /app/models

# Port 7860 is required by Hugging Face Spaces
# Do NOT change this for HF deployment
EXPOSE 7860

# Start the app with Gunicorn
# 1 worker = safest for free-tier (avoids TF graph corruption with multiple workers)
# 300s timeout = allows time for model download on first cold start
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300"]
