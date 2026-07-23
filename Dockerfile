FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Local Interior Studio" \
      description="Fully offline interior design tool — AI restyling with zero cloud dependency"

# System deps for OpenCV headless + git for pip install from VCS
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/models \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps (copy requirements first for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p /app/output

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

ENTRYPOINT ["python", "app.py"]
