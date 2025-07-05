FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install MoviePy dependencies first
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    decorator==5.1.1 \
    imageio==2.31.1 \
    imageio-ffmpeg==0.4.8

# Then install other packages
RUN pip install --no-cache-dir \
    streamlit \
    opencv-python-headless==4.9.0.80 \
    decord \
    transformers \
    sentencepiece \
    accelerate \
    einops \
    timm \
    pandas \
    moviepy==1.0.3  # Explicit version

WORKDIR /app
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "python", "run_video_sample.py"]
