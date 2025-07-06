# Dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    streamlit==1.28.0 \
    opencv-python-headless==4.8.0.76 \
    decord==0.6.0 \
    transformers==4.35.0 \
    sentencepiece==0.1.99 \
    accelerate==0.25.0 \
    einops==0.7.0 \
    timm==0.9.12 \
    moviepy==1.0.3 \
    pandas==2.1.3

# Create app directory
WORKDIR /app

# Copy application files
COPY . .

# Create directories
RUN mkdir -p /app/model_cache && \
    mkdir -p /data/videos && \
    mkdir -p /data/results

# Expose Streamlit port
EXPOSE 8501

# Set default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
