# run.sh
#!/bin/bash

# Build Docker image
docker build -t journey-summarizer .

# Create data directories
mkdir -p ./data/videos ./data/results

# Run container with GPU support
docker run -d --gpus all \
  --name journey-container \
  -p 8501:8501 \
  -v $(pwd)/data/videos:/data/videos \
  -v $(pwd)/data/results:/data/results \
  -v journey-model-cache:/app/model_cache \
  journey-summarizer

echo "Web UI: http://localhost:8501"
echo ""
echo "CLI Examples:"
echo "  docker exec journey-container python cli.py /data/videos/drive.mp4"
echo "  docker exec journey-container python cli.py /data/videos/trip.mp4 --output /data/results/summary.json"