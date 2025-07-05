# JourneyText

Here are the complete instructions to run the journey summarization application on an **Azure Lab with NVIDIA V100 GPU**:

---

## **1. Set Up Azure Lab Environment**

### **A. Create Azure GPU Instance**
1. Log in to [Azure Portal](https://portal.azure.com/)
2. Select the vm and start

### **B. Connect to the VM**
```bash
ssh -i ~/.ssh/your_key.pem azureuser@<VM_Public_IP>
```

### **C. Install NVIDIA Drivers & CUDA**
```bash
# Install NVIDIA drivers (for Ubuntu 20.04/22.04)
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-dkms-535
sudo reboot

# Verify GPU detection
nvidia-smi
```
(Expected output: Should show Tesla V100 GPU)

```bash
# Install CUDA 12.1 (compatible with PyTorch)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt install -y cuda-12-1
```

### **D. Install Docker & NVIDIA Container Toolkit**
```bash
# Install Docker
sudo apt install -y docker.io
sudo systemctl enable --now docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
      && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# Verify NVIDIA Docker works
sudo docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```
(Should show GPU info)

---

## **2. Deploy the Journey Summarization App (GPU Version)**

### **A. Clone the Repository**
```bash
git clone https://github.com/your-repo/journey-summarization.git
cd journey-summarization
```

### **B. Build Docker Image (GPU Optimized)**
```bash
sudo docker build -t journey-summarizer-gpu .
```

### **C. Run the Container with GPU Support**
```bash
sudo docker run -d \
  --gpus all \
  -p 8501:8501 \
  --name journey-summarizer \
  journey-summarizer-gpu
```

### **D. Access the Web UI**
- Open in browser:  
  `http://<AZURE_VM_PUBLIC_IP>:8501`

---

## **3. Test GPU Performance**
### **A. Expected Performance (V100)**
| Video Length | Processing Time | GPU Utilization |
|--------------|----------------|----------------|
| 10 sec       | ~10-15 sec     | 70-80%         |
| 30 sec       | ~20-30 sec     | 80-90%         |
| 1 min        | ~45-60 sec     | 90-100%        |

### **B. Verify GPU Usage**
```bash
# Check GPU usage inside container
sudo docker exec -it journey-summarizer nvidia-smi
```
(Should show `python3` process using GPU)

---

## **4. Troubleshooting**
### **A. CUDA Errors**
If you see `CUDA out of memory`:
```bash
# Reduce batch size in utils.py (modify before building Docker)
BATCH_SIZE = 1  # Instead of 4 or 8
```

### **B. Docker Permission Issues**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### **C. Slow Processing**
- Use shorter videos (≤30 sec)
- Reduce `num_frames` in `utils.py` (e.g., from 16 → 8)

---

## **5. Shutting Down**
```bash
# Stop container
sudo docker stop journey-summarizer

# Delete container
sudo docker rm journey-summarizer

# (Optional) Delete image
sudo docker rmi journey-summarizer-gpu
```

---
### **Final Notes**
✅ **Works best with**:  
- Short videos (10-60 sec)  
- Well-lit driving footage  
- 1080p or 720p resolution  

🚀 **For better performance**:  
- Use **NC12s_v3** (2x V100) for longer videos  
- Increase `num_frames` in `utils.py` for more detailed analysis  

Now you can upload driving videos and get AI-generated navigation summaries! 🚗💨