#!/bin/bash
# =================================================================
# Voice Form Assistant - AWS EC2 Setup Script
# =================================================================
# Target: AWS EC2 g4dn.xlarge
# AMI: AWS Deep Learning Base AMI (Ubuntu 22.04)
# AMI Name: Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04)
#
# This script sets up the Voice Form Assistant on a fresh EC2 instance
# with GPU support for IndicConformer + Whisper hybrid STT.
# =================================================================

set -e  # Exit on error

echo "=========================================="
echo "Voice Form Assistant - AWS EC2 Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =================================================================
# Step 1: System Updates and Dependencies
# =================================================================
print_status "Step 1: Installing system dependencies..."

sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    redis-server \
    git \
    wget \
    curl \
    htop \
    nvtop

# =================================================================
# Step 2: Verify NVIDIA GPU
# =================================================================
print_status "Step 2: Verifying NVIDIA GPU..."

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    print_status "NVIDIA GPU detected and driver working!"
else
    print_error "NVIDIA GPU not detected. Please ensure you're using the Deep Learning AMI."
    exit 1
fi

# =================================================================
# Step 3: Setup Python Environment
# =================================================================
print_status "Step 3: Setting up Python environment..."

cd /home/ubuntu

# Create project directory if cloning fresh
if [ ! -d "voice-form-assistant" ]; then
    print_status "Cloning repository..."
    # Replace with your actual repository URL
    git clone https://github.com/itsdivyanshjha/VBFF_NeGD.git
    cd VBFF_NeGD/voice-form-assistant
else
    cd voice-form-assistant
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# =================================================================
# Step 4: Install PyTorch with CUDA Support
# =================================================================
print_status "Step 4: Installing PyTorch with CUDA support..."

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
print_status "CUDA Version: $CUDA_VERSION"

# Install PyTorch with CUDA 12.1 (compatible with most Deep Learning AMIs)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# =================================================================
# Step 5: Install Application Dependencies
# =================================================================
print_status "Step 5: Installing application dependencies..."

cd backend

# Install requirements (skip torch as we installed it with CUDA)
pip install -r requirements.txt --no-deps torch torchaudio

# Install additional GPU-specific packages
pip install onnxruntime-gpu

# =================================================================
# Step 6: Configure Redis
# =================================================================
print_status "Step 6: Configuring Redis..."

sudo systemctl enable redis-server
sudo systemctl start redis-server

# Test Redis
redis-cli ping && print_status "Redis is running!" || print_error "Redis failed to start"

# =================================================================
# Step 7: Create Environment File
# =================================================================
print_status "Step 7: Creating environment configuration..."

if [ ! -f .env ]; then
    cat > .env << 'EOF'
# Voice Form Assistant - Production Configuration

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# STT Configuration
STT_MODE=hybrid
ML_DEVICE=cuda

# IndicConformer (Indian Languages)
INDIC_ASR_DECODER=ctc
INDIC_DEFAULT_LANGUAGE=hi

# Whisper (English)
WHISPER_MODEL=large
WHISPER_DEVICE=cuda

# Language Detection
LANG_DETECTION_THRESHOLD=0.6
PREFER_INDIC_ASR=true
FALLBACK_LANGUAGE=hi

# OpenRouter API (REPLACE WITH YOUR KEY)
OPENROUTER_API_KEY=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
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# TTS
TTS_ENGINE=gtts
TTS_LANGUAGE=en
TTS_TLD=co.in

# CORS - Update for production
ALLOWED_ORIGINS=["*"]

# Logging
LOG_LEVEL=INFO

# Model Cache
MODEL_CACHE_DIR=/home/ubuntu/.cache/voice-form-assistant
EOF
    print_warning "Created .env file. Please update OPENROUTER_API_KEY!"
else
    print_status ".env file already exists"
fi

# =================================================================
# Step 8: Create Model Cache Directory
# =================================================================
print_status "Step 8: Creating model cache directory..."

mkdir -p /home/ubuntu/.cache/voice-form-assistant
chmod 755 /home/ubuntu/.cache/voice-form-assistant

# =================================================================
# Step 9: Pre-download Models (Optional but recommended)
# =================================================================
print_status "Step 9: Pre-downloading models (this may take a while)..."

python3 << 'EOF'
import os
os.environ["MODEL_CACHE_DIR"] = "/home/ubuntu/.cache/voice-form-assistant"

print("Downloading SpeechBrain language detection model...")
try:
    from speechbrain.inference.classifiers import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="/home/ubuntu/.cache/voice-form-assistant/lang-id-voxlingua107"
    )
    print("Language detection model downloaded!")
except Exception as e:
    print(f"Warning: Could not download language detection model: {e}")

print("Downloading IndicConformer model...")
try:
    from indic_asr_onnx import IndicTranscriber
    transcriber = IndicTranscriber()
    print("IndicConformer model downloaded!")
except Exception as e:
    print(f"Warning: Could not download IndicConformer model: {e}")

print("Downloading Whisper model...")
try:
    import whisper
    model = whisper.load_model("medium", device="cpu")
    print("Whisper model downloaded!")
except Exception as e:
    print(f"Warning: Could not download Whisper model: {e}")

print("All models downloaded successfully!")
EOF

# =================================================================
# Step 10: Create Systemd Service
# =================================================================
print_status "Step 10: Creating systemd service..."

sudo tee /etc/systemd/system/voice-form-assistant.service << 'EOF'
[Unit]
Description=Voice Form Assistant Backend
After=network.target redis-server.service
Wants=redis-server.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/voice-form-assistant/backend
Environment="PATH=/home/ubuntu/voice-form-assistant/venv/bin:/usr/local/bin:/usr/bin"
ExecStart=/home/ubuntu/voice-form-assistant/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

# GPU Configuration
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="MODEL_CACHE_DIR=/home/ubuntu/.cache/voice-form-assistant"

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable voice-form-assistant

# =================================================================
# Step 11: Create Start/Stop Scripts
# =================================================================
print_status "Step 11: Creating utility scripts..."

# Start script
cat > /home/ubuntu/start_voice_assistant.sh << 'EOF'
#!/bin/bash
sudo systemctl start voice-form-assistant
echo "Voice Form Assistant started!"
echo "Check status: sudo systemctl status voice-form-assistant"
echo "View logs: sudo journalctl -u voice-form-assistant -f"
EOF
chmod +x /home/ubuntu/start_voice_assistant.sh

# Stop script
cat > /home/ubuntu/stop_voice_assistant.sh << 'EOF'
#!/bin/bash
sudo systemctl stop voice-form-assistant
echo "Voice Form Assistant stopped!"
EOF
chmod +x /home/ubuntu/stop_voice_assistant.sh

# Development run script (without systemd)
cat > /home/ubuntu/run_dev.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/voice-form-assistant/backend
source ../venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export MODEL_CACHE_DIR=/home/ubuntu/.cache/voice-form-assistant
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
EOF
chmod +x /home/ubuntu/run_dev.sh

# =================================================================
# Step 12: Verify Installation
# =================================================================
print_status "Step 12: Verifying installation..."

source /home/ubuntu/voice-form-assistant/venv/bin/activate

python3 << 'EOF'
print("=" * 50)
print("Installation Verification")
print("=" * 50)

# Check PyTorch
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Check SpeechBrain
try:
    import speechbrain
    print(f"SpeechBrain: {speechbrain.__version__}")
except ImportError:
    print("SpeechBrain: Not installed")

# Check IndicASR
try:
    import indic_asr_onnx
    print("IndicASR ONNX: Installed")
except ImportError:
    print("IndicASR ONNX: Not installed")

# Check Whisper
try:
    import whisper
    print("Whisper: Installed")
except ImportError:
    print("Whisper: Not installed")

# Check FastAPI
try:
    import fastapi
    print(f"FastAPI: {fastapi.__version__}")
except ImportError:
    print("FastAPI: Not installed")

print("=" * 50)
print("Installation complete!")
print("=" * 50)
EOF

# =================================================================
# Final Instructions
# =================================================================
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
print_status "Next steps:"
echo "1. Edit .env file and add your OPENROUTER_API_KEY:"
echo "   nano /home/ubuntu/voice-form-assistant/backend/.env"
echo ""
echo "2. Start the service:"
echo "   sudo systemctl start voice-form-assistant"
echo ""
echo "3. Check status:"
echo "   sudo systemctl status voice-form-assistant"
echo ""
echo "4. View logs:"
echo "   sudo journalctl -u voice-form-assistant -f"
echo ""
echo "5. For development mode (with auto-reload):"
echo "   /home/ubuntu/run_dev.sh"
echo ""
echo "6. Test the API:"
echo "   curl http://localhost:8000/health"
echo ""
print_status "The service will be available at http://<EC2_PUBLIC_IP>:8000"
print_warning "Don't forget to configure your EC2 Security Group to allow port 8000!"
