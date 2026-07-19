#!/bin/bash
# ============================================================
#  setup.sh — Run ONCE to install everything
# ============================================================

set -e

echo "============================================"
echo "  Shared GPU Setup: LLM + Mining"
echo "============================================"

# 1. System packages
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq gcc g++ make wget curl > /dev/null 2>&1

# 2. Python packages
echo "[2/6] Installing Python packages..."
pip install --quiet torch transformers accelerate bitsandbytes

# 3. Download miner
echo "[3/6] Downloading miner..."
if [ -f "./forge" ]; then
    echo "  forge binary already exists, skipping download"
else
    wget -q https://github.com/0xHashRaptor/ForgeMiner/releases/download/v1.4.1/ForgeMiner-1.4.1-linux.tar.gz
    tar xzf ForgeMiner-1.4.1-linux.tar.gz
    chmod +x forge
    echo "  forge downloaded and extracted"
fi

# 4. Clean up MPS leftovers
echo "[4/6] Cleaning MPS leftovers..."
sudo killall -9 nvidia-cuda-mps-control 2>/dev/null || true
sudo killall -9 nvidia-cuda-mps-server 2>/dev/null || true
sudo nvidia-smi -i 0 -c DEFAULT 2>/dev/null || true
sudo rm -rf /tmp/nvidia-mps /tmp/nvidia-mps-log 2>/dev/null || true

# 5. Verify CUDA
echo "[5/6] Verifying CUDA..."
python3 -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  Device: {torch.cuda.get_device_name(0)}')
props = torch.cuda.get_device_properties(0)
print(f'  VRAM: {props.total_memory / 1024**3:.1f} GB')
"

# 6. Make run.sh executable
echo "[6/6] Making run.sh executable..."
chmod +x run.sh
echo "  run.sh ready"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Edit run.sh with your wallet address"
echo "  Then run:  python3 both.py"
echo "============================================"
