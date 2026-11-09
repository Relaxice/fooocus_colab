#!/bin/bash
# Fooocus Colab — one-time setup
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FOOOCUS_DIR="$(dirname "$SCRIPT_DIR")/Fooocus"
VENV="$SCRIPT_DIR/../fooocus_env"

echo "=== Fooocus Colab Setup ==="

# 1. Create venv if missing
if [ ! -f "$VENV/bin/activate" ]; then
  echo "Creating Python 3.11 venv..."
  python3.11 -m venv "$VENV" --prompt fooocus
fi
source "$VENV/bin/activate"

# 2. Install torch (CPU)
echo "Installing PyTorch (CPU)..."
pip install -q torch torchvision

# 3. Install Fooocus requirements
echo "Installing Fooocus dependencies..."
pip install -q -r "$FOOOCUS_DIR/requirements_versions.txt"

# 4. Create model directories
mkdir -p "$FOOOCUS_DIR/models/"{checkpoints,loras,embeddings,vae_approx,upscale_models,inpaint,controlnet,clip_vision}
mkdir -p "$FOOOCUS_DIR/models/prompt_expansion/fooocus_expansion"
mkdir -p "$FOOOCUS_DIR/outputs"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Models directory: $FOOOCUS_DIR/models/checkpoints"
echo "Put .safetensors model files there, or Fooocus will download them on first launch."
echo ""
echo "To launch:"
echo "  ./launch.sh              # local access on :7865"
echo "  ./launch.sh --share      # public Gradio URL"
echo "  ./launch.sh --threads=8  # more CPU threads"
