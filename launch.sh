#!/bin/bash
# Fooocus Colab — launcher
# Usage: ./launch.sh [--share] [--port PORT] [--cpu-threads N]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FOOOCUS_DIR="$(dirname "$SCRIPT_DIR")/Fooocus"
VENV="$SCRIPT_DIR/../fooocus_env"
SHARE=""
PORT="7865"
CPU_THREADS="4"

for arg in "$@"; do
  case "$arg" in
    --share)      SHARE="--share" ;;
    --port=*)     PORT="${arg#*=}" ;;
    --threads=*)  CPU_THREADS="${arg#*=}" ;;
  esac
done

echo "=== Fooocus Colab Launcher ==="
echo "Dir:     $FOOOCUS_DIR"
echo "Venv:    $VENV"
echo "Port:    $PORT"
echo "Threads: $CPU_THREADS (CPU mode)"
echo ""

# Activate venv
if [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
else
  echo "ERROR: venv not found at $VENV. Run setup.sh first."
  exit 1
fi

# Install / verify deps
pip install -q -r "$FOOOCUS_DIR/requirements_versions.txt" 2>/dev/null || true

cd "$FOOOCUS_DIR"

echo "Starting Fooocus on port $PORT (CPU mode — image gen will be slow)..."
python launch.py \
  --always-cpu "$CPU_THREADS" \
  --port "$PORT" \
  --disable-analytics \
  --theme dark \
  $SHARE
