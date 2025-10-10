#!/usr/bin/env bash
# ============================================================
# download_checkpoints.sh
# Usage:
#   bash download_checkpoints.sh model      # download all DA-V2 model weights
#   bash download_checkpoints.sh dragbench  # download DragBench dataset
#   bash download_checkpoints.sh all        # download both models and DragBench
# ============================================================

set -euo pipefail

SUBCMD="${1:-help}"

# ---------- utils ----------
need() { command -v "$1" >/dev/null 2>&1 || { echo "'$1' not found. Please install it."; exit 1; }; }
fetch() {
  local url="$1" out="$2"
  if [ -f "$out" ]; then
    echo "  $out already exists. Skipping download."
  else
    echo "  Downloading $out ..."
    wget --show-progress -c "$url?download=true" -O "$out"
  fi
}

# ---------- downloads ----------
download_models() {
  echo "Downloading Depth Anything V2 (ViT) checkpoints..."
  mkdir -p checkpoints && cd checkpoints

  # NOTE: we only use the Large model in our experiments
  # You can uncomment the other two lines to download all models if needed
  # fetch "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth" "depth_anything_v2_vits.pth"
  # fetch "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"  "depth_anything_v2_vitb.pth"
  fetch "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth" "depth_anything_v2_vitl.pth"

  echo "Models saved to: $(pwd)"
}

download_dragbench() {
  echo "Downloading DragBench dataset..."
  mkdir -p datasets && cd datasets

  local url="https://github.com/Yujun-Shi/DragDiffusion/releases/download/v0.1.1/DragBench.zip"
  local file="DragBench.zip"

  if [ -f "$file" ]; then
      echo "  $file already exists. Skipping download."
  else
      echo "  Downloading DragBench dataset..."
      wget -q --show-progress -c "$url" -O "$file"
  fi

  echo "  Extracting..."
  unzip -q -o "$file"
  rm -f "$file"

  echo "DragBench dataset is ready at: $(pwd)/DragBench"
}

# ---------- main ----------
need wget
need unzip

case "$SUBCMD" in
  model)
    download_models
    ;;
  dragbench)
    download_dragbench
    ;;
  all)
    download_models
    cd - >/dev/null
    download_dragbench
    ;;
  help|*)
    cat <<'EOF'
Usage:
  bash download_checkpoints.sh model      # download all DA-V2 model weights (vits/vitb/vitl)
  bash download_checkpoints.sh dragbench  # download DragBench dataset
  bash download_checkpoints.sh all        # download both models and DragBench

Examples:
  bash download_checkpoints.sh model
  bash download_checkpoints.sh dragbench
EOF
    ;;
esac