#!/usr/bin/env bash
# One-step setup: the main environment, CUT3R, and (optionally) the habitat-sim environment for the Matterport3D frames.
#   bash scripts/setup.sh              # environment "prospace" + CUT3R
#   bash scripts/setup.sh --habitat    # also environment "habitat" for scripts/render_mp3d.py
# Uses conda, mamba or micromamba, whichever is installed.  PROSPACE_ENV=<name> and HABITAT_ENV=<name> change the environment names.
# Every step is skipped when it is already done, so the script can be run again after a failure.
set -eu
cd "$(dirname "$0")/.."
ENV_NAME=${PROSPACE_ENV:-prospace}; HABITAT_ENV=${HABITAT_ENV:-habitat}; HABITAT=0
[ "${1:-}" = "--habitat" ] && HABITAT=1
say() { printf '\n== %s\n' "$*"; }

for m in conda mamba micromamba; do command -v $m > /dev/null && { MGR=$m; break; }; done
[ -n "${MGR:-}" ] || { echo "Install conda, mamba or micromamba first."; exit 1; }
run() { $MGR run -n "$1" "${@:2}"; }
has_env() { $MGR env list 2>/dev/null | awk '{print $1}' | grep -qx "$1"; }

say "GPU driver"
if command -v nvidia-smi > /dev/null; then
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  echo "driver $drv"
  if [ "${drv%%.*}" -lt 580 ]; then
    echo "torch 2.13.0 from PyPI is a CUDA 13 build and needs driver 580 or newer."
    echo "With this driver, install the CUDA 12.8 build after this script:"
    echo "  $MGR run -n $ENV_NAME pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu128"
    echo "(our numbers were produced with the CUDA 13 build)"
  fi
else
  echo "nvidia-smi not found; evaluation and training need an NVIDIA GPU"
fi

say "environment $ENV_NAME (Python 3.11)"
has_env "$ENV_NAME" || $MGR create -y -n "$ENV_NAME" python=3.11
run "$ENV_NAME" pip install -r requirements.txt

say "CUT3R"
if [ ! -d third_party/CUT3R/.git ]; then
  git clone https://github.com/Jian-Zhang-3DV/CUT3R.git third_party/CUT3R
  git -C third_party/CUT3R checkout 51244364af3566d6473559f71a81b4accc75c424
  git -C third_party/CUT3R apply ../../patches/cut3r_rope.patch
fi
[ -f third_party/CUT3R/src/cut3r_512_dpt_4_64.pth ] ||
  run "$ENV_NAME" gdown 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD -O third_party/CUT3R/src/cut3r_512_dpt_4_64.pth   # 3.2 GB

if [ $HABITAT = 1 ]; then
  say "environment $HABITAT_ENV (habitat-sim 0.3.1, Python 3.9)"
  has_env "$HABITAT_ENV" || $MGR create -y -n "$HABITAT_ENV" python=3.9
  # libopengl and libegl provide libOpenGL.so.0 and libEGL, which headless habitat-sim loads at import
  $MGR install -y -n "$HABITAT_ENV" -c conda-forge -c aihabitat habitat-sim=0.3.1 headless libopengl libegl
  run "$HABITAT_ENV" pip install pillow
  run "$HABITAT_ENV" python -c "import habitat_sim; print('habitat-sim', habitat_sim.__version__)"
fi

say "check"
run "$ENV_NAME" python - << 'EOF'
import sys
import torch, transformers
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__)
print("CUDA available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
sys.path.insert(0, "third_party/CUT3R/src")
import dust3r  # noqa: F401
print("CUT3R import: ok")
EOF
echo
echo "Done. Activate with: $MGR activate $ENV_NAME"
