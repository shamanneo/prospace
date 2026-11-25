# Setup

`bash scripts/setup.sh` does everything below in one step; this page lists the steps for a manual install and for troubleshooting.

## What `setup.sh` installs

| Step | Details |
|---|---|
| Environment `prospace` | Python 3.11 and `requirements.txt` (the versions our numbers were produced with), and ffmpeg 4.4.2 from conda-forge (to rebuild some ScanNet++ training frames byte for byte) |
| CUT3R | `third_party/CUT3R` at commit `5124436`, `patches/cut3r_rope.patch` applied, weights `src/cut3r_512_dpt_4_64.pth` (3.2 GB) |
| Environment `habitat` | Python 3.9, habitat-sim 0.3.1 (headless), `libopengl`, `libegl`, pillow. Only needed to render the Matterport3D frames; `SKIP_HABITAT=1 bash scripts/setup.sh` leaves it out |

It works with conda, mamba or micromamba, skips the steps that are already done, and ends with a check of torch, the GPUs and CUT3R. `PROSPACE_ENV` and `HABITAT_ENV` change the environment names.

## Manual install

```bash
conda create -n prospace python=3.11 && conda activate prospace
pip install -r requirements.txt
conda install -n prospace -c conda-forge ffmpeg=4.4.2   # only for the ScanNet++ training frames

git clone https://github.com/Jian-Zhang-3DV/CUT3R.git third_party/CUT3R
cd third_party/CUT3R
git checkout 51244364af3566d6473559f71a81b4accc75c424
git apply ../../patches/cut3r_rope.patch
gdown 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD -O src/cut3r_512_dpt_4_64.pth
cd ../..

# only for the Matterport3D frames
conda create -n habitat python=3.9
conda install -n habitat -c conda-forge -c aihabitat habitat-sim=0.3.1 headless libopengl libegl
conda run -n habitat pip install pillow
```

## Requirements

- An NVIDIA GPU. Evaluation runs on one GPU with about 40 GB of memory; we trained on four H200 GPUs (141 GB each, peak about 55 GiB per GPU).
- `torch==2.13.0` from PyPI is a CUDA 13 build and needs NVIDIA driver 580 or newer. With an older driver, install the CUDA 12.8 build instead: `pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu128` (our numbers were produced with the CUDA 13 build).
- Disk: about 6 GB for the environment, 3 GB for CUT3R, 16 GB for the base model `Qwen/Qwen2.5-VL-7B-Instruct` (downloaded on first use), 8 GB for the benchmark frames and about 100 GB for the training frames.
- `wandb` is optional (`pip install wandb`), for training curves with `WANDB=1`.

## Troubleshooting

| Message | Fix |
|---|---|
| `ImportError: libOpenGL.so.0` when importing `habitat_sim` | `conda install -n habitat -c conda-forge libopengl libegl`, or `apt install libopengl0 libegl1` |
| `cannot find cuda-compiled version of RoPE2D` | Expected: the CUT3R patch uses the PyTorch implementation |
| `unauthenticated requests to the HF Hub` | Expected: all files are public, no login is needed |
| `some frames are missing` | Build the frames as in [data.md](data.md) and run `python scripts/check_frames.py` |
| `some streams have no record yet` from `score.py` | The run has not finished; run the same command again, or score with `--partial` |
