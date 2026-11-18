# Evaluate other models

Run every command from the repository root. The Qwen-family models run in the environment of this repository. Every other model runs in its own environment, with its repository cloned under `third_party/`:

| Model | `third_party/` folder | Repository @ commit | Environment we used |
|---|---|---|---|
| VLM-3R | `VLM-3R` | `VITA-Group/VLM-3R` @ `206afd0` | Python 3.10, torch 2.1.1, transformers 4.40.0 |
| Cambrian-S-7B | `cambrian-s` | `cambrian-mllm/cambrian-s` @ `058b1b4` | Python 3.10, torch 2.6.0, transformers 4.37.0 |
| Spatial-TTT | `Spatial-TTT` | `THU-SI/Spatial-TTT` @ `e2e33a6` | Python 3.12, torch 2.8.0, transformers 4.57.0 |
| TimeChat-Online | `TimeChat-Online` | `yaolinli/TimeChat-Online` @ `25a1578` | Python 3.10, torch 2.4.0, transformers 4.49.0 |
| LiveCC | `livecc` | `showlab/livecc` @ `fdaa91e` | Python 3.11, torch 2.6.0, transformers 4.50.0 |
| Dispider | `Dispider` | `Mark12Ding/Dispider` @ `803f6b4` | Python 3.10, torch 2.2.0, transformers 4.41.2 |
| Stream3D-VLM | `Stream3D-VLM` | `hanxunyu/Stream3D-VLM` @ `6a7929a`, then `git apply ../../patches/stream3d_vlm_eval.patch` | Python 3.10, torch 2.5.1, transformers 4.50.0 |

**Prefix polling**: every second the model sees the question and all frames so far and answers Yes / No.

```bash
bash scripts/eval_polling.sh <tag> <family> <model> 0,1,2,3
```

| Model | `<family>` | `<model>` |
|---|---|---|
| Qwen2.5-VL-7B / 32B | `qwen25` | `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-32B-Instruct` |
| Qwen3-VL-8B / 32B | `qwen3` | `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-32B-Instruct` |
| Cosmos-Reason2-32B | `qwen3` | `nvidia/Cosmos-Reason2-32B` |
| SenseNova-SI-1.1-8B | `qwen3` | `sensenova/SenseNova-SI-1.1-Qwen3-VL-8B` |
| VLM-3R | `vlm3r_video` | `Journey9ni/vlm-3r-llava-qwen2-lora` |
| Cambrian-S-7B | `cambrians_video` | `nyu-visionx/Cambrian-S-7B` |
| Spatial-TTT | `spatial_ttt` | `THU-SI/Spatial-TTT-nano` |

**Native streaming**: the model runs its own streaming inference with the request given at 0 s.

```bash
bash scripts/eval_streaming.sh <tag> <timechat_online | livecc | dispider | stream3d_vlm> 0
```

Both write `results/<tag>/` and print the row. Any other Hugging Face chat model runs with `<family>` = `images`.
