# Evaluate other models

Run every command from the repository root. The weights are the Hugging Face repositories linked below: the runners load them by name, so they go to the Hugging Face cache and no folder has to be filled in. The Qwen-family models run in the environment of this repository. Every other model runs in its own environment, with its repository cloned under `third_party/`:

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
bash scripts/eval_polling.sh <tag> <family> <weights> 0,1,2,3
```

| Model | `<family>` | `<weights>` |
|---|---|---|
| Qwen2.5-VL-7B | `qwen25` | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Qwen2.5-VL-32B | `qwen25` | [`Qwen/Qwen2.5-VL-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) |
| Qwen3-VL-8B | `qwen3` | [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen3-VL-32B | `qwen3` | [`Qwen/Qwen3-VL-32B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| Cosmos-Reason2-32B | `qwen3` | [`nvidia/Cosmos-Reason2-32B`](https://huggingface.co/nvidia/Cosmos-Reason2-32B) |
| SenseNova-SI-1.1-8B | `qwen3` | [`sensenova/SenseNova-SI-1.1-Qwen3-VL-8B`](https://huggingface.co/sensenova/SenseNova-SI-1.1-Qwen3-VL-8B) |
| VLM-3R | `vlm3r_video` | [`Journey9ni/vlm-3r-llava-qwen2-lora`](https://huggingface.co/Journey9ni/vlm-3r-llava-qwen2-lora), on the base [`lmms-lab/LLaVA-NeXT-Video-7B-Qwen2`](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B-Qwen2) |
| Cambrian-S-7B | `cambrians_video` | [`nyu-visionx/Cambrian-S-7B`](https://huggingface.co/nyu-visionx/Cambrian-S-7B) |
| Spatial-TTT | `spatial_ttt` | [`THU-SI/Spatial-TTT-nano`](https://huggingface.co/THU-SI/Spatial-TTT-nano), on the base [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |

**Native streaming**: the model runs its own streaming inference with the request given at 0 s.

```bash
bash scripts/eval_streaming.sh <tag> <model> 0
```

| Model | `<model>` | Weights |
|---|---|---|
| TimeChat-Online | `timechat_online` | [`wyccccc/TimeChatOnline-7B`](https://huggingface.co/wyccccc/TimeChatOnline-7B) |
| LiveCC | `livecc` | [`chenjoya/LiveCC-7B-Instruct`](https://huggingface.co/chenjoya/LiveCC-7B-Instruct) |
| Dispider | `dispider` | [`Mar2Ding/Dispider`](https://huggingface.co/Mar2Ding/Dispider) |
| Stream3D-VLM | `stream3d_vlm` | [`JonnyYu828/Stream3D-VLM-4B`](https://huggingface.co/JonnyYu828/Stream3D-VLM-4B) |

Both write `results/<tag>/` and print the row. Any other Hugging Face chat model runs with `<family>` = `images`.
