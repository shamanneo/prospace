# Evaluate other models

Run every command from the repository root. The weights are the Hugging Face repositories linked below: the runners load them by name, so they go to the Hugging Face cache and no folder has to be filled in. The models of the `qwen25`, `qwen3` and `images` families run in the environment of this repository; InternVideo3 needs transformers 4.57.3 instead. Every other model runs in its own environment, with its repository cloned under `third_party/`:

| Model | `third_party/` folder | Repository @ commit | Environment we used |
|---|---|---|---|
| VLM-3R | `VLM-3R` | `VITA-Group/VLM-3R` @ `206afd0` | Python 3.10, torch 2.1.1, transformers 4.40.0 |
| Cambrian-S-7B | `cambrian-s` | `cambrian-mllm/cambrian-s` @ `058b1b4` | Python 3.10, torch 2.6.0, transformers 4.37.0 |
| Spatial-TTT | `Spatial-TTT` | `THU-SI/Spatial-TTT` @ `e2e33a6` | Python 3.12, torch 2.8.0, transformers 4.57.0 |
| TimeChat-Online | `TimeChat-Online` | `yaolinli/TimeChat-Online` @ `25a1578` | Python 3.10, torch 2.4.0, transformers 4.49.0 |
| LiveCC | `livecc` | `showlab/livecc` @ `fdaa91e` | Python 3.11, torch 2.6.0, transformers 4.50.0 |
| Dispider | `Dispider` | `Mark12Ding/Dispider` @ `803f6b4` | Python 3.10, torch 2.2.0, transformers 4.41.2 |
| Flash-VStream | `Flash-VStream` | `IVGSZ/Flash-VStream` @ `8f6bde2` | Python 3.10, torch 2.6.0, transformers 4.45.0, flash-attn |
| MMDuet | `MMDuet` | `yellow-binary-tree/MMDuet` @ `33be138` | Python 3.10, torch 2.1.2, transformers 4.44.2, peft 0.12.0 |
| STRIDE | `STRIDE` | `interlive-team/STRIDE` @ `153adb7`, then `git apply ../../patches/stride_eval.patch` | Python 3.11, torch 2.5.1, transformers 5.15.1, flash-attn 2.7.4.post1 |
| Stream3D-VLM | `Stream3D-VLM` | `hanxunyu/Stream3D-VLM` @ `6a7929a`, then `git apply ../../patches/stream3d_vlm_eval.patch` | Python 3.10, torch 2.5.1, transformers 4.50.0 |

**Prefix polling**: every second the model sees the question and all frames so far and answers Yes / No.

```bash
bash scripts/eval.sh <model> 0,1,2,3
```

| Model | `<model>` | Weights |
|---|---|---|
| Qwen2.5-VL-7B | `qwen2.5-vl-7b` | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Qwen2.5-VL-32B | `qwen2.5-vl-32b` | [`Qwen/Qwen2.5-VL-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) |
| Qwen3-VL-2B | `qwen3-vl-2b` | [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| Qwen3-VL-8B | `qwen3-vl-8b` | [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen3-VL-32B | `qwen3-vl-32b` | [`Qwen/Qwen3-VL-32B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| Qwen3-VL-30B-A3B | `qwen3-vl-30b-a3b` | [`Qwen/Qwen3-VL-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) |
| Gemma-3-27B | `gemma-3-27b` | [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it) |
| InternVideo3-8B | `internvideo3-8b` | [`yanziang/InternVideo3-8B-Instruct`](https://huggingface.co/yanziang/InternVideo3-8B-Instruct) |
| Cosmos-Reason2-8B | `cosmos-reason2-8b` | [`nvidia/Cosmos-Reason2-8B`](https://huggingface.co/nvidia/Cosmos-Reason2-8B) |
| Cosmos-Reason2-32B | `cosmos-reason2-32b` | [`nvidia/Cosmos-Reason2-32B`](https://huggingface.co/nvidia/Cosmos-Reason2-32B) |
| SenseNova-SI-1.1-8B | `sensenova-si-1.1-8b` | [`sensenova/SenseNova-SI-1.1-Qwen3-VL-8B`](https://huggingface.co/sensenova/SenseNova-SI-1.1-Qwen3-VL-8B) |
| RoboBrain2.5-4B | `robobrain2.5-4b` | [`BAAI/RoboBrain2.5-4B`](https://huggingface.co/BAAI/RoboBrain2.5-4B) |
| RoboBrain2.5-8B-NV | `robobrain2.5-8b-nv` | [`BAAI/RoboBrain2.5-8B-NV`](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV) |
| VLM-3R | `vlm-3r` | [`Journey9ni/vlm-3r-llava-qwen2-lora`](https://huggingface.co/Journey9ni/vlm-3r-llava-qwen2-lora), on the base [`lmms-lab/LLaVA-NeXT-Video-7B-Qwen2`](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B-Qwen2) |
| Cambrian-S-7B | `cambrian-s-7b` | [`nyu-visionx/Cambrian-S-7B`](https://huggingface.co/nyu-visionx/Cambrian-S-7B) |
| Spatial-TTT | `spatial-ttt-nano` | [`THU-SI/Spatial-TTT-nano`](https://huggingface.co/THU-SI/Spatial-TTT-nano), on the base [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |

**Native streaming**: the model runs its own streaming inference with the request given at 0 s. Flash-VStream has no response trigger of its own, so it is asked the per-second Yes / No question on its streaming memory.

```bash
bash scripts/eval.sh <model> 0      # one GPU
```

| Model | `<model>` | Weights |
|---|---|---|
| TimeChat-Online | `timechat-online-7b` | [`wyccccc/TimeChatOnline-7B`](https://huggingface.co/wyccccc/TimeChatOnline-7B) |
| LiveCC | `livecc-7b` | [`chenjoya/LiveCC-7B-Instruct`](https://huggingface.co/chenjoya/LiveCC-7B-Instruct) |
| Dispider | `dispider` | [`Mar2Ding/Dispider`](https://huggingface.co/Mar2Ding/Dispider) |
| Flash-VStream | `flash-vstream-7b` | [`zhang9302002/Flash-VStream-Qwen-7b`](https://huggingface.co/zhang9302002/Flash-VStream-Qwen-7b) |
| MMDuet | `mmduet` | [`wangyueqian/MMDuet`](https://huggingface.co/wangyueqian/MMDuet), on the base [`lmms-lab/llava-onevision-qwen2-7b-ov`](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov) |
| STRIDE-2B | `stride-2b` | [`interlive/STRIDE-2B`](https://huggingface.co/interlive/STRIDE-2B) |
| Stream3D-VLM | `stream3d-vlm-4b` | [`JonnyYu828/Stream3D-VLM-4B`](https://huggingface.co/JonnyYu828/Stream3D-VLM-4B) |

Both write `results/<model>/` and print the scores. Any other Hugging Face chat model runs with `bash scripts/eval_polling.sh <tag> images <weights> 0,1,2,3`.
