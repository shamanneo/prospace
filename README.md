<div align="center" style="font-family: charter;">

<h1>🌊 <i>Thinking in Streams</i>:</br> Proactive Spatial Reasoning with Vision-Language Models</h1>

<img src="assets/fig_teaser.png" width="100%"/>

<p align="justify"><i>Recognizing when a requested spatial condition holds in streaming video can require maintaining and updating spatial knowledge across changing viewpoints. We study this capability through proactive spatial reasoning, where a model receives a query in advance and determines whether and when to respond as the stream unfolds. We introduce ProSpace-Bench, comprising nine tasks across object, camera, and scene reasoning, and ProSpace-Train for supervised training. To support spatial decisions, we propose ProSpace-VLM, which injects explicit geometric encodings estimated causally from incoming frames into intermediate VLM layers, using the existing KV cache as spatial memory and spatial token selection for efficient streaming. Experiments reveal a substantial gap between VLMs and humans, while ProSpace-VLM improves timely responses and correct abstention over fine-tuned baselines. Analyses examine how ProSpace-VLM represents previously observed spatial information across changing viewpoints.</i></p>

</div>

## 📢 News

- **[2026-11-20]** Paper released on arXiv: [Thinking in Streams](https://arxiv.org/abs/2611.25865).
- **[2026-11-19]** Released 🤗 [ProSpace-Bench](https://huggingface.co/datasets/shamanneo/ProSpace-Bench), 🤗 [ProSpace-Train](https://huggingface.co/datasets/shamanneo/ProSpace-Train), 🤗 [ProSpace-VLM](https://huggingface.co/shamanneo/ProSpace-VLM), and the code.

## 📌 Overview

- **ProSpace-Bench**: 3,636 streams, nine tasks. A model watches a 1 fps stream with a standing request and has to respond at the right second, or stay silent when the event never happens.
- **ProSpace-Train**: supervised training streams, a balanced 100K set and an extended 400K set.
- **ProSpace-VLM**: a streaming VLM with explicit geometric encodings injected into its intermediate layers.

## ⚙️ Setup

**1. Environment** (Python 3.11)

```bash
pip install -r requirements.txt
```

**2. CUT3R**

```bash
git clone https://github.com/Jian-Zhang-3DV/CUT3R.git third_party/CUT3R
cd third_party/CUT3R
git checkout 51244364af3566d6473559f71a81b4accc75c424
git apply ../../patches/cut3r_rope.patch
gdown 1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD -O src/cut3r_512_dpt_4_64.pth   # CUT3R weights, 3.2 GB
cd ../..
```

**3. Benchmark and checkpoint** (Hugging Face)

```bash
hf download shamanneo/ProSpace-Bench prospace_bench.jsonl --repo-type dataset --local-dir data
hf download shamanneo/ProSpace-VLM prospace_vlm.pth --local-dir checkpoints
```

The base model `Qwen/Qwen2.5-VL-7B-Instruct` is downloaded on first use.

**4. Frames**

The frames are not redistributed. Download each dataset from its official source, then build `data/frames/` from it:

```bash
python scripts/prepare_frames.py --dataset scannetpp --src <ScanNet++ root>   # <root>/data/<scene_id>/iphone/rgb.mkv
python scripts/prepare_frames.py --dataset scannet   --src <ScanNet root>     # <root>/scans/<scene_id>/<scene_id>.sens
python scripts/prepare_frames.py --dataset ca1m      --src <CA-1M root>       # <root>/ca1m-val-<capture>.tar
python scripts/render_mp3d.py --src <Matterport3D root>                       # <root>/<house>/<house>.glb
python scripts/check_frames.py                                                # every frame present, md5 against data/frames_manifest.tsv
```

- The benchmark uses 46 ScanNet++ scenes, 147 ScanNet scenes, 49 CA-1M validation captures and 18 Matterport3D houses. Each script reads only the scenes it needs and names the source files it cannot find.
- `render_mp3d.py` renders with habitat-sim 0.3.1 (headless), which needs its own Python 3.9 environment: `conda create -n habitat python=3.9 && conda install -n habitat habitat-sim=0.3.1 headless -c conda-forge -c aihabitat`, plus `pip install pillow`. `<Matterport3D root>` is the habitat release of the dataset (`mp3d/`).
- `check_frames.py` passes when every frame is md5-identical to ours. The exception is 13 Matterport3D frames, which the renderer reproduces only to within a few pixels; they are reported as `near`. We checked the render on RTX 2080 Ti GPUs.

<details>
<summary>Folder layout after setup</summary>

The places marked `<-` are the ones the steps above fill in.

```
prospace/
├── data/
│   ├── prospace_bench.jsonl         <- the benchmark, one stream per line
│   ├── frames/                      <- benchmark frames
│   │   ├── scannetpp/<scene_id>/frame_XXXXXX.jpg
│   │   ├── scannet/<scene_id>/frame_XXXXXX.jpg
│   │   ├── ca1m/<scene_id>/frame_XXXXXX.jpg
│   │   └── mp3d/<scene_id>/frame_XXXXXX.jpg
│   ├── train/                       <- ProSpace-Train files (only for training)
│   ├── train_frames/                <- training frames (only for training)
│   ├── frames_manifest.tsv          md5 of every benchmark frame
│   └── human_subset_900.json        ids of the 900 streams of the human study
├── checkpoints/prospace_vlm.pth     <- ProSpace-VLM (LoRA + the four per-layer MLPs)
├── third_party/CUT3R/               <- CUT3R
├── prospace_vlm/                    model, native-streaming trainer and evaluation
├── prospace_bench/scoring.py        scoring rules
├── runners/                         prefix-polling and native-streaming runners for other models
├── training/                        prefix-polling trainer
└── scripts/
```

`XXXXXX` is the six-digit `frame_ids` entry of the stream, e.g. `frame_000120.jpg`.

</details>

## 📊 Evaluate ProSpace-VLM

```bash
bash scripts/eval_prospace_vlm.sh 0,1,2,3        # GPU ids; one GPU also works
```

The run is resumable: run the same command again if it stops. For a quick test of the pipeline (the first 20 streams), `LIMIT=20 OUT=results/quick bash scripts/eval_prospace_vlm.sh 0`.

## 🧪 Evaluate other models

Prefix-polling and native-streaming runners for other models: see [runners/README.md](runners/README.md).

## 🔥 Train

We trained ProSpace-VLM on four NVIDIA H200 GPUs (141 GB each).

**1. Data**

```bash
hf download shamanneo/ProSpace-Train --repo-type dataset --local-dir data/train
python scripts/prepare_train.py --set 100k --polling --check     # --set 400k for the extended set
```

Training frames go to `data/train_frames/<dataset>/`, with the same file naming as the benchmark frames. `--check` counts the frames it cannot find. The scripts that build the training frames from the official downloads are TBA.

**2. Native streaming** (ProSpace-VLM, or the Qwen2.5-VL-7B baseline without geometry)

```bash
bash scripts/train_streaming.sh prospace_vlm 0,1,2,3 100k        # or: qwen25vl
bash scripts/eval_prospace_vlm.sh 0,1,2,3 runs/prospace_vlm_100k/best
```

**3. Prefix polling** (Qwen3-VL-8B; `FAMILY=qwen25` for Qwen2.5-VL-7B)

```bash
bash scripts/train_polling.sh 0,1,2,3 100k
ADAPTER=runs/qwen3vl8b_polling_100k/adapter_best bash scripts/eval_polling.sh qwen3vl8b_sft qwen3 Qwen/Qwen3-VL-8B-Instruct 0,1,2,3
```

## 🎯 Scoring

Each cell is `timely / abstention`:

- **timely**: share of the positive streams where the response falls inside the timely window. The other outcomes are premature, delayed and miss.
- **abstention**: share of the absence streams where the model never responds.
- **average**: mean over the nine tasks.

A record is one line of JSON with `task`, `episode_id` and `response_i` (the response second, or `null`). To score your own model, write one record per stream and run `python scripts/score.py <file or folder>`.

## 📄 License

The code in this repository is released under the [Apache License 2.0](LICENSE). ScanNet++, ScanNet, CA-1M and Matterport3D keep their own terms of use; this repository does not redistribute their data. CUT3R, Qwen2.5-VL and the other models keep their own licenses.

## 📝 Citation

```bibtex
@misc{hong2026thinkinginstreams,
      title={Thinking in Streams: Proactive Spatial Reasoning with Vision-Language Models},
      author={Junyoung Hong and Minsu Kim and Jaehyun Kim and Jeonghwan Cho and Jiwoo Lee and Seon Joo Kim},
      year={2026},
      eprint={2611.25865},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2611.25865},
}
```
