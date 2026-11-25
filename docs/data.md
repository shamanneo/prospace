# Data

The streams and labels are on Hugging Face. The video frames are not redistributed: build them from the official datasets with the scripts below.

## Download

```bash
hf download shamanneo/ProSpace-Bench prospace_bench.jsonl --repo-type dataset --local-dir data    # benchmark
hf download shamanneo/ProSpace-VLM prospace_vlm.pth --local-dir checkpoints                       # ProSpace-VLM
hf download shamanneo/ProSpace-Train --repo-type dataset --local-dir data/train                   # only for training
```

## Source datasets

| Dataset | Where | What the scripts read |
|---|---|---|
| ScanNet++ | [kaldir.vc.in.tum.de/scannetpp](https://kaldir.vc.in.tum.de/scannetpp/) | `<root>/data/<scene_id>/iphone/rgb.mkv` (the iPhone video) |
| ScanNet v2 | [scan-net.org](http://www.scan-net.org/) | `<root>/scans/<scene_id>/<scene_id>.sens` |
| CA-1M | [apple/ml-cubifyanything](https://github.com/apple/ml-cubifyanything) | `<root>/ca1m-val-<capture>.tar` (validation split) |
| Matterport3D | [niessner.github.io/Matterport](https://niessner.github.io/Matterport/) | `<root>/<house>/<house>.glb` (the Habitat release) |

Each dataset has its own terms of use, which you accept when you request it.

## Benchmark frames

```bash
python scripts/prepare_frames.py --dataset scannetpp --src <ScanNet++ root>
python scripts/prepare_frames.py --dataset scannet   --src <ScanNet root>
python scripts/prepare_frames.py --dataset ca1m      --src <CA-1M root>
conda run -n habitat python scripts/render_mp3d.py --src <Matterport3D root>
python scripts/check_frames.py
```

- The benchmark uses 46 ScanNet++ scenes, 147 ScanNet scenes, 49 CA-1M validation captures and 18 Matterport3D houses. Each script reads only what it needs, names the source files it cannot find, and skips frames that already exist, so it can be run again.
- `check_frames.py` compares every frame with our md5. 13 Matterport3D frames are reproduced only to within a few pixels and are reported as `near`; this is expected.

## Training frames

Training frames go to `data/train_frames/<dataset>/`, with the benchmark naming except for Matterport3D (`mp3d/<scene_id>/frames/frame_XXXXXX.jpg`). The 100K set uses about 1.65M frames, most of them Matterport3D renders. The scripts that build them from the official downloads are coming soon; `python scripts/prepare_train.py --set 100k --check` counts the frames that are still missing.

## Folder layout

```
prospace/
├── data/
│   ├── prospace_bench.jsonl      benchmark, one stream per line
│   ├── frames/<dataset>/<scene_id>/frame_XXXXXX.jpg
│   ├── train/                    ProSpace-Train files
│   ├── train_frames/             training frames
│   ├── frames_manifest.tsv       md5 of every benchmark frame
│   └── human_subset_900.json     the 900 streams of the human study
├── checkpoints/prospace_vlm.pth
└── third_party/CUT3R/
```

`XXXXXX` is the six-digit `frame_ids` entry of the stream, e.g. `frame_000120.jpg`.
