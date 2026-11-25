# Data

The streams, labels, camera poses and md5 lists are on Hugging Face. The video frames are not redistributed: build them from the official datasets with the scripts below.

## Download

```bash
hf download shamanneo/ProSpace-Bench --repo-type dataset --local-dir data/prospace_bench   # benchmark
hf download shamanneo/ProSpace-VLM prospace_vlm.pth --local-dir checkpoints               # ProSpace-VLM
hf download shamanneo/ProSpace-Train --repo-type dataset --local-dir data/prospace_train   # only for training
```

The scripts also fetch a missing file of the two datasets on first use, so the benchmark and checkpoint lines are enough to evaluate. No login is needed.

## Source datasets

| Dataset | Where | What the scripts read |
|---|---|---|
| ScanNet++ | [kaldir.vc.in.tum.de/scannetpp](https://kaldir.vc.in.tum.de/scannetpp/) | `<root>/data/<scene_id>/iphone/rgb.mkv` (the iPhone video) |
| ScanNet v2 | [scan-net.org](http://www.scan-net.org/) | `<root>/scans/<scene_id>/<scene_id>.sens` |
| CA-1M | [apple/ml-cubifyanything](https://github.com/apple/ml-cubifyanything) | `<root>/ca1m-<split>-<capture>.tar` (validation split for the benchmark, training split for ProSpace-Train) |
| Matterport3D | [niessner.github.io/Matterport](https://niessner.github.io/Matterport/) | `<root>/<house>/<house>.glb` (the Habitat release) |

Each dataset has its own terms of use, which you accept when you request it.

**Matterport3D**: the Matterport3D data is for non-commercial academic use only, under the [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf). We release camera poses, not rendered images; the frames are rendered from your own copy of the dataset, and the same terms apply to them.

## Benchmark frames

```bash
python scripts/prepare_frames.py --dataset scannetpp --src <ScanNet++ root>
python scripts/prepare_frames.py --dataset scannet   --src <ScanNet root>
python scripts/prepare_frames.py --dataset ca1m      --src <CA-1M root>
conda run -n habitat python scripts/render_mp3d.py --src <Matterport3D root>
python scripts/check_frames.py
```

- The benchmark uses 46 ScanNet++ scenes, 147 ScanNet scenes, 49 CA-1M validation captures and 18 Matterport3D houses (118,577 frames). Each script reads only what it needs, names the source files it cannot find, and skips frames that already exist, so it can be run again.
- `check_frames.py` compares every frame with our md5. 13 Matterport3D frames are reproduced only to within a few pixels and are reported as `near`; this is expected.
- A `near` result needs no action: it is either a frame with the same pixels and a different encoder signature, or one of the 13 benchmark and 1 training Matterport3D renders that differ by a few pixels.

## Training frames

The same commands with `--set train` build the frames of ProSpace-Train (both sets and the validation files):

```bash
python scripts/prepare_frames.py --dataset scannetpp --src <ScanNet++ root> --set train
python scripts/prepare_frames.py --dataset scannet   --src <ScanNet root>   --set train
python scripts/prepare_frames.py --dataset ca1m      --src <CA-1M root>     --set train
conda run -n habitat python scripts/render_mp3d.py --src <Matterport3D root> --set train
python scripts/check_frames.py --set train
python scripts/prepare_train.py --set 100k --check    # counts the frames the 100K set still misses
```

- ProSpace-Train uses 1,663,926 frames: 139,040 ScanNet++, 21,496 ScanNet, 176,461 CA-1M (training split) and 1,326,929 Matterport3D renders from 45 houses. The benchmark and ProSpace-Train come from disjoint official splits, so their frames share `data/frames/`. One Matterport3D training frame is reproduced only to within a few pixels and is reported as `near`.
- CA-1M portrait captures are kept at their own size (768x1024); the loaders pillar-box every frame to 512x384.
- ScanNet++ training frames were extracted by video timestamp, which runs up to a few frames behind the decode order, at JPEG quality 85, 90 or 92; `data/prospace_train/scannetpp_frame_sources.json` records both per frame, so `--set train` reproduces them exactly. 3,986 of them come from the ScanNet++ toolkit's extraction (`ffmpeg -q:v 1`): `prepare_frames.py` calls ffmpeg for these, by default the ffmpeg 4.4.2 that `setup.sh` installs into the `prospace` environment (`PROSPACE_FFMPEG=<path>` uses another one), and they are then byte-identical. Another ffmpeg writes its own version into the file, so the md5 differs although the pixels are the same (checked with ffmpeg 7.1.1); `check_frames.py` reports these frames as `near`.
- The published stream files name the Matterport3D training frames `mp3d/<scene_id>/frames/frame_XXXXXX.jpg`, and the history before a mid-stream request `mp3d/<scene_id>/history_b/frame_JJJJ.jpg` (or `history/`). `prepare_train.py` maps them to `data/frames/mp3d/<scene_id>/frame_XXXXXX.jpg` and `data/frames/mp3d/<scene_id>_history/frame_<JJJJ>.jpg` (six digits) when it writes `cache/`.
- `render_mp3d.py --houses <id> ...` renders only some houses, e.g. to split the work over GPUs (`--gpu`).
- The training frames take about 100 GB. On one machine with 128 CPU cores and three H200 GPUs, the Matterport3D training render took about one hour (three processes, one per GPU, split with `--houses`), the ScanNet++ training frames about 45 minutes, CA-1M about 4 minutes, ScanNet a few seconds, and `check_frames.py --set train` about 7 minutes.

## Folder layout

```
prospace/
├── data/
│   ├── prospace_bench/           = Hugging Face shamanneo/ProSpace-Bench
│   │   ├── prospace_bench.jsonl      benchmark, one stream per line
│   │   ├── mp3d_camera_poses.npz     camera poses of the Matterport3D frames
│   │   └── frames_manifest.tsv       md5 of every benchmark frame
│   ├── prospace_train/           = Hugging Face shamanneo/ProSpace-Train
│   │   ├── prospace_train_100k.jsonl.gz, prospace_train_400k.jsonl.gz, val_poll.jsonl, val_es.jsonl
│   │   ├── mp3d_camera_poses.npz     camera poses of the Matterport3D training frames
│   │   ├── ca1m_frame_timestamps.json, scannetpp_frame_sources.json
│   │   └── frames_manifest.tsv.gz    md5 of every training frame
│   ├── frames/<dataset>/<scene_id>/frame_XXXXXX.jpg    benchmark and training frames
│   ├── ca1m_frame_timestamps.json    CA-1M frame id -> timestamp of the benchmark captures
│   └── human_subset_900.json         the 900 streams of the human study
├── checkpoints/prospace_vlm.pth
└── third_party/CUT3R/
```

`XXXXXX` is the six-digit `frame_ids` entry of the stream, e.g. `frame_000120.jpg`. A checkout that still has the earlier layout (`data/prospace_bench.jsonl`, `data/train/`, `data/train_frames/`) stops with a message saying what to move.
