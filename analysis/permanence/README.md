# Object permanence

Does ProSpace-VLM keep track of where an object is after it has left the view? A linear probe reads the bearing of the asked object (the angle from the camera heading to the object) from the model's decision-token state at every second, and the error is measured on the seconds when the object is out of view. This folder reproduces the ProSpace-VLM numbers of the object permanence analysis in the paper: the bearing error of the probe (overall and by seconds since the object was last seen), the read-out accuracy, the rotation correlation, the error curve over 60 s and the two top-view examples.

The streams are Object Direction streams of held-out validation scenes (the probe is fitted on streams of training scenes); they are not part of ProSpace-Bench, so they are listed here: `streams.jsonl.gz` (one line per stream and question, see `common.py`) and `frames.tsv.gz` (md5 and source of every frame).

## Run

Run from the repository root in the `prospace` environment, after the setup and the data of [docs/data.md](../../docs/data.md) (the checkpoint and CUT3R are needed; ProSpace-Bench and ProSpace-Train frames are reused).

```bash
# 1. frames: most are benchmark or training frames; build the rest (held-out validation scenes)
python analysis/permanence/frames.py --check
python analysis/permanence/frames.py --dataset scannetpp --src <ScanNet++ root> --workers 16
python analysis/permanence/frames.py --dataset ca1m      --src <CA-1M root>
python analysis/permanence/frames.py --dataset scannet   --src <ScanNet root>

# 2. ground truth (camera poses, object positions) from your copy of the datasets -> cache/permanence/truth/
python analysis/permanence/truth.py --dataset scannetpp --src <ScanNet++ root>
python analysis/permanence/truth.py --dataset ca1m      --src <CA-1M root>
python analysis/permanence/truth.py --dataset scannet   --src <ScanNet root>

# 3. model states, one shard per GPU (resumable) -> cache/permanence/states/
bash analysis/permanence/run.sh 0,1,2,3

# 4. probe, read-out, rotation correlation, error curve (CPU) -> results/permanence/
python analysis/permanence/probe.py

# 5. top views of the two examples
python analysis/permanence/topview.py A --mesh <ARKitScenes>/raw/<split>/47331084/47331084_3dod_mesh.ply
python analysis/permanence/topview.py B --mesh <ARKitScenes>/raw/<split>/45261495/45261495_3dod_mesh.ply
```

- Besides the files of docs/data.md, `truth.py` reads `iphone/pose_intrinsic_imu.json` and `scans/segments_anno.json` (ScanNet++), the `.sens`, `_vh_clean_2.ply`, `_vh_clean_2.0.010000.segs.json` and `.aggregation.json` files (ScanNet), and the CA-1M tars of both splits (`ca1m-val-*` and `ca1m-train-*`).
- Both examples are CA-1M captures, which have no mesh: `topview.py` places the ARKitScenes mesh of the same capture (`<id>_3dod_mesh.ply`, with `lowres_wide.traj` next to it) in the CA-1M frame by a rigid fit of the two camera paths. Without `--mesh`, the CUT3R points of the stream are drawn as the floor plan. A ScanNet++ example takes `scans/mesh_aligned_0.05.ply` as it is.
- Step 3 runs the model over 1,790 runs (790 streams, 120,692 seconds). On three H200 GPUs it took about 50 minutes; `cache/geom_cut3r/` is shared with the benchmark evaluation.

## Outputs

`probe.py` prints and writes to `results/permanence/`:

| File | Content |
|---|---|
| `summary.json` | `bearing_probe`: median bearing error (deg) on the out-of-view seconds of the 78 held-out streams (three questions each), and by seconds since the object was last seen; `readout`: three-way accuracy (%) of the response probabilities; `rotation`: circular correlation of the scene-cross-fitted probe; `examples`: median error of the two top-view streams |
| `curve.json`, `curve.png` | median bearing error at each second since the object left the view (1-60 s, 378 streams), with the 95 % stream-bootstrap band |
| `examples.json` | decoded and true bearing at every out-of-view second of the examples A and B |

`topview.py` writes `topview_A` and `topview_B` (`.png`, `.svg`, `.pdf`) and prints the error of each drawn arrow.

The probe and the curve are deterministic given the states; the states are recomputed by the model on your GPU, so the numbers can differ from the paper in the last digit.
