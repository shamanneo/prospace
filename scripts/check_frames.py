"""Check data/frames against the md5 of every frame a set reads.

    python scripts/check_frames.py                 # the benchmark: every frame present and md5-identical
    python scripts/check_frames.py --set train     # ProSpace-Train (both sets and the validation files)
    python scripts/check_frames.py --no-md5        # only that the files are there (fast)
    python scripts/check_frames.py --remove-different   # delete the frames that differ, then build them again

The md5 lists are data/prospace_bench/frames_manifest.tsv and data/prospace_train/frames_manifest.tsv.gz (fetched from
Hugging Face when missing).  Layout: data/frames/<dataset>/<scene_id>/frame_<frame_id:06d>.jpg with dataset in
{scannetpp, scannet, ca1m, mp3d}.

Frames that can come out a few pixels away from our copy are counted as "near" and do not fail the check: the
Matterport3D frames with md5_exact = False in the set's mp3d_camera_poses.npz (13 benchmark frames and 1 training frame,
below 2 % of the pixels of a frame) and, for --set train, the 3,986 ScanNet++ frames made with ffmpeg, which are
byte-identical only with ffmpeg 4.4.2 (another version writes its own version tag into the file; the pixels are the
same, checked with 7.1.1).
"""

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prospace_bench.data import FRAMES, check_layout, fetch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="bench", choices=["bench", "train"])
    ap.add_argument("--frames", default=str(FRAMES))
    ap.add_argument("--manifest", default=None, help="default: the md5 list of --set")
    ap.add_argument("--no-md5", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--list", default=None, help="write the paths of all missing and different frames to this file")
    ap.add_argument("--remove-different", action="store_true", help="delete the different frames so the build scripts make them again")
    a = ap.parse_args()
    check_layout()
    man = a.manifest or fetch(a.set, "frames_manifest.tsv" if a.set == "bench" else "frames_manifest.tsv.gz")
    with gzip.open(man, "rt") if str(man).endswith(".gz") else open(man) as f:
        rows = [l.rstrip("\n").split("\t") for l in f][1:]
    near = set()  # (dataset, scene_id, frame_id) that may come out a few pixels away
    if not a.no_md5 and any(r[0] == "mp3d" for r in rows):
        z = np.load(fetch(a.set, "mp3d_camera_poses.npz"))
        bad = ~z["md5_exact"]
        near |= {("mp3d", str(s), int(i)) for s, i in zip(z["scene_id"][bad], z["frame_id"][bad])}
    if not a.no_md5 and a.set == "train" and any(r[0] == "scannetpp" for r in rows):
        src = json.loads(Path(fetch("train", "scannetpp_frame_sources.json")).read_text())
        near |= {("scannetpp", s, int(f)) for s, d in src.items() for f, (_, q) in d.items() if q == 0}

    def one(r):
        ds, scene, fid, md5 = r
        p = Path(a.frames) / ds / scene / f"frame_{int(fid):06d}.jpg"
        if not p.exists():
            return ds, "missing", str(p)
        if a.no_md5 or hashlib.md5(p.read_bytes()).hexdigest() == md5:
            return ds, "ok", str(p)
        return ds, "near" if (ds, scene, int(fid)) in near else "different", str(p)

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, rows, chunksize=256))
    c = Counter((ds, s) for ds, s, _ in res)
    for ds in sorted({r[0] for r in rows}):
        print(
            f"{ds:10s} ok {c[(ds, 'ok')]:7d}  missing {c[(ds, 'missing')]:7d}  different {c[(ds, 'different')]:7d}"
            + (f"  near {c[(ds, 'near')]:5d}" if c[(ds, "near")] else "")
        )
    bad = [(s, p) for _, s, p in res if s not in ("ok", "near")]
    for s, p in bad[:10]:
        print(s, p)
    if a.list:
        Path(a.list).write_text("".join(f"{s}\t{p}\n" for s, p in bad))
        print(f"{len(bad)} paths written to {a.list}")
    if a.remove_different:
        diff = [p for s, p in bad if s == "different"]
        for p in diff:
            Path(p).unlink()
        print(f"removed {len(diff)} different frames: run the build commands of docs/data.md again")
    ok = sum(s == "ok" for _, s, _ in res)
    print(f"{a.set}: {ok} / {len(res)} frames ok" + ("" if a.no_md5 else " (md5)"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
