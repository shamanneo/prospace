"""ProSpace-Train files -> the folders the two trainers read.

    data/prospace_train/prospace_train_<set>.jsonl.gz, val_poll.jsonl, val_es.jsonl   (fetched when missing)
    data/frames/<dataset>/<scene_id>/frame_XXXXXX.jpg                                  (frames: docs/data.md)

writes cache/train_<set>/{train.jsonl, val_poll.jsonl, val_es.jsonl, stats.json} for the native-streaming trainer and,
with --polling, cache/train_<set>_polling/ (prefix-polling samples of the same streams, training/polling_samples.py).
The frame paths of the published files are mapped to data/frames on the way (prospace_bench.data.train_frame: the
Matterport3D training frames have no frames/ level there, and the generated history before a mid-stream request is the
scene <scene_id>_history).  With --check, every frame the set uses is looked up and missing ones are counted per dataset.

    python scripts/prepare_train.py [--set 100k] [--frames data/frames] [--polling] [--check]
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prospace_bench.data import DATASETS, FRAMES, TRAIN, check_layout, fetch, train_frame  # noqa: E402


def convert(src: Path, dst: Path) -> int:
    """Copy a stream file with its frame paths mapped to data/frames; returns the number of streams."""
    n = 0
    with (gzip.open(src, "rt") if src.suffix == ".gz" else open(src)) as f, open(str(dst) + ".tmp", "w") as g:
        for line in f:
            e = json.loads(line)
            e["frames"] = [train_frame(p) for p in e["frames"]]
            g.write(json.dumps(e) + "\n")
            n += 1
    Path(str(dst) + ".tmp").rename(dst)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="100k", choices=["100k", "400k"])
    ap.add_argument("--src", default=str(TRAIN), help="the ProSpace-Train files (default data/prospace_train)")
    ap.add_argument("--frames", default=str(FRAMES))
    ap.add_argument("--out", default=None, help="default: cache/train_<set>")
    ap.add_argument("--polling", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check_layout()
    src, out = Path(a.src), Path(a.out or ROOT / "cache" / f"train_{a.set}")
    out.mkdir(parents=True, exist_ok=True)
    roots = {ds: str((Path(a.frames) / ds).absolute()) for ds in DATASETS}

    names = [f"prospace_train_{a.set}.jsonl.gz", "val_poll.jsonl", "val_es.jsonl"]
    if a.polling and a.set == "100k":
        names.append("prospace_train_400k.jsonl.gz")
    for name in names:
        if not (src / name).exists() and src == TRAIN:
            fetch("train", name)
    if not (out / "train.jsonl").exists():
        convert(src / f"prospace_train_{a.set}.jsonl.gz", out / "train.jsonl")
    for name in ("val_poll.jsonl", "val_es.jsonl"):
        convert(src / name, out / name)
    (out / "stats.json").write_text(json.dumps({"frame_roots": roots}, indent=1))
    n = sum(1 for _ in open(out / "train.jsonl"))
    print(f"{out}: {n} training streams, {sum(1 for _ in open(out / 'val_poll.jsonl'))} validation streams")

    if a.check:
        seen, missing, total = set(), Counter(), Counter()
        for name in ("train.jsonl", "val_poll.jsonl", "val_es.jsonl"):
            for line in open(out / name):
                e = json.loads(line)
                for p in e["frames"]:
                    if p not in seen:
                        seen.add(p)
                        total[e["dataset"]] += 1
                        if not (Path(roots[e["dataset"]]) / p.split("/", 1)[1]).exists():
                            missing[e["dataset"]] += 1
        for ds in DATASETS:
            print(f"  {ds}: {total[ds]} frames, {missing[ds]} missing")
        if sum(missing.values()):
            sys.exit("some frames are missing: build them as in docs/data.md")

    if a.polling:
        pout = out.parent / f"{out.name}_polling"
        cmd = [sys.executable, "-m", "training.polling_samples", "--src", str(out), "--out", str(pout)]
        cmd += ["--stats", str(out / "stats.json")]
        if a.set == "100k":  # the wording of a mid-request copy comes from its first-frame original in the 400K set
            cmd += ["--orig-src", str(src / "prospace_train_400k.jsonl.gz")]
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
