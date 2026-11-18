"""ProSpace-Train files -> the folders the two trainers read.

    data/train/prospace_train_<set>.jsonl.gz, val_poll.jsonl, val_es.jsonl   (downloaded)
    data/train_frames/<dataset>/...                                          (frames, same naming as the stream records)

writes cache/train_<set>/{train.jsonl, val_poll.jsonl, val_es.jsonl, stats.json} for the native-streaming trainer and,
with --polling, cache/train_<set>_polling/ (prefix-polling samples of the same streams, training/polling_samples.py).
With --check, every frame the set uses is looked up and missing ones are counted per dataset.

    python scripts/prepare_train.py [--set 100k] [--frames data/train_frames] [--polling] [--check]
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["scannetpp", "scannet", "ca1m", "mp3d"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="100k", choices=["100k", "400k"])
    ap.add_argument("--src", default=str(ROOT / "data" / "train"))
    ap.add_argument("--frames", default=str(ROOT / "data" / "train_frames"))
    ap.add_argument("--out", default=None, help="default: cache/train_<set>")
    ap.add_argument("--polling", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out or ROOT / "cache" / f"train_{a.set}")
    out.mkdir(parents=True, exist_ok=True)
    roots = {ds: str((Path(a.frames) / ds).absolute()) for ds in DATASETS}

    if not (out / "train.jsonl").exists():
        with gzip.open(src / f"prospace_train_{a.set}.jsonl.gz", "rb") as f, open(out / "train.jsonl.tmp", "wb") as g:
            shutil.copyfileobj(f, g, 1 << 24)
        (out / "train.jsonl.tmp").rename(out / "train.jsonl")
    for name in ("val_poll.jsonl", "val_es.jsonl"):
        shutil.copyfile(src / name, out / name)
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
            sys.exit(1)

    if a.polling:
        pout = out.parent / f"{out.name}_polling"
        cmd = [sys.executable, "-m", "training.polling_samples", "--src", str(out), "--out", str(pout)]
        cmd += ["--stats", str(out / "stats.json")]
        if a.set == "100k":  # the wording of a mid-request copy comes from its first-frame original in the 400K set
            cmd += ["--orig-src", str(src / "prospace_train_400k.jsonl.gz")]
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
