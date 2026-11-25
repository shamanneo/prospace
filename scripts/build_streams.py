"""ProSpace-Bench episodes -> the stream records the ProSpace-VLM evaluation reads.

Reads data/prospace_bench/prospace_bench.jsonl (3,636 streams, never edited; fetched from Hugging Face when missing) and writes <out>/val_poll.jsonl (one stream per line:
frames, question, timely window w0 / w1, scoring start, kind), <out>/val_es.jsonl (empty) and <out>/roots.json.
  - scoring start = gap_start_i for Object Re-finding, 0 otherwise
  - frames: data/frames/<dataset>/<scene_id>/frame_<frame_id:06d>.jpg, every frame of the stream
With --check, every frame is looked up and missing ones are reported per dataset.

    python scripts/build_streams.py [--frames data/frames] [--out cache/streams] [--check]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prospace_bench.data import FRAMES, check_layout, fetch  # noqa: E402

TASKS = [
    "ego_rotation",
    "object_direction",
    "visual_homing",
    "origin_return",
    "object_finding",
    "object_refinding",
    "place_entry",
    "place_revisit",
    "place_transition",
]
DATASETS = ["scannetpp", "scannet", "ca1m", "mp3d"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=None, help="default data/prospace_bench/prospace_bench.jsonl")
    ap.add_argument("--frames", default=str(FRAMES))
    ap.add_argument("--out", default=str(ROOT / "cache" / "streams"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check_layout()
    a.bench = a.bench or fetch("bench", "prospace_bench.jsonl")
    roots = {ds: str((Path(a.frames) / ds).absolute()) for ds in DATASETS}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs, counts, missing = [], Counter(), Counter()
    bench = [json.loads(line) for line in open(a.bench)]
    for task in TASKS:
        for e in (e for e in bench if e["task"] == task):
            ds, frames = e["dataset"], e["frames"]
            w0, w1 = e.get("timely_start_i"), e.get("timely_end_i")
            start = int(e.get("gap_start_i") or 0) if task == "object_refinding" else 0
            r = dict(
                episode_id=e["episode_id"],
                task=task,
                dataset=ds,
                scene_id=e["scene_id"],
                frames=frames,
                frame_ids=[int(i) for i in e["frame_ids"]],
                question=e["question"],
                frame0_bbox=e.get("frame0_bbox"),
                w0=None if w0 is None else int(w0),
                w1=None if w1 is None else int(w1),
                start=start,
                kind="absence" if w0 is None else "positive",
            )
            if a.check and any(not os.path.exists(os.path.join(roots[ds], f.split("/", 1)[1])) for f in frames):
                missing[ds] += 1
            recs.append(r)
            counts[(task, r["kind"])] += 1
    with (out / "val_poll.jsonl").open("w") as w:
        for r in recs:
            w.write(json.dumps(r) + "\n")
    (out / "val_es.jsonl").write_text("")
    (out / "roots.json").write_text(json.dumps(roots, indent=1))
    summ = {
        "scored_episodes": len(recs),
        "per_task": {t: {k: counts[(t, k)] for k in ("positive", "absence")} for t in TASKS},
        "frames_total": sum(len(r["frames"]) for r in recs),
        "max_frames": max(len(r["frames"]) for r in recs),
    }
    if a.check:
        summ["streams_with_missing_frames"] = dict(missing)
    (out / "bench_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    if a.check and missing:
        raise SystemExit("some frames are missing: build them as in docs/data.md")


if __name__ == "__main__":
    main()
