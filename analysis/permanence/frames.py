"""The frames of the permanence streams: check them, and build the ones that are missing from the official datasets.

Most of these frames are benchmark or ProSpace-Train frames (docs/data.md); about 1,800 come from held-out validation
scenes that neither set uses.  frames.tsv.gz lists all 28,864 with their md5 and where each comes from (ScanNet++:
decode order and JPEG quality; CA-1M: timestamp; ScanNet: .sens index), and this script reuses scripts/prepare_frames.py
to write them into data/frames.

    python analysis/permanence/frames.py --check                                   # missing / different frames
    python analysis/permanence/frames.py --dataset scannetpp --src <ScanNet++ root> --workers 16
    python analysis/permanence/frames.py --dataset scannet   --src <ScanNet root>
    python analysis/permanence/frames.py --dataset ca1m      --src <CA-1M root>     # validation and training tars
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def table():
    with gzip.open(HERE / "frames.tsv.gz", "rt") as f:
        next(f)
        return [l.rstrip("\n").split("\t") for l in f]


def check(frames: Path, workers: int) -> int:
    rows = table()

    def one(r):
        ds, scene, fid, md5, _ = r
        p = frames / ds / scene / f"frame_{int(fid):06d}.jpg"
        if not p.exists():
            return ds, "missing"
        return ds, "ok" if hashlib.md5(p.read_bytes()).hexdigest() == md5 else "different"

    with ThreadPoolExecutor(workers) as ex:
        c = Counter(ex.map(one, rows))
    for ds in ("scannetpp", "scannet", "ca1m"):
        print(f"{ds:10s} " + "  ".join(f"{k} {c[(ds, k)]}" for k in ("ok", "missing", "different")))
    bad = sum(v for (_, k), v in c.items() if k != "ok")
    print("all frames present and identical" if not bad else f"{bad} frames to build (see the commands at the top)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["scannetpp", "scannet", "ca1m"])
    ap.add_argument("--src", help="root of the official download (layout: scripts/prepare_frames.py)")
    ap.add_argument("--frames", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    frames = Path(a.frames)
    if a.check:
        sys.exit(1 if check(frames, max(a.workers, 16)) else 0)
    assert a.dataset and a.src, "--dataset and --src (or --check)"
    spec = importlib.util.spec_from_file_location("prepare_frames", ROOT / "scripts" / "prepare_frames.py")
    pf = importlib.util.module_from_spec(spec)
    sys.modules["prepare_frames"] = pf  # the worker processes look the job function up by its module name
    spec.loader.exec_module(pf)
    want, lookup = defaultdict(list), defaultdict(dict)
    for ds, scene, fid, _, src in table():
        if ds != a.dataset:
            continue
        want[scene].append(int(fid))
        if ds == "scannetpp":
            k, q = src.split(":")
            lookup[scene][fid] = [int(k), int(q)]
        elif ds == "ca1m":
            lookup[scene][fid] = int(src)
    jobs = [
        (a.dataset, a.src, s, str(frames / a.dataset / s), ids, (lookup[s],) if a.dataset != "scannet" else ())
        for s, ids in sorted(want.items())
    ]
    print(f"{a.dataset}: {len(jobs)} scenes, {sum(len(j[4]) for j in jobs)} frames", flush=True)
    missing = 0
    with Pool(a.workers) as pool:
        for k, (scene, n, todo, miss, note) in enumerate(pool.imap_unordered(pf.one, jobs), 1):
            missing += miss
            if todo or miss:
                print(f"[{k}/{len(jobs)}] {scene}: {n} frames, {todo - miss} written, {miss} missing {note}".rstrip(), flush=True)
    print(f"done, {missing} frames missing; check with --check")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
