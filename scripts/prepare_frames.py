"""Rebuild the benchmark frames of ScanNet++, ScanNet v2 and CA-1M from the official downloads.

    python scripts/prepare_frames.py --dataset scannetpp --src <ScanNet++ root>    # <src>/data/<scene_id>/iphone/rgb.mkv
    python scripts/prepare_frames.py --dataset scannet   --src <ScanNet root>      # <src>/scans/<scene_id>/<scene_id>.sens
    python scripts/prepare_frames.py --dataset ca1m      --src <CA-1M tars>        # <src>/ca1m-val-<capture>.tar

Writes data/frames/<dataset>/<scene_id>/frame_<frame_id:06d>.jpg for every frame of that dataset in
data/frames_manifest.tsv and skips the files that are already there, so an interrupted run can simply be started again.
Check the result with scripts/check_frames.py.

scannetpp   frame id = decode order of iphone/rgb.mkv (the k-th decoded frame is frame k); saved as JPEG, quality 90
scannet     frame id = frame index in the .sens file; the JPEG stored there is copied byte for byte
ca1m        frame id -> timestamp in data/ca1m_frame_timestamps.json; <capture>/<timestamp>.wide/image.png saved as
JPEG, quality 85
"""

import argparse
import io
import json
import os
import struct
import sys
import tarfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def name(dst, fid):
    return dst / f"frame_{fid:06d}.jpg"


def write(path, data):
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def jpeg(im, quality):
    b = io.BytesIO()
    im.convert("RGB").save(b, format="JPEG", quality=quality)
    return b.getvalue()


def first(*paths):
    return next((p for p in paths if p.exists()), None)


def scannetpp(src, scene, dst, todo):
    import av

    path = first(src / "data" / scene / "iphone/rgb.mkv", src / scene / "iphone/rgb.mkv")
    if path is None:
        return f"no {src}/data/{scene}/iphone/rgb.mkv"
    last = max(todo)
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for k, fr in enumerate(c.decode(st)):
            if k in todo:
                write(name(dst, k), jpeg(fr.to_image(), 90))
            if k >= last:
                break


def scannet(src, scene, dst, todo):
    path = first(*(src / d / scene / f"{scene}.sens" for d in ("scans", "scans_test", ".")))
    if path is None:
        return f"no {src}/scans/{scene}/{scene}.sens"
    todo = set(todo)
    with open(path, "rb") as f:  # layout of ScanNet's SensorData.py
        (version,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        f.read(n + 64 * 4)  # sensor name, colour and depth intrinsics and extrinsics
        colour, _ = struct.unpack("<ii", f.read(8))
        if version != 4 or colour != 2:  # 2 = jpeg, what every ScanNet v2 scan uses
            return f"{path}: version {version}, colour compression {colour}"
        f.read(16 + 4)  # colour and depth size, depth shift
        (frames,) = struct.unpack("<Q", f.read(8))
        for i in range(frames):
            f.read(64)  # camera to world
            _, _, cs, ds = struct.unpack("<QQQQ", f.read(32))
            if i in todo:
                write(name(dst, i), f.read(cs))
                f.seek(ds, 1)
                todo.discard(i)
                if not todo:
                    break
            else:
                f.seek(cs + ds, 1)


def ca1m(src, scene, dst, todo):
    from PIL import Image

    ts = json.loads((ROOT / "data/ca1m_frame_timestamps.json").read_text())[scene]
    want = {int(ts[str(f)]): f for f in todo}
    if (src / scene).is_dir():  # an unpacked tar
        for t, f in want.items():
            p = src / scene / f"{t}.wide" / "image.png"
            if p.exists():
                write(name(dst, f), jpeg(Image.open(p), 85))
        return
    tars = sorted(src.glob(f"ca1m-*-{scene}.tar"))
    if not tars:
        return f"no {src}/ca1m-val-{scene}.tar"
    with tarfile.open(tars[0], "r|") as tar:  # one pass, only the wanted images are decoded
        for m in tar:
            p = m.name.split("/")
            if not m.isfile() or len(p) != 3 or not p[1].endswith(".wide") or p[2] != "image.png":
                continue
            f = want.pop(int(p[1].split(".")[0]), None)
            if f is not None:
                write(name(dst, f), jpeg(Image.open(io.BytesIO(tar.extractfile(m).read())), 85))
                if not want:
                    break


def one(job):
    dataset, src, scene, dst, ids = job
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    todo = {f for f in ids if not name(dst, f).exists()}
    note = globals()[dataset](Path(src), scene, dst, todo) if todo else None
    missing = sum(not name(dst, f).exists() for f in ids)
    return scene, len(ids), len(todo), missing, note or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["scannetpp", "scannet", "ca1m"])
    ap.add_argument("--src", required=True, help="root of the official download (layout: see the top of this file)")
    ap.add_argument("--out", default=str(ROOT / "data/frames"))
    ap.add_argument("--manifest", default=str(ROOT / "data/frames_manifest.tsv"))
    ap.add_argument("--scenes", nargs="*", help="only these scene ids (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    want = defaultdict(list)
    for l in list(open(a.manifest))[1:]:
        ds, scene, fid, _ = l.rstrip("\n").split("\t")
        if ds == a.dataset and (not a.scenes or scene in a.scenes):
            want[scene].append(int(fid))
    jobs = [(a.dataset, a.src, s, str(Path(a.out) / a.dataset / s), ids) for s, ids in sorted(want.items())]
    print(f"{a.dataset}: {len(jobs)} scenes, {sum(len(j[4]) for j in jobs)} frames", flush=True)
    missing = 0
    with Pool(a.workers) as pool:
        for k, (scene, n, todo, miss, note) in enumerate(pool.imap_unordered(one, jobs), 1):
            missing += miss
            print(
                f"[{k}/{len(jobs)}] {scene}: {n} frames, {todo - miss} written, {miss} missing {note}".rstrip(),
                flush=True,
            )
    print(f"done, {missing} frames missing")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
