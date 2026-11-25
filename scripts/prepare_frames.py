"""Rebuild the frames of ScanNet++, ScanNet v2 and CA-1M from the official downloads.

    python scripts/prepare_frames.py --dataset scannetpp --src <ScanNet++ root>    # <src>/data/<scene_id>/iphone/rgb.mkv
    python scripts/prepare_frames.py --dataset scannet   --src <ScanNet root>      # <src>/scans/<scene_id>/<scene_id>.sens
    python scripts/prepare_frames.py --dataset ca1m      --src <CA-1M tars>        # <src>/ca1m-<split>-<capture>.tar
    add --set train for the frames of ProSpace-Train (default --set bench, the benchmark)

Writes data/frames/<dataset>/<scene_id>/frame_<frame_id:06d>.jpg for every frame of that dataset in the set's md5 list
(data/prospace_bench/frames_manifest.tsv or data/prospace_train/frames_manifest.tsv.gz, fetched from Hugging Face when
missing) and skips the files that are already there, so an interrupted run can simply be started again.  The benchmark
uses validation scenes and ProSpace-Train training scenes, so both sets share data/frames.  Check the result with
scripts/check_frames.py [--set train].

scannetpp   benchmark: frame id = decode order of iphone/rgb.mkv (the k-th decoded frame is frame k), JPEG quality 90;
            training: data/prospace_train/scannetpp_frame_sources.json gives per frame id the decode order and the JPEG
            quality of our copy (the training frames were extracted by video timestamp, round(pts * 59.98), which runs up
            to a few frames behind the decode order, and at quality 85 or 90; quality 0 marks the frames copied from the
            ScanNet++ toolkit's extraction, ffmpeg -q:v 1; byte-identical with ffmpeg 4.4.2,
            which scripts/setup.sh installs into the prospace environment and which is used by default
            (PROSPACE_FFMPEG=<path> overrides); other versions give the same pixels with their own version tag)
scannet     frame id = frame index in the .sens file; the JPEG stored there is copied byte for byte
ca1m        frame id -> timestamp (data/ca1m_frame_timestamps.json for the benchmark, data/prospace_train/
            ca1m_frame_timestamps.json for training); <capture>/<timestamp>.wide/image.png saved as JPEG, quality 85, at
            its own size (portrait captures stay 768x1024; the loaders pillar-box them to 512x384)
"""

import argparse
import gzip
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prospace_bench.data import FRAMES, check_layout, fetch  # noqa: E402


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


def ffmpeg():
    """PROSPACE_FFMPEG, else the ffmpeg of this Python's environment (setup.sh installs 4.4.2 there), else the PATH's."""
    if os.environ.get("PROSPACE_FFMPEG"):
        return shutil.which(os.environ["PROSPACE_FFMPEG"])
    own = Path(sys.executable).parent / "ffmpeg"
    return str(own) if own.exists() else shutil.which("ffmpeg")


def scannetpp(src, scene, dst, todo, sources=None):
    import av

    path = first(src / "data" / scene / "iphone/rgb.mkv", src / scene / "iphone/rgb.mkv")
    if path is None:
        return f"no {src}/data/{scene}/iphone/rgb.mkv"
    plan, ff = defaultdict(list), defaultdict(list)  # decode order -> [(frame id, JPEG quality)] / [frame id]
    for f in todo:
        k, q = sources[str(f)] if sources else (f, 90)
        (ff[k].append(f) if q == 0 else plan[k].append((f, q)))
    if plan:
        last = max(plan)
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            st.thread_type = "AUTO"
            for k, fr in enumerate(c.decode(st)):
                if k in plan:
                    im = fr.to_image()
                    for f, q in plan[k]:
                        write(name(dst, f), jpeg(im, q))
                if k >= last:
                    break
    if ff:  # frames of the ScanNet++ toolkit's extraction: ffmpeg -i rgb.mkv -start_number 0 -q:v 1 frame_%06d.jpg
        exe = ffmpeg()
        if exe is None:
            return f"{sum(map(len, ff.values()))} frames need ffmpeg (bash scripts/setup.sh installs ffmpeg 4.4.2)"
        tmp = Path(tempfile.mkdtemp(dir=dst, prefix=".ffmpeg"))
        ks = sorted(ff)
        sel = "+".join(f"eq(n\\,{k})" for k in ks)
        subprocess.run([exe, "-loglevel", "error", "-i", str(path), "-vf", f"select={sel}", "-vsync", "0", "-q:v", "1",
                        "-start_number", "0", str(tmp / "%06d.jpg")], check=True)
        for i, k in enumerate(ks):
            data = (tmp / f"{i:06d}.jpg").read_bytes()
            for f in ff[k]:
                write(name(dst, f), data)
        shutil.rmtree(tmp)


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


def ca1m(src, scene, dst, todo, ts):
    from PIL import Image

    want = {int(ts[str(f)]): f for f in todo}
    if (src / scene).is_dir():  # an unpacked tar
        for t, f in want.items():
            p = src / scene / f"{t}.wide" / "image.png"
            if p.exists():
                write(name(dst, f), jpeg(Image.open(p), 85))
        return
    tars = sorted(src.glob(f"ca1m-*-{scene}.tar"))
    if not tars:
        return f"no {src}/ca1m-<split>-{scene}.tar"
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
    dataset, src, scene, dst, ids, extra = job
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    todo = {f for f in ids if not name(dst, f).exists()}
    note = globals()[dataset](Path(src), scene, dst, todo, *extra) if todo else None
    missing = sum(not name(dst, f).exists() for f in ids)
    return scene, len(ids), len(todo), missing, note or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["scannetpp", "scannet", "ca1m"])
    ap.add_argument("--src", required=True, help="root of the official download (layout: see the top of this file)")
    ap.add_argument("--set", default="bench", choices=["bench", "train"])
    ap.add_argument("--out", default=str(FRAMES))
    ap.add_argument("--manifest", default=None, help="default: the md5 list of --set")
    ap.add_argument("--scenes", nargs="*", help="only these scene ids (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    check_layout()
    man = a.manifest or fetch(a.set, "frames_manifest.tsv" if a.set == "bench" else "frames_manifest.tsv.gz")
    want = defaultdict(list)
    with (gzip.open(man, "rt") if str(man).endswith(".gz") else open(man)) as f:
        next(f)
        for l in f:
            ds, scene, fid, _ = l.rstrip("\n").split("\t")
            if ds == a.dataset and (not a.scenes or scene in a.scenes):
                want[scene].append(int(fid))
    lookup = {}  # per-scene argument of the dataset's function
    if a.dataset == "ca1m":
        tsf = ROOT / "data/ca1m_frame_timestamps.json" if a.set == "bench" else fetch("train", "ca1m_frame_timestamps.json")
        lookup = json.loads(Path(tsf).read_text())
    elif a.dataset == "scannetpp" and a.set == "train":
        lookup = json.loads(Path(fetch("train", "scannetpp_frame_sources.json")).read_text())
    jobs = [
        (a.dataset, a.src, s, str(Path(a.out) / a.dataset / s), ids, (lookup[s],) if lookup else ())
        for s, ids in sorted(want.items())
    ]
    print(f"{a.dataset} ({a.set}): {len(jobs)} scenes, {sum(len(j[4]) for j in jobs)} frames", flush=True)
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
