"""Ground truth of the permanence streams from your copy of the official datasets: the camera pose of every frame and
the position of the asked object.  Nothing of it is redistributed; it is read here and written to cache/permanence/truth/.

    python analysis/permanence/truth.py --dataset scannetpp --src <ScanNet++ root>
    python analysis/permanence/truth.py --dataset scannet   --src <ScanNet root>
    python analysis/permanence/truth.py --dataset ca1m      --src <CA-1M root>

What is read (the frame of the Object Direction rule; z is up in all three):
    scannetpp  <root>/data/<scene>/iphone/pose_intrinsic_imu.json   aligned_pose of frame_<id>
               <root>/data/<scene>/scans/segments_anno.json          obb.centroid of the object
    scannet    <root>/scans/<scene>/<scene>.sens                      camera_to_world of frame <id>
               <scene>_vh_clean_2.ply, _vh_clean_2.0.010000.segs.json, .aggregation.json   mean of the object's vertices
    ca1m       <root>/ca1m-<split>-<capture>.tar                      <timestamp>.gt/RT.json, world.gt/instances.json position
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
import tarfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, HERE, load_runs  # noqa: E402


def first(*paths):
    return next((p for p in paths if p.exists()), None)


def scannetpp(src: Path, scene: str, fids, oids, _):
    d = first(src / "data" / scene, src / scene)
    if d is None:
        return f"no {src}/data/{scene}"
    pose = json.loads((d / "iphone" / "pose_intrinsic_imu.json").read_text())
    anno = json.loads((d / "scans" / "segments_anno.json").read_text())
    cent = {str(g["objectId"]): g["obb"]["centroid"] for g in anno["segGroups"]}
    return {
        "c2w": {str(f): pose[f"frame_{f:06d}"]["aligned_pose"] for f in fids},
        "objects": {o: [float(v) for v in cent[o]] for o in oids},
    }


def read_ply_vertices(path: Path) -> np.ndarray:
    """x, y, z of a binary little-endian PLY (the ScanNet meshes)."""
    with open(path, "rb") as f:
        head = []
        while True:
            line = f.readline().decode("ascii").strip()
            head.append(line)
            if line == "end_header":
                break
        assert "format binary_little_endian 1.0" in head, path
        n = next(int(l.split()[2]) for l in head if l.startswith("element vertex"))
        i = head.index(next(l for l in head if l.startswith("element vertex")))
        props = []
        for l in head[i + 1 :]:
            if not l.startswith("property"):
                break
            props.append(l.split())
        types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}
        dt = np.dtype([(p[2], types[p[1]]) for p in props])
        v = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)


def scannet(src: Path, scene: str, fids, oids, _):
    d = first(src / "scans" / scene, src / scene)
    if d is None:
        return f"no {src}/scans/{scene}"
    want, c2w = set(fids), {}
    with open(d / f"{scene}.sens", "rb") as f:  # layout of ScanNet's SensorData.py; payloads are skipped
        (version,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        f.read(n + 64 * 4 + 8 + 16 + 4)
        (frames,) = struct.unpack("<Q", f.read(8))
        for i in range(frames):
            pose = np.array(struct.unpack("<16f", f.read(64)), np.float64).reshape(4, 4)
            _, _, cs, ds = struct.unpack("<QQQQ", f.read(32))
            f.seek(cs + ds, 1)
            if i in want:
                c2w[str(i)] = pose.tolist()
                if len(c2w) == len(want):
                    break
    verts = read_ply_vertices(d / f"{scene}_vh_clean_2.ply")
    seg = np.asarray(json.loads((d / f"{scene}_vh_clean_2.0.010000.segs.json").read_text())["segIndices"])
    groups = {str(g["objectId"]): g["segments"] for g in json.loads((d / f"{scene}.aggregation.json").read_text())["segGroups"]}
    objects = {o: verts[np.isin(seg, groups[o])].mean(0).astype(np.float32).tolist() for o in oids}
    return {"c2w": c2w, "objects": objects}


def ca1m(src: Path, scene: str, fids, oids, ts):
    tars = sorted(src.glob(f"ca1m-*-{scene}.tar"))
    if not tars:
        return f"no {src}/ca1m-<split>-{scene}.tar"
    want = {str(ts[str(f)]): str(f) for f in fids}
    c2w, objects = {}, None
    with tarfile.open(tars[0], "r") as tar:  # random access: only the headers are read on the way
        for m in tar:
            p = m.name.split("/")
            if len(p) == 3 and p[1] == "world.gt" and p[2] == "instances.json":
                objects = {str(x["id"]): [float(v) for v in x["position"]] for x in json.load(tar.extractfile(m))}
            elif len(p) == 3 and p[1].endswith(".gt") and p[2] == "RT.json" and p[1][:-3] in want:
                c2w[want[p[1][:-3]]] = json.load(tar.extractfile(m))
    if objects is None:
        return f"{tars[0]}: no world.gt/instances.json"
    return {"c2w": c2w, "objects": {o: objects[o] for o in oids}}


def job(args):
    dataset, src, scene, fids, oids, ts = args
    try:
        out = globals()[dataset](Path(src), scene, fids, oids, ts)
    except (OSError, KeyError, ValueError) as e:
        out = f"{type(e).__name__}: {e}"
    return scene, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["scannetpp", "scannet", "ca1m"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    fids, oids = defaultdict(set), defaultdict(set)
    for r in load_runs():
        if r["dataset"] == a.dataset:
            fids[r["scene_id"]].update(int(f) for f in r["frame_ids"])
            oids[r["scene_id"]].add(r["object_id"])
    ts = defaultdict(dict)  # CA-1M frame id -> timestamp of its image
    with gzip.open(HERE / "frames.tsv.gz", "rt") as f:
        next(f)
        for l in f:
            ds, scene, fid, _, src = l.rstrip("\n").split("\t")
            if ds == "ca1m":
                ts[scene][fid] = src
    jobs = [(a.dataset, a.src, s, sorted(fids[s]), sorted(oids[s]), ts.get(s)) for s in sorted(fids)]
    print(f"{a.dataset}: {len(jobs)} scenes", flush=True)
    out, bad = {}, []
    with Pool(a.workers) as pool:
        for k, (scene, res) in enumerate(pool.imap_unordered(job, jobs), 1):
            if isinstance(res, str):
                bad.append(scene)
                print(f"  {scene}: {res}", flush=True)
                continue
            lost = len(fids[scene]) - sum(np.isfinite(np.asarray(v, float)).all() for v in res["c2w"].values())
            if lost:
                print(f"  {scene}: {lost} frames without a valid pose", flush=True)
            out[scene] = res
            if k % 20 == 0:
                print(f"  [{k}/{len(jobs)}]", flush=True)
    dst = CACHE / "truth" / f"{a.dataset}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out))
    print(f"{len(out)} scenes -> {dst}" + (f"; {len(bad)} scenes could not be read" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
