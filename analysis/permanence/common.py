"""Shared pieces of the object permanence analysis: the stream list, frame paths, the ground-truth bearing.

streams.jsonl.gz has one line per run (a stream asked one question):
    set          fit | test | long | readout
    episode_id   the Object Direction episode; its last field is the direction the stream was generated for
    dataset, scene_id, frame_ids, w0, w1
    asked        the direction the question names (left | right | behind)
    question

fit / test: the probe's fitting streams (one question each, the direction drawn at random) and its 78 held-out
streams (three questions each).  long: 300 more held-out streams that stay out of view for at least 30 s (three
questions each).  readout: the 83 held-out streams whose response probabilities give the read-out accuracy.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / "cache" / "permanence"
RESULTS = ROOT / "results" / "permanence"
DIRS = ("left", "right", "behind")
HALF_HFOV = 34.0  # the object is in view when |bearing| < 34 deg (512 x 384 frames)


def load_runs() -> list[dict]:
    with gzip.open(HERE / "streams.jsonl.gz", "rt") as f:
        runs = [json.loads(l) for l in f]
    for i, r in enumerate(runs):
        r["run"] = i
        r["own"] = r["episode_id"].rsplit(":", 1)[1].rsplit("_v", 1)[0]
        r["stream"] = ("fit:" if r["set"] == "fit" else "test:") + r["episode_id"]
        r["object_id"] = r["episode_id"].split(":")[4]
    return runs


def frame_names(r: dict) -> list[str]:
    return [f"{r['dataset']}/{r['scene_id']}/frame_{int(f):06d}.jpg" for f in r["frame_ids"]]


def frame_roots(frames: str | Path) -> dict:
    return {ds: str((Path(frames) / ds).absolute()) for ds in ("scannetpp", "scannet", "ca1m")}


def frame_paths(r: dict, roots: dict) -> list[str]:
    """Absolute paths, as the model and the CUT3R cache see them."""
    return [str(Path(roots[r["dataset"]]) / n.split("/", 1)[1]) for n in frame_names(r)]


def bearing_deg(c2w: np.ndarray, p_xy: np.ndarray) -> np.ndarray:
    """(T,) signed ground-plane angle from the camera heading to point p, left positive (the Object Direction rule)."""
    cam = c2w[:, :2, 3]
    fwd = c2w[:, :2, 2]
    d = p_xy[None] - cam
    return np.degrees(np.arctan2(fwd[:, 0] * d[:, 1] - fwd[:, 1] * d[:, 0], (fwd * d).sum(1)))


def load_truth() -> dict:
    """{dataset: {scene: {"c2w": {frame_id: 4x4}, "objects": {object_id: [x, y, z]}}}} written by truth.py."""
    out = {}
    for f in sorted((CACHE / "truth").glob("*.json")):
        out[f.stem] = json.loads(f.read_text())
    return out


def geometry(r: dict, truth: dict):
    """(T, 4, 4) camera-to-world of the stream's frames and the object's ground-plane position, in the dataset's
    z-up world; None when the dataset's ground truth is not there."""
    sc = truth.get(r["dataset"], {}).get(r["scene_id"])
    if sc is None:
        return None, None
    c2w = np.array([sc["c2w"][str(f)] for f in r["frame_ids"]], np.float64)
    return c2w, np.asarray(sc["objects"][r["object_id"]][:2], np.float64)


def per_second(r: dict, truth: dict):
    """Per second of the run: true bearing, in view, seconds since first seen (-1 before), since last seen (-1 before)."""
    c2w, obj = geometry(r, truth)
    if c2w is None:
        return None
    th = bearing_deg(c2w, obj)
    vis = np.abs(th) < HALF_HFOV
    first = int(np.nonzero(vis)[0][0]) if vis.any() else -1
    since_first = np.array([t - first if first >= 0 and t >= first else -1 for t in range(len(th))])
    last, since_last = -1, []
    for t in range(len(th)):
        if vis[t]:
            last = t
        since_last.append(t - last if last >= 0 else -1)
    return {"theta": th, "vis": vis, "since_first": since_first, "since_last": np.array(since_last), "c2w": c2w, "obj": obj}


def wrap(x):
    return (np.asarray(x, float) + 180.0) % 360.0 - 180.0
