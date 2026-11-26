"""Top view of one example: the camera path, the asked object (star) and, at ten out-of-view seconds, the bearing the
probe reads from ProSpace-VLM's state (arrows).  Arrows, dots and the path are coloured by the bearing error at that
second (green 0 deg -> amber -> red >= 90 deg); grey where the object is in view or not seen yet.

    python analysis/permanence/topview.py A --mesh <ARKitScenes>/<split>/47331084/47331084_3dod_mesh.ply
    python analysis/permanence/topview.py B --mesh <ARKitScenes>/<split>/45261495/45261495_3dod_mesh.ply
    python analysis/permanence/topview.py A            # no mesh: the CUT3R points of the stream as the floor plan

Reads results/permanence/examples.json (probe.py) and the ground truth of truth.py; writes
results/permanence/topview_<example>.{png,svg,pdf}.  The defaults (VIEWS) give the framing of the paper's two examples.

Background mesh: ScanNet++ mesh_aligned_0.05.ply is in the frame of the poses and is used as it is.  CA-1M has no mesh;
the ARKitScenes mesh of the same capture (<id>_3dod_mesh.ply) is moved into the CA-1M frame by a rigid fit of the
ARKitScenes camera centres (lowres_wide.traj next to the mesh, or --traj) to the CA-1M ones, matched by timestamp.
Faces between 1.6 m below and 0.3 m above the first camera are drawn, so the ceiling does not hide the floor.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, RESULTS, ROOT, frame_paths, frame_roots, load_runs, load_truth, per_second, wrap  # noqa: E402

VIEWS = {  # view rotation (deg); crop of the square view around the whole path (left, top, side, as fractions); height / width
    "A": dict(rotate=45.0, crop=(0.10, 0.04, 0.86), aspect=1.0),
    "B": dict(rotate=-40.5, crop=(0.0877, 0.2156, 0.6182), aspect=444 / 440),
}
K_ARROWS, EMAX, GREY = 10, 90.0, "#BDBDBD"
BAND = (-1.6, 0.3)
WHITE, DESAT = 0.50, 0.40  # mesh colours: desaturate, then blend toward white


def ramp():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("bearing_error", ["#1E9E5A", "#EFA12E", "#D7301F"])


def read_ply(path):
    """Vertices (N, 3), triangles (M, 3) and vertex colours (N, 3) in 0..1 (or None) of a binary little-endian PLY."""
    with open(path, "rb") as f:
        head = []
        while not head or head[-1] != "end_header":
            head.append(f.readline().decode("ascii").strip())
        assert "format binary_little_endian 1.0" in head, path
        types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}
        els, cur = [], None
        for l in head:
            p = l.split()
            if p and p[0] == "element":
                cur = [p[1], int(p[2]), []]
                els.append(cur)
            elif p and p[0] == "property":
                cur[2].append(p[1:])
        V = F = None
        for name, n, props in els:
            if name == "vertex":
                dt = np.dtype([(q[1], types[q[0]]) for q in props])
                V = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
            elif name == "face":
                assert len(props) == 1, "faces with extra properties"
                cnt, idx, _ = props[0][1:]
                dt = np.dtype([("n", types[cnt]), ("i", types[idx], 3)])
                F = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
                assert (F["n"] == 3).all(), "triangles only"
                break
    xyz = np.stack([V["x"], V["y"], V["z"]], 1).astype(np.float64)
    col = np.stack([V["red"], V["green"], V["blue"]], 1) / 255.0 if "red" in V.dtype.names else None
    return xyz, F["i"].astype(np.int64), col


def axis_angle(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def arkit_to_ca1m(traj, ts_s, centres):
    """Rigid transform (scale 1) from the ARKitScenes frame to the CA-1M frame: camera centres of lowres_wide.traj
    (world -> camera, axis-angle) interpolated at the CA-1M frame times, fitted to the CA-1M centres (Kabsch)."""
    rows = np.loadtxt(traj)
    C = np.array([-axis_angle(r[1:4]).T @ r[4:7] for r in rows])
    ok = (ts_s > rows[0, 0]) & (ts_s < rows[-1, 0])
    A = np.stack([np.interp(ts_s[ok], rows[:, 0], C[:, j]) for j in range(3)], 1)
    B = centres[ok]
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    t = mb - R @ ma
    res = np.linalg.norm(A @ R.T + t - B, axis=1)
    print(f"ARKitScenes -> CA-1M: {ok.sum()} frames, residual median {100 * np.median(res):.1f} cm, max {100 * res.max():.1f} cm")
    return R, t


def mesh_layer(path, traj, run, g, lo, hi):
    V, F, VC = read_ply(path)
    if run["dataset"] == "ca1m":
        ts = {}
        with gzip.open(HERE / "frames.tsv.gz", "rt") as f:
            for l in f:
                p = l.rstrip("\n").split("\t")
                if p[0] == "ca1m" and p[1] == run["scene_id"]:
                    ts[int(p[2])] = int(p[4])
        R, t = arkit_to_ca1m(traj or Path(path).parent / "lowres_wide.traj",
                             np.array([ts[int(f)] / 1e9 for f in run["frame_ids"]]), g["c2w"][:, :3, 3])
        V = V @ R.T + t
    z0 = g["c2w"][0, 2, 3]
    T = V[F]
    inb = ((T[..., 0] > lo[0] - 0.3) & (T[..., 0] < hi[0] + 0.3) & (T[..., 1] > lo[1] - 0.3) & (T[..., 1] < hi[1] + 0.3)).any(1)
    keep = (T[..., 2].max(1) < z0 + BAND[1]) & (T[..., 2].min(1) > z0 + BAND[0]) & inb
    T, Fk = T[keep], F[keep]
    order = np.argsort(T[..., 2].mean(1))
    T, Fk = T[order], Fk[order]
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.35, 0.45, 0.82])
    lam = np.clip(np.abs(nrm @ (light / np.linalg.norm(light))), 0, 1)
    if VC is None:
        c = np.repeat((0.80 + 0.15 * lam)[:, None], 3, 1)
    else:
        c = VC[Fk].mean(1) * (0.85 + 0.15 * lam[:, None])
    grey = c.mean(1, keepdims=True)
    col = (1 - WHITE) * ((1 - DESAT) * c + DESAT * grey) + WHITE
    return T[..., :2], np.clip(col, 0, 1)


def point_layer(run, g, cache):
    """The stream's CUT3R points (first camera = origin) moved into the dataset frame by the first frame's pose."""
    paths = frame_paths(run, frame_roots(ROOT / "data" / "frames"))
    f = Path(cache) / (hashlib.md5("\n".join(paths).encode()).hexdigest() + ".npz")
    if not f.exists():
        print(f"no CUT3R cache {f}: drawing without a floor plan")
        return None
    xyz = np.load(f)["xyz"].astype(np.float64).reshape(-1, 3)
    xyz = xyz[np.isfinite(xyz).all(1)]
    g0 = g["c2w"][0]
    w = xyz @ g0[:3, :3].T + g0[:3, 3]
    k = (w[:, 2] < g0[2, 3] + BAND[1]) & (w[:, 2] > g0[2, 3] + BAND[0])
    return w[k]


def smooth(cam):
    """Centripetal Catmull-Rom through the 1 Hz camera positions (12 steps per second) -> points and their times."""
    P = np.vstack([2 * cam[0] - cam[1], cam, 2 * cam[-1] - cam[-2]])
    pts, ts = [], []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1 : i + 3]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-4) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-4) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-4) ** 0.5
        for u in np.linspace(t1, t2, 12, endpoint=False):
            A1 = (t1 - u) / (t1 - t0) * p0 + (u - t0) / (t1 - t0) * p1
            A2 = (t2 - u) / (t2 - t1) * p1 + (u - t1) / (t2 - t1) * p2
            A3 = (t3 - u) / (t3 - t2) * p2 + (u - t2) / (t3 - t2) * p3
            B1 = (t2 - u) / (t2 - t0) * A1 + (u - t0) / (t2 - t0) * A2
            B2 = (t3 - u) / (t3 - t1) * A2 + (u - t1) / (t3 - t1) * A3
            pts.append((t2 - u) / (t2 - t1) * B1 + (u - t1) / (t2 - t1) * B2)
            ts.append(i - 1 + (u - t1) / (t2 - t1))
    pts.append(cam[-1])
    ts.append(len(cam) - 1)
    return np.array(pts), np.array(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("example", choices=list(VIEWS))
    ap.add_argument("--mesh", default=None, help="scene mesh (see the top of this file); default: CUT3R points")
    ap.add_argument("--traj", default=None, help="CA-1M: ARKitScenes lowres_wide.traj (default: next to the mesh)")
    ap.add_argument("--rotate", type=float, default=None, help="view rotation in deg (default: the example's)")
    ap.add_argument("--geom-cache", default=str(ROOT / "cache" / "geom_cut3r"))
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import to_hex

    view = VIEWS[a.example]
    ex = json.loads((RESULTS / "examples.json").read_text())[a.example]
    run = next(r for r in load_runs() if r["episode_id"] == ex["episode_id"] and r["asked"] == r["own"] and r["set"] in ("test", "long"))
    g = per_second(run, load_truth())
    assert g is not None, f"no ground truth for {run['dataset']} (truth.py)"
    err = {int(t): float(abs(wrap(b - th))) for t, b, th in ex["seconds"]}
    bel = {int(t): float(b) for t, b, _ in ex["seconds"]}
    ts = np.array(sorted(err))
    cam, fwd, obj = g["c2w"][:, :2, 3], g["c2w"][:, :2, 2], g["obj"]
    ang = np.radians(view["rotate"] if a.rotate is None else a.rotate)
    Rz = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    piv = cam.mean(0)
    turn = lambda p: (p - piv) @ Rz.T + piv
    cam2, obj2, fwd2 = turn(cam), turn(obj[None])[0], fwd @ Rz.T
    pts = np.vstack([cam2, obj2[None]])
    lo, hi = pts.min(0) - 1.6, pts.max(0) + 1.6
    side = max(hi - lo)
    mid = (lo + hi) / 2
    lo, hi = mid - side / 2, mid + side / 2
    L = 0.45 * side / 4.0 + 0.9  # arrow length (m)
    fx, fy, fs = view["crop"]
    lo, hi = np.array([lo[0] + fx * side, hi[1] - (fy + fs) * side]), np.array([lo[0] + (fx + fs) * side, hi[1] - fy * side])
    yc, hh = (lo[1] + hi[1]) / 2, (hi[0] - lo[0]) / 2 * view["aspect"]
    lo, hi = np.array([lo[0], yc - hh]), np.array([hi[0], yc + hh])

    rm = ramp()
    ecol = lambda e: to_hex(rm(min(e, EMAX) / EMAX))

    def col(t):  # colour of the path at time t: error of the nearest out-of-view second, grey if none within 1 s
        j = int(np.argmin(np.abs(ts - t)))
        return ecol(err[int(ts[j])]) if abs(ts[j] - t) <= 1.0 else GREY

    fig, ax = plt.subplots(figsize=(6.0, 6.0 * view["aspect"]))
    if a.mesh:
        corners = (np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]]) - piv) @ Rz + piv
        tri, c = mesh_layer(a.mesh, a.traj, run, g, corners.min(0), corners.max(0))
        tri = (tri - piv) @ Rz.T + piv
        ax.add_collection(PolyCollection(tri, facecolors=c, edgecolors=c, linewidths=0.15, antialiased=False, zorder=1, rasterized=True))
    else:
        w = point_layer(run, g, a.geom_cache)
        if w is not None:
            w2 = turn(w[:, :2])
            ax.scatter(w2[:, 0], w2[:, 1], s=1.2, c="0.85", linewidths=0, rasterized=True, zorder=1)
    P, Pt = smooth(cam2)
    ax.add_collection(LineCollection(np.stack([P[:-1], P[1:]], 1), colors=[col(x) for x in Pt[:-1]], linewidths=3.2,
                                     capstyle="round", joinstyle="round", zorder=3))
    ax.scatter(*cam2[0], s=90, color=col(0), edgecolors="k", linewidths=1.0, zorder=4)
    shown = ts[np.unique(np.linspace(0, len(ts) - 1, min(K_ARROWS, len(ts))).round().astype(int))]
    for t in shown:
        f = fwd2[t] / (np.linalg.norm(fwd2[t]) + 1e-9)
        b = np.radians(bel[int(t)])
        d = np.array([np.cos(b) * f[0] - np.sin(b) * f[1], np.sin(b) * f[0] + np.cos(b) * f[1]])
        ax.scatter(*cam2[t], s=40, color=ecol(err[int(t)]), edgecolors="k", linewidths=0.6, zorder=6)
        ax.annotate("", xy=cam2[t] + L * d, xytext=cam2[t], zorder=5,
                    arrowprops=dict(arrowstyle="-|>,head_length=1.2,head_width=0.6", color=ecol(err[int(t)]), lw=3.9,
                                    alpha=0.95, shrinkA=0, shrinkB=0))
    ax.scatter(*obj2, marker="*", s=950, c="#F0E442", edgecolors="k", linewidths=1.4, zorder=7)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"topview_{a.example}.{ext}", dpi=a.dpi, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    plt.close(fig)
    arrows = ", ".join(f"{int(t)} s: {err[int(t)]:.1f}" for t in shown)
    print(f"example {a.example}: {run['dataset']} {run['scene_id']}, median bearing error {np.median(list(err.values())):.1f} deg over "
          f"{len(ts)} out-of-view seconds; arrows (second: error deg) {arrows}")
    print(f"written: {out}/topview_{a.example}.png / .svg / .pdf")


if __name__ == "__main__":
    main()
