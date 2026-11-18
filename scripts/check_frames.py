"""Check data/frames against data/frames_manifest.tsv (md5 of every frame the benchmark reads).

    python scripts/check_frames.py                 # every frame: present and md5-identical
    python scripts/check_frames.py --no-md5        # only that the files are there (fast)

Layout: data/frames/<dataset>/<scene_id>/frame_<frame_id:06d>.jpg with dataset in {scannetpp, scannet, ca1m, mp3d}.

13 Matterport3D frames come out of scripts/render_mp3d.py a few pixels away from the benchmark's own copy (below 2% of
the pixels of a frame); they are counted as "near" and do not fail the check.
"""

import argparse
import hashlib
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MP3D_NEAR = {
    ("QUCTc6BB5sX_ng100045", 550),
    ("QUCTc6BB5sX_po2154", 360),
    ("YFuZgdQ5vWj_ne100417", 70),
    ("jtcxE69GiFV_po2344", 850),
    ("rqfALeAoiTq_en3148", 350),
    ("rqfALeAoiTq_re501468", 50),
    ("wc2JMjhGNzB_po2552", 1310),
    ("wc2JMjhGNzB_re3820", 460),
    ("wc2JMjhGNzB_re3822", 590),
    ("wc2JMjhGNzB_re3834", 120),
    ("zsNo4HB9uLZ_nr300657", 750),
    ("zsNo4HB9uLZ_re3919", 430),
    ("zsNo4HB9uLZ_re501739", 1080),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "data/frames"))
    ap.add_argument("--manifest", default=str(ROOT / "data/frames_manifest.tsv"))
    ap.add_argument("--no-md5", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = [l.rstrip("\n").split("\t") for l in open(a.manifest)][1:]

    def one(r):
        ds, scene, fid, md5 = r
        p = Path(a.frames) / ds / scene / f"frame_{int(fid):06d}.jpg"
        if not p.exists():
            return ds, "missing", str(p)
        if a.no_md5 or hashlib.md5(p.read_bytes()).hexdigest() == md5:
            return ds, "ok", str(p)
        return ds, "near" if ds == "mp3d" and (scene, int(fid)) in MP3D_NEAR else "different", str(p)

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, rows))
    c = Counter((ds, s) for ds, s, _ in res)
    for ds in sorted({r[0] for r in rows}):
        print(
            f"{ds:10s} ok {c[(ds, 'ok')]:6d}  missing {c[(ds, 'missing')]:6d}  different {c[(ds, 'different')]:6d}"
            + (f"  near {c[(ds, 'near')]:3d}" if c[(ds, "near")] else "")
        )
    bad = [(s, p) for _, s, p in res if s not in ("ok", "near")]
    for s, p in bad[:10]:
        print(s, p)
    ok = sum(s == "ok" for _, s, _ in res)
    print(f"{ok} / {len(res)} frames ok" + ("" if a.no_md5 else " (md5)"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
