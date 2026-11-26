"""Run ProSpace-VLM over the permanence streams and keep, for every second, the decision-token state of the last layer
(the hidden state the response head reads) and p(respond).

    python analysis/permanence/dump.py --ckpt checkpoints/prospace_vlm.pth --shard 0/4     # one process per GPU
    bash analysis/permanence/run.sh 0,1,2,3                                                    # the same, all shards

One file per run, cache/permanence/states/<run>.npz (h: T x 3584 float16, p: T), so a stopped dump continues where it
was.  The read-out runs keep p only.  The streams are read exactly as in the benchmark evaluation (prospace_vlm/eval.py):
one forward pass, the question before the first frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, ROOT, frame_names, frame_roots, load_runs  # noqa: E402

sys.path.insert(0, str(ROOT))


def main() -> None:
    from prospace_vlm.data import loader
    from prospace_vlm.model import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "prospace_vlm.pth"))
    ap.add_argument("--geom-cache", default=str(ROOT / "cache" / "geom_cut3r"))
    ap.add_argument("--frames", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--out", default=str(CACHE / "states"))
    ap.add_argument("--shard", default="0/1", help="k/n: the streams k, k + n, ... (all questions of a stream together)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k, n = (int(v) for v in a.shard.split("/"))
    runs = load_runs()
    eps = sorted({r["episode_id"] for r in runs})
    mine = set(eps[k::n])
    todo = [r for r in runs if r["episode_id"] in mine and not (out / f"{r['run']:04d}.npz").exists()]
    print(f"shard {k}/{n}: {len(todo)} runs to dump", flush=True)
    if not todo:
        print("DUMP_SHARD_DONE", flush=True)
        return
    roots = frame_roots(a.frames)
    model, processor = load_model(a.ckpt, a.geom_cache)
    box = {}
    model.core.lm_head.register_forward_hook(lambda m, i, o: box.__setitem__("h", i[0].detach()))
    eps_in = [
        dict(
            episode_id=f"run{r['run']:04d}",
            task="object_direction",
            dataset=r["dataset"],
            scene_id=r["scene_id"],
            frames=frame_names(r),
            frame_ids=r["frame_ids"],
            question=r["question"],
            w0=r["w0"],
            w1=r["w1"],
            run=r["run"],
            keep_h=r["set"] != "readout",
        )
        for r in todo
    ]
    device = torch.device("cuda")
    with torch.no_grad():
        for j, item in enumerate(loader(eps_in, processor, roots, a.workers, train=False)):
            ep = item["ep"]
            p = torch.sigmoid(model(item, device)["final"].float()).cpu().numpy()
            save = {"p": p.astype(np.float32)}
            if ep["keep_h"]:
                save["h"] = box["h"].float().cpu().numpy().astype(np.float16)
                assert save["h"].shape[0] == len(p)
            tmp = out / f"{ep['run']:04d}.tmp.npz"
            np.savez(tmp, **save)
            tmp.rename(out / f"{ep['run']:04d}.npz")
            if (j + 1) % 25 == 0:
                print(f"  {j + 1}/{len(eps_in)}", flush=True)
    print("DUMP_SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
