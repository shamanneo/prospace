"""Evaluate a native-streaming model on ProSpace-Bench (scripts/eval_prospace_vlm.sh).

The model reads a stream once.  Its response is the first second, from the second the request is asked at, where
p(respond) >= 0.5; a stream where that never happens gets no response.

    python -m prospace_vlm.eval --data cache/streams --ckpt checkpoints/prospace_vlm.pth --out results/x --shard 0/4
    python -m prospace_vlm.eval --out results/x --merge

A shard writes one line per stream as it goes, so a stopped run continues where it was.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch

RULE = 0.5
TASKS = [
    "ego_rotation",
    "origin_return",
    "visual_homing",
    "object_finding",
    "object_refinding",
    "object_direction",
    "place_transition",
    "place_entry",
    "place_revisit",
]


def response_second(ep: dict, final_logits: torch.Tensor):
    """The first second >= ep["start"] with p(respond) >= RULE, or None."""
    prob = torch.sigmoid(final_logits)
    return next((t for t in range(int(ep["start"]), len(prob)) if prob[t] >= RULE), None)


def verdict_of(ep: dict, final_logits: torch.Tensor) -> str:
    fire = response_second(ep, final_logits)
    if ep["w0"] is None:
        return "correct_abstention" if fire is None else "false_alarm"
    if fire is None:
        return "miss"
    return "premature" if fire < ep["w0"] else "timely" if fire <= ep["w1"] else "delayed"


def run_shard(a) -> None:
    from prospace_vlm.data import loader
    from prospace_vlm.model import load_model

    data = Path(a.data)
    roots = json.loads(Path(a.roots or data / "roots.json").read_text())
    k, n = (int(v) for v in a.shard.split("/"))
    eps = [json.loads(l) for l in open(data / "val_poll.jsonl")][k::n]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"verdicts.shard{k}.jsonl"
    done = {(r["task"], r["episode_id"]) for r in map(json.loads, open(f))} if f.exists() else set()
    eps = [e for e in eps if (e["task"], e["episode_id"]) not in done]
    if a.limit:
        eps = eps[: a.limit]
    print(f"shard {k}/{n}: {len(eps)} streams to score ({len(done)} done)", flush=True)
    model, processor = load_model(a.ckpt, a.geom_cache)
    device = torch.device("cuda")
    with torch.no_grad(), open(f, "a") as w:
        for j, item in enumerate(loader(eps, processor, roots, a.workers, train=False)):
            ep = item["ep"]
            logits = model(item, device)["final"].float().cpu()
            p = torch.sigmoid(logits)
            w.write(
                json.dumps(
                    {
                        "episode_id": ep["episode_id"],
                        "task": ep["task"],
                        "kind": ep["kind"],
                        "w0": ep["w0"],
                        "w1": ep["w1"],
                        "start": ep["start"],
                        "fire": response_second(ep, logits),
                        "verdict": verdict_of(ep, logits),
                        "p": [round(float(x), 4) for x in p],
                    }
                )
                + "\n"
            )
            w.flush()
            if (j + 1) % 50 == 0:
                print(f"  {j + 1}/{len(eps)}", flush=True)
    print("BENCH_SHARD_DONE", flush=True)


def merge(out: Path) -> None:
    """All shards -> <out>/eval_bench.json and the printed row: timely / abstention, average and per task."""
    rows = {}
    for f in sorted(out.glob("verdicts.shard*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            rows[(r["task"], r["episode_id"])] = r  # an episode id is unique inside its task only
    by = collections.defaultdict(list)
    for r in rows.values():
        by[r["task"]].append(r)
    m = {"n_streams": len(rows)}
    for t in TASKS:
        pos = [r for r in by[t] if r["w0"] is not None]
        neg = [r for r in by[t] if r["w0"] is None]
        m[t] = {
            "n_pos": len(pos),
            "n_neg": len(neg),
            **{
                v: 100 * sum(r["verdict"] == v for r in pos) / max(1, len(pos))
                for v in ("timely", "premature", "delayed", "miss")
            },
            "silence": 100 * sum(r["verdict"] == "correct_abstention" for r in neg) / max(1, len(neg)),
        }
    done = [t for t in TASKS if m[t]["n_pos"]]
    m["task_mean_timely"] = sum(m[t]["timely"] for t in done) / max(1, len(done))
    m["task_mean_silence"] = sum(m[t]["silence"] for t in done if m[t]["n_neg"]) / max(
        1, sum(1 for t in done if m[t]["n_neg"])
    )
    (out / "eval_bench.json").write_text(json.dumps(m, indent=1))
    print(f"streams {len(rows)} | task-mean timely {m['task_mean_timely']:.1f} silence {m['task_mean_silence']:.1f}")
    print(
        " ".join(
            [f"{m['task_mean_timely']:.1f}", f"{m['task_mean_silence']:.1f}"]
            + [f"{m[t][k]:.1f}" for t in TASKS for k in ("timely", "silence")]
        )
    )
    for t in TASKS:
        print(
            f"  {t:18s} pos {m[t]['n_pos']:4d} timely {m[t]['timely']:5.1f} prem {m[t]['premature']:5.1f} | neg {m[t]['n_neg']:4d} silence {m[t]['silence']:5.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cache/streams", help="folder of scripts/build_streams.py")
    ap.add_argument("--roots", default=None, help="json: dataset -> frame folder (default <data>/roots.json)")
    ap.add_argument("--ckpt", default=None, help="prospace_vlm.pth, or the best/ folder of a training run")
    ap.add_argument("--geom-cache", default=None, help="folder for the CUT3R outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="k/n: every n-th stream starting at k")
    ap.add_argument("--merge", action="store_true", help="merge the shards of --out and print the row")
    ap.add_argument("--limit", type=int, default=0, help="only the first N streams of the shard")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.merge:
        return merge(Path(a.out))
    assert a.ckpt, "--ckpt"
    run_shard(a)


if __name__ == "__main__":
    main()
