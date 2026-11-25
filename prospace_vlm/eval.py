"""Evaluate a native-streaming model on ProSpace-Bench (scripts/eval_prospace_vlm.sh).

The model reads a stream once.  Its response is the first second, from the second the request is asked at, where
p(respond) >= 0.5; a stream where that never happens gets no response.

    python -m prospace_vlm.eval --data cache/streams --ckpt checkpoints/prospace_vlm.pth --out results/x --shard 0/4
    python -m prospace_vlm.eval --out results/x --merge

A shard writes one line per stream as it goes, so a stopped run continues where it was.

By default the stream is read in one forward pass, which equals streaming with an unbounded KV cache.  With
`--token-selection 32:16` (Spatial Token Selection) the stream is read frame by frame with a bounded KV cache; see
streaming_forward.
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


# ----------------------------------------------------------------------------- Spatial Token Selection
def frame_spans(ids: torch.Tensor, slots: torch.Tensor, vision_start_id: int):
    """ids (L,), slots (T,) = the <|vision_end|> positions -> (end of the request prefix, [(a, b) span of frame t]).
    Frame t = <|vision_start|> ... <|vision_end|> and its decision token; the frames are contiguous."""
    T = int(slots.shape[0])
    first = int(torch.nonzero(ids == vision_start_id, as_tuple=False)[0, 0])
    starts = [first] + [int(slots[t - 1]) + 2 for t in range(1, T)]
    spans = [(starts[t], int(slots[t]) + 2) for t in range(T)]
    for a, b in spans:
        assert int(ids[a]) == vision_start_id and b <= ids.shape[0], f"frame span ({a}, {b})"
    return first, spans


def select_tokens(frame, vision, xyz, t: int, budget: int, recent: int, voxel: float = 0.25) -> torch.Tensor:
    """The cached tokens after the request prefix (in cache order): frame index (n,), is-vision (n,), 3D point (n, 3)
    (NaN for none).  Returns the sorted indices kept before frame t is fed: every token of the last `recent` - 1 frames;
    of the older frames, only vision tokens with a 3D point, the earliest one in each voxel, the voxel grown x1.5 from
    0.25 m until they fit the rest of the budget."""
    is_recent = frame >= t - (recent - 1)
    keep = is_recent.clone()
    room = budget - int(is_recent.sum())
    cand = torch.nonzero((~is_recent) & vision & torch.isfinite(xyz).all(-1), as_tuple=False)[:, 0]
    if room > 0 and cand.numel():
        P, s = xyz[cand].double(), voxel
        while True:
            _, inv = torch.unique(torch.floor(P / s).long(), dim=0, return_inverse=True)
            if int(inv.max()) + 1 <= room:
                break
            s *= 1.5
        first = torch.full((int(inv.max()) + 1,), cand.numel(), dtype=torch.long, device=inv.device)
        first = first.scatter_reduce(0, inv, torch.arange(cand.numel(), device=inv.device), reduce="amin")
        keep[cand[first]] = True
    return torch.nonzero(keep, as_tuple=False)[:, 0]


def keep_cache(cache, idx: torch.Tensor) -> None:
    """Keep only the cache positions idx in every layer."""
    for layer in cache.layers:
        layer.keys = layer.keys.index_select(-2, idx)
        layer.values = layer.values.index_select(-2, idx)


def streaming_forward(model, budget_frames: int, recent_frames: int):
    """Spatial Token Selection: replaces the backbone's forward so that model(item, device) reads the stream frame by
    frame with a KV cache that never holds more than the request prefix + `budget_frames` frames of tokens.

    The request prefix (system prompt, instruction, request) is always kept.  Before frame t is fed, if the cached
    frames exceed `budget_frames` - 1 frames of tokens, the last `recent_frames` - 1 frames stay whole and of the older
    frames only one vision token per voxel stays (select_tokens; the 3D point of a token is its CUT3R point, the one the
    geometric encoding uses).  Evicted tokens are never put back, and the kept tokens keep their original M-RoPE
    positions.  Inference only, on a model trained with the full context; ProSpace-VLM checkpoints only (the 3D points
    come from its geometry)."""
    m = model.core.model
    assert model.inject is not None, "--token-selection needs a ProSpace-VLM checkpoint (the 3D points of its geometry)"
    assert m.config._attn_implementation == "sdpa", "explicit attention masks need sdpa attention"
    assert 1 <= recent_frames <= budget_frames
    # the cache length changes almost every frame; cuDNN's SDPA would build a new plan for each (this is also the
    # kernel setting of our runs)
    torch.backends.cuda.enable_cudnn_sdp(False)

    def forward(
        input_ids=None, pixel_values=None, image_grid_thw=None, attention_mask=None, mm_token_type_ids=None, **kw
    ):
        assert input_ids.shape[0] == 1 and (attention_mask is None or bool(attention_mask.all())), "one stream"
        emb = m.get_input_embeddings()(input_ids)
        with torch.no_grad():
            img = torch.cat(m.get_image_features(pixel_values, image_grid_thw).pooler_output, 0).to(emb.dtype)
        is_img = input_ids == m.config.image_token_id
        emb = emb.masked_scatter(is_img[..., None], img)
        pos = m.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=emb,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )
        ids = input_ids[0]
        slots = torch.nonzero(ids == m.config.vision_end_token_id, as_tuple=False)[:, 0]
        prefix, spans = frame_spans(ids, slots, m.config.vision_start_token_id)
        full = model._ctx  # the geometry of the whole stream (ProSpaceVLM._set_geometry)
        vision = is_img[0]
        xyz = torch.full((ids.shape[0], 3), float("nan"), device=emb.device)
        xyz[full["img_pos"]] = full["xyz"].float()
        dev = emb.device
        H = emb.new_zeros(1, ids.shape[0], emb.shape[-1])
        cache = None
        fr_c = torch.empty(0, dtype=torch.long, device=dev)
        vis_c = torch.empty(0, dtype=torch.bool, device=dev)
        xyz_c = torch.empty(0, 3, dtype=torch.float32, device=dev)
        try:
            for a, b, t in [(0, prefix, None)] + [(a, b, t) for t, (a, b) in enumerate(spans)]:
                q = b - a
                if t is None:
                    model._ctx = None
                else:
                    budget = (budget_frames - 1) * q
                    if fr_c.numel() > budget:
                        keep = select_tokens(fr_c, vis_c, xyz_c, t, budget, recent_frames)
                        if keep.numel() < fr_c.numel():
                            keep_cache(cache, torch.cat([torch.arange(prefix, device=dev), prefix + keep]))
                            fr_c, vis_c, xyz_c = fr_c[keep], vis_c[keep], xyz_c[keep]
                    sel = (full["img_pos"] >= a) & (full["img_pos"] < b)  # the injection hooks index the chunk
                    model._ctx = {**full, "img_pos": full["img_pos"][sel] - a, "code": full["code"][sel]}
                past = 0 if cache is None else int(cache.get_seq_length())
                mask = torch.ones(q, past + q, dtype=torch.bool, device=dev)  # every cached token + causal in the chunk
                mask[:, past:] = torch.tril(torch.ones(q, q, dtype=torch.bool, device=dev))
                mask = mask[None, None]
                out = m.language_model(
                    inputs_embeds=emb[:, a:b],
                    position_ids=pos[..., a:b],
                    attention_mask={"full_attention": mask, "sliding_attention": mask},
                    past_key_values=cache,
                    use_cache=True,
                )
                H[:, a:b], cache = out.last_hidden_state, out.past_key_values
                if t is not None:
                    fr_c = torch.cat([fr_c, torch.full((q,), t, dtype=torch.long, device=dev)])
                    vis_c = torch.cat([vis_c, vision[a:b]])
                    xyz_c = torch.cat([xyz_c, xyz[a:b]])
        finally:
            model._ctx = full
        from types import SimpleNamespace

        return SimpleNamespace(last_hidden_state=H)

    m.forward = forward  # an instance attribute: the module call dispatches here


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
    if a.token_selection:
        streaming_forward(model, *(int(v) for v in a.token_selection.split(":")))
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
    ap.add_argument(
        "--token-selection",
        default=None,
        metavar="B:R",
        help="Spatial Token Selection: a KV cache of at most the request + B frames of tokens, the last R frames whole "
        "(the paper: 32:16); default off (unbounded cache)",
    )
    a = ap.parse_args()
    if a.merge:
        return merge(Path(a.out))
    assert a.ckpt, "--ckpt"
    run_shard(a)


if __name__ == "__main__":
    main()
