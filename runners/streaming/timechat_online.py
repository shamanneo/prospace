"""TimeChat-Online-7B streaming runner (Object Finding, first-trigger protocol).

Official code wrapped (third_party/TimeChat-Online, no edits):
  * eval/scripts/ovobench.py : build_prompt("CRR", ...)  -- OVO-Bench Forward Active Responding prompt
                               get_response(prompt, video, model, processor) -- per-query inference
  * eval/qwen2_5_vl          : Qwen2_5_VLForConditionalGeneration with Differential Token Drop (DTD)

How streaming works for this model: there is no LLM-side KV-cache streaming and no working online trigger in the
repo (demo/web_demo.py computes `trigger_time` from DTD drop ratios over the *whole* uploaded video, top-10 %
lowest ratio, and never uses it).  The repo's documented forward-active-responding evaluation (OVO-Bench CRR/SSR)
re-queries the model on the video prefix [0, t] at each probe time with a Yes/No prompt.  We do exactly that at
every second t = 0..n-1 (frames 0..t, qwen_vl_utils pads an odd prefix by repeating the last frame); DTD drops
redundant visual tokens of each prefix before the LLM.  response_i = first t whose reply parses to "Yes".

Run in the TimeChat-Online repository's own environment (requirement.txt pins + flash-attn 2.7.4.post1):
  CUDA_VISIBLE_DEVICES=0 python runners/streaming/timechat_online.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "third_party" / "TimeChat-Online"
sys.path.insert(0, str(ROOT / "runners"))
sys.path.insert(0, str(REPO / "eval" / "scripts"))  # ovobench.py appends eval/ itself for `qwen2_5_vl`

import protocol  # noqa: E402  (torch-free)


def wait_for_gpu(need_gb: float, log, poll_s: int = 60):
    """The GPU may be shared: wait (never crash) until enough memory is free."""
    import subprocess

    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    while True:
        try:
            free = (
                float(
                    subprocess.check_output(
                        ["nvidia-smi", "-i", dev, "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
                    )
                    .decode()
                    .strip()
                )
                / 1024
            )
        except Exception as e:
            log.warning(f"nvidia-smi failed ({e}); proceeding")
            return
        if free >= need_gb:
            log.info(f"GPU {dev}: {free:.1f} GB free (need {need_gb})")
            return
        log.info(f"GPU {dev}: only {free:.1f} GB free, need {need_gb}; waiting {poll_s}s")
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="out/epsiodes/object_finding_stream10.json")
    ap.add_argument("--frames", default=protocol.DEFAULT_FRAMES)
    ap.add_argument("--out", default="out/runs/streaming/stream10/timechat_online.jsonl")
    ap.add_argument("--log", default=None)
    ap.add_argument("--ckpt", default="wyccccc/TimeChatOnline-7B")
    ap.add_argument("--task", default="object_finding")
    ap.add_argument("--drop_method", default="feature")  # ovobench.py / eval_ovobench.sh defaults
    ap.add_argument("--drop_threshold", type=float, default=0.5)
    # 512x384 frames -> smart_resize(min=max=pixels) gives exactly 504x392 (18x14 merged tokens / 2-frame grid);
    # the official eval budget is 448*448 = 200,704 px, this is 197,568 px.
    ap.add_argument("--pixels", type=int, default=504 * 392)
    ap.add_argument("--episode_id", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--need_gb", type=float, default=22.0)
    ap.add_argument("--stop_at_first", action="store_true", help="stop polling at the first Yes (default: full trace)")
    ap.add_argument(
        "--score_from_field",
        default=None,
        help="episode field; a Yes before that second is not a response (Re-finding: gap_start_i)",
    )
    args = ap.parse_args()
    args.out = str(ROOT / args.out) if not os.path.isabs(args.out) else args.out
    log_path = args.log or os.path.splitext(args.out)[0] + ".log"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    log = logging.getLogger("timechat_online")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s | %(message)s")
    for h in (logging.FileHandler(log_path), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)

    import torch
    from PIL import Image
    from transformers import AutoProcessor

    # ovobench.py imports moviepy (video chunking) and pandas at module level; neither is used by build_prompt /
    # get_response.  Stub them only if absent -- no repo code is changed.
    import importlib.util
    import types

    if importlib.util.find_spec("moviepy") is None:
        editor = types.ModuleType("moviepy.editor")
        editor.VideoFileClip = None
        sys.modules["moviepy"] = types.ModuleType("moviepy")
        sys.modules["moviepy.editor"] = editor
    if importlib.util.find_spec("pandas") is None:
        sys.modules["pandas"] = types.ModuleType("pandas")
    import ovobench  # official eval script (module globals = its CLI args)
    from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    ovobench.DROP_METHOD = args.drop_method
    ovobench.DROP_THRESHOLD = args.drop_threshold
    ovobench.DROP_ABSOLUTE = True
    ovobench.MIN_PIXELS = ovobench.MAX_PIXELS = args.pixels
    dr_path = os.path.splitext(args.out)[0] + ".drop.txt"  # DTD: one {drop,total,ratio} line per prefill
    ovobench.DR_SAVE_PATH = dr_path

    eps = protocol.select_episodes(args)
    if Path(args.out).exists():  # a killed run continues where it stopped (same as livecc.py)
        done = {json.loads(l)["episode_id"] for l in open(args.out) if l.strip()}
        eps = [e for e in eps if e["episode_id"] not in done]
        log.info(f"resume: {len(done)} episode(s) already in {args.out}")
    log.info(f"episodes to run: {len(eps)} -> {args.out}")
    if not eps:
        return

    wait_for_gpu(args.need_gb, log)
    torch.manual_seed(1234)
    try:
        import flash_attn  # noqa: F401

        attn = "flash_attention_2"  # official setting
    except ImportError:
        attn = "sdpa"  # repo supports sdpa too (QWEN2_5_VL_*ATTENTION_CLASSES)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.ckpt, min_pixels=args.pixels, max_pixels=args.pixels)
    # checkpoint generation_config has no sampling keys (greedy already); keep its repetition_penalty=1.05 as in
    # the official eval, only make greedy explicit
    model.generation_config.do_sample = False
    log.info(f"model loaded: {args.ckpt}; attn={attn}; generation_config={model.generation_config.to_diff_dict()}")

    task = protocol.TASKS[args.task]
    for ep in eps:
        request = protocol.fill(task["request"], ep)
        prompt = ovobench.build_prompt(
            task="CRR", question=None, options=None, _anno_={"question": request, "answer": None}, index=None
        )
        paths = protocol.episode_frames(
            ep, protocol.frames_root_for(ep, args.frames), burn_bbox=task.get("frame0_bbox", False)
        )
        frames = [
            (p if isinstance(p, Image.Image) else Image.open(p))
            .convert("RGB")
            .resize((protocol.FRAME_W, protocol.FRAME_H), Image.BICUBIC)
            for p in paths
        ]
        score_from = int(ep.get(args.score_from_field, 0)) if args.score_from_field else 0
        n = len(frames)
        t0 = time.time()
        trace, response_i = [], None
        for t in range(n):
            ts = time.time()
            reply = ovobench.get_response(prompt=prompt, video_path=frames[: t + 1], model=model, processor=processor)
            action = protocol.parse_action(reply, "state", ("Yes", "No"))
            dr = None
            try:
                with open(dr_path) as f:
                    dr = json.loads(f.readlines()[-1])
            except Exception:
                pass
            trace.append(
                {
                    "t": t,
                    "reply": reply,
                    "action": action,
                    "sec": round(time.time() - ts, 3),
                    "drop_ratio": round(dr["ratio"], 4) if dr else None,
                    "kept_tokens": (dr["total"] - dr["drop"]) if dr else None,
                }
            )
            if (
                action == "RESPOND" and response_i is None and t >= score_from
            ):  # a Yes before score_from is not a response
                response_i = t
                log.info(f"{ep['episode_id']} t={t}: RESPOND reply={reply!r}")
                if args.stop_at_first:
                    break
        v = protocol.verdict(ep, response_i, has_event=protocol.has_event(ep))
        rec = {
            "task": args.task,
            "episode_id": ep["episode_id"],
            "label": ep.get("label"),
            "scene_id": ep["scene_id"],
            "n_frames": n,
            "timely_start_i": ep["timely_start_i"],
            "timely_end_i": ep["timely_end_i"],
            "response_i": response_i,
            "verdict": v,
            "score_from": score_from,
            "n_yes": sum(s["action"] == "RESPOND" for s in trace),
            "n_invalid": sum(s["action"] is None for s in trace),
            "response_reply": trace[response_i]["reply"] if response_i is not None else None,
            "request": request,
            "prompt": prompt,
            "runtime_sec": round(time.time() - t0, 1),
            "config": {
                "ckpt": args.ckpt,
                "drop_method": args.drop_method,
                "drop_threshold": args.drop_threshold,
                "pixels": args.pixels,
                "frame_wh": [protocol.FRAME_W, protocol.FRAME_H],
                "decoding": "greedy (repetition_penalty 1.05 from checkpoint)",
                "max_new_tokens": 128,
                "attn": attn,
                "mode": "prefix polling every 1 s (OVO-Bench CRR)",
            },
            "trace": trace,
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info(
            f"{ep['episode_id']} ({ep['label']}) n={n} window=[{ep['timely_start_i']},{ep['timely_end_i']}] "
            f"response_i={response_i} verdict={v} yes={rec['n_yes']} invalid={rec['n_invalid']} "
            f"runtime={rec['runtime_sec']}s"
        )


if __name__ == "__main__":
    main()
