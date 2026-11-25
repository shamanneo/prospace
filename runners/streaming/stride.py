"""STRIDE-2B (HF interlive/STRIDE-2B) streaming runner for ProSpace.

Wraps the official streaming trigger detector, third_party/STRIDE/eval/model/qwen3_stride.py
(`STRIDEQwen3VL.from_pretrained_with_processor` -> `start_stream(query)` -> `detect(frames, timestamps)`),
the same entry point used by the official ET-Bench evaluation (eval/task/etbench/bench.py, scripts/etbench.sh).

How the official detector streams: the query + video are prefilled once into a KV cache ("mega window",
max_window_size frames); at every cursor step the cache is cropped to the frames seen so far (causal), and a
masked-diffusion activation head labels a sliding window of time slots <active>/<inactive>
(unmasking_steps iterations, retention of previous-step slot labels above confidence_threshold).  A trigger
is emitted when an active span is complete (its end is more than margin_seconds before the newest frame).

No repository code is modified in behaviour.  This file only subclasses the detector to record, per step, the slot
labels and the moment the first span completes (official trigger) and the first step with any active slot.

    response_i        = frame index of the step at which the first span completes (official trigger)
    first_active_i    = frame index of the first step at which any slot is labelled active (recorded only)

Run in the STRIDE repository's environment, with flash-attn.  We ran it with torch 2.5.1, flash-attn 2.7.4.post1 and
transformers 5.15.1, which needs patches/stride_eval.patch (the model passes the multimodal token types that
transformers 5.x uses for M-RoPE):
  CUDA_VISIBLE_DEVICES=0 python runners/streaming/stride.py --task object_finding --episodes data/prospace_bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "runners"))
sys.path.insert(0, str(ROOT / "third_party" / "STRIDE"))

import protocol as P  # noqa: E402  (torch-free)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from eval.model.qwen3_stride import STRIDEQwen3VL, STRIDETriggerArguments  # noqa: E402


class RecordingSTRIDE(STRIDEQwen3VL):
    """Official detector + per-step recording hooks (behaviour unchanged)."""

    def start_stream(self, query, is_multi_event=False):
        super().start_stream(query, is_multi_event)
        self.rec_steps = []
        self.rec_trigger = None  # dict for the first completed span
        self.rec_first_active = None  # dict for the first step with an active slot
        self._rec_last_i = None

    def _diffuse_shifted(self, cache, crop_len, last_frame_time, time_offset):
        out = super()._diffuse_shifted(cache, crop_len, last_frame_time, time_offset)
        slot_labels, starts, ends, unseen = out
        i = int(round(last_frame_time + time_offset))
        seen = ~unseen
        active = np.where((slot_labels == 1) & seen)[0]
        step = {
            "i": i,
            "active_slots": [[round(float(starts[k]), 2), round(float(ends[k]), 2)] for k in active],
            "seen_labels": "".join(str(int(v)) for v in slot_labels[seen]),
        }
        self.rec_steps.append(step)
        self._rec_last_i = i
        if len(active) and self.rec_first_active is None:
            self.rec_first_active = {"i": i, "slot_start_sec": step["active_slots"][0][0]}
        return out

    def _update_segments(self, spans, first_seen_start, last_abs_time):
        newly = super()._update_segments(spans, first_seen_start, last_abs_time)
        if newly and int(round(last_abs_time)) < getattr(self, "rec_score_from", 0):
            return newly  # trigger before the scored part (Re-finding show phase): not the response
        if newly and self.rec_trigger is None:
            seg = newly[0]
            self.rec_trigger = {
                "i": int(round(last_abs_time)),
                "start_sec": seg.start_sec,
                "end_sec": seg.end_sec,
                "finalized_at_end": False,
            }
        return newly


def load_frames(ep: dict, root: Path, burn_bbox: bool = False) -> np.ndarray:
    paths = P.episode_frames(ep, root, burn_bbox=burn_bbox)  # Re-finding: frame 0 carries the red box
    return np.stack(
        [
            np.asarray(
                (p if isinstance(p, Image.Image) else Image.open(p))
                .convert("RGB")
                .resize((P.FRAME_W, P.FRAME_H), Image.BICUBIC)
            )
            for p in paths
        ]
    )  # the boxed frame 0 comes back as an image, the rest as paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="interlive/STRIDE-2B")
    ap.add_argument("--task", default="object_finding", choices=sorted(P.TASKS))
    ap.add_argument(
        "--score_from_field",
        default=None,
        help="episode field: triggers before this frame index are ignored (Re-finding: gap_start_i)",
    )
    ap.add_argument("--episodes", default="data/prospace_bench.jsonl")
    ap.add_argument("--frames", default=P.DEFAULT_FRAMES)
    ap.add_argument("--out", default="results/stride/object_finding.jsonl")
    ap.add_argument("--episode_id", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=0)
    # official trigger arguments (scripts/etbench.sh), except chunk_size: 2 = one decision per temporal patch
    ap.add_argument("--chunk_size", type=int, default=2)
    ap.add_argument("--max_window_size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--unmasking_steps", type=int, default=8)
    ap.add_argument("--confidence_threshold", default="0.75")
    args = ap.parse_args()
    args.out = str(ROOT / args.out) if not os.path.isabs(args.out) else args.out
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    eps = P.select_episodes(args)
    print(f"[stride] {len(eps)} episodes -> {args.out}", flush=True)
    if not eps:
        return

    t0 = time.time()
    model = RecordingSTRIDE.from_pretrained_with_processor(args.model, device="cuda")
    model.trigger_args = STRIDETriggerArguments(
        chunk_size=args.chunk_size,
        max_window_size=args.max_window_size,
        stride=args.stride,
        unmasking_steps=args.unmasking_steps,
        confidence_threshold=args.confidence_threshold,
    )
    print(
        f"[stride] loaded in {time.time() - t0:.1f}s  trigger_args={model.trigger_args}  "
        f"window_past={model.config.trigger_window_past} future={model.config.trigger_window_future} "
        f"res={model.config.trigger_temporal_resolution}",
        flush=True,
    )

    for ep in eps:
        query = P.fill(P.TASKS[args.task]["request"], ep)
        frames = load_frames(
            ep, P.frames_root_for(ep, args.frames), burn_bbox=P.TASKS[args.task].get("frame0_bbox", False)
        )
        score_from = int(ep.get(args.score_from_field) or 0) if args.score_from_field else 0
        ts = np.arange(len(frames), dtype=np.float64) / P.SAMPLE_FPS
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        t1 = time.time()
        model.rec_score_from = score_from
        model.start_stream(query=query, is_multi_event=score_from > 0)  # a pre-gap trigger must not end the stream
        segments = model.detect(frames, ts)
        runtime = time.time() - t1

        trig = model.rec_trigger
        if trig is None and score_from == 0 and not any(s.get("fallback") for s in segments):
            # official end-of-video finalisation of an incomplete span (no in-stream trigger)
            s = segments[0]
            trig = {
                "i": len(frames) - 1,
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
                "finalized_at_end": True,
            }
        response_i = trig["i"] if trig else None
        first = model.rec_first_active
        rec = {
            "episode_id": ep["episode_id"],
            "label": ep.get("label"),
            "task": args.task,
            "n_frames": len(frames),
            "score_from": score_from,
            "timely_start_i": ep.get("timely_start_i"),
            "timely_end_i": ep.get("timely_end_i"),
            "response_i": response_i,
            "verdict": P.verdict(ep, response_i, has_event=P.has_event(ep)),
            "trigger": trig,
            "first_active_i": first["i"] if first else None,
            "first_active": first,
            "first_active_verdict": P.verdict(ep, first["i"] if first else None, P.has_event(ep)),
            "segments": segments,
            "query": query,
            "runtime_s": round(runtime, 2),
            "steps": model.rec_steps,
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"[stride] {ep['episode_id']} {ep.get('label')!r} n={len(frames)} timely=[{ep.get('timely_start_i')},"
            f"{ep.get('timely_end_i')}] response_i={response_i} ({rec['verdict']}) first_active_i="
            f"{rec['first_active_i']} segments={segments} {runtime:.1f}s",
            flush=True,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
