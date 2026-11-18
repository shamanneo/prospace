#!/usr/bin/env python3
"""Dispider (CVPR 2025, HF Mar2Ding/Dispider) streaming runner for ProSpace.

Wraps the official streaming entry point `third_party/Dispider/stream_inference.py:run` unchanged:
  - backend  = ReleasedPerceptionDecisionReactionBackend.from_pretrained (loaded once, reset per episode)
  - adapter  = DispiderStreamingAdapter (Decision head, native threshold 0.0, Decision KV cache 'auto')
  - session  = StreamingSession (16-second windows x 16 frames, history 8)
Only the video source is replaced: instead of decord reading an mp4, the 1 fps episode frames are pushed
one per second (timestamp = frame index i in seconds) through the official `drive_frame_source`.
The question is fixed at session creation, i.e. before the first frame.

Response time = frame index of the first Dispider answer event (event.timestamp_s = timestamp of the last
frame of the window whose Decision score crossed the threshold).  Scoring: runners/protocol.py:verdict.

Run in the Dispider repository's own environment:
  CUDA_VISIBLE_DEVICES=0 python runners/streaming/dispider.py [--limit 1] [--frames ROOT]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "third_party" / "Dispider"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "runners"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import protocol  # noqa: E402

TASK = "object_finding"


def build_args(model: str, prompt: str) -> argparse.Namespace:
    """Official stream_inference.py CLI defaults, with our prompt and a placeholder video."""
    import stream_inference

    return stream_inference.build_parser().parse_args(["<frames>", "--model", model, "--prompt", prompt])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="out/epsiodes/object_finding_stream10.json")
    ap.add_argument("--frames", default=protocol.DEFAULT_FRAMES)
    ap.add_argument("--model", default="Mar2Ding/Dispider")
    ap.add_argument("--out", default="out/runs/streaming/stream10/dispider.jsonl")
    ap.add_argument("--episode_id", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--min_free_gib", type=float, default=30.0)
    ap.add_argument("--task", default=TASK, help="ProSpace task whose request text is asked (any of the nine tasks)")
    ap.add_argument(
        "--score_from_field",
        default=None,
        help="episode field; an answer event before that second is not a response (Re-finding: gap_start_i). "
        "The stream always runs to the end, so the first event at or after it is taken, as the polling runners do",
    )
    ap.add_argument(
        "--sliding",
        action="store_true",
        help="a decision every second: the official session is run 16 times with the window grid shifted by 0..15 s "
        "(timeline origin a-16), so some session closes a 16 s window at every second; response = earliest event. "
        "Each session is the unmodified official path (non-overlapping 16 s windows, history 8).",
    )
    a = ap.parse_args()
    a.out = str(ROOT / a.out)

    import torch
    import stream_inference
    import dataclasses
    from dispider.streaming import (
        DispiderStreamingAdapter,
        FrameChunk,
        ReleasedPerceptionDecisionReactionBackend,
        drive_frame_source,
    )
    from dispider.streaming.session import StreamingSession

    episodes = protocol.select_episodes(a)
    print(f"[dispider] {len(episodes)} episodes -> {a.out}", flush=True)
    if not episodes:
        return

    # --- shared GPU: wait until enough memory is free before loading (fp16 weights ~18.2 GiB) ------------
    need = a.min_free_gib * 2**30
    while torch.cuda.mem_get_info()[0] < need:
        print(f"[dispider] waiting for GPU memory: free {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB", flush=True)
        time.sleep(60)

    # --- load once through the official loader, with the official CLI kwargs ---------------------------
    t0 = time.time()
    proto_args = build_args(a.model, "placeholder")
    runtime = ReleasedPerceptionDecisionReactionBackend.from_pretrained(
        proto_args.model,
        device=proto_args.device,
        device_map=None if proto_args.device_map.lower() == "none" else proto_args.device_map,
        generation_kwargs={"max_new_tokens": proto_args.max_new_tokens},
        cache_oracle_atol=proto_args.cache_oracle_atol,
        cache_fallback_margin=proto_args.cache_fallback_margin,
    )
    print(
        f"[dispider] loaded in {time.time() - t0:.1f}s, cuda mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB",
        flush=True,
    )

    # Observation-only hook: keep the raw Reaction text (special tokens kept).  The official runtime decodes with
    # skip_special_tokens=True, so a Reaction of "<keep_silent><|im_end|>" becomes "" and no event is emitted.
    reaction_raw = []
    _generate = runtime.reaction_model.generate

    def generate_logged(input_ids, **kw):
        out = _generate(input_ids, **kw)
        seq = getattr(out, "sequences", out)
        reaction_raw.append(runtime.tokenizer.decode(seq[0], skip_special_tokens=False))
        return out

    runtime.reaction_model.generate = generate_logged

    # Shifted grids start before 0 s: the first window of phase a is [a-16, a) and holds frames 0..a-1 (the official
    # nearest-sample rule repeats them to 16).  Only the clip's time sentence is clamped, so it reads "0 to a seconds".
    _perceive_clip = runtime.perceive_clip

    def perceive_clip_clamped(window):
        return _perceive_clip(dataclasses.replace(window, start_s=0.0) if window.start_s < 0 else window)

    runtime.perceive_clip = perceive_clip_clamped

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        prompt = protocol.fill(protocol.TASKS[a.task]["request"], ep)
        paths = protocol.episode_frames(
            ep, protocol.frames_root_for(ep, a.frames), burn_bbox=protocol.TASKS[a.task].get("frame0_bbox", False)
        )
        score_from = int(ep.get(a.score_from_field, 0)) if a.score_from_field else 0
        args = build_args(a.model, prompt)
        L = len(paths)
        frames_np = [
            np.asarray(
                (Image.open(p) if isinstance(p, str) else p)
                .convert("RGB")
                .resize((protocol.FRAME_W, protocol.FRAME_H), Image.BICUBIC)
            )
            for p in paths
        ]
        reaction_raw.clear()
        t_ep = time.time()
        torch.cuda.reset_peak_memory_stats()
        events, decisions, pushed_at, stderr_all = [], [], [], []
        best = None  # earliest scored response over the phases run so far

        for phase in range(16) if a.sliding else [0]:
            captured, cur, n_before = {}, {"i": -1}, len(reaction_raw)
            origin = float(phase - 16) if phase else 0.0

            def loader(*_a, **_k):
                runtime.reset()  # drop the previous stream's Decision KV cache
                return runtime

            def adapter_type(*ad_a, **ad_k):
                ad = DispiderStreamingAdapter(*ad_a, **ad_k)
                if phase:  # same official session, window grid shifted by `phase` seconds
                    ad.new_session = lambda **kw: StreamingSession(ad, ad, ad, timeline_origin_s=origin, **kw)
                captured["adapter"] = ad
                return ad

            def window_end(i):  # last frame index of the window that frame i falls in
                return int(origin) + ((i - int(origin)) // 16) * 16 + 15

            def streamer(session, _video, chunk_frames=1):
                """1 fps live source: frame i arrives at t = i seconds; official drive_frame_source feeds it."""

                def chunks():
                    for i in range(L):
                        # sliding: a window that can only close at or after the best response so far cannot improve it
                        if a.sliding and best is not None and window_end(i) >= best:
                            if i > 0 and window_end(i - 1) != window_end(i) and window_end(i - 1) < best:
                                cur["i"] = i  # this frame only closes the previous window, which can still beat `best`
                                yield FrameChunk(frames=(frames_np[i],), timestamps_s=(float(i),))
                            return
                        cur["i"] = i
                        yield FrameChunk(frames=(frames_np[i],), timestamps_s=(float(i),))

                # official flush closes the trailing partial window; in sliding mode only the phase whose window ends on
                # the last frame is flushed (the other phases' trailing windows are covered by that phase)
                fl = (not a.sliding) or (window_end(L - 1) == L - 1 and (best is None or L - 1 < best))
                for ev in drive_frame_source(session, chunks(), flush=fl):
                    pushed_at.append(
                        cur["i"]
                    )  # frame whose arrival closed the triggering window (last+1, or last at flush)
                    yield ev
                    if a.sliding and int(round(ev.timestamp_s)) >= score_from:
                        return  # this phase has its first scored response

            out_buf, err_buf = io.StringIO(), io.StringIO()
            stream_inference.run(
                args,
                backend_loader=loader,
                adapter_type=adapter_type,
                streamer=streamer,
                stdout=out_buf,
                stderr=err_buf,
            )
            ev_p = [json.loads(l) for l in out_buf.getvalue().splitlines() if l.strip()]
            dec_p = [
                {
                    "phase": phase,
                    "clip_position": d.clip_position,
                    "window_last_i": int(round(d.timestamp_s)),
                    "score": round(d.score, 4),
                    "triggered": d.triggered,
                    "kv_cache_used": d.kv_cache_used,
                    "cached_score": None if d.cached_score is None else round(d.cached_score, 4),
                    "oracle_score": None if d.oracle_score is None else round(d.oracle_score, 4),
                    "used_oracle_for_decision": d.used_oracle_for_decision,
                }
                for d in captured["adapter"].decisions
            ]
            for d, raw in zip(
                [d for d in dec_p if d["triggered"]], reaction_raw[n_before:]
            ):  # Reaction runs once per trigger
                d["reaction_raw"] = raw
            for e in ev_p:
                e["phase"] = phase
            events += ev_p
            decisions += dec_p
            stderr_all.append(err_buf.getvalue().strip())
            sc = [int(round(e["timestamp_s"])) for e in ev_p if int(round(e["timestamp_s"])) >= score_from]
            if sc and (best is None or sc[0] < best):
                best = sc[0]
        runtime_s = time.time() - t_ep

        events.sort(key=lambda e: e["timestamp_s"])
        decisions.sort(key=lambda d: (d["window_last_i"], d["phase"]))
        scored = [e for e in events if int(round(e["timestamp_s"])) >= score_from]
        response_i = int(round(scored[0]["timestamp_s"])) if scored else None
        trig = [d["window_last_i"] for d in decisions if d["triggered"]]
        err_buf = io.StringIO(stderr_all[0] if stderr_all else "")
        rec = {
            "episode_id": ep["episode_id"],
            "scene_id": ep["scene_id"],
            "label": ep.get("label"),
            "n_frames": len(paths),
            "timely_start_i": ep["timely_start_i"],
            "timely_end_i": ep["timely_end_i"],
            "response_i": response_i,
            "verdict": protocol.verdict(ep, response_i, has_event=protocol.has_event(ep)),
            "response_text": scored[0]["answer"] if scored else None,
            "task": a.task,
            "score_from": score_from,
            # supplementary (not the scored response): first Decision-head trigger, even if Reaction kept silent
            "decision_trigger_i": trig[0] if trig else None,
            "prompt": prompt,
            "events": events,
            "event_emitted_after_push_i": pushed_at,
            "decisions": decisions,
            "stderr": err_buf.getvalue().strip(),
            "config": {
                "sliding": bool(a.sliding),
                "window_seconds": args.window_seconds,
                "frames_per_window": args.frames_per_window,
                "max_history_windows": args.max_history_windows,
                "decision_kv_cache": args.decision_kv_cache,
                "threshold": runtime.native_decision_threshold,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "frame_size": [protocol.FRAME_W, protocol.FRAME_H],
            },
            "runtime_s": round(runtime_s, 2),
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        with open(a.out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"[dispider] {ep['episode_id']} {ep.get('label')!r} n={len(paths)} timely=[{ep['timely_start_i']},"
            f"{ep['timely_end_i']}] resp={response_i} -> {rec['verdict']} | scores="
            f"{[(d['window_last_i'], d['score']) for d in decisions][:40]} | answer={rec['response_text']!r} "
            f"| {runtime_s:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
