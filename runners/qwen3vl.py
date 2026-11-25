"""ProSpace runner for Qwen3-VL - one file for every task, only the task text differs.

Two protocols, same model call (512 x 384, 1 fps, [text][video] order, greedy):

  polling  (main, proactive)  At every polled second t the stream 0..t is given - never a future frame -
           together with a standing request, and the model answers WAIT or RESPOND.  The first RESPOND is
           the response time; it is scored on the frame index against the timely window.  P(RESPOND) from the
           first answer token is stored at every step (readiness curve).  Invalid output stops the episode
           and is recorded as such; it is never silently treated as WAIT.
  oneshot  (offline event localisation, kept for comparison)  The whole episode is given once and the model
           answers the second; first number floored, same window scoring.

Task table: episode file, how the request is filled from the episode (label / direction / x-y), and any input
tweak (Re-finding burns the red frame-0 box in).  Add a task = add a row.

Odd prefix lengths: qwen_vl_utils pads an odd frame list by repeating the last frame and gives the copy the
next frame index; the metadata is corrected here so the copy keeps the last *seen* second and
total_num_frames = frames seen.

    CUDA_VISIBLE_DEVICES=0 python runners/qwen3vl.py --task object_finding --mode polling --step 1 \\
        --episodes data/prospace_bench/prospace_bench.jsonl --frames-by-dataset --all --shard 0/4 \\
        --out results/qwen3vl_8b/object_finding.s0.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


# --- protocol (shared with every other runner) -----------------------------------------------
from protocol import (
    ROOT,
    FRAME_W,
    FRAME_H,
    SAMPLE_FPS,
    DEFAULT_FRAMES,
    DATASET_FRAMES,  # noqa: F401,E402
    POLL_PROMPT,
    STATE_PROMPT,
    ONESHOT_PREFIX,
    ONESHOT_SUFFIX,
    DIRECTION,
    TASKS,
    fill,
    episode_frames,
    select_episodes,
    verdict,
    parse_action,
    parse_seconds,
)


# --- model ----------------------------------------------------------------------------------
def load(model_path: str, adapter: str | None = None):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map="auto").eval()
    if adapter:  # LoRA from training/train_polling.py, merged so inference is unchanged
        import sys
        from peft import PeftModel
        import peft.tuners.lora.model  # noqa: F401

        for mod in list(sys.modules.values()):  # shared env: gptqmodel + old optimum makes peft's check raise
            if getattr(mod, "__name__", "").startswith("peft") and hasattr(mod, "is_gptqmodel_available"):
                mod.is_gptqmodel_available = lambda *a, **k: False
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    return model, processor


# --frame-cache: memoise the per-frame load+resize qwen_vl_utils does for a frame list (fetch_image), so a frame is
# decoded once per episode instead of once per polled second.  fetch_video then re-runs fetch_image on the cached
# PIL image, which is a same-size copy, so the model input is unchanged.  Cleared at every episode start.
FRAME_CACHE: dict | None = None
FRAMES_FIRST = False


def _cached_frame(f):
    from qwen_vl_utils.vision_process import fetch_image, SPATIAL_MERGE_SIZE

    key = f if isinstance(f, str) else id(f)
    if key not in FRAME_CACHE:
        FRAME_CACHE[key] = fetch_image(
            {"image": f, "resized_height": FRAME_H, "resized_width": FRAME_W, "sample_fps": SAMPLE_FPS},
            16 * SPATIAL_MERGE_SIZE,
        )
    return FRAME_CACHE[key]


def build_inputs(processor, frames, text: str, indices: list[int] | None = None):
    """[text][video] message -> processor inputs; pads odd prefixes without inventing a future second.
    `indices` gives each frame's real second when the clip is not simply frames 0..n-1 (anchored polling)."""
    n_seen = len(frames)
    indices = list(range(n_seen)) if indices is None else list(indices)
    assert len(indices) == n_seen
    if FRAME_CACHE is not None:
        frames = [_cached_frame(f) for f in frames]
    video = list(frames) if n_seen >= 2 else list(frames) * 2  # the video path needs >= 2 frames
    parts = [
        {"type": "text", "text": text},
        {"video": video, "resized_height": FRAME_H, "resized_width": FRAME_W, "sample_fps": SAMPLE_FPS},
    ]
    if FRAMES_FIRST:  # [video][question] (the Qwen cookbook order) instead of the default [question][video]
        parts = parts[::-1]
    messages = [{"role": "user", "content": parts}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True
    )
    video_inputs, metas = map(list, zip(*video_inputs))
    n_enc = int(video_inputs[0].shape[0])
    m = dict(metas[0])
    m["fps"] = SAMPLE_FPS
    m["frames_indices"] = [indices[min(j, n_seen - 1)] for j in range(n_enc)]  # padded copies keep the last seen second
    m["total_num_frames"] = indices[-1] + 1
    metas = [m]
    inputs = processor(
        text=[chat],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=metas,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt",
    ).to(model_device())
    return inputs


_DEVICE = "cuda"


def model_device():
    return _DEVICE


@torch.no_grad()
def generate(model, processor, inputs, max_new_tokens: int, first_token_choices: list[int] | None = None):
    """Greedy generation; optionally the softmax over the given first-token ids (restricted) is returned."""
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        output_scores=first_token_choices is not None,
        return_dict_in_generate=True,
    )
    gen = out.sequences[0, inputs.input_ids.shape[1] :]
    text = processor.batch_decode(gen[None], skip_special_tokens=True)[0]
    probs = None
    if first_token_choices is not None:
        logits = out.scores[0][0].float()
        sub = torch.softmax(logits[first_token_choices], dim=-1).tolist()
        full = torch.softmax(logits, dim=-1)[first_token_choices].tolist()
        probs = {"restricted": sub, "full": full}
    return text, probs


# --- episodes -------------------------------------------------------------------------------


# --- protocols ------------------------------------------------------------------------------
def run_polling(
    model,
    processor,
    frames,
    request: str,
    step: int,
    choice_ids,
    style: str = "alert",
    question: str | None = None,
    single_frame: bool = False,
    stop_at_first: bool = True,
    start: int = 0,
    words: tuple[str, str] = ("Yes", "No"),
    anchor: int = 0,
    recent: int = 0,
    score_from: int = 0,
) -> dict:
    """style="alert": WAIT/RESPOND wrapper with the standing request.  style="state": current-state Yes/No question
    (first Yes = response).  single_frame=True gives only frame t instead of the prefix 0..t (recognition control)."""
    trace, response_i, invalid = [], None, None
    n = len(frames)
    for t in range(start, n, step):
        if style == "alert":
            text = POLL_PROMPT.format(request=request, t=t)
        else:
            text = STATE_PROMPT.format(question=question, t=t, pos=words[0], neg=words[1])
        idx = None
        if single_frame:
            clip = [frames[t]]
            idx = [t]
        elif (
            anchor and recent
        ):  # anchored polling: the first `anchor` frames plus the last `recent` seconds, never beyond t
            idx = sorted(set(range(0, min(anchor, t + 1))) | set(range(max(0, t - recent + 1), t + 1)))
            clip = [frames[i] for i in idx]
        else:
            clip = frames[: t + 1]
        inputs = build_inputs(processor, clip, text, idx)
        reply, probs = generate(model, processor, inputs, 4, choice_ids)
        action = parse_action(reply, style, words)
        trace.append(
            {
                "t": t,
                "action": action,
                "p_respond": round(probs["restricted"][0], 4),
                "p_respond_full": round(probs["full"][0], 4),
                "p_wait_full": round(probs["full"][1], 4),
                "reply": reply,
                "tokens": int(inputs.input_ids.shape[1]),
            }
        )
        if action is None:
            invalid = {"t": t, "reply": reply}
            break
        if action == "RESPOND" and response_i is None and t >= score_from:  # a Yes before score_from is not a response
            response_i = t
            if stop_at_first:
                break
    return {"response_i": response_i, "invalid": invalid, "n_calls": len(trace), "trace": trace}


def run_oneshot(model, processor, frames, question: str) -> dict:
    inputs = build_inputs(processor, frames, ONESHOT_PREFIX + question + ONESHOT_SUFFIX)
    reply, _ = generate(model, processor, inputs, 64)
    s = parse_seconds(reply)
    return {"response_i": int(s) if s is not None else None, "reply": reply, "tokens": int(inputs.input_ids.shape[1])}


# --- main -----------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--mode", default="polling", choices=["polling", "oneshot"])
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument(
        "--adapter", default=None, help="LoRA adapter dir (out/training/runs/<run>/adapter_stepN), merged at load"
    )
    ap.add_argument("--episodes", default=None, help="defaults to the task's episode file")
    ap.add_argument("--frames", default=DEFAULT_FRAMES)
    ap.add_argument(
        "--frames-by-dataset",
        action="store_true",
        help="mixed-dataset episode files: pick the frame root from each episode's `dataset` field",
    )
    ap.add_argument("--episode-id", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--step", type=int, default=1, help="polling interval in seconds")
    ap.add_argument(
        "--style",
        default="alert",
        choices=["alert", "state"],
        help="alert: WAIT/RESPOND standing-request wrapper; state: current-state Yes/No question",
    )
    ap.add_argument("--single-frame", action="store_true", help="give only frame t (no prefix) - recognition control")
    ap.add_argument(
        "--start-field",
        default=None,
        help="episode field giving the first second to poll (e.g. show_len for Re-finding: skip the initial showing); "
        "default 0",
    )
    ap.add_argument(
        "--score-from-field",
        default=None,
        help="episode field giving the first second whose answer is scored (e.g. show_len for Re-finding); polling "
        "still starts at 0 and earlier Yes answers are recorded but do not end the episode; default 0",
    )
    ap.add_argument(
        "--frames-first", action="store_true", help="put the video before the question (default: question first)"
    )
    ap.add_argument(
        "--no-frame-cache",
        dest="frame_cache",
        action="store_false",
        help="turn off the per-episode frame cache (default on: each frame is decoded+resized once per "
        "episode; model input verified identical, A/B: P(Yes) diff 0 over 2,293 calls)",
    )
    ap.add_argument(
        "--oneshot-override", default=None, help="oneshot question template override (fill() fields allowed)"
    )
    ap.add_argument(
        "--frame0-crop",
        action="store_true",
        help="ablation: frame 0 = target crop only (scene context removed), Re-finding",
    )
    ap.add_argument(
        "--no-stop",
        action="store_true",
        help="keep polling after the first Yes (full P(Yes) curve; response = first Yes)",
    )
    ap.add_argument(
        "--anchor",
        type=int,
        default=0,
        help="anchored polling: keep only the first N frames as the start-of-episode anchor",
    )
    ap.add_argument(
        "--recent",
        type=int,
        default=0,
        help="anchored polling: plus the last M frames as the current context (needs --anchor)",
    )
    ap.add_argument(
        "--answer-words",
        default=None,
        help="comma-separated trigger,wait answer words for --style state (default Yes,No); "
        "e.g. Different,Same asks the model to pick between two words instead of Yes/No",
    )
    ap.add_argument(
        "--state-override",
        default=None,
        help="replace the task's state question (template with {label} {x} {y} {direction}) for prompt ablations",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip episodes already written to --out (a killed run restarts where it stopped)",
    )
    args = ap.parse_args()
    cfg = TASKS[args.task]
    args.episodes = args.episodes or cfg["episodes"]
    eps = select_episodes(args)
    if args.resume and args.out and Path(args.out).exists():
        done = {json.loads(l)["episode_id"] for l in open(args.out) if l.strip()}
        eps = [e for e in eps if e["episode_id"] not in done]
        print(f"resume: {len(done)} episode(s) already in {args.out}, {len(eps)} left", flush=True)
    print(
        f"{args.task} {args.mode}: {len(eps)} episode(s) shard {args.shard}  {FRAME_W}x{FRAME_H} @ {SAMPLE_FPS} fps  "
        f"step {args.step}  style {args.style}{' single-frame' if args.single_frame else ''}  model {args.model}\n"
        f"text: {(cfg['request'] if args.style == 'alert' else cfg['state']) if args.mode == 'polling' else cfg['oneshot']}\n",
        flush=True,
    )
    model, processor = load(args.model, args.adapter)
    tok = processor.tokenizer
    pos, neg = ("RESPOND", "WAIT") if args.style == "alert" else ("Yes", "No")
    if args.answer_words and args.style == "state":
        pos, neg = (w.strip() for w in args.answer_words.split(","))
    choice_ids = [tok.encode(pos, add_special_tokens=False)[0], tok.encode(neg, add_special_tokens=False)[0]]

    counts: dict[str, int] = {}
    for k, ep in enumerate(eps, 1):
        has_event = bool(ep.get("has_event", ep.get("timely_start_i") is not None))  # absence episodes have no window
        t0 = time.time()
        rec = {
            "model": args.model,
            "task": args.task,
            "mode": args.mode,
            "episode_id": ep["episode_id"],
            "label": ep.get("label"),
            "frames": len(ep["frame_ids"]),
            "size": [FRAME_W, FRAME_H],
            "step": args.step,
            "has_event": has_event,
            "timely": [ep["timely_start_i"], ep["timely_end_i"]] if has_event else None,
        }
        rec["step"] = args.step
        rec["poll_start"] = int(ep.get(args.start_field, 0)) if args.start_field else 0
        rec["score_from"] = int(ep.get(args.score_from_field, 0)) if args.score_from_field else 0
        try:
            root = Path(DATASET_FRAMES[ep["dataset"]]) if args.frames_by_dataset else Path(args.frames)
            frames = episode_frames(ep, root, cfg.get("frame0_bbox", False), frame0_crop=args.frame0_crop)
            rec["frame0_crop"] = bool(args.frame0_crop)
            global FRAME_CACHE, FRAMES_FIRST
            FRAME_CACHE = {} if args.frame_cache else None
            FRAMES_FIRST = args.frames_first
            rec["order"] = "frames_first" if args.frames_first else "question_first"
            if args.mode == "polling":
                request = fill(cfg["request"], ep)
                question = fill(args.state_override or cfg["state"], ep)
                res = run_polling(
                    model,
                    processor,
                    frames,
                    request,
                    args.step,
                    choice_ids,
                    args.style,
                    question,
                    args.single_frame,
                    stop_at_first=not args.no_stop,
                    start=int(ep.get(args.start_field, 0)) if args.start_field else 0,
                    words=(pos, neg),
                    anchor=args.anchor,
                    recent=args.recent,
                    score_from=int(ep.get(args.score_from_field, 0)) if args.score_from_field else 0,
                )
                rec.update(
                    request=request if args.style == "alert" else None,
                    question=question if args.style == "state" else None,
                    style=args.style,
                    single_frame=args.single_frame,
                    answer_words=[pos, neg],
                    anchor=args.anchor,
                    recent=args.recent,
                    **res,
                )
                v = "invalid" if res["invalid"] else verdict(ep, res["response_i"], has_event)
            else:
                question = fill(args.oneshot_override or cfg["oneshot"], ep)
                res = run_oneshot(model, processor, frames, question)
                rec.update(question=question, **res)
                v = verdict(ep, res["response_i"], has_event)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            v = "error"
            rec["error"] = f"OOM: {e}"[:200]
        except Exception as e:
            v = "error"
            rec["error"] = traceback.format_exc()[-400:]
        rec["verdict"] = v
        rec["wall_s"] = round(time.time() - t0, 1)
        counts[v] = counts.get(v, 0) + 1
        extra = f"{rec.get('n_calls', 1)} calls" if args.mode == "polling" else repr(rec.get("reply", ""))[:20]
        print(
            f"[{k}/{len(eps)}] {ep['episode_id']:26s} {str(rec['label'])[:14]:14s} {rec['frames']:3d}f  timely {rec['timely']}"
            f" -> response {rec.get('response_i')}  {v:18s} ({extra}, {rec['wall_s']:.0f}s)",
            flush=True,
        )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = sum(counts.values())
    if n:
        correct = counts.get("timely", 0) + counts.get("correct_abstention", 0)
        print(
            "\nRESULT "
            + "  ".join(f"{k} {v} ({100*v/n:.0f}%)" for k, v in sorted(counts.items()))
            + f"  | correct {correct}/{n} ({100*correct/n:.0f}%)"
            + (f"  -> {args.out}" if args.out else "")
        )


if __name__ == "__main__":
    main()
