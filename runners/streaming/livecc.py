"""LiveCC-7B-Instruct streaming runner (ProSpace, Object Finding).

Wraps the official real-time commentary entrypoint `LiveCCDemoInfer.live_cc` from
third_party/livecc/demo/infer.py unchanged.  That function reads frames through a decord
VideoReader cached in `infer._cached_video_readers_with_hw[video_path]`; we pre-populate that cache
with an in-memory reader over the episode's 1 fps JPEGs (each frame exposed twice, so the model sees
its native 2 fps / 2-frames-per-second turn layout with correct Time=a-bs stamps), and drive
`state['video_timestamp']` exactly like demo/cli.py (t = 1, 2, 3, ...).

Turn layout (official): first turn = frames of 0-3 s (our frames 0,1,2) + the query; then one turn
per second (frame i, duplicated) with KV cache carried across turns.  Each turn the model generates
<=16 tokens; an output of just " ..." means "nothing to say" (training target for silence,
data/lmm_dataset.py).  response_i = index of the last frame in the first turn whose output, after
removing the trailing " ...", is non-empty.

--mode ovobench: the authors' own forward-active-responding evaluation,
third_party/livecc/evaluation/ovobench/{transfer_annotation_format.py,distributed_evaluate_ovobench.py}.  At every
second t the clip 0..t (1 fps JPEGs sampled to the model's 2 fps exactly like _read_may1fps_video_decord:
linspace(0, m-1, 2m).round()) goes through _spatial_resize_video; the user turn is [video][CRR question built around
the standing request]; the chat template ends with the answer prefix "The answer is:\n"; the prediction is the argmax
of the last-position logits over the official option list (No, Yes, 0-9, A-E; ids of "\n"+option).  First "Yes" =
response_i.  The whole trace is kept.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "third_party" / "livecc"
sys.path.insert(0, str(ROOT / "runners"))
from protocol import (
    FRAME_H,
    FRAME_W,
    TASKS,
    fill,
    verdict,
    frames_root_for,
    episode_frames,
    load_episodes,
    has_event as ep_has_event,
)  # noqa: E402

DEFAULT_EPISODES = ROOT / "out/epsiodes/object_finding_stream10.json"
DEFAULT_FRAMES = ROOT / "data/frames/scannetpp"
DEFAULT_OUT = ROOT / "out/runs/streaming/stream10/livecc.jsonl"


class _Batch:
    def __init__(self, a):
        self.a = a

    def asnumpy(self):
        return self.a


class FramesAs2fpsReader:
    """Stand-in for decord.VideoReader: frame k of a 2 fps stream = JPEG k//2 (1 fps)."""

    def __init__(self, paths):
        import numpy as np
        from PIL import Image

        self.frames = np.stack(
            [np.asarray(Image.open(p).convert("RGB").resize((FRAME_W, FRAME_H), Image.BICUBIC)) for p in paths]
        )

    def get_batch(self, idxs):
        import numpy as np

        return _Batch(self.frames[np.asarray(idxs, dtype=int) // 2])


def commentary(text: str) -> str:
    s = text.strip()
    if s.endswith("..."):
        s = s[:-3].strip()
    return s


def run_episode(infer, ep, frames_root, max_pixels, rp, max_turns=None):
    import torch
    from livecc_utils.video_process_patch import IMAGE_FACTOR, VIDEO_MIN_PIXELS, smart_resize

    paths = [frames_root / ep["scene_id"] / f"frame_{fid:06d}.jpg" for fid in ep["frame_ids"]]
    n = len(paths)
    if max_turns:
        n = min(n, max_turns + 2)
        paths = paths[:n]
    key = f"prospace://{ep['episode_id']}"
    h, w = smart_resize(FRAME_H, FRAME_W, factor=IMAGE_FACTOR, min_pixels=VIDEO_MIN_PIXELS, max_pixels=max_pixels)
    infer._cached_video_readers_with_hw = {key: (FramesAs2fpsReader(paths), h, w)}
    state = {"video_path": key, "video_pts": torch.arange(2 * n, dtype=torch.float64) / 2, "last_video_pts_index": -1}
    query = fill(TASKS["object_finding"]["request"], ep)

    steps, response_i, label_i = [], None, None
    t0 = time.time()
    t = 1
    while t <= n + 5:
        state["video_timestamp"] = t
        ts = time.time()
        for (start_t, stop_t), resp, state in infer.live_cc(
            message=query, state=state, max_pixels=max_pixels, do_sample=False, repetition_penalty=rp
        ):
            i = int(round(stop_t)) - 1
            c = commentary(resp)
            steps.append(
                {"i": i, "time": [round(start_t, 1), round(stop_t, 1)], "raw": resp, "sec": round(time.time() - ts, 3)}
            )
            if c and response_i is None:
                response_i = i
            if label_i is None and ep["label"].lower() in c.lower():
                label_i = i
            ts = time.time()
        if state.get("video_end"):
            break
        t += 1
    ctx = int(state["past_ids"].shape[1]) if state.get("past_ids") is not None else 0
    return dict(
        episode_id=ep["episode_id"],
        label=ep["label"],
        n_frames=len(ep["frame_ids"]),
        timely_start_i=ep["timely_start_i"],
        timely_end_i=ep["timely_end_i"],
        query=query,
        response_i=response_i,
        verdict=verdict(ep, response_i, has_event=ep_has_event(ep)),
        label_mention_i=label_i,
        label_mention_verdict=verdict(ep, label_i, has_event=ep_has_event(ep)),
        n_turns=len(steps),
        final_context_tokens=ctx,
        frame_hw=[h, w],
        steps=steps,
        runtime_sec=round(time.time() - t0, 2),
    )


OVO_OPTIONS = ["No", "Yes", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E"]
OVO_ANSWER_PREFIX = "The answer is:\n"


def ovo_crr_question(request: str) -> str:
    """transfer_annotation_format.Transfer.format_crr, verbatim (including its typos)."""
    return f"""You're responsible of answering questions based on the video content. The following question are relevant to the latest frames, i.e. the end of the video.\n\n{request}\n\nDecide whether existing visual content, especially latest frames, i.e frames that near the end of the video, provide enough information for answering the question.\nReturn "Yes" if existing visual content has provided enough information;\nReturn "No" otherwise."""


def run_episode_ovobench(
    model, processor, ep, frames_root, stop_at_first=False, task="object_finding", score_from_field=None
):
    import numpy as np
    import torch
    from PIL import Image
    from livecc_utils import _spatial_resize_video

    cfg = TASKS[task]
    paths = episode_frames(
        ep, Path(frames_root), burn_bbox=cfg.get("frame0_bbox", False)
    )  # Re-finding: red target box on frame 0
    frames = np.stack(
        [
            np.asarray(
                (p if isinstance(p, Image.Image) else Image.open(p))
                .convert("RGB")
                .resize((FRAME_W, FRAME_H), Image.BICUBIC)
            )
            for p in paths
        ]
    )
    n = len(paths)
    score_from = int(ep.get(score_from_field, 0)) if score_from_field else 0
    request = fill(cfg["request"], ep)
    question = ovo_crr_question(request)
    strict_ids = [processor.tokenizer(f"\n{o}").input_ids[-1] for o in OVO_OPTIONS]
    yes_k, no_k = OVO_OPTIONS.index("Yes"), OVO_OPTIONS.index("No")
    trace, response_i = [], None
    t0 = time.time()
    for t in range(n):
        m = t + 1
        idx = np.linspace(0, m - 1, 2 * m).round().astype(int)  # 1 fps -> the model's 2 fps (FRAME_FACTOR 2)
        video = _spatial_resize_video(torch.from_numpy(frames[idx]).permute(0, 3, 1, 2))
        conv = [{"role": "user", "content": [{"type": "video", "video": video}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) + OVO_ANSWER_PREFIX
        inputs = processor(text=[text], images=None, videos=[video], padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()
        sel = logits[strict_ids]
        pred = OVO_OPTIONS[int(sel.argmax())]
        p_yes = float(torch.softmax(sel[[no_k, yes_k]], dim=0)[1])
        trace.append({"t": t, "pred": pred, "p_yes": round(p_yes, 4)})
        if pred == "Yes" and response_i is None and t >= score_from:  # a Yes before score_from is not a response
            response_i = t
            if stop_at_first:  # the response is the first Yes either way; later seconds are not needed for scoring
                break
    return dict(
        episode_id=ep["episode_id"],
        label=ep.get("label"),
        n_frames=n,
        mode="ovobench",
        stopped_at_first=stop_at_first,
        task=task,
        score_from=score_from,
        timely_start_i=ep["timely_start_i"],
        timely_end_i=ep["timely_end_i"],
        question=question,
        response_i=response_i,
        verdict=verdict(ep, response_i, has_event=ep_has_event(ep)),
        trace=trace,
        runtime_sec=round(time.time() - t0, 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(DEFAULT_EPISODES))
    ap.add_argument("--frames", default=str(DEFAULT_FRAMES))
    ap.add_argument("--model", default="chenjoya/LiveCC-7B-Instruct")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_turns", type=int, default=0, help="debug: truncate stream")
    ap.add_argument("--repetition_penalty", type=float, default=1.05)  # demo/cli.py + inference.md value
    ap.add_argument("--max_pixels", type=int, default=384 * 28 * 28)  # live_cc default
    ap.add_argument("--min_free_gb", type=float, default=30)
    ap.add_argument(
        "--mode",
        default="commentary",
        choices=["commentary", "ovobench"],
        help="commentary: official real-time commentary (first utterance); ovobench: the authors' OVO-Bench "
        "forward-responding eval (prefix polling, option-logit argmax)",
    )
    ap.add_argument(
        "--stop_at_first",
        action="store_true",
        help="ovobench mode: stop polling an episode at its first Yes (bench runs)",
    )
    ap.add_argument("--task", default="object_finding", choices=sorted(TASKS))
    ap.add_argument(
        "--score_from_field",
        default=None,
        help="episode field; a Yes before that second is not a response (Re-finding: gap_start_i)",
    )
    args = ap.parse_args()
    args.out, args.episodes, args.frames = (str(Path(v).resolve()) for v in (args.out, args.episodes, args.frames))

    import torch

    while True:  # shared GPU: wait instead of crashing when memory is short
        free = torch.cuda.mem_get_info()[0] / 2**30
        if free >= args.min_free_gb:
            break
        print(f"GPU free {free:.1f} GB < {args.min_free_gb} GB, waiting 60 s", flush=True)
        time.sleep(60)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    if args.mode == "ovobench":
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        try:
            import flash_attn  # noqa: F401

            attn = "flash_attention_2"  # official eval setting
        except ImportError:
            attn = "sdpa"
        model = (
            Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype="auto", attn_implementation=attn)
            .to("cuda")
            .eval()
        )
        processor = AutoProcessor.from_pretrained(args.model, padding_side="left")
        print(f"ovobench mode: attn={attn}", flush=True)
    else:
        from demo.infer import LiveCCDemoInfer

        infer = LiveCCDemoInfer(model_path=args.model, device="cuda")

    eps = load_episodes(args.episodes, args.task)
    if args.limit:
        eps = eps[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["episode_id"] for l in open(out) if l.strip()} if out.exists() else set()
    for ep in eps:
        if ep["episode_id"] in done:
            continue
        if args.mode == "ovobench":
            rec = run_episode_ovobench(
                model,
                processor,
                ep,
                frames_root_for(ep, args.frames),
                args.stop_at_first,
                args.task,
                args.score_from_field,
            )
            rec["task"] = args.task
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                f"{rec['episode_id']} [{rec['label']}] n={rec['n_frames']} window={rec['timely_start_i']}-{rec['timely_end_i']} "
                f"response_i={rec['response_i']} -> {rec['verdict']} | {rec['runtime_sec']}s",
                flush=True,
            )
            continue
        rec = run_episode(
            infer, ep, frames_root_for(ep, args.frames), args.max_pixels, args.repetition_penalty, args.max_turns
        )
        rec["task"] = args.task
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        first = [s["raw"] for s in rec["steps"][:4]]
        print(
            f"{rec['episode_id']} [{rec['label']}] n={rec['n_frames']} window={rec['timely_start_i']}-"
            f"{rec['timely_end_i']} response_i={rec['response_i']} -> {rec['verdict']} | label_i={rec['label_mention_i']}"
            f" | ctx={rec['final_context_tokens']} {rec['runtime_sec']}s | first={first}",
            flush=True,
        )


if __name__ == "__main__":
    main()
