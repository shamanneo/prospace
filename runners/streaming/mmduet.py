"""MMDuet (v1, wangyueqian/MMDuet LoRA on lmms-lab/llava-onevision-qwen2-7b-ov) streaming runner for ProSpace.

Wraps the official streaming inference loop `test.inference.LiveInferForBenchmark.inference()` from
third_party/MMDuet unchanged: the query is queued at time 0, frames are fed one per step into the LLM KV cache
(49 tokens per frame), and after every frame the informative / relevance heads give a score; when the score sum
exceeds the threshold the model generates a free-form reply.  response_i = index of the frame after which the
first reply was generated.

Official settings used (scripts/inference/magqa.sh, the repo's query-conditioned "answer at the right time" task):
  --stream_end_prob_threshold 0.5 --score_heads informative_score,relevance_score --remove_assistant_turns true
With remove_assistant_turns the KV cache never contains replies, so the per-frame score trace does not depend on
the threshold; the demo default (informative_score > 0.3, demo/app.py) is also reported post hoc from the trace.

Frames: protocol 512x384 resize, then the official loader's letterbox (long side 384, black pad to 384x384,
test/datasets.py load_video) and the model's image processor.

Run in the MMDuet repository's own environment (we used Python 3.10, torch 2.1.2, transformers 4.44.2, peft 0.12.0),
with HF_HOME holding lmms-lab/llava-onevision-qwen2-7b-ov, google/siglip-so400m-patch14-384 and wangyueqian/MMDuet:
  CUDA_VISIBLE_DEVICES=0 python runners/streaming/mmduet.py --task object_finding --episodes data/prospace_bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "third_party" / "MMDuet"
sys.path.insert(0, str(ROOT / "runners"))
sys.path.insert(0, str(REPO))  # repo's `test` package must shadow the stdlib one

import protocol  # noqa: E402

DEMO_THRESHOLD = 0.3  # demo/app.py default: single-frame informative_score


def load_frames(ep: dict, frames_root: Path, res: int = 384, burn_bbox: bool = False):
    import cv2
    import numpy as np
    import torch

    out = []
    # Re-finding: frame 0 carries the red box
    for k, p in enumerate(protocol.episode_frames(ep, frames_root, burn_bbox=burn_bbox)):
        img = (
            cv2.imread(str(p)) if isinstance(p, str) else cv2.cvtColor(np.asarray(p.convert("RGB")), cv2.COLOR_RGB2BGR)
        )
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.resize(img, (protocol.FRAME_W, protocol.FRAME_H), interpolation=cv2.INTER_AREA)
        # official letterbox (test/datasets.py FastAndAccurateStreamingVideoQADataset.load_video)
        h, w = img.shape[:2]
        if w > h:
            nw, nh = res, int((h / w) * res)
        else:
            nh, nw = res, int((w / h) * res)
        r = cv2.resize(img, (nw, nh))
        canvas = cv2.copyMakeBorder(
            r,
            top=(res - nh) // 2,
            bottom=(res - nh + 1) // 2,
            left=(res - nw) // 2,
            right=(res - nw + 1) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        out.append(np.transpose(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (2, 0, 1)))
    return torch.tensor(np.stack(out))


def build(args):
    from models.arguments_live import LiveTestArguments
    from test.inference import LiveInferForBenchmark

    targs = LiveTestArguments(
        output_dir=str(Path(args.out).parent / "_mmduet_tmp"),
        llm_pretrained="lmms-lab/llava-onevision-qwen2-7b-ov",
        lora_pretrained="wangyueqian/MMDuet",
        bf16=True,
        attn_implementation=args.attn,
        frame_fps=protocol.SAMPLE_FPS,
        max_num_frames=400,
        stream_end_prob_threshold=args.threshold,
        score_heads=args.score_heads,
        remove_assistant_turns=True,
        report_to="none",
    )
    return LiveInferForBenchmark(targs), targs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="data/prospace_bench.jsonl")
    ap.add_argument("--task", default="object_finding")
    ap.add_argument("--out", default=str(ROOT / "results/mmduet/object_finding.jsonl"))
    ap.add_argument("--frames", default=protocol.DEFAULT_FRAMES)
    ap.add_argument("--episode_id", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--score_heads", default="informative_score,relevance_score")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument(
        "--score_from_field",
        default=None,
        help="episode field; a reply before that second is not a response (Re-finding: gap_start_i). Replies never "
        "enter the KV cache (remove_assistant_turns), so taking the first reply at or after it is exact",
    )
    args = ap.parse_args()

    import torch

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    eps = protocol.select_episodes(args)
    print(f"{len(eps)} episodes to run", flush=True)
    if not eps:
        return
    infer, targs = build(args)
    print(
        f"threshold={infer.stream_end_prob_threshold} heads={infer.score_heads} "
        f"rm_ass_turns={infer.remove_assistant_turns} attn={args.attn} system_prompt={infer.system_prompt!r}",
        flush=True,
    )

    for ep in eps:
        t0 = time.time()
        query = protocol.fill(protocol.TASKS[args.task]["request"], ep)
        frames = load_frames(
            ep,
            protocol.frames_root_for(ep, args.frames),
            burn_bbox=protocol.TASKS[args.task].get("frame0_bbox", False),
        )
        score_from = int(ep.get(args.score_from_field, 0)) if args.score_from_field else 0
        infer.reset()
        infer.set_fps(fps=protocol.SAMPLE_FPS)
        infer.input_video_stream(frames)
        infer.input_query_stream([{"role": "user", "time": 0.0, "content": query}])
        with torch.no_grad():
            turns = infer.inference()
        replies = [
            {"i": int(round(t["time"] * protocol.SAMPLE_FPS)), "text": t["content"]}
            for t in turns
            if t["role"] == "assistant"
        ]
        scored = [r for r in replies if r["i"] >= score_from]
        response_i = scored[0]["i"] if scored else None
        scores = [
            {"i": k, "informative": round(d["informative_score"], 4), "relevance": round(d["relevance_score"], 4)}
            for k, d in enumerate(infer.debug_data_list)
        ]
        demo_i = next((s["i"] for s in scores if s["informative"] > DEMO_THRESHOLD), None)
        rec = dict(
            episode_id=ep["episode_id"],
            label=ep.get("label"),
            n_frames=len(ep["frame_ids"]),
            timely_start_i=ep["timely_start_i"],
            timely_end_i=ep["timely_end_i"],
            response_i=response_i,
            verdict=protocol.verdict(ep, response_i, has_event=protocol.has_event(ep)),
            response_text=scored[0]["text"] if scored else None,
            task=args.task,
            score_from=score_from,
            response_mentions_label=(ep.get("label", "").lower() in scored[0]["text"].lower()) if scored else None,
            query=query,
            threshold=args.threshold,
            score_heads=args.score_heads,
            posthoc_demo_informative_gt_0p3=dict(
                response_i=demo_i, verdict=protocol.verdict(ep, demo_i, protocol.has_event(ep))
            ),
            replies=replies,
            scores=scores,
            runtime_s=round(time.time() - t0, 2),
            peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        )
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"{ep['episode_id']} [{ep.get('label')}] n={rec['n_frames']} timely=[{ep['timely_start_i']},"
            f"{ep['timely_end_i']}] resp={response_i} -> {rec['verdict']} | demo0.3 resp={demo_i} | "
            f"{len(replies)} replies | {rec['runtime_s']}s | first reply: {rec['response_text']!r}",
            flush=True,
        )


if __name__ == "__main__":
    main()
