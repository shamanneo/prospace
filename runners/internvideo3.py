"""InternVideo3-8B-Instruct (HF yanziang/InternVideo3-8B-Instruct) on ProSpace-Bench by prefix polling.

InternVideo3 is an offline video LLM (Qwen3-VL-style processor + M^2LA LM).  Neither the official repository
(OpenGVLab/InternVideo, InternVideo3/) nor the model card provides a streaming / proactive mode, so the protocol is
prefix polling: at every second t the model sees frames 0..t as one video and the STATE_PROMPT question; the response
is the first greedy "Yes" (runners/protocol.py:poll_loop, the same rule as runners/qwen3vl.py).

Model loading / input construction follow the model card's "Video Understanding" example (AutoModelForCausalLM +
AutoProcessor, trust_remote_code, bf16, sdpa, video content then text, processor.video_processor.size override).
Because our input is an already-sampled 1 fps frame list rather than an mp4, frames are passed as a (T, H, W, 3) array
with video_metadata(fps=1) and do_sample_frames=False, so the processor's per-temporal-patch "<x.x seconds>"
timestamps stay correct.

The model's remote code needs transformers 4.57 (we used 4.57.3); it does not load with transformers 5.x.  Launcher:
    bash scripts/eval_polling.sh internvideo3 internvideo3 yanziang/InternVideo3-8B-Instruct 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import (  # noqa: E402
    FRAME_H,
    FRAME_W,
    ROOT,
    SAMPLE_FPS,
    TASKS,
    episode_frames,
    fill,
    frames_root_for,
    has_event,
    poll_loop,
    select_episodes,
    verdict,
)

MODEL = "yanziang/InternVideo3-8B-Instruct"
log = logging.getLogger("internvideo3")


def gpu_free_gb() -> float:
    """Free memory on the (single) visible GPU, as the driver sees it (the GPU may be shared)."""
    free, _ = torch.cuda.mem_get_info(0)
    return free / 2**30


def wait_for_gpu(need_gb: float, poll_s: int = 60) -> None:
    while (free := gpu_free_gb()) < need_gb:
        log.info("GPU free %.1f GB < %.1f GB needed; waiting %ds", free, need_gb, poll_s)
        time.sleep(poll_s)


def load(model_path: str):
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0", trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # model card: processor.video_processor.size = {"longest_edge": max_pixels * max_frames, "shortest_edge": ...}
    # budget large enough that 512x384 frames (multiples of 32) are never downscaled for our episode lengths
    processor.video_processor.size = {"longest_edge": FRAME_W * FRAME_H * 512, "shortest_edge": 128 * 32 * 32}
    return model, processor


def load_frames(ep: dict, root: Path, burn_bbox: bool = False) -> list[np.ndarray]:
    """Frames at the model resolution; with burn_bbox (Object Re-finding) frame 0 carries the red target box."""
    out = []
    for f in episode_frames(ep, Path(root), burn_bbox):  # paths, or a PIL image for the boxed frame 0
        im = f if isinstance(f, Image.Image) else Image.open(f)
        out.append(np.asarray(im.convert("RGB").resize((FRAME_W, FRAME_H), Image.BICUBIC)))
    return out


class Asker:
    def __init__(self, model, processor):
        from transformers.video_utils import VideoMetadata

        self.VideoMetadata = VideoMetadata
        self.model, self.processor = model, processor
        tok = processor.tokenizer
        self.yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = tok.encode("No", add_special_tokens=False)[0]

    @torch.inference_mode()
    def __call__(self, clip: list[np.ndarray], text: str):
        n = len(clip)
        idx = list(range(n))
        if n == 1:  # the Qwen3-VL video processor rejects T < temporal_patch_size (2); repeat frame 0 (its own padding)
            clip, idx = clip * 2, [0, 0]
        video = np.stack(clip)
        meta = self.VideoMetadata(
            total_num_frames=len(clip), fps=SAMPLE_FPS, width=FRAME_W, height=FRAME_H, frames_indices=idx
        )
        messages = [{"role": "user", "content": [{"type": "video", "video": "frames"}, {"type": "text", "text": text}]}]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[chat], videos=[video], video_metadata=[meta], do_sample_frames=False, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
            output_logits=True,
            return_dict_in_generate=True,
        )
        gen = out.sequences[0, inputs["input_ids"].shape[1] :]
        reply = self.processor.batch_decode([gen], skip_special_tokens=True)[0].strip()
        probs = torch.softmax(out.logits[0][0].float(), dim=-1)
        p_yes, p_no = probs[self.yes_id].item(), probs[self.no_id].item()
        return reply, p_yes / max(p_yes + p_no, 1e-12), p_yes, p_no, int(inputs["input_ids"].shape[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--task", default="object_finding", choices=sorted(TASKS))
    ap.add_argument("--episodes", default="data/prospace_bench.jsonl")
    ap.add_argument("--frames", default=str(ROOT / "data" / "frames" / "scannetpp"))
    ap.add_argument(
        "--frames-by-dataset",
        action="store_true",
        help="mixed-dataset episode files: pick the frame root from each episode's `dataset` field",
    )
    ap.add_argument("--all", action="store_true", help="every episode of the task (the default; as runners/qwen3vl.py)")
    ap.add_argument("--step", type=int, default=1, help="polling interval in seconds (only 1)")
    ap.add_argument("--episode-id", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--need-gb", type=float, default=21.0, help="free GPU memory required before loading")
    ap.add_argument("--out", default="results/internvideo3/object_finding.jsonl")
    ap.add_argument(
        "--score-from-field", default=None, help="episode field with the first scored second (Re-finding: gap_start_i)"
    )
    args = ap.parse_args()
    assert args.step == 1, "the benchmark polls every second"
    if args.frames_by_dataset:
        os.environ["PROSPACE_FRAMES_BY_DATASET"] = "1"
    args.out = str(ROOT / args.out) if not Path(args.out).is_absolute() else args.out
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler()])

    eps = select_episodes(args)
    log.info("episodes to run: %d -> %s", len(eps), args.out)
    if not eps:
        return
    wait_for_gpu(args.need_gb)
    t0 = time.time()
    model, processor = load(args.model)
    asker = Asker(model, processor)

    def ask(clip, text, tries=30):
        for k in range(tries):  # shared GPU: on OOM free the cache, wait and retry the same step
            try:
                return asker(clip, text)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.info("OOM at %d frames (free %.1f GB); retry %d in 60s", len(clip), gpu_free_gb(), k + 1)
                time.sleep(60)
        raise RuntimeError("persistent OOM")

    log.info("model loaded in %.1fs, gpu mem %.1f GB", time.time() - t0, torch.cuda.max_memory_allocated() / 2**30)
    question = TASKS[args.task]["state"]
    burn = bool(TASKS[args.task].get("frame0_bbox", False))
    for ep in eps:
        t1 = time.time()
        frames = load_frames(ep, frames_root_for(ep, args.frames), burn)
        q = fill(question, ep)
        score_from = int(ep.get(args.score_from_field, 0)) if args.score_from_field else 0
        res = poll_loop(ask, frames, q, words=("Yes", "No"), step=1, stop_at_first=True, score_from=score_from)
        v = verdict(ep, res["response_i"], has_event=has_event(ep)) if res["invalid"] is None else "invalid"
        rec = {
            "episode_id": ep["episode_id"],
            "label": ep.get("label"),
            "n_frames": len(frames),
            "timely_start_i": ep["timely_start_i"],
            "timely_end_i": ep["timely_end_i"],
            "response_i": res["response_i"],
            "verdict": v,
            "invalid": res["invalid"],
            "mode": "prefix_polling_state",
            "task": args.task,
            "score_from": score_from,
            "question": q,
            "model": args.model,
            "n_calls": res["n_calls"],
            "trace": res["trace"],
            "runtime_s": round(time.time() - t1, 1),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info(
            "%s %s n=%d window=[%s,%s] response=%s verdict=%s calls=%d %.1fs peak=%.1fGB",
            ep["episode_id"],
            ep.get("label"),
            len(frames),
            ep["timely_start_i"],
            ep["timely_end_i"],
            res["response_i"],
            v,
            res["n_calls"],
            rec["runtime_s"],
            rec["peak_gpu_gb"],
        )


if __name__ == "__main__":
    main()
