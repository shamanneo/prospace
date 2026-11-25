"""Flash-VStream-Qwen-7b (HF zhang9302002/Flash-VStream-Qwen-7b) streaming runner for ProSpace.

Wraps the official real-time path of third_party/Flash-VStream/Flash-VStream-Qwen/cli_server_2gpu.py in one process:
  * frame_memory_manager(): processor.image_processor(videos=clip, additional_pool_size=...) ->
    FlashVStreamQwen2VLModel.embed_new_video_clip(**video_inputs, start_idx=frame_cnt)   (models/vstream_qwen2vl_realtime.py)
    i.e. every new frame is ViT-encoded once and merged into the Flash Memory (CSM k-means + DAM retrieval).
  * main(): "<|vision_start|><|video_pad|><|vision_end|>" + question -> apply_chat_template ->
    processor(text, dummy_video_tokens=...) -> model.generate(max_new_tokens=1, use_cache=False); the forward pass
    reads the video embeddings from the memory (prepare_realtime_inference, AM-RoPE).
The model has no proactive trigger (the CLI only answers questions asked by the user), so we ask the protocol's
per-second state question after each new frame and take the first Yes (protocol.poll_loop).

Runner-side shims (no repository file is edited):
  1. single process: model.video_embedding_memory is a plain list instead of a Manager().list() shared between processes.
  2. dummy_video_tokens = 4 * (current memory token count) instead of the CLI's hard-coded 10800*4 (which assumes
     the CLI's 120-frame warm-up clip); so frames can be fed one by one from t=0.
  3. lm_head wrapped to project only the last position (generate only reads logits[:, -1]); identical outputs, saves
     ~10 GB of full-sequence float32 logits.

Run in the Flash-VStream repository's own environment (we used Python 3.10, torch 2.6.0, transformers 4.45.0,
flash-attn):
  CUDA_VISIBLE_DEVICES=0 python runners/streaming/flash_vstream.py --task object_finding --episodes data/prospace_bench/prospace_bench.jsonl
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
REPO = ROOT / "third_party" / "Flash-VStream" / "Flash-VStream-Qwen"
sys.path.insert(0, str(ROOT / "runners"))
sys.path.insert(0, str(REPO))

import protocol as P  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from models import FlashVStreamQwen2VLConfig, FlashVStreamQwen2VLProcessor  # noqa: E402
from models.vstream_qwen2vl_realtime import FlashVStreamQwen2VLModel  # noqa: E402
from qwen_vl_utils.vision_process import fetch_image, VIDEO_MIN_PIXELS  # noqa: E402

# cli_server_2gpu.py __main__ / video_stream_similator settings
FLASH_MEMORY_DICT = dict(
    flash_memory_temporal_length=120,
    flash_memory_temporal_method="kmeans_ordered",
    flash_memory_temporal_poolsize=2,
    flash_memory_temporal_pca_dim=32,
    flash_memory_spatial_length=60,
    flash_memory_spatial_method="klarge_retrieve",
)
CLI_MAX_PIXELS = 4 * 224 * 224
VIDEO_PREFIX = "<|vision_start|><|video_pad|><|vision_end|>"
TASK = "object_finding"

log = logging.getLogger("flash_vstream")


def wait_for_gpu(need_gb: float, poll_s: int = 30):
    """The GPU may be shared: block until the visible device has `need_gb` free (nvidia-smi)."""
    import subprocess

    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    while True:
        try:
            free = (
                float(
                    subprocess.check_output(
                        ["nvidia-smi", "-i", dev, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
                    ).strip()
                )
                / 1024
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"nvidia-smi failed: {e}")
            free = need_gb
        if free >= need_gb:
            return free
        log.info(f"GPU {dev}: {free:.1f} GB free < {need_gb} GB, waiting {poll_s}s")
        time.sleep(poll_s)


def retry_oom(fn, *a, tries: int = 20, wait_s: int = 30, **k):
    for i in range(tries):
        try:
            return fn(*a, **k)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.warning(f"CUDA OOM in {fn.__name__} (attempt {i + 1}/{tries}); retrying in {wait_s}s")
            time.sleep(wait_s)
    raise RuntimeError(f"{fn.__name__}: persistent CUDA OOM")


class LastTokenHead(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, h):
        return self.head(h[:, -1:, :])


class FlashVStreamStream:
    def __init__(self, model_path: str, max_new_tokens: int = 1):
        cfg = FlashVStreamQwen2VLConfig.from_pretrained(model_path, trust_remote_code=True)
        cfg.vision_config.flash_memory_config = FLASH_MEMORY_DICT
        self.model = FlashVStreamQwen2VLModel.from_pretrained(
            model_path,
            config=cfg,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        self.model.lm_head = LastTokenHead(self.model.lm_head)
        self.model.use_video_streaming_mode = True
        self.processor = FlashVStreamQwen2VLProcessor.from_pretrained(model_path)
        self.fm = FLASH_MEMORY_DICT
        self.max_new_tokens = max_new_tokens
        tok = self.processor.tokenizer
        # the model often answers in lower case ("yes"/"no"); parse_action accepts both, so probabilities sum both cases
        ids = {w: tok.encode(w, add_special_tokens=False) for w in ("Yes", "yes", "No", "no")}
        assert all(len(v) == 1 for v in ids.values()), ids
        self.yes_ids, self.no_ids = [ids["Yes"][0], ids["yes"][0]], [ids["No"][0], ids["no"][0]]
        self.eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
        self.reset()

    def reset(self):
        self.model.video_embedding_memory = []
        self.frame_cnt = 0

    @torch.inference_mode()
    def add_frame(self, frame) -> float:
        """frame_memory_manager(): one 1-fps frame -> encoder -> Flash Memory update."""
        img = frame if isinstance(frame, Image.Image) else Image.open(frame)
        img = img.convert("RGB").resize((P.FRAME_W, P.FRAME_H), Image.BICUBIC)
        # video_stream_similator(): process_vision_info with max_pixels=4*224*224 (fetch_image per frame is identical)
        img = fetch_image({"image": img, "min_pixels": VIDEO_MIN_PIXELS, "max_pixels": CLI_MAX_PIXELS})
        t0 = time.perf_counter()
        vi = self.processor.image_processor(
            images=None,
            videos=[img],
            return_tensors="pt",
            additional_pool_size=self.fm["flash_memory_temporal_poolsize"],
        )
        self.model.embed_new_video_clip(**vi, start_idx=self.frame_cnt)
        torch.cuda.synchronize()
        self.frame_cnt += 1
        self.frame_hw = tuple(int(v) for v in vi["video_grid_thw"][0][1:] * 14)
        return time.perf_counter() - t0

    def n_mem_tokens(self) -> int:
        return int(self.model.video_embedding_memory[-1][0])

    @torch.inference_mode()
    def ask(self, question: str, max_new_tokens: int | None = None):
        """main(): text query against the current memory.  Greedy.  Multi-token replies are produced by repeating the
        official single-token call on the extended sequence (the repo's generate path only supports 1 new token)."""
        max_new_tokens = max_new_tokens or self.max_new_tokens
        n_tok = self.n_mem_tokens()
        messages = [{"role": "user", "content": [{"type": "text", "text": VIDEO_PREFIX + question}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
            flash_memory_config=self.fm,
            dummy_video_tokens=n_tok * 4,
        ).to("cuda")
        out_ids, first_scores = [], None
        for _ in range(max_new_tokens):
            g = self.model.generate(
                **inputs,
                max_new_tokens=1,
                use_cache=False,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                output_scores=True,
                return_dict_in_generate=True,
            )
            nid = int(g.sequences[0, -1])
            if first_scores is None:
                first_scores = g.scores[0][0].float()
            if nid in self.eos:
                break
            out_ids.append(nid)

            def one(x, v):
                return torch.cat([x, torch.full((1, 1), v, dtype=x.dtype, device=x.device)], dim=1)

            inputs["input_ids"] = one(inputs["input_ids"], nid)
            inputs["attention_mask"] = one(inputs["attention_mask"], 1)
            inputs["visual_position_ids"] = one(inputs["visual_position_ids"], -1)
        reply = self.processor.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        prob = torch.softmax(first_scores, -1)
        p_yes, p_no = float(prob[self.yes_ids].sum()), float(prob[self.no_ids].sum())
        top = torch.topk(prob, 3)
        top3 = [
            (self.processor.tokenizer.decode([int(i)]), round(float(p), 4)) for p, i in zip(top.values, top.indices)
        ]
        return reply, p_yes / max(p_yes + p_no, 1e-12), p_yes, p_no, n_tok, top3


def run_episode(
    stream: FlashVStreamStream,
    ep: dict,
    frame_root: Path,
    seed: int,
    task: str = TASK,
    score_from_field: str | None = None,
) -> dict:
    torch.manual_seed(seed)  # Flash Memory k-means uses torch.randperm for centroid init
    stream.reset()
    cfg = P.TASKS[task]
    frames = P.episode_frames(ep, frame_root, burn_bbox=cfg.get("frame0_bbox", False))  # Re-finding: red box on frame 0
    question = P.fill(cfg["state"], ep)
    score_from = int(ep.get(score_from_field, 0)) if score_from_field else 0
    extra = []

    def ask(clip, text):
        assert len(clip) == stream.frame_cnt + 1, "polling must advance exactly one frame per call"
        emb_s = retry_oom(stream.add_frame, clip[-1])  # memory is written only at the end of embed_new_video_clip
        t0 = time.perf_counter()
        reply, p_pos, p_yes, p_no, n_tok, top3 = retry_oom(stream.ask, text)
        q_s = time.perf_counter() - t0
        extra.append({"embed_s": round(emb_s, 3), "query_s": round(q_s, 3), "top3": top3})
        log.info(
            f"{ep['episode_id']} t={len(clip) - 1} mem_tokens={n_tok} reply={reply!r} p_yes={p_pos:.3f} "
            f"embed={emb_s:.2f}s query={q_s:.2f}s"
        )
        return reply, p_pos, p_yes, p_no, n_tok

    t0 = time.perf_counter()
    res = P.poll_loop(ask, frames, question, words=("Yes", "No"), step=1, stop_at_first=True, score_from=score_from)
    runtime = time.perf_counter() - t0
    for tr, ex in zip(res["trace"], extra):
        tr["mem_tokens"] = tr.pop("tokens")
        tr.update(ex)
    return {
        "episode_id": ep["episode_id"],
        "scene_id": ep.get("scene_id"),
        "label": ep.get("label"),
        "n_frames": len(frames),
        "timely_start_i": ep["timely_start_i"],
        "timely_end_i": ep["timely_end_i"],
        "response_i": res["response_i"],
        "verdict": P.verdict(ep, res["response_i"], has_event=P.has_event(ep)),
        "invalid": res["invalid"],
        "n_calls": res["n_calls"],
        "task": task,
        "score_from": score_from,
        "mode": "state_polling_on_streaming_memory",
        "question": question,
        "prompt_template": P.STATE_PROMPT,
        "model_input_hw": list(stream.frame_hw),
        "steps": res["trace"],
        "runtime_s": round(runtime, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zhang9302002/Flash-VStream-Qwen-7b", help="HF id or local folder")
    ap.add_argument("--task", default=TASK, choices=sorted(P.TASKS))
    ap.add_argument(
        "--score-from-field",
        default=None,
        help="episode field; a Yes before that second is not a response (Re-finding: gap_start_i)",
    )
    ap.add_argument("--episodes", default="data/prospace_bench/prospace_bench.jsonl")
    ap.add_argument("--frames", default=P.DEFAULT_FRAMES)
    ap.add_argument("--out", default="results/flash_vstream/object_finding.jsonl")
    ap.add_argument("--episode-id", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-new-tokens", type=int, default=1, help="1 = official CLI (cli_server_2gpu.py main)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--need-gb", type=float, default=22.0, help="free GPU memory required before loading")
    args = ap.parse_args()

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(out.with_suffix(".log")), logging.StreamHandler(sys.stdout)],
    )
    for noisy in ("transformers", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.out = str(out)
    eps = P.select_episodes(args)

    model_path = args.model
    if not Path(model_path).is_dir():
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(model_path)
    log.info(f"GPU free before load: {wait_for_gpu(args.need_gb):.1f} GB")
    stream = FlashVStreamStream(model_path, max_new_tokens=args.max_new_tokens)
    log.info(f"loaded {model_path}; gpu mem {torch.cuda.memory_allocated() / 2**30:.1f} GB; {len(eps)} episodes")

    with open(out, "a") as fh:
        for ep in eps:
            rec = run_episode(
                stream, ep, P.frames_root_for(ep, args.frames), args.seed, args.task, args.score_from_field
            )
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            log.info(
                f"DONE {rec['episode_id']} label={rec['label']} response_i={rec['response_i']} "
                f"window=[{rec['timely_start_i']},{rec['timely_end_i']}] verdict={rec['verdict']} "
                f"calls={rec['n_calls']} runtime={rec['runtime_s']}s "
                f"peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
            )


if __name__ == "__main__":
    main()
