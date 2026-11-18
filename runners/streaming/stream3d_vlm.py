"""Stream3D-VLM-4B streaming runner (ProSpace, proactive response-time protocol).

Wraps the official streaming decoder from third_party/Stream3D-VLM/scripts/eval/evaluate.py:
`StreamingEvaluator.generate_with_streaming_decision(all_images, query_text, num_images_before_query)`.
The model sees the query text, then one frame at a time with a growing LLM KV cache; after every frame it emits a
decision token: "," = continue (take the next frame), "\\n" = stop and answer (any other token is treated as STOP
by the official code).  Response index = the frame after which the model first emits STOP.  If every frame gets
"," the official code still generates an answer after the last frame; we score that as a miss.

The official function reports decisions only through stdout, so we capture stdout per episode and parse the
"Decision: CONTINUE/STOP/UNEXPECTED" lines (one per frame, in order) to recover per-step decisions and to tell a
STOP on the last frame apart from running out of frames.  The repository needs patches/stream3d_vlm_eval.patch
(Object Re-finding: no STOP before `min_stop_frame`; long streams: geometry encoded frame by frame).

Env: the Stream3D-VLM repository's own environment (torch 2.5.1+cu121, flash-attn 2.7.4.post1, transformers==4.50.0).

  CUDA_VISIBLE_DEVICES=0 python runners/streaming/stream3d_vlm.py --limit 1
"""

from __future__ import annotations

import argparse
import os
import contextlib
import importlib.util
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runners"))
import protocol  # noqa: E402  (torch-free)

REPO = ROOT / "third_party" / "Stream3D-VLM"
MODEL = os.environ.get(
    "STREAM3D_MODEL", "JonnyYu828/Stream3D-VLM-4B"
)  # a local snapshot path or the Hugging Face id (revision 28a5245b)

# Official Stream3D-Bench object_finding_forward phrasing (examples/stream3d_bench_examples.json).  Default
# (--query_style native): with the literal protocol.py request the model STOPs on frame 0 and answers "1.0"
# (out-of-distribution query), so the native template is used and the protocol-text run is kept as a comparison.
NATIVE_OF = (
    "I am controlling the camera to explore. Locate the {label} for me. "
    "Once visible, tell me its distance and clock position."
)
# --query-text: the ProSpace question rewritten in the model's own forward-query style (condition clause, no
# timer wording).  Chosen once on out/epsiodes/dev30/<task>.json and then frozen.
# {label} / {room} / {direction_cam} are filled from the episode by protocol.fill.


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def wait_for_gpu_memory(need_mib: int, poll_s: int = 60, log=None):
    """GPU is shared: block until the visible GPU has `need_mib` free (nvidia-smi), instead of crashing on load."""
    import os
    import subprocess

    idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    while True:
        free = int(
            subprocess.check_output(
                ["nvidia-smi", "-i", idx, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
            ).strip()
        )
        if free >= need_mib:
            print(f"GPU {idx}: {free} MiB free (need {need_mib})", file=log, flush=True)
            return
        print(f"GPU {idx}: only {free} MiB free (need {need_mib}), retry in {poll_s}s", file=log, flush=True)
        time.sleep(poll_s)


def load_official_evaluator():
    sys.path.insert(0, str(REPO / "src"))
    # load by path: the module name `evaluate` would collide with the HF `evaluate` package
    spec = importlib.util.spec_from_file_location(
        "stream3d_official_evaluate", REPO / "scripts" / "eval" / "evaluate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StreamingEvaluator


def parse_decisions(stdout: str) -> list[str]:
    out = []
    for m in re.finditer(r"Decision: (CONTINUE|STOP|UNEXPECTED)(?: \((.*?)\))?", stdout, flags=re.S):
        out.append({"CONTINUE": ",", "STOP": "\\n"}.get(m.group(1), f"UNEXPECTED:{m.group(2)!r}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="out/epsiodes/object_finding_stream10.json")
    ap.add_argument("--frames", default=str(ROOT / "data/frames/scannetpp"))
    ap.add_argument(
        "--frames-by-dataset",
        action="store_true",
        help="pick the frame root per episode `dataset` (protocol.DATASET_FRAMES)",
    )
    ap.add_argument("--out", default=str(ROOT / "out/runs/streaming/stream10/stream3d_vlm.jsonl"))
    ap.add_argument("--task", default="object_finding")
    ap.add_argument(
        "--query-text",
        default=None,
        help="native-style query template for this task ({label} etc. filled per episode); overrides --query_style",
    )
    ap.add_argument("--query_style", choices=["protocol", "native"], default="native")
    ap.add_argument(
        "--score-from-field",
        default=None,
        help="episode field: a STOP before that second is not a response (Object Re-finding: show_len / gap_start_i); "
        "the streaming decision is forced to CONTINUE there so the stream goes on, as runners/qwen3vl.py ignores an "
        "early Yes",
    )
    ap.add_argument("--episode_id", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--need_mib", type=int, default=16000)  # observed peak 10.7 GB on 117-frame episodes
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = open(out.with_suffix(".log"), "a", buffering=1)
    eps = protocol.select_episodes(args)
    print(f"{len(eps)} episodes to run -> {out}", file=log, flush=True)
    if not eps:
        return

    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # 6464461276/frame_007200.jpg is short by 53 bytes
    StreamingEvaluator = load_official_evaluator()
    wait_for_gpu_memory(args.need_mib, log=log)
    t0 = time.time()
    with contextlib.redirect_stdout(Tee(sys.stdout, log)):
        ev = StreamingEvaluator(
            model_path=MODEL, local_rank=-1, world_size=1, max_new_tokens=args.max_new_tokens, temperature=0.0
        )  # official defaults otherwise (grad-ckpt flag on, eval mode)
    print(f"model loaded in {time.time() - t0:.1f}s", file=log, flush=True)

    for ep in eps:
        rot = {
            "left": "the direction that was on your left",
            "right": "the direction that was on your right",
            "behind": "the opposite direction",
        }.get(ep.get("direction"), "")
        request = (
            protocol.fill(args.query_text.replace("{rot_target}", rot), ep)
            if args.query_text
            else (
                protocol.fill(protocol.TASKS[args.task]["request"], ep)
                if args.query_style == "protocol"
                else NATIVE_OF.format(label=ep["label"])
            )
        )
        cfg = protocol.TASKS.get(args.task, {})
        paths = protocol.episode_frames(
            ep,
            (
                Path(protocol.DATASET_FRAMES.get(ep.get("dataset"), args.frames))
                if args.frames_by_dataset
                else Path(args.frames)
            ),
            burn_bbox=cfg.get("frame0_bbox", False),
        )  # Re-finding: frame 0 carries the red target box, as for every other model
        images = [
            (Image.open(p) if isinstance(p, str) else p)
            .convert("RGB")
            .resize((protocol.FRAME_W, protocol.FRAME_H), Image.BICUBIC)
            for p in paths
        ]
        score_from = int(ep.get(args.score_from_field, 0)) if args.score_from_field else 0
        ev.min_stop_frame = score_from
        # Question at 0 s: the query precedes every frame, in the official conversation format.
        images_before, query_text, images_after = ev.parse_conversation_with_query(request + "<image>" * len(images))
        assert len(images_before) == 0 and len(images_after) == len(images)

        buf = io.StringIO()
        ts = time.time()
        with contextlib.redirect_stdout(Tee(buf, log)):
            print(
                f"\n##### episode {ep['episode_id']} label={ep['label']} n={len(images)} "
                f"timely=[{ep['timely_start_i']},{ep['timely_end_i']}] query={query_text!r}"
            )
            last_idx, text, timing = ev.generate_with_streaming_decision(images, query_text, 0)
        runtime = time.time() - ts

        decisions = parse_decisions(buf.getvalue())
        stopped = bool(decisions) and decisions[-1] != ","
        assert len(decisions) == last_idx + 1, (len(decisions), last_idx)
        response_i = last_idx if stopped else None
        rec = {
            "task": args.task,
            "episode_id": ep["episode_id"],
            "scene_id": ep["scene_id"],
            "label": ep["label"],
            "n_frames": len(images),
            "timely_start_i": ep["timely_start_i"],
            "timely_end_i": ep["timely_end_i"],
            "response_i": response_i,
            "verdict": protocol.verdict(ep, response_i, has_event=protocol.has_event(ep)),
            "score_from": score_from,
            "query_style": args.query_style,
            "query_text": query_text,
            "step_decisions": decisions,  # one decision token per frame fed, index = frame i
            "answer_text": text if stopped else None,
            "answer_text_after_last_frame": None if stopped else text,
            "timing": timing,
            "runtime_s": round(runtime, 2),
        }
        with open(out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"[{ep['episode_id']}] response_i={response_i} window=[{ep['timely_start_i']},{ep['timely_end_i']}] "
            f"verdict={rec['verdict']} runtime={runtime:.1f}s answer={text!r}",
            file=log,
            flush=True,
        )
        print(f"[{ep['episode_id']}] response_i={response_i} verdict={rec['verdict']} {runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()
