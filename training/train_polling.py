"""LoRA SFT of Qwen3-VL / Qwen2.5-VL on prefix-polling samples (the "SFT, prefix polling" rows).

The model input is built by the SAME function the benchmark runner uses (runners/qwen3vl.build_inputs: [question][video],
frames 0..t resized to 512 x 384, 1 fps metadata), so training and evaluation see identical prompts.  The target is the
single answer word ("Yes" / "No") + end-of-turn; the loss covers only those tokens.  Object Re-finding samples get the
red box burnt into frame 0 exactly like runners/protocol.episode_frames.

    bash scripts/train_polling.sh qwen3 0,1,2,3               # the launch line of the paper row
    python -m training.train_polling --dry-run 8              # CPU: build a few samples, print shapes / label mask

Logged to wandb (project prospace, entity from WANDB_ENTITY; --no-wandb turns it off) and to <out>/log.jsonl:
  train/loss, lr, grad_norm, tokens/s, frames per sample
  val/loss, val/acc, val/acc_<task>, val/recall_yes|no, val/acc_<sample_type>, val/acc_reg_<E|T|P>, val/acc_<dataset>,
  val/yes_rate_t<bin> (elapsed-time shortcut check)
  poll/<verdict> and poll/timely_<task>: per-second polling on validation streams, scored like the benchmark
The adapter with the lowest validation loss is kept as <out>/adapter_best.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runners"))

import qwen3vl as R  # noqa: E402  (the bench runner: build_inputs, FRAME_W/H)
from prospace_vlm import data as FF  # noqa: E402
from protocol import STATE_PROMPT, FRAME_W  # noqa: E402
from PIL import ImageFile  # noqa: E402

# A frame file that is short by a few bytes is decoded as far as it goes instead of stopping the run (the streaming
# trainer, prospace_vlm/train.py, does the same).
ImageFile.LOAD_TRUNCATED_IMAGES = True

WORDS = ("Yes", "No")

# Hooks so a second backend can reuse this trainer without a copy:
# BUILD_INPUTS(processor, frames, question) -> model inputs, LOAD_MODEL(args, device) -> a ready model.
BUILD_INPUTS = None
LOAD_MODEL = None
# PROSPACE_QWEN_FAMILY=qwen25 -> the Qwen2.5-VL input builder of the benchmark runner (runners/vlm.py
# --family qwen25: the qwen_vl_utils video path without the Qwen3 per-frame metadata), so the offline polling SFT row can
# use the streaming rows' backbone (Qwen2.5-VL-7B).  Read from the environment so the spawned DataLoader workers, which
# re-import this module, build the same inputs as the main process.  Unset = the Qwen3 path, unchanged.
if os.environ.get("PROSPACE_QWEN_FAMILY"):
    import vlm as _VLM  # noqa: E402  (runners/vlm.py)

    BUILD_INPUTS = _VLM.BUILDERS[os.environ["PROSPACE_QWEN_FAMILY"]]
    # With left-padded micro-batches torch 2.13 routes the language model's
    # masked attention to the cuDNN SDPA kernel, whose BACKWARD returned NaN for every LoRA tensor on a padded batch whose
    # samples are finite one by one.  The
    # cuDNN SDPA backend is switched off for this path; the memory-efficient kernel serves the masked attention instead.
    torch.backends.cuda.enable_cudnn_sdp(False)
T_BINS = (0, 5, 10, 20, 40, 80, 10**9)


# ----------------------------------------------------------------------------- data
def frames_of(sample: dict, roots: dict) -> list:
    from PIL import Image

    paths = [str(Path(roots[sample["dataset"]]) / p.split("/", 1)[1]) for p in sample["frames"]]
    if "boxed_frame" in sample and "_boxed" in roots:  # the boxed request frame lives in the sample folder
        q = sample["boxed_frame"]
        paths[q] = str(Path(roots["_boxed"]) / sample["frames"][q])
    frames = list(paths)
    bbox = sample.get("frame0_bbox")
    if sample["task"] == "object_refinding" and bbox:
        from PIL import ImageDraw

        im = Image.open(paths[0]).convert("RGB")
        sx, sy = im.width / 1920, im.height / 1440
        lw = max(4, round(6 * im.width / FRAME_W))
        ImageDraw.Draw(im).rectangle(
            [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy], outline=(255, 0, 0), width=lw
        )
        frames[0] = im
    # portrait frames (CA-1M upright captures, 768 x 1024) are pillarboxed to 512 x 384 instead of stretched;
    # landscape frames keep the runner's own resize
    replay = sample["task"].startswith("replay")  # generic replay images / video frames: any aspect, never stretched
    for i, f in enumerate(frames):
        im = Image.open(f) if isinstance(f, str) else f
        if replay:
            frames[i] = FF.fit_any(im)
        elif FF.is_portrait(im):
            frames[i] = FF.fit(im)
    return frames


def read_jsonl(path, limit=None, max_frames=None):
    out = []
    for line in open(path):
        s = json.loads(line)
        if max_frames and len(s["frames"]) > max_frames:
            continue
        out.append(s)
        if limit and len(out) >= limit:
            break
    return out


MAX_ANS = 256  # free-text (replay) answers are cut to this many tokens before the end-of-turn token
# --replay-weight: loss multiplier for replay_* samples (their gradient share at weight 1 was ~13 % for a 10 % sample
# share).  Read from the environment so the spawned DataLoader workers,
# which re-import this module, see the same value as the main process.
REPLAY_WEIGHT = float(os.environ.get("PROSPACE_REPLAY_WEIGHT", "1"))


def answer_tokens(tok, answer: str, answer_ids: dict) -> list:
    """Yes / No: the fixed ids (word + <|im_end|>).  Any other answer (generic replay): its tokens + <|im_end|>."""
    if answer in answer_ids:
        return answer_ids[answer]
    return tok.encode(answer, add_special_tokens=False)[:MAX_ANS] + [answer_ids[WORDS[0]][-1]]


def encode(processor, sample, roots, answer_ids=None, keep_cache=False):
    """Runner inputs for the prompt; with answer_ids the answer tokens are appended and `labels` masks the prompt.
    Frames go through the runner's frame cache (one PIL load + resize per frame, same model input as without it)."""
    if not keep_cache or R.FRAME_CACHE is None:
        R.FRAME_CACHE = {}
    build = BUILD_INPUTS or R.build_inputs
    inputs = build(processor, frames_of(sample, roots), sample["messages"][0]["content"])
    if answer_ids is None:
        return inputs
    ans = torch.tensor(
        [answer_tokens(processor.tokenizer, sample["answer"], answer_ids)], device=inputs["input_ids"].device
    )
    n_prompt = inputs["input_ids"].shape[1]
    inputs["input_ids"] = torch.cat([inputs["input_ids"], ans], 1)
    for k in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
        if k in inputs:
            pad = torch.ones_like(ans) if k == "attention_mask" else torch.zeros_like(ans)
            inputs[k] = torch.cat([inputs[k], pad.to(inputs[k].dtype)], 1)
    labels = inputs["input_ids"].clone()
    labels[:, :n_prompt] = -100
    inputs["labels"] = labels
    inputs["sample_weight"] = torch.tensor([REPLAY_WEIGHT if sample["task"].startswith("replay") else 1.0])
    return inputs


def collate(xs: list, pad_id: int) -> dict:
    """LEFT-pad the token tensors of several encoded samples (so the answer tokens are the last N_ANS positions of
    every row and only their logits need to be computed), concatenate their video patches."""
    n = max(x["input_ids"].shape[1] for x in xs)
    out = {}
    for k, fill in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100), ("mm_token_type_ids", 0)):
        if k in xs[0]:
            out[k] = torch.cat([torch.nn.functional.pad(x[k], (n - x[k].shape[1], 0), value=fill) for x in xs], 0)
    for k in ("pixel_values_videos", "video_grid_thw", "sample_weight"):
        if k in xs[0]:
            out[k] = torch.cat([x[k] for x in xs], 0)
    if "second_per_grid_ts" in xs[0]:  # Qwen2.5-VL: seconds per temporal grid of every video (its time position ids);
        out["second_per_grid_ts"] = torch.cat(
            [torch.as_tensor(x["second_per_grid_ts"], dtype=torch.float32).reshape(-1) for x in xs], 0
        )  # without it the model assumes 1 s, the runner passes 2 s
    return out


class Encoded(torch.utils.data.Dataset):
    """Micro-batches encoded on CPU by DataLoader workers (decode + resize + processor are the bottleneck, not the GPU).
    `batches` = lists of sample indices; one item = one padded micro-batch."""

    def __init__(self, samples, batches, processor, roots, answer_ids):
        self.samples, self.batches, self.processor, self.roots, self.answer_ids = (
            samples,
            batches,
            processor,
            roots,
            answer_ids,
        )
        self.pad_id = processor.tokenizer.pad_token_id or 0

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, b):
        idx = self.batches[b]
        return idx, collate(
            [encode(self.processor, self.samples[i], self.roots, self.answer_ids) for i in idx], self.pad_id
        )


FRAME_BUDGET = 320  # frames per micro-batch (about 38K tokens): long prefixes get smaller batches


def make_batches(samples, bs: int, window: int = 256) -> list[list[int]]:
    """Micro-batches of similar length: inside windows of `window` samples sort by frame count, then cut at `bs`
    samples or FRAME_BUDGET frames, whichever comes first."""
    out = []
    for w in range(0, len(samples), window):
        idx = sorted(range(w, min(w + window, len(samples))), key=lambda i: len(samples[i]["frames"]))
        cur, frames = [], 0
        for i in idx:
            n = len(samples[i]["frames"])
            if cur and (len(cur) >= bs or frames + n > FRAME_BUDGET):
                out.append(cur)
                cur, frames = [], 0
            cur.append(i)
            frames += n
        if cur:
            out.append(cur)
    random.Random(len(samples)).shuffle(out)
    return out


def _worker_init(_):
    R._DEVICE = "cpu"
    torch.set_num_threads(2)


def loader(samples, processor, roots, answer_ids, workers, bs=1, skip=0):
    """`skip` drops the first N micro-batches so a restarted run continues on the sample it stopped at: the batch
    order is a pure function of the sample list (make_batches seeds on len(samples)), so slicing reproduces it."""
    ds = Encoded(samples, make_batches(samples, bs)[skip:], processor, roots, answer_ids)
    # spawn, not fork: forking after CUDA / tokenizer threads exist segfaults the workers
    return torch.utils.data.DataLoader(
        ds,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=4,
        worker_init_fn=_worker_init,
        persistent_workers=False,
        multiprocessing_context="spawn",
    )


def to_dev(x, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in x.items()}


N_ANS = 2  # answer word + <|im_end|>


def answer_forward(net, x):
    """Loss over the answer tokens only.  The full-vocabulary logits of a whole batch of video prompts do not fit
    in memory (8 x 3K tokens x 152K vocab), so only the last n + 1 positions go through the LM head, n = the longest
    answer in the batch (N_ANS = 2 for a Yes / No batch; longer for generic replay answers, which are left-aligned to
    the end by the padding and masked with -100 where shorter).  The loss is the mean over samples of each sample's
    mean token loss, so a long free-text answer does not outweigh the Yes / No samples of the same batch."""
    ids = x["input_ids"]
    n = int((x["labels"] != -100).sum(1).max()) if "labels" in x else N_ANS
    inp = {k: v for k, v in x.items() if k not in ("labels", "sample_weight")}
    logits = net(**inp, logits_to_keep=n + 1).logits.float()  # (B, n + 1, V): positions -(n+1) .. -1
    pred = logits[:, :n]  # position -(n+1) predicts the first answer token
    tgt = x["labels"][:, -n:] if "labels" in x else ids[:, -n:]
    tok_loss = torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1), ignore_index=-100, reduction="none"
    ).view(tgt.shape)
    valid = (tgt != -100).float()
    per_sample = (tok_loss * valid).sum(1) / valid.sum(1).clamp(min=1)
    if "sample_weight" in x:  # replay samples x REPLAY_WEIGHT, divided by B (not by
        per_sample = per_sample * x["sample_weight"].to(
            per_sample.dtype
        )  # the weight sum), so the ProSpace gradient is unchanged
    loss = per_sample.mean()
    return loss, logits[:, n - N_ANS]  # the position that predicts the Yes / No word


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def p_yes(model, processor, sample, roots, yes_id, no_id):
    inputs = encode(processor, sample, roots, keep_cache=True)
    logits = model(**inputs).logits[0, -1].float()
    two = torch.softmax(logits[[yes_id, no_id]], -1)
    return float(two[0])


@torch.no_grad()
def validate(model, processor, samples, roots, answer_ids, yes_id, no_id, rank, world, workers=6, bs=4):
    model.eval()
    rows = []
    mine = samples[rank::world]
    dev = next(model.parameters()).device
    for idx, x in loader(mine, processor, roots, answer_ids, workers, bs):
        x = to_dev(x, dev)
        _, first = answer_forward(model, x)
        tgt = x["input_ids"][:, -N_ANS]
        losses = torch.nn.functional.cross_entropy(first, tgt, reduction="none")
        py = torch.softmax(first[:, [yes_id, no_id]], -1)[:, 0]
        for j, i in enumerate(idx):
            s = mine[i]
            rows.append(
                {
                    "loss": float(losses[j]),
                    "p_yes": float(py[j]),
                    "answer": s["answer"],
                    "task": s["task"],
                    "type": s["sample_type"],
                    "reg": s["question_id"].split(":")[-1][0],
                    "dataset": s["dataset"],
                    "t": s["t"],
                }
            )
    model.train()
    return rows


def summarize_val(rows):
    m = {}
    ok = lambda r: (r["p_yes"] > 0.5) == (r["answer"] == "Yes")
    m["val/loss"] = sum(r["loss"] for r in rows) / len(rows)
    m["val/acc"] = sum(map(ok, rows)) / len(rows)
    for a in WORDS:
        g = [r for r in rows if r["answer"] == a]
        if g:
            m[f"val/recall_{a.lower()}"] = sum(map(ok, g)) / len(g)
    for key, name in (("task", "acc_"), ("type", "acc_"), ("reg", "acc_reg_"), ("dataset", "acc_")):
        by = defaultdict(list)
        for r in rows:
            by[r[key]].append(ok(r))
        for k, v in by.items():
            m[f"val/{name}{k}"] = sum(v) / len(v)
    for lo, hi in zip(T_BINS, T_BINS[1:]):
        g = [r for r in rows if lo <= r["t"] < hi]
        if g:
            m[f"val/yes_rate_t{lo}"] = sum(r["p_yes"] > 0.5 for r in g) / len(g)
    return m


@torch.no_grad()
def polling(model, processor, episodes, roots, yes_id, no_id, rank, world):
    """Per-second prefix polling, first Yes = the response, scored like runners/protocol.verdict."""
    model.eval()
    rows = []
    for ep in episodes[rank::world]:
        sub = "frames/" if ep["dataset"] == "mp3d" else ""  # place clips: <scene>/frames/frame_XXXXXX.jpg
        # ProSpace-Train records carry their frame list; for the others it is rebuilt from the frame ids
        frames = ep.get("frames") or [
            f"{ep['dataset']}/{ep['scene_id']}/{sub}frame_{int(i):06d}.jpg" for i in ep["frame_ids"]
        ]
        resp = None
        R.FRAME_CACHE = {}  # one decode per frame for the whole episode
        start = int(ep.get("gap_start_i") or 0) if ep["task"] == "object_refinding" else 0
        for t in range(len(frames)):
            s = {
                "dataset": ep["dataset"],
                "task": ep["task"],
                "frames": frames[: t + 1],
                "frame0_bbox": ep.get("frame0_bbox"),
                "messages": [{"content": STATE_PROMPT.format(question=ep["question"], t=t, pos="Yes", neg="No")}],
            }
            if p_yes(model, processor, s, roots, yes_id, no_id) > 0.5 and t >= start:
                resp = t
                break
        ts, te = ep.get("timely_start_i"), ep.get("timely_end_i")
        if ts is None:
            v = "correct_abstention" if resp is None else "false_alarm"
        elif resp is None:
            v = "miss"
        else:
            v = "premature" if resp < ts else "delayed" if resp > te else "timely"
        rows.append({"task": ep["task"], "verdict": v, "has_event": ts is not None})
    model.train()
    return rows


def summarize_poll(rows):
    m = {}
    pos = [r for r in rows if r["has_event"]]
    neg = [r for r in rows if not r["has_event"]]
    for v in ("timely", "premature", "delayed", "miss"):
        m[f"poll/{v}"] = sum(r["verdict"] == v for r in pos) / max(1, len(pos))
    m["poll/false_alarm"] = sum(r["verdict"] == "false_alarm" for r in neg) / max(1, len(neg))
    m["poll/overall"] = sum(r["verdict"] in ("timely", "correct_abstention") for r in rows) / max(1, len(rows))
    by = defaultdict(list)
    for r in pos:
        by[r["task"]].append(r["verdict"] == "timely")
    for k, v in by.items():
        m[f"poll/timely_{k}"] = sum(v) / len(v)
    return m


def gather(obj, world):
    if world == 1:
        return obj
    out = [None] * world
    torch.distributed.all_gather_object(out, obj)
    return [x for part in out for x in part]


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--data", default="cache/train_100k_polling")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="qwen3vl8b-polling-sft", help="wandb run name and output folder")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=16, help="samples per rank per optimizer step")
    ap.add_argument("--max-frames", type=int, default=120, help="skip samples with a longer prefix")
    ap.add_argument("--limit", type=int, default=None, help="samples per epoch (smoke test)")
    ap.add_argument("--val-every", type=int, default=250, help="optimizer steps")
    ap.add_argument("--val-n", type=int, default=1200)
    ap.add_argument("--poll-every", type=int, default=1000)
    ap.add_argument("--poll-n", type=int, default=120)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument(
        "--initial-poll", action="store_true", help="also run the (slow) per-second polling evaluation before training"
    )
    ap.add_argument("--workers", type=int, default=8, help="CPU encoding workers per rank")
    ap.add_argument(
        "--micro-batch", type=int, default=4, help="samples per forward pass (length-bucketed, right padded)"
    )
    ap.add_argument("--no-ckpt", action="store_true", help="no gradient checkpointing (faster, more memory)")
    ap.add_argument("--dry-run", type=int, default=0, help="CPU: encode N samples and print the label mask, then exit")
    ap.add_argument("--val-file", default="val.jsonl", help="validation samples inside --data")
    ap.add_argument(
        "--train-file",
        default=None,
        help="one training file inside --data (paper protocol: train_full.jsonl = every episode once)",
    )
    ap.add_argument(
        "--early-stop",
        type=int,
        default=0,
        metavar="PATIENCE",
        help="stop when val/loss has not improved by --early-stop-rel for PATIENCE validations in a row; "
        "the adapter with the lowest val/loss is kept as adapter_best (0 = off)",
    )
    ap.add_argument("--early-stop-rel", type=float, default=0.01, help="relative improvement that counts (1 %%)")
    ap.add_argument(
        "--init-adapter",
        default=None,
        metavar="DIR",
        help="start from this LoRA adapter instead of a fresh one (restart after a crash)",
    )
    ap.add_argument("--replay-weight", type=float, default=1.0, help="loss multiplier for replay_* samples (mixed set)")
    ap.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="with --init-adapter: the optimizer step that adapter was saved at; the data pointer and the "
        "learning-rate schedule are advanced to it so the run continues instead of repeating",
    )
    a = ap.parse_args()
    global REPLAY_WEIGHT
    REPLAY_WEIGHT = a.replay_weight
    os.environ["PROSPACE_REPLAY_WEIGHT"] = str(a.replay_weight)

    from transformers import AutoProcessor

    data = Path(a.data)
    stats = json.loads((data / "stats.json").read_text())
    roots = stats["frame_roots"]
    if "boxed_root" in stats:
        roots = {**roots, "_boxed": stats["boxed_root"]}
    processor = AutoProcessor.from_pretrained(a.model)
    tok = processor.tokenizer
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    answer_ids = {w: tok.encode(w, add_special_tokens=False) + [eot] for w in WORDS}
    yes_id, no_id = answer_ids["Yes"][0], answer_ids["No"][0]

    if a.dry_run:
        R._DEVICE = "cpu"
        for s in read_jsonl(data / (a.train_file or "train_epoch0.jsonl"), a.dry_run, a.max_frames):
            x = encode(processor, s, roots, answer_ids)
            lab = x["labels"][0]
            print(
                s["task"],
                s["answer"],
                "frames",
                len(s["frames"]),
                "tokens",
                x["input_ids"].shape[1],
                "weight",
                float(x["sample_weight"][0]),
                "| supervised:",
                repr(tok.decode(lab[lab != -100])),
            )
        return

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        import datetime

        torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    torch.cuda.set_device(local)
    R._DEVICE = f"cuda:{local}"
    out = Path(a.out or f"runs/{a.name}")
    out.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    # the shared env has gptqmodel with an old optimum: peft's availability check raises instead of returning False.
    # No GPTQ layers here, so answer "not available" (the env itself is left untouched - the bench runners use it).
    import peft.tuners.lora.model  # noqa: F401

    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("peft") and hasattr(mod, "is_gptqmodel_available"):
            mod.is_gptqmodel_available = lambda *a, **k: False
    model = (
        LOAD_MODEL(a, R._DEVICE)
        if LOAD_MODEL
        else AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to(R._DEVICE)
    )
    if not a.no_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(
        r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, cfg)
    if os.environ.get("PROSPACE_QWEN_FAMILY") == "qwen25":  # frozen vision tower: no autograd, batched window attention
        from training import qwen25_fast_vision  # (same numbers; see the module and its check script)

        qwen25_fast_vision.install(model)
    if a.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        sd = load_file(str(Path(a.init_adapter) / "adapter_model.safetensors"))
        info = set_peft_model_state_dict(model, sd)
        if rank == 0:
            print(
                f"resumed from {a.init_adapter} at step {a.start_step}: {len(sd)} tensors, "
                f"unexpected {len(getattr(info, 'unexpected_keys', []))}",
                flush=True,
            )
    if rank == 0:
        model.print_trainable_parameters()
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if world > 1 else model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))

    val = read_jsonl(data / a.val_file, None, a.max_frames)
    random.Random(0).shuffle(val)
    val = val[: a.val_n]
    poll_eps = [json.loads(l) for l in open(data / "val_polling_episodes.jsonl")]
    poll_eps = [e for e in poll_eps if len(e["frame_ids"]) <= a.max_frames]
    random.Random(0).shuffle(poll_eps)
    poll_eps = poll_eps[: a.poll_n]

    epoch_files = [data / a.train_file] if a.train_file else sorted(data.glob("train_epoch*.jsonl"))
    first = read_jsonl(epoch_files[0], a.limit, a.max_frames)[rank::world]
    n_batches = len(make_batches(first, a.micro_batch))
    per_step = a.grad_accum * len(first) / max(1, n_batches)  # mean samples per rank per optimizer step
    total_steps = a.epochs * (n_batches // a.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, int(0.03 * total_steps)))
        * 0.5
        * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps)))),
    )
    wb = None
    if rank == 0 and not a.no_wandb:
        import wandb

        wb = wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project="prospace",
            name=a.name,
            config={
                **vars(a),
                "world": world,
                "total_steps": total_steps,
                "data_stats": {
                    k: stats[k] for k in ("task_share", "neg_share", "episodes_per_epoch", "protocol") if k in stats
                },
                "train_set": stats.get("train_full") or stats.get("train_epoch0", {}),
            },
        )
    log = (out / "log.jsonl").open("a") if rank == 0 else None

    def emit(metrics, step):
        if rank == 0:
            log.write(json.dumps({"step": step, **metrics}) + "\n")
            log.flush()
            if wb:
                wb.log(metrics, step=step)
            print(
                step, {k: round(v, 4) if isinstance(v, float) else v for k, v in list(metrics.items())[:8]}, flush=True
            )

    es = {"best": float("inf"), "best_step": 0, "stale": 0}

    def evaluate(step, do_poll):
        """Validation (+ optional polling).  Returns True when the early-stopping rule says stop.  Every rank gets the
        same gathered rows, so every rank takes the same decision."""
        rows = gather(validate(model, processor, val, roots, answer_ids, yes_id, no_id, rank, world), world)
        metrics = summarize_val(rows)
        if do_poll:
            metrics.update(
                summarize_poll(gather(polling(model, processor, poll_eps, roots, yes_id, no_id, rank, world), world))
            )
        stop = False
        if a.early_stop and step > 0:
            vl = metrics["val/loss"]
            if vl < es["best"] and rank == 0:  # lowest so far: this is the model that is kept
                model.save_pretrained(out / "adapter_best")
                (out / "adapter_best" / "step.json").write_text(json.dumps({"step": step, "val_loss": vl}))
            es["stale"] = (
                0 if vl < es["best"] * (1 - a.early_stop_rel) else es["stale"] + 1
            )  # a real improvement resets it
            if vl < es["best"]:
                es["best"], es["best_step"] = vl, step
            metrics.update(
                {
                    "early_stop/best_val_loss": es["best"],
                    "early_stop/best_step": es["best_step"],
                    "early_stop/stale": es["stale"],
                }
            )
            stop = es["stale"] >= a.early_stop
        emit(metrics, step)
        return stop

    step = a.start_step
    n_skipped = 0
    for _ in range(a.start_step):  # the cosine schedule must resume where it was, not warm up again
        sched.step()
    evaluate(step, do_poll=a.initial_poll)  # the base model's numbers, before any update (polling a base model
    # that never answers walks every episode to its end: > 1 h, so opt-in)
    model.train()
    for ep in range(a.epochs):
        samples = read_jsonl(epoch_files[ep % len(epoch_files)], a.limit, a.max_frames)[rank::world]
        t0, n_tok, n_frames, run_loss, n_samp, seen = time.time(), 0, 0, 0.0, 0, 0
        dev = torch.device(R._DEVICE)
        tm = Counter()
        t_mark = time.time()
        # Every rank must run the SAME number of micro-batches: the length-bucketed batches differ by a few per rank,
        # and a rank that finishes first walks into the final evaluation's all_gather while the others are still in a
        # gradient all-reduce (seen in the 2-GPU smoke test: "tried to allocate more than 1EB").
        skip = a.start_step * a.grad_accum if ep == 0 else 0
        n_mb = len(make_batches(samples, a.micro_batch)) - skip
        if world > 1:
            t_mb = torch.tensor([n_mb], device=dev)
            torch.distributed.all_reduce(t_mb, op=torch.distributed.ReduceOp.MIN)
            n_mb = int(t_mb.item())
        n_mb -= n_mb % a.grad_accum
        for i, (idx, x) in enumerate(loader(samples, processor, roots, answer_ids, a.workers, a.micro_batch, skip)):
            if i >= n_mb:
                break
            tm["data_wait"] += time.time() - t_mark
            t_mark = time.time()
            sync = (i + 1) % a.grad_accum == 0
            x = to_dev(x, dev)
            torch.cuda.synchronize()
            tm["h2d"] += time.time() - t_mark
            t_mark = time.time()
            ctx = net.no_sync() if (world > 1 and not sync) else torch.autocast("cuda", enabled=False)
            with ctx:
                loss = answer_forward(net, x)[0] / a.grad_accum
                loss.backward()
            torch.cuda.synchronize()
            tm["fwd_bwd"] += time.time() - t_mark
            t_mark = time.time()
            run_loss += float(loss)
            n_tok += int(x["attention_mask"].sum())
            n_frames += sum(len(samples[j]["frames"]) for j in idx)
            n_samp += len(idx)
            seen += len(idx)
            if not sync:
                t_mark = time.time()
                continue
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            # Non-finite guard (one NaN gradient would go through clipping and
            # AdamW and turn every weight NaN).  A step whose loss or gradient norm is not finite is dropped: gradients
            # zeroed, no optimizer update; the schedule and the data pointer still advance, so a restart lines up.  Finite
            # steps are unchanged.  Every rank sees the same all-reduced norm, so every rank takes the same decision.
            if not (math.isfinite(float(gn)) and math.isfinite(run_loss)):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                if rank == 0:
                    print(
                        f"NON-FINITE step {step + 1}: loss {run_loss} grad norm {float(gn)} - skipped ({n_skipped} so far)",
                        flush=True,
                    )
            else:
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            emit(
                {
                    "train/loss": run_loss,
                    "train/lr": sched.get_last_lr()[0],
                    "train/grad_norm": float(gn),
                    "train/tokens_per_s": n_tok * world / (time.time() - t0),
                    "train/frames_per_sample": n_frames / max(1, n_samp),
                    "train/samples_per_s": n_samp * world / (time.time() - t0),
                    "time/data_wait_s": tm["data_wait"],
                    "time/h2d_s": tm["h2d"],
                    "time/fwd_bwd_s": tm["fwd_bwd"],
                    "train/epoch": ep + (seen + skip * a.micro_batch) / len(samples),
                    "train/skipped_nonfinite": n_skipped,
                },
                step,
            )
            if n_skipped > 0.01 * max(step - a.start_step, 100):
                raise RuntimeError(
                    f"{n_skipped} of {step - a.start_step} optimizer steps had a non-finite loss or gradient "
                    f"(more than 1 %): stopping"
                )
            t0, n_tok, n_frames, run_loss, n_samp = time.time(), 0, 0, 0.0, 0
            tm = Counter()
            t_mark = time.time()
            stopped = False
            if step % a.val_every == 0:
                stopped = evaluate(step, do_poll=step % a.poll_every == 0)
            if rank == 0 and step % a.save_every == 0:
                model.save_pretrained(out / f"adapter_step{step}")
            if stopped:
                if rank == 0:
                    print(
                        f"EARLY STOP at step {step}: val/loss has not improved by {a.early_stop_rel:.0%} for "
                        f"{a.early_stop} validations; best {es['best']:.4f} at step {es['best_step']}",
                        flush=True,
                    )
                break
        else:
            if rank == 0:
                model.save_pretrained(out / f"adapter_epoch{ep}")
            continue
        break  # early stop: leave the epoch loop as well
    evaluate(step, do_poll=True)
    if rank == 0:
        model.save_pretrained(out / "adapter_final")
        if wb:
            wb.finish()


if __name__ == "__main__":
    main()
