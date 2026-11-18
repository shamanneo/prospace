"""Native-streaming training on ProSpace-Train (scripts/train_streaming.sh).

One stream per forward pass.  The loss is the cross-entropy of the decision token ("," keep watching / "\\n" respond
now) at every second from the request to the end of the timely window, or to the end of a stream where the event
never happens.  One pass over the training set, in file order (--seed 0).

    --geometry     ProSpace-VLM: Qwen2.5-VL-7B + LoRA + CUT3R geometry injected at --layers
    (without it)   the Qwen2.5-VL-7B + LoRA native-streaming baseline

Validation runs every --val-every steps on <data>/val_poll.jsonl; the checkpoint with the lowest validation loss
is kept in <out>/best.  <out>/last holds the state a stopped run continues from (run the same command again).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from prospace_vlm.data import loader, request_second
from prospace_vlm.eval import verdict_of
from prospace_vlm.model import BACKBONE, LORA_TARGETS, GeometryInjection, ProSpaceVLM, patch_peft


def supervised_seconds(ep: dict) -> int:
    """Seconds 0 .. n-1 carry a decision that is trained: up to the end of the timely window, or the whole stream."""
    n = len(ep["frames"])
    return n if ep["w0"] is None else min(int(ep["w1"]), n - 1) + 1


def loss_of(out: dict, item: dict, device) -> tuple[torch.Tensor, dict]:
    loss = torch.nn.functional.cross_entropy(out["logits"], item["targets"].to(device))  # -100: before the request
    return loss, {"loss/yes_no": float(loss), "loss/total": float(loss), **out["stats"]}


@torch.no_grad()
def validate(model, eps, processor, roots, device, rank, world, workers) -> list[dict]:
    """Every validation stream of this rank, in full: its loss (as in training: the mean over the asked seconds up
    to the end of the timely window) and its outcome under the response rule."""
    model.eval()
    rows = []
    for item in loader(eps[rank::world], processor, roots, workers, train=False):
        ep = item["ep"]
        out = model(item, device)
        ce = torch.nn.functional.cross_entropy(out["logits"], item["targets"].to(device), reduction="none").cpu()
        n_sup, P = supervised_seconds(ep), request_second(ep)
        loss = float(ce[:n_sup].mean())  # the seconds before the request count as 0 here ...
        if P:
            assert n_sup > P, f"{ep['episode_id']}: nothing is supervised after the request"
            loss = loss * n_sup / (n_sup - P)  # ... and are taken out of the mean here
        rows.append(
            {
                "task": ep["task"],
                "has_event": ep["w0"] is not None,
                "verdict": verdict_of(ep, out["final"].cpu()),
                "loss": loss,
                "request": ep.get("request") or ("mid" if P else "first"),
            }
        )
    model.train()
    return rows


def summarize(rows: list[dict], prefix: str = "valid") -> dict:
    """Every average is the mean over tasks of the task's own rate.

    <prefix>/loss     the stream's loss
    <prefix>/timely   streams with an event answered inside the timely window (also premature, delayed, miss)
    <prefix>/silence  streams without an event that got no response
    <prefix>/acc      streams handled correctly (timely, or silent when nothing happens)
    and the same per task (<prefix>/timely_<task>, ...).
    """
    mean = lambda xs: sum(xs) / len(xs)
    per, cols = {}, defaultdict(list)
    for k in sorted({v["task"] for v in rows}):
        g = [v for v in rows if v["task"] == k]
        pos = [v for v in g if v["has_event"]]
        neg = [v for v in g if not v["has_event"]]
        r = {
            "loss": mean([v["loss"] for v in g]),
            "acc": mean([v["verdict"] in ("timely", "correct_abstention") for v in g]),
        }
        if pos:
            for verdict in ("timely", "premature", "delayed", "miss"):
                r[verdict] = mean([v["verdict"] == verdict for v in pos])
        if neg:
            r["silence"] = mean([v["verdict"] == "correct_abstention" for v in neg])
        for c, x in r.items():
            per[f"{prefix}/{c}_{k}"] = x
            cols[c].append(x)
        per[f"{prefix}/n_{k}"] = len(g)
    m = {
        f"{prefix}/{c}": mean(cols[c])
        for c in ("loss", "acc", "timely", "silence", "premature", "delayed", "miss")
        if cols[c]
    }
    m.update(per)
    return m


def summarize_by_request(rows: list[dict]) -> dict:
    """valid/* on all validation streams, plus the same numbers for the streams whose request comes with the first
    frame (valid_first/*) and for those asked mid-stream (valid_mid/*)."""
    m = summarize(rows, "valid")
    for g in ("first", "mid"):
        sub = [v for v in rows if v["request"] == g]
        if sub:
            m.update(summarize(sub, f"valid_{g}"))
    return m


def gather(obj, world):
    if world == 1:
        return obj
    out = [None] * world
    torch.distributed.all_gather_object(out, obj)
    return [x for part in out for x in part]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train.jsonl and val_poll.jsonl (prepare_train.py)")
    ap.add_argument("--stats", default=None, help="json with frame_roots (default <data>/stats.json)")
    ap.add_argument("--roots", default=None, help="json dict dataset -> frame folder (instead of --stats)")
    ap.add_argument("--model", default=BACKBONE)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None, help="default runs/<name>")
    ap.add_argument("--geometry", action="store_true", help="ProSpace-VLM (without it: the Qwen2.5-VL baseline)")
    ap.add_argument("--layers", default="4,8,12,16", help="layers whose output gets the geometric encodings")
    ap.add_argument("--geom-cache", default=None, help="folder for the CUT3R outputs")
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA")
    ap.add_argument("--geom-lr", type=float, default=1e-3, help="the per-layer MLPs of the geometric encodings")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=16, help="streams per GPU per optimizer step")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=0, help="resumable state every N steps (0 = at every validation)")
    ap.add_argument("--keep-every", type=int, default=0, help="keep the weights of every N-th step in <out>/step_<N>")
    ap.add_argument("--early-stop", type=int, default=0, help="stop after N validations without improvement (0 = off)")
    ap.add_argument("--early-stop-rel", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0, help="only the first N training streams")
    ap.add_argument("--val-limit", type=int, default=0, help="only the first N validation streams")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0, help="initialisation; other than 0 also shuffles the stream order")
    ap.add_argument("--resume", default="auto", choices=["auto", "never"])
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    data = Path(a.data)
    if a.roots:
        roots = json.loads(a.roots)
    else:
        roots = json.loads(Path(a.stats or data / "stats.json").read_text())["frame_roots"]
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(a.model)
    train_eps = [json.loads(l) for l in open(data / "train.jsonl")]
    val_eps = [json.loads(l) for l in open(data / "val_poll.jsonl")]
    if a.limit:
        train_eps = train_eps[: a.limit]
    if a.val_limit:
        val_eps = val_eps[: a.val_limit]

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    out = Path(a.out or f"runs/{a.name}")
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------- model
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    patch_peft()
    qwen = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to(device)
    qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    qwen.enable_input_require_grads()
    qwen = get_peft_model(
        qwen,
        LoraConfig(
            r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=LORA_TARGETS
        ),
    )
    geom = inject = None
    if a.geometry:
        from prospace_vlm.geometry import Cut3r

        geom = Cut3r(cache_dir=a.geom_cache).to(device)
        inject = GeometryInjection(
            qwen.config.get_text_config().hidden_size, layers=tuple(int(v) for v in a.layers.split(","))
        ).to(device)
    model = ProSpaceVLM(qwen, geom, inject)

    groups = [{"params": [p for p in qwen.parameters() if p.requires_grad], "lr": a.lr}]
    if inject is not None:
        groups.append({"params": list(inject.parameters()), "lr": a.geom_lr})
    if rank == 0:
        print(
            f"trainable: LoRA {sum(p.numel() for p in groups[0]['params']) / 1e6:.1f}M"
            + (
                f", geometry MLPs {sum(p.numel() for p in inject.parameters()) / 1e6:.1f}M (layers {a.layers})"
                if inject is not None
                else ""
            ),
            flush=True,
        )
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.999))
    net = (
        torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
        if world > 1
        else model
    )

    # ------------------------------------------------------------------------- data split and schedule
    order = list(range(len(train_eps)))
    if a.seed:
        random.Random(a.seed).shuffle(order)  # seed 0 keeps the file order
    train_eps = [train_eps[i] for i in order]
    mine = train_eps[rank::world]  # GPU r reads streams r, r + world, ...
    n_mine = len(mine)
    if world > 1:
        t_n = torch.tensor([n_mine], device=device)
        torch.distributed.all_reduce(t_n, op=torch.distributed.ReduceOp.MIN)
        n_mine = int(t_n.item())
    n_mine -= n_mine % a.grad_accum
    mine = mine[:n_mine]
    total_steps = n_mine // a.grad_accum
    # linear warm-up over the first 3% of the steps, then cosine decay to 0
    schedule = (
        lambda s: min(1.0, (s + 1) / max(1, int(0.03 * total_steps)))
        * 0.5
        * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps))))
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    wb = None
    if rank == 0 and not a.no_wandb:
        import wandb

        wb = wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project="prospace",
            name=a.name,
            config={**vars(a), "world": world, "total_steps": total_steps},
        )
    log = (out / "log.jsonl").open("a") if rank == 0 else None

    def emit(metrics, step):
        if rank == 0:
            log.write(json.dumps({"step": step, **metrics}) + "\n")
            log.flush()
            if wb:
                wb.log(metrics, step=step)
            print(
                step, {k: round(v, 4) if isinstance(v, float) else v for k, v in list(metrics.items())[:9]}, flush=True
            )

    # ------------------------------------------------------------------------- checkpoints
    def weights() -> dict:
        """prospace_vlm.pt, next to the PEFT adapter: the run's arguments and the per-layer MLPs ("memory")."""
        extra = {"args": vars(a)}
        if inject is not None:
            extra["memory"] = inject.state_dict()
        return extra

    def save(tag):
        if rank != 0:
            return
        qwen.save_pretrained(out / tag)
        torch.save(weights(), out / tag / "prospace_vlm.pt")

    es = {"best": float("inf"), "best_step": 0, "stale": 0}

    def save_state(step):
        """<out>/last: weights + optimizer, schedule, step, early-stop state.  Written to last.tmp, then renamed."""
        if rank == 0:
            tmp, dst = out / "last.tmp", out / "last"
            shutil.rmtree(tmp, ignore_errors=True)
            qwen.save_pretrained(tmp)
            torch.save(weights(), tmp / "prospace_vlm.pt")
            torch.save(
                {
                    "step": step,
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "es": dict(es),
                    "world": world,
                    "grad_accum": a.grad_accum,
                },
                tmp / "state.pt",
            )
            shutil.rmtree(dst, ignore_errors=True)
            tmp.rename(dst)
        if world > 1:
            torch.distributed.barrier()

    start_step = 0
    last = out / "last" / "state.pt"
    if a.resume == "auto" and last.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        st = torch.load(last, map_location="cpu", weights_only=False)
        assert (
            st["world"] * st["grad_accum"] == world * a.grad_accum
        ), "resume needs the same effective batch (GPUs x accumulation)"
        set_peft_model_state_dict(qwen, load_file(str(out / "last" / "adapter_model.safetensors")))
        if inject is not None:
            extra = torch.load(out / "last" / "prospace_vlm.pt", map_location="cpu", weights_only=False)
            inject.load_state_dict(extra["memory"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        es.update(st["es"])
        start_step = int(st["step"])
        if rank == 0:
            print(
                f"RESUMED from {last} at step {start_step} (best {es['best']:.4f} at step {es['best_step']})",
                flush=True,
            )

    def evaluate(step) -> bool:
        """Validation; keeps <out>/best.  True = stop early."""
        rows = validate(model, val_eps, processor, roots, device, rank, world, a.workers)
        metrics = summarize_by_request(gather(rows, world))
        stop = False
        if step > 0:
            vl = metrics["valid/loss"]
            if vl < es["best"]:
                save("best")
            es["stale"] = 0 if vl < es["best"] * (1 - a.early_stop_rel) else es["stale"] + 1
            if vl < es["best"]:
                es["best"], es["best_step"] = vl, step
            metrics.update(
                {
                    "early_stop/best_val_loss": es["best"],
                    "early_stop/best_step": es["best_step"],
                    "early_stop/stale": es["stale"],
                }
            )
            stop = a.early_stop > 0 and es["stale"] >= a.early_stop
        emit(metrics, step)
        return stop

    # ------------------------------------------------------------------------- training
    step = start_step
    if start_step == 0:
        evaluate(0)
    mine = mine[start_step * a.grad_accum :]  # a resumed run: every GPU continues at its own next stream
    model.train()
    t0 = time.time()
    acc = Counter()
    n_acc = 0
    t_mark = time.time()
    for i, item in enumerate(loader(mine, processor, roots, a.workers, train=True)):
        acc["time/data_wait_s"] += time.time() - t_mark
        t_mark = time.time()
        sync = (i + 1) % a.grad_accum == 0
        # gradients are accumulated locally and exchanged between the GPUs on the last stream of the step
        ctx = net.no_sync() if (world > 1 and not sync) else torch.autocast("cuda", enabled=False)
        with ctx:
            out_ = net(item, device)
            torch.cuda.synchronize()
            acc["time/forward_s"] += time.time() - t_mark
            acc["time/geom_s"] += out_["geom_s"]
            t_mark = time.time()
            loss, m = loss_of(out_, item, device)
            (loss / a.grad_accum).backward()
            torch.cuda.synchronize()
            acc["time/backward_s"] += time.time() - t_mark
            t_mark = time.time()
        for k, v in m.items():
            acc[k] += v
        acc["frames"] += len(item["ep"]["frames"])
        n_acc += 1
        acc["time/tokens"] += int(item["inputs"]["input_ids"].shape[1])
        if not sync:
            continue
        for gi, g in enumerate(groups):  # gradient norm per parameter group, before clipping
            sq = sum(float((p.grad.float() ** 2).sum()) for p in g["params"] if p.grad is not None)
            acc[f"gnorm_group{gi}"] += n_acc * math.sqrt(sq)
        # each group is clipped to norm 1 on its own: the LoRA update is clipped as in the baseline
        gns = [torch.nn.utils.clip_grad_norm_(g["params"], 1.0) for g in groups]
        gn = torch.sqrt(sum(x.float() ** 2 for x in gns))
        if not torch.isfinite(gn):  # non-finite gradients: skip the update
            opt.zero_grad(set_to_none=True)
            acc["train/skipped_nonfinite"] += 1
            if rank == 0:
                print(f"step {step + 1}: non-finite grad norm, update skipped", flush=True)
        else:
            opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        emit(
            {
                **{(k if k.startswith("time/") else f"train/{k}"): v / n_acc for k, v in acc.items()},
                "train/lr": sched.get_last_lr()[0],
                "train/grad_norm": float(gn),
                "train/episodes_per_s": n_acc * world / (time.time() - t0),
                "train/progress": step / max(1, total_steps),
                "train/peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
            step,
        )
        t0 = time.time()
        acc = Counter()
        n_acc = 0
        t_mark = time.time()
        stop_now = step % a.val_every == 0 and evaluate(step)
        if step % (a.save_every or a.val_every) == 0:
            save_state(step)
        if a.keep_every and step % a.keep_every == 0:
            save(f"step_{step:05d}")
        if a.max_steps and step >= a.max_steps:
            if rank == 0:
                print(f"--max-steps {a.max_steps}: stop", flush=True)
            break
        if stop_now:
            if rank == 0:
                print(f"EARLY STOP at step {step}; best {es['best']:.4f} at step {es['best_step']}", flush=True)
            break
    else:
        evaluate(step)
    save("final")
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
