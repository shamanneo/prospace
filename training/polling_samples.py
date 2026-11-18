"""Prefix-polling SFT samples from a ProSpace-Train stream set (the "SFT, prefix polling" rows).

The polling rows are trained on exactly the streams of the native-streaming rows; only the form of the supervision
differs.  One sample = (frames 0..t, question) -> "Yes" if second t is a moment to respond, else "No"
(runners/protocol.STATE_PROMPT, the prompt of the benchmark's prefix-polling runner).

    positive stream   Yes x1: 50 % uniform over the first 3 s of the timely window, 50 % uniform over the whole window
                      No  x2: hard = 2-3 s before the window start, random = a premature second drawn from the task's
                              Yes-time distribution (so that elapsed time alone cannot answer)
                      seconds after the window are never sampled
    absence stream    No x ABSENCE_NO (10 for the three large tasks, 2 otherwise), placed where the task's P(Yes | t)
                      is still too high
    mid-request copy  (q_at = P > 0) the prompt counts seconds from the start of the video and the question carries the
                      request-time wording (mid_text)

Every stream is used once; the file is shuffled over all tasks.  The draw is seeded (SEED), so the output is reproducible.

Output (--out): train_full.jsonl (training/train_polling.py format), val.jsonl (the validation seconds of
val_es.jsonl, one sample per (stream, second)), val_polling_episodes.jsonl (the validation streams of val_poll.jsonl),
stats.json (frame_roots + counts).

    python -m training.polling_samples --src <stream set dir> --stats <stats.json with frame_roots> --out <dir> \
        [--orig-src <larger set's train.jsonl(.gz)>]
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.protocol import FRAME_W, STATE_PROMPT  # noqa: E402

EARLY_S = 3
HARD_BACK = (2, 3)
SEED = 2027
ABSENCE_NO = {"ego_rotation": 10, "object_finding": 10, "object_direction": 10}  # seconds asked per absence stream

# Mid-request copies: this sentence goes before the question, and the question's references to the first frame / the
# start are re-anchored to the request.
REQUEST_SENTENCE = (
    "The request below was given at second {q} of this video; the frames before it are earlier "
    "observations. Treat the moment the request was given as the reference."
)
MID_REPLACE = [
    (
        "the first frame is the view from where you start",
        "the frame at second {q}, when the request was given, is the view from where you start",
    ),
    (
        "The first frame is the view from the starting spot",
        "The frame at second {q}, when the request was given, is the view from the starting spot",
    ),
    ("The red box in the first frame", "The red box in the frame at second {q}, when the request was given,"),
    ("Since the first frame", "Since second {q}, when the request was given"),
    ("in the first frame", "at second {q}, when the request was given"),
    ("of the first frame", "of the frame at second {q}, when the request was given"),
    ("the first frame", "the frame at second {q}, when the request was given"),
    ("from its viewing direction at the start", "from its viewing direction when the request was given"),
    ("returns to its starting location", "returns to the location where it was when the request was given"),
    ("The first frame", "The frame at second {q}, when the request was given,"),
]


def mid_text(text: str, q: int, sentence: bool = True, table=MID_REPLACE) -> str:
    if not q:
        return text
    for a, b in table:
        text = text.replace(a, b.format(q=q))
    return (REQUEST_SENTENCE.format(q=q) + " " + text) if sentence else text


def adapt(e: dict) -> dict:
    """Stream record -> the fields the samplers read (window as timely_start_i / timely_end_i).
    A copy with q_at = P carries P context frames before the request.  The samplers see the stream FROM the request
    (frames[P:], window shifted back by P), so every asked second t is >= 0 relative to the request, exactly as in the
    streaming runs (nothing is asked before the question); `sample` then hands the model all frames up to that second,
    context included."""
    e = dict(e)
    P = int(e.get("q_at", 0) or 0)
    if P:
        e["frames_full"], e["frames"], e["frame_ids"] = e["frames"], e["frames"][P:], e["frame_ids"][P:]
        e["w0"], e["w1"] = (None if e["w0"] is None else e["w0"] - P), (None if e["w1"] is None else e["w1"] - P)
        e["start"] = int(e.get("start", 0)) - P
    e["timely_start_i"], e["timely_end_i"] = e["w0"], e["w1"]
    e["event_id"] = e["episode_id"]
    e["question_id"] = f"{e['task']}:given"  # the stream's own question, no paraphrase draw
    return e


def sample(ep: dict, t: int, answer: str, kind: str, rng: random.Random) -> dict:
    """One training record; the frames and the question are the stream's own.
    Mid-request copies (q_at = P > 0): the prompt's second count is the ABSOLUTE position of the last frame (P + t, as
    the polling runner counts at evaluation)."""
    P = int(ep.get("q_at", 0) or 0)
    text = STATE_PROMPT.format(question=ep["question"], t=P + t, pos="Yes", neg="No")
    return {
        "id": f"{ep['episode_id']}@{t}",
        "task": ep["task"],
        "dataset": ep["dataset"],
        "scene_id": ep["scene_id"],
        "event_id": ep["event_id"],
        "episode_kind": "absence" if ep["timely_start_i"] is None else "positive",
        "sample_type": kind,
        "t": int(t),
        "answer": answer,
        "question_id": ep["question_id"],
        "direction": ep.get("direction"),
        "orientation": ep.get("orientation"),
        "frames": (
            ep["frames_full"][: int(ep.get("q_at", 0) or 0) + t + 1] if "frames_full" in ep else ep["frames"][: t + 1]
        ),
        "q_at": int(ep.get("q_at", 0) or 0),
        "frame0_bbox": ep.get("frame0_bbox"),
        "messages": [{"role": "user", "content": text}, {"role": "assistant", "content": answer}],
    }


def positive_samples(ep: dict, rng: random.Random, yes_times: list[int] | None = None) -> list[dict]:
    ts, te = ep["timely_start_i"], ep["timely_end_i"]
    early = rng.random() < 0.5
    ty = rng.randint(ts, min(te, ts + EARLY_S - 1)) if early else rng.randint(ts, te)
    out = [sample(ep, ty, "Yes", "yes_early" if early else "yes_window", rng)]
    th = ts - rng.choice(HARD_BACK)
    if th >= 0:
        out.append(sample(ep, th, "No", "no_hard", rng))
    pool = [t for t in range(0, max(0, ts - HARD_BACK[0])) if t != th]
    if pool:
        # the random No follows the task's Yes-time distribution (truncated to this premature stretch) instead of
        # uniform, so that P(Yes | t) is flat and elapsed time alone cannot answer
        cand = [t for t in (yes_times or []) if t in set(pool)]
        out.append(sample(ep, rng.choice(cand) if cand else rng.choice(pool), "No", "no_random", rng))
    return out


_NP_RNG: dict = {}


def np_rng(rng: random.Random) -> np.random.Generator:
    if id(rng) not in _NP_RNG:
        _NP_RNG[id(rng)] = np.random.default_rng(rng.randint(0, 2**31 - 1))
    return _NP_RNG[id(rng)]


def build_full(pos: dict, neg: dict, rng: random.Random, cap: dict | None = None) -> list[dict]:
    """EVERY stream exactly once (the whole training set, one epoch, natural task proportions).
    A positive gives its Yes, its hard No and its random No; an absence stream gives No's whose times fill the seconds
    where this task's P(Yes | t) is still too high.  `cap` = {task: (n_pos, n_neg)} limits a split.
    The result is shuffled over all tasks, so a run that stops early has still seen every task in proportion."""
    out = []
    for task in list(pos) + [t for t in neg if t not in pos]:
        P, N = list(pos.get(task, [])), list(neg.get(task, []))
        if cap:
            rng.shuffle(P)
            rng.shuffle(N)
            P, N = P[: cap[task][0]], N[: cap[task][1]]
        if not P:
            continue
        yes_all = sorted(rng.randint(e["timely_start_i"], e["timely_end_i"]) for e in rng.sample(P, min(len(P), 5000)))
        mine = []
        for e in P:
            mine += positive_samples(e, rng, yes_all)
        if N:
            # Inside a positive every No precedes the Yes, so elapsed time alone predicts the answer; only the absence
            # streams can supply late No's.  With the natural proportions the three large tasks have too few of them,
            # so their absence streams are asked at ABSENCE_NO seconds instead of 2.  Each stream is still used once;
            # the seconds are drawn, within the stream's own length, where this task's P(Yes | t) is still too high.
            k = ABSENCE_NO.get(task, 2)
            yes = np.bincount([s["t"] for s in mine if s["answer"] == "Yes"])
            T = max(len(yes), max(len(e["frame_ids"]) for e in N))
            yes = np.pad(yes, (0, T - len(yes))).astype(float)
            no = np.bincount([s["t"] for s in mine if s["answer"] == "No"], minlength=T)[:T].astype(float)
            rate = yes.sum() / max(1.0, yes.sum() + no.sum() + k * len(N))
            deficit = np.maximum(yes * (1 - rate) / max(rate, 1e-9) - no, 0.0) + 1e-9
            g = np_rng(rng)
            for e in N:
                n = len(e["frame_ids"])
                pr = deficit[:n] / deficit[:n].sum()
                for tt in sorted(int(x) for x in g.choice(n, size=min(k, n), replace=False, p=pr)):
                    mine.append(sample(e, tt, "No", "no_absence", rng))
        out += mine
    rng.shuffle(out)
    return out


def time_auc(samples: list[dict]) -> dict:
    res = {}
    for task in sorted({s["task"] for s in samples}):
        y = np.array([s["t"] for s in samples if s["task"] == task and s["answer"] == "Yes"])
        n = np.array([s["t"] for s in samples if s["task"] == task and s["answer"] == "No"])
        if len(y) and len(n):
            g = np.random.default_rng(0)
            a, b = g.choice(y, 100000), g.choice(n, 100000)
            res[task] = round(float((a > b).mean() + 0.5 * (a == b).mean()), 3)
    return res


def describe(samples: list[dict]) -> dict:
    c = Counter((s["task"], s["answer"]) for s in samples)
    return {
        "samples": len(samples),
        "yes": sum(v for (t, a), v in c.items() if a == "Yes"),
        "no": sum(v for (t, a), v in c.items() if a == "No"),
        "by_task": {t: {"Yes": c[(t, "Yes")], "No": c[(t, "No")]} for t in sorted({k[0] for k in c})},
        "sample_types": dict(Counter(s["sample_type"] for s in samples)),
        "datasets": dict(Counter(s["dataset"] for s in samples)),
        "frames_mean": round(float(np.mean([len(s["frames"]) for s in samples])), 1),
        "frames_p95": int(np.percentile([len(s["frames"]) for s in samples], 95)),
        "time_only_auc": time_auc(samples),
    }


def boxed_rel(frame_rel: str, bbox) -> str:
    """'<scene>/frame_000123.jpg' -> '<scene>/frame_000123_box_x0_y0_x1_y1.jpg' (bbox in the 1920 x 1440 rule frame)."""
    stem = frame_rel[: -len(".jpg")]
    return f"{stem}_box_{'_'.join(str(int(round(v))) for v in bbox)}.jpg"


def draw_box(src: Path, bbox):
    """The benchmark runner's drawing (runners/protocol.episode_frames) on the full-size frame."""
    from PIL import Image, ImageDraw

    im = Image.open(src).convert("RGB")
    sx, sy = im.width / 1920, im.height / 1440
    lw = max(4, round(6 * im.width / FRAME_W))
    ImageDraw.Draw(im).rectangle(
        [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy], outline=(255, 0, 0), width=lw
    )
    return im


def box_on_request_frame(samples: list[dict], roots: dict) -> int:
    """Object Re-finding, mid-request copies: the red box marks the object on the REQUEST frame (index q_at), while the
    trainer draws `frame0_bbox` on frame 0.  The request frame of these samples is therefore replaced by a boxed file,
    written once next to the plain frame, and `frame0_bbox` is cleared."""
    todo = {}
    for s in samples:
        q = int(s.get("q_at", 0) or 0)
        if q and s.get("frame0_bbox") and s["task"] == "object_refinding":
            ds, rel = s["frames"][q].split("/", 1)
            new = boxed_rel(rel, s["frame0_bbox"])
            todo[Path(roots[ds]) / new] = (Path(roots[ds]) / rel, s["frame0_bbox"])
            s["frames"][q] = ds + "/" + new
            s["boxed_frame"] = q
            s["frame0_bbox"] = None
    for dst, (src, bbox) in todo.items():
        if not dst.exists():
            draw_box(src, bbox).save(dst, quality=95)
    return len(todo)


def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="stream set dir: train.jsonl, val_es.jsonl, val_poll.jsonl")
    ap.add_argument("--stats", required=True, help="json with frame_roots (dataset -> frame folder)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--orig-src",
        default=None,
        help="a larger set's train.jsonl(.gz) (or its dir): the first-frame originals of mid-request copies whose "
        "original is not in --src (ProSpace-Train-100K is a subset of ProSpace-Train-400K)",
    )
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = json.loads(Path(a.stats).read_text())["frame_roots"]

    rows = [json.loads(l) for l in open(src / "train.jsonl")]
    orig_q = {r["episode_id"]: r["question"] for r in rows if not int(r.get("q_at", 0) or 0)}
    if a.orig_src:
        op = Path(a.orig_src)
        op = op / "train.jsonl" if op.is_dir() else op
        for l in _open(op):
            r = json.loads(l)
            if not int(r.get("q_at", 0) or 0):
                orig_q.setdefault(r["episode_id"], r["question"])
    pos, neg = defaultdict(list), defaultdict(list)
    for r in rows:
        P = int(r.get("q_at", 0) or 0)
        if P:
            r = {**r, "question": mid_text(orig_q[r["orig_episode_id"]], P)}
        e = adapt(r)
        assert len(e["frames"]) == len(e["frame_ids"])
        (neg if e["w0"] is None else pos)[e["task"]].append(e)
    rng = random.Random(a.seed)
    S = build_full(pos, neg, rng)
    n_box = box_on_request_frame(S, roots)
    with (out / "train_full.jsonl").open("w") as f:
        for s in S:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    n_pos, n_neg = sum(map(len, pos.values())), sum(map(len, neg.values()))
    print(f"train: {len(S)} samples from {n_pos} positive + {n_neg} absence streams, {n_box} boxed frames", flush=True)

    # validation samples = the shared validation seconds of the streaming rows (val_es: val_t / val_answer per stream)
    V = []
    for l in open(src / "val_es.jsonl"):
        e = adapt(json.loads(l))
        for t, ans in zip(e.get("val_t", []), e.get("val_answer", [])):
            if t < len(e["frames"]):
                kind = ("yes_window" if ans == "Yes" else "no_hard") if e["w0"] is not None else "no_absence"
                V.append(sample(e, int(t), ans, kind, rng))
    rng.shuffle(V)
    with (out / "val.jsonl").open("w") as f:
        for s in V:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # polling streams (the validation streams of the streaming rows) in the trainer's polling record form
    n_poll = 0
    with (out / "val_polling_episodes.jsonl").open("w") as f:
        for l in open(src / "val_poll.jsonl"):
            e = adapt(json.loads(l))
            e["gap_start_i"] = e.get("start", 0)
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
            n_poll += 1

    stats = {
        "frame_roots": roots,
        "source": str(src),
        "protocol": "prefix samples of the stream set: every stream once, the stream's own question and frames",
        "streams": {"positive": {t: len(v) for t, v in pos.items()}, "absence": {t: len(v) for t, v in neg.items()}},
        "train_full": describe(S),
        "val": describe(V),
        "val_polling_episodes": n_poll,
        "train_prefix_over_120": sum(len(s["frames"]) > 120 for s in S),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(
        f"val samples {len(V)} | polling streams {n_poll} | prefixes over 120 frames {stats['train_prefix_over_120']} -> {out}"
    )


if __name__ == "__main__":
    main()
