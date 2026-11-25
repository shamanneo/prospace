"""ProSpace-Bench scoring.

A stream is positive when it has a timely window [timely_start_i, timely_end_i] (seconds = frame indices at 1 fps) and
an absence stream otherwise. A model gives at most one response second per stream (None = it never responded).

  positive stream:  timely (inside the window) | premature (before) | delayed (after) | miss (no response)
  absence stream:   correct_abstention (no response) | false_alarm (any response)

A table cell is "timely % of the positive streams / correct-abstention % of the absence streams" for one task; the
average is the mean of the nine task cells.
"""

import json
from pathlib import Path

TASKS = [
    "object_finding",
    "object_refinding",
    "object_direction",
    "ego_rotation",
    "origin_return",
    "visual_homing",
    "place_transition",
    "place_entry",
    "place_revisit",
]  # the column order of Table 1
SHORT = dict(zip(TASKS, ["OF", "RF", "OD", "ER", "OR", "VH", "PT", "PE", "PR"]))


def load_bench(path=Path(__file__).resolve().parent.parent / "data/prospace_bench/prospace_bench.jsonl"):
    """{task: {episode_id: episode}} for the 3,636 streams (one per line of the benchmark file)."""
    E = {t: {} for t in TASKS}
    for line in open(path):
        e = json.loads(line)
        E[e["task"]][e["episode_id"]] = e
    return E


def is_positive(ep):
    return ep.get("timely_start_i") is not None


def verdict(ep, response_i):
    if not is_positive(ep):
        return "correct_abstention" if response_i is None else "false_alarm"
    if response_i is None:
        return "miss"
    return (
        "premature" if response_i < ep["timely_start_i"] else "delayed" if response_i > ep["timely_end_i"] else "timely"
    )


def read_records(path):
    """Records {task, episode_id, response_i (or fire)} from a .jsonl file or a folder of .jsonl files."""
    path = Path(path)
    files = sorted(f for f in path.glob("*.jsonl") if not f.name.endswith(".err.jsonl")) if path.is_dir() else [path]
    out = []
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if "response_i" not in r:
                    r["response_i"] = r["fire"]
                r["task"] = {"ego_rotation_lrb": "ego_rotation"}.get(
                    r["task"], r["task"]
                )  # the harness name of the task
                out.append(r)
    return out


def cells(correct, E):
    """correct {task: {episode_id: value in [0, 1]}} -> {task: (timely %, abstention % or None)} plus "average"."""
    pct = lambda v: 100 * sum(v) / len(v) if v else None
    out = {}
    for t in TASKS:
        out[t] = (
            pct([v for k, v in correct[t].items() if is_positive(E[t][k])]),
            pct([v for k, v in correct[t].items() if not is_positive(E[t][k])]),
        )
    mean = lambda v: None if None in v else sum(v) / len(v)  # the average needs all nine tasks
    out["average"] = (mean([out[t][0] for t in TASKS]), mean([out[t][1] for t in TASKS]))
    return out


def score(records, E, require_all=True):
    """Score one model. A record marked verdict == "invalid" (no parsable output) counts as wrong."""
    correct, counts = {t: {} for t in TASKS}, {t: {} for t in TASKS}
    for r in records:
        t, k = r["task"], r["episode_id"]
        if k not in E[t]:
            continue
        v = "invalid" if r.get("verdict") == "invalid" else verdict(E[t][k], r["response_i"])
        correct[t][k] = float(v in ("timely", "correct_abstention"))
        counts[t][v] = counts[t].get(v, 0) + 1
    if require_all:
        miss = {t: len(E[t]) - len(correct[t]) for t in TASKS if len(correct[t]) != len(E[t])}
        assert not miss, f"streams without a record: {miss}"
    return cells(correct, E), counts


def score_human(records, E):
    """Records carry one response second per annotator; a stream's value is the share of annotators who were right."""
    correct = {t: {} for t in TASKS}
    for r in records:
        e = E[r["task"]][r["episode_id"]]
        vs = [verdict(e, x) for x in r["response_i"]]
        correct[r["task"]][r["episode_id"]] = sum(v in ("timely", "correct_abstention") for v in vs) / len(vs)
    return cells(correct, E)


def random_baseline(E):
    """A response second drawn uniformly over the stream: expected timely rate (positive streams only)."""
    correct = {
        t: {
            k: (e["timely_end_i"] - e["timely_start_i"] + 1) / len(e["frame_ids"])
            for k, e in E[t].items()
            if is_positive(e)
        }
        for t in TASKS
    }
    return cells(correct, E)


def fixed_time_oracle(E):
    """The single best response second per task (chosen on the benchmark itself; clamped to the last second of short
    streams)."""
    correct, best = {}, {}
    for t in TASKS:
        pos = {k: e for k, e in E[t].items() if is_positive(e)}
        hit = lambda s: {
            k: float(e["timely_start_i"] <= min(s, len(e["frame_ids"]) - 1) <= e["timely_end_i"])
            for k, e in pos.items()
        }
        best[t] = max(range(max(len(e["frame_ids"]) for e in pos.values())), key=lambda s: sum(hit(s).values()))
        correct[t] = hit(best[t])
    return cells(correct, E), best


def fmt(cell):
    return " / ".join("-" if x is None else f"{x:.1f}" for x in cell)
