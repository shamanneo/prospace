"""Score one model's records on ProSpace-Bench.

    python scripts/score.py results/prospace_vlm_repro           # a folder of .jsonl files
    python scripts/score.py results/my_model.jsonl           # or one .jsonl file
    python scripts/score.py results/my_run --partial             # score only the streams that have a record

Each record needs task, episode_id and the response second (response_i or fire; null = never responded).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prospace_bench import scoring as S  # noqa: E402
from prospace_bench.data import fetch  # noqa: E402

E = S.load_bench(fetch("bench", "prospace_bench.jsonl"))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
partial = "--partial" in sys.argv[1:]
records = S.read_records(args[0])
if not partial and len({(r["task"], r["episode_id"]) for r in records}) < sum(len(v) for v in E.values()):
    sys.exit("some streams have no record yet; finish the run, or add --partial to score only the streams that have one")
c, counts = S.score(records, E, require_all=not partial)
kinds = ["timely", "premature", "delayed", "miss", "correct_abstention", "false_alarm", "invalid"]
print(
    f"{'task':18s} {'timely / abstention':>20s}   "
    + " ".join(f"{k:>10s}" for k in ["timely", "premature", "delayed", "miss", "abstain", "false_alarm", "invalid"])
)
for t in S.TASKS:
    print(f"{t:18s} {S.fmt(c[t]):>20s}   " + " ".join(f"{counts[t].get(k, 0):10d}" for k in kinds))
print(f"{'average':18s} {S.fmt(c['average']):>20s}" + ("   (partial: only streams with a record)" if partial else ""))
