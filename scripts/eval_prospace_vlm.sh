#!/usr/bin/env bash
# ProSpace-VLM on ProSpace-Bench (3,636 streams): one shard per GPU, per-second records written as they come (resumable).
#   bash scripts/eval_prospace_vlm.sh [gpus, default 0] [checkpoint, default checkpoints/prospace_vlm.pth]
# With LIMIT=<n> (the first n streams per GPU) only the frames of those streams have to exist.
# Output: results/prospace_vlm_repro/verdicts.shard*.jsonl, eval_bench.json and the Table 1 row on the screen.
# TOKEN_SELECTION=32:16 runs with Spatial Token Selection (a KV cache of the request + at most 32 frames of tokens, the
# last 16 frames whole); use another OUT for it.
set -eu
cd "$(dirname "$0")/.."
GPUS=${1:-0}; CK=${2:-checkpoints/prospace_vlm.pth}; OUT=${OUT:-results/prospace_vlm_repro}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/build_streams.py $([ -n "${LIMIT:-}" ] || echo --check) > cache_streams.log 2>&1 || { cat cache_streams.log; exit 1; }
mv cache_streams.log cache/streams/build.log
mkdir -p "$OUT" cache/geom_cut3r
n=$(echo "$GPUS" | tr ',' '\n' | wc -l); k=0
for g in $(echo "$GPUS" | tr ',' ' '); do
  CUDA_VISIBLE_DEVICES=$g python -m prospace_vlm.eval --data cache/streams --ckpt "$CK" --geom-cache cache/geom_cut3r \
    --shard $k/$n --out "$OUT" ${LIMIT:+--limit $LIMIT} ${TOKEN_SELECTION:+--token-selection $TOKEN_SELECTION} > "$OUT/shard$k.log" 2>&1 &
  k=$((k + 1))
done
wait
grep -L BENCH_SHARD_DONE "$OUT"/shard*.log | sed 's/^/NOT FINISHED (run the same command again to resume): /'
python -m prospace_vlm.eval --out "$OUT" --merge | tee "$OUT/summary.txt"
