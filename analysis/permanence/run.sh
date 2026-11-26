#!/usr/bin/env bash
# ProSpace-VLM states of the permanence streams (analysis/permanence/dump.py), one shard per GPU, resumable.
#   bash analysis/permanence/run.sh [gpus, default 0] [checkpoint, default checkpoints/prospace_vlm.pth]
set -eu
cd "$(dirname "$0")/../.."
GPUS=${1:-0}; CK=${2:-checkpoints/prospace_vlm.pth}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p cache/permanence/logs cache/geom_cut3r
n=$(echo "$GPUS" | tr ',' '\n' | wc -l); k=0
for g in $(echo "$GPUS" | tr ',' ' '); do
  CUDA_VISIBLE_DEVICES=$g python analysis/permanence/dump.py --ckpt "$CK" --shard $k/$n > cache/permanence/logs/dump$k.log 2>&1 &
  k=$((k + 1))
done
wait
grep -L DUMP_SHARD_DONE cache/permanence/logs/dump*.log | sed 's/^/NOT FINISHED (run the same command again to resume): /'
echo "states: $(ls cache/permanence/states/*.npz | wc -l) of 1790 runs"
