#!/usr/bin/env bash
# Native-streaming SFT on ProSpace-Train: ProSpace-VLM, or the Qwen2.5-VL-7B baseline (the same trainer without geometry).
#   bash scripts/train_streaming.sh <prospace_vlm | qwen25vl> [gpus, default 0,1,2,3] [set: 100k | 400k, default 100k]
# The settings of the paper rows: LoRA r 16, lr 1e-4, 16 streams per optimizer step (1, 2, 4, 8 or 16 GPUs), one epoch,
# validation and a checkpoint every 250 steps.  Output: runs/<name>/best (lowest validation loss), final, log.jsonl.
# Run the same command again to resume.  Curves go to wandb with WANDB=1 (entity from WANDB_ENTITY).
# Benchmark afterwards:  bash scripts/eval_prospace_vlm.sh 0,1,2,3 runs/<name>/best
set -eu
cd "$(dirname "$0")/.."
ROW=$1; GPUS=${2:-0,1,2,3}; SET=${3:-100k}; NAME=${NAME:-${ROW}_$SET}; DATA=cache/train_$SET
case $ROW in
  prospace_vlm) ARGS="--geometry --layers 4,8,12,16 --geom-lr 1e-3" ;;
  qwen25vl) ARGS="" ;;
  *) echo "first argument: prospace_vlm | qwen25vl"; exit 1 ;;
esac
[ -f $DATA/train.jsonl ] || python scripts/prepare_train.py --set $SET
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p cache/geom_cut3r_train runs
python -m torch.distributed.run --standalone --nproc_per_node $NG --master_port ${PORT:-29517} \
  -m prospace_vlm.train --data $DATA --geom-cache cache/geom_cut3r_train $ARGS \
  --grad-accum $((16 / NG)) --val-every 250 --keep-every 250 --workers 6 \
  --name $NAME --out runs/$NAME $([ -n "${WANDB:-}" ] || echo --no-wandb) ${EXTRA:-}
