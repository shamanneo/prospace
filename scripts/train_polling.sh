#!/usr/bin/env bash
# Prefix-polling SFT on ProSpace-Train: Qwen3-VL-8B on (frames 0..t, question) -> Yes / No samples of the same streams.
#   bash scripts/train_polling.sh [gpus, default 0,1,2,3] [set: 100k | 400k, default 100k]
# The settings of the paper row: LoRA r 16, lr 1e-4, 64 samples per optimizer step (1, 2, 4 or 8 GPUs), one epoch,
# validation every 500 steps.  Output: runs/<name>/adapter_best (lowest validation loss), log.jsonl.
# FAMILY=qwen25 trains Qwen2.5-VL-7B instead.  Curves go to wandb with WANDB=1 (entity from WANDB_ENTITY).
# Benchmark afterwards:  ADAPTER=runs/<name>/adapter_best bash scripts/eval_polling.sh <tag> qwen3 Qwen/Qwen3-VL-8B-Instruct 0,1,2,3
set -eu
cd "$(dirname "$0")/.."
GPUS=${1:-0,1,2,3}; SET=${2:-100k}; FAMILY=${FAMILY:-qwen3}; DATA=cache/train_${SET}_polling
case $FAMILY in
  qwen3) MODEL=Qwen/Qwen3-VL-8B-Instruct; NAME=${NAME:-qwen3vl8b_polling_$SET} ;;
  qwen25) MODEL=Qwen/Qwen2.5-VL-7B-Instruct; NAME=${NAME:-qwen25vl7b_polling_$SET}; export PROSPACE_QWEN_FAMILY=qwen25 ;;
  *) echo "FAMILY: qwen3 | qwen25"; exit 1 ;;
esac
[ -f $DATA/train_full.jsonl ] || python scripts/prepare_train.py --set $SET --polling
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p runs
python -m torch.distributed.run --standalone --nproc_per_node $NG --master_port ${PORT:-29518} \
  -m training.train_polling --name $NAME --out runs/$NAME --model $MODEL --data $DATA --train-file train_full.jsonl \
  --val-file val.jsonl --epochs 1 --micro-batch 8 --grad-accum $((64 / NG / 8)) --val-every 500 --val-n 1200 \
  --early-stop 100000 --early-stop-rel 0.01 --poll-every 2000 --poll-n 45 --save-every 500 --max-frames 200 \
  $([ -n "${WANDB:-}" ] || echo --no-wandb) ${EXTRA:-}
