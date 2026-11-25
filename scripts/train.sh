#!/usr/bin/env bash
# Fine-tune a model on ProSpace-Train by its name, with the settings of the paper rows.
#   bash scripts/train.sh <model> [gpus, default 0,1,2,3] [set: 100k | 400k, default 100k]
#   prospace-vlm               ProSpace-VLM (native streaming, with geometry)
#   qwen2.5-vl-7b-streaming    Qwen2.5-VL-7B, the same native-streaming trainer without geometry
#   qwen3-vl-8b-polling        Qwen3-VL-8B, prefix polling
#   qwen2.5-vl-7b-polling      Qwen2.5-VL-7B, prefix polling
# Output: runs/<model>/.  Run the same command again to resume.  The command to evaluate the result is printed at the end.
set -eu
cd "$(dirname "$0")/.."
MODEL=${1:-}; GPUS=${2:-0,1,2,3}; SET=${3:-100k}
export NAME=${RUN_NAME:-$MODEL}
case $MODEL in
  prospace-vlm)            bash scripts/train_streaming.sh prospace_vlm "$GPUS" "$SET"; W=runs/$NAME/best ;;
  qwen2.5-vl-7b-streaming) bash scripts/train_streaming.sh qwen25vl "$GPUS" "$SET"; W=runs/$NAME/best ;;
  qwen3-vl-8b-polling)     FAMILY=qwen3 bash scripts/train_polling.sh "$GPUS" "$SET"; W=runs/$NAME/adapter_best ;;
  qwen2.5-vl-7b-polling)   FAMILY=qwen25 bash scripts/train_polling.sh "$GPUS" "$SET"; W=runs/$NAME/adapter_best ;;
  *) sed -n '3,7p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
echo
echo "Evaluate it with:  bash scripts/eval.sh $MODEL $GPUS $W"
