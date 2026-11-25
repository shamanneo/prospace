#!/usr/bin/env bash
# Evaluate a model on ProSpace-Bench by its name.
#   bash scripts/eval.sh <model> [gpus, default 0] [weights]
#   bash scripts/eval.sh prospace-vlm 0,1,2,3                          # the released checkpoint
#   bash scripts/eval.sh qwen3-vl-8b 0,1,2,3                           # a zero-shot row
#   bash scripts/eval.sh prospace-vlm 0,1,2,3 runs/prospace-vlm/best   # your own training run
#   bash scripts/eval.sh --list                                        # all model names
# Results go to results/<model> (or results/<model>-<run> with your own weights) and the row is printed at the end.
# Every run is resumable: run the same command again if it stops.  LIMIT=<n> runs the first n streams per GPU.
set -eu
cd "$(dirname "$0")/.."

# name | how | arguments
MODELS="
prospace-vlm            ours       checkpoints/prospace_vlm.pth
prospace-vlm-tss        ours_tss   checkpoints/prospace_vlm.pth
qwen2.5-vl-7b-streaming ours       -
qwen3-vl-8b-polling     polling    qwen3 Qwen/Qwen3-VL-8B-Instruct
qwen2.5-vl-7b-polling   polling    qwen25 Qwen/Qwen2.5-VL-7B-Instruct
qwen2.5-vl-7b           polling    qwen25 Qwen/Qwen2.5-VL-7B-Instruct
qwen2.5-vl-32b          polling    qwen25 Qwen/Qwen2.5-VL-32B-Instruct
qwen3-vl-2b             polling    qwen3 Qwen/Qwen3-VL-2B-Instruct
qwen3-vl-8b             polling    qwen3 Qwen/Qwen3-VL-8B-Instruct
qwen3-vl-32b            polling    qwen3 Qwen/Qwen3-VL-32B-Instruct
qwen3-vl-30b-a3b        polling    qwen3 Qwen/Qwen3-VL-30B-A3B-Instruct
gemma-3-27b             polling    images google/gemma-3-27b-it
internvideo3-8b         polling    internvideo3 yanziang/InternVideo3-8B-Instruct
cosmos-reason2-8b       polling    qwen3 nvidia/Cosmos-Reason2-8B
cosmos-reason2-32b      polling    qwen3 nvidia/Cosmos-Reason2-32B
sensenova-si-1.1-8b     polling    qwen3 sensenova/SenseNova-SI-1.1-Qwen3-VL-8B
robobrain2.5-4b         polling    qwen3 BAAI/RoboBrain2.5-4B
robobrain2.5-8b-nv      polling    qwen3 BAAI/RoboBrain2.5-8B-NV
vlm-3r                  polling    vlm3r_video Journey9ni/vlm-3r-llava-qwen2-lora
cambrian-s-7b           polling    cambrians_video nyu-visionx/Cambrian-S-7B
spatial-ttt-nano        polling    spatial_ttt THU-SI/Spatial-TTT-nano
timechat-online-7b      streaming  timechat_online
livecc-7b               streaming  livecc
dispider                streaming  dispider
flash-vstream-7b        streaming  flash_vstream
mmduet                  streaming  mmduet
stride-2b               streaming  stride
stream3d-vlm-4b         streaming  stream3d_vlm
"

if [ "${1:-}" = "" ] || [ "$1" = "--list" ]; then
  echo "Models (see runners/README.md for the environment each one needs):"
  echo "$MODELS" | awk 'NF {k = $2 == "polling" ? "prefix polling" : $2 == "streaming" ? "native streaming" : "ProSpace-VLM evaluator"; printf "  %-24s %s\n", $1, k}'
  exit 0
fi
NAME=$1; GPUS=${2:-0}; WEIGHTS=${3:-}
row=$(echo "$MODELS" | awk -v n="$NAME" '$1 == n')
[ -n "$row" ] || { echo "unknown model: $NAME (bash scripts/eval.sh --list)"; exit 1; }
set -- $row
HOW=$2; shift 2
TAG=$NAME; [ -n "$WEIGHTS" ] && TAG=$NAME-$(basename "$(dirname "$WEIGHTS")")
export OUT=${OUT:-results/$TAG}

case $HOW in
  ours | ours_tss)
    CK=${WEIGHTS:-$1}
    [ "$CK" != - ] || { echo "$NAME needs the weights of your training run as the third argument"; exit 1; }
    [ $HOW = ours_tss ] && export TOKEN_SELECTION=${TOKEN_SELECTION:-32:16}
    exec bash scripts/eval_prospace_vlm.sh "$GPUS" "$CK" ;;
  polling)
    [ -n "$WEIGHTS" ] && export ADAPTER=$WEIGHTS
    exec bash scripts/eval_polling.sh "$TAG" "$1" "$2" "$GPUS" ;;
  streaming)
    exec bash scripts/eval_streaming.sh "$TAG" "$1" "${GPUS%%,*}" ;;
esac
