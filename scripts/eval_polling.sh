#!/usr/bin/env bash
# Prefix polling on ProSpace-Bench (the zero-shot protocol of Table 1) for an open model, one shard per GPU, resumable.
# Every second t the model sees the question and frames 0..t (512x384, 1 fps) and answers Yes / No; the first Yes is its response.
#   bash scripts/eval_polling.sh <tag> <family> <model> [gpus, default 0]
#   bash scripts/eval_polling.sh qwen25vl_7b qwen25 Qwen/Qwen2.5-VL-7B-Instruct 0,1,2,3
# Families and the Table 1 rows they give (run each in an environment that can load the model):
#   qwen25           Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen2.5-VL-32B-Instruct
#   qwen3            Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3-VL-32B-Instruct, nvidia/Cosmos-Reason2-32B,
#                    sensenova/SenseNova-SI-1.1-Qwen3-VL-8B
#   images           any other Hugging Face chat model (frames as an image list)
#   internvideo3     yanziang/InternVideo3-8B-Instruct      needs transformers 4.57 (its remote code)
#   vlm3r_video      Journey9ni/vlm-3r-llava-qwen2-lora     needs third_party/VLM-3R      (VITA-Group/VLM-3R @ 206afd0)
#   cambrians_video  nyu-visionx/Cambrian-S-7B              needs third_party/cambrian-s  (cambrian-mllm/cambrian-s @ 058b1b4)
#   spatial_ttt      THU-SI/Spatial-TTT-nano                needs third_party/Spatial-TTT (THU-SI/Spatial-TTT @ e2e33a6)
# Output: results/<tag>/<task>.s<k>.jsonl; then  python scripts/score.py results/<tag>
# EPISODES=<file> runs another episode file (same format), LIMIT=<n> only the first n streams of each task.
# ADAPTER=<dir> merges a LoRA adapter from scripts/train_polling.sh into the model first (qwen25 / qwen3 families).
set -eu
cd "$(dirname "$0")/.."
TAG=$1; FAMILY=$2; MODEL=$3; GPUS=${4:-0}; OUT=${OUT:-results/$TAG}; EPISODES=${EPISODES:-data/prospace_bench.jsonl}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
case $FAMILY in
  vlm3r_video | cambrians_video | spatial_ttt)
    # Spatial-TTT loads its weights from a local folder
    [ $FAMILY = spatial_ttt ] && [ ! -d "$MODEL" ] && MODEL=$(python -c "from huggingface_hub import snapshot_download as d; print(d('$MODEL'))")
    run="runners/offline_spatial.py --backend $FAMILY" ;;
  internvideo3) run="runners/internvideo3.py" ;;
  *) run="runners/vlm.py --family $FAMILY --mode polling --style state --resume${ADAPTER:+ --adapter $ADAPTER}" ;;
esac
n=$(echo "$GPUS" | tr ',' '\n' | wc -l)
for t in object_finding object_refinding object_direction ego_rotation origin_return visual_homing place_transition place_entry place_revisit; do
  task=$t; extra=""
  [ $t = ego_rotation ] && task=ego_rotation_lrb                       # the harness name of the benchmark's Ego Rotation prompt
  [ $t = object_refinding ] && extra="--score-from-field gap_start_i"   # answers during the initial showing are not scored
  k=0
  for g in $(echo "$GPUS" | tr ',' ' '); do
    CUDA_VISIBLE_DEVICES=$g python $run --model "$MODEL" --task $task --step 1 $extra \
      --episodes "$EPISODES" --frames-by-dataset --all ${LIMIT:+--limit $LIMIT} --shard $k/$n --out "$OUT/$t.s$k.jsonl" > "$OUT/$t.s$k.log" 2>&1 &
    k=$((k + 1))
  done
  wait
  echo "$t done: $(cat "$OUT"/$t.s*.jsonl | wc -l) records"
done
partial=${LIMIT:+--partial}; [ "$EPISODES" != data/prospace_bench.jsonl ] && partial=--partial
python scripts/score.py "$OUT" $partial
