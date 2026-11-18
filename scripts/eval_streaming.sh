#!/usr/bin/env bash
# Native streaming on ProSpace-Bench (the streaming rows of Table 1): each model runs its own streaming inference on the
# 1 fps stream with the request given at 0 s; the second of its first response is scored.  One GPU, resumable.
#   bash scripts/eval_streaming.sh <tag> <model: timechat_online | livecc | dispider | stream3d_vlm> [gpu, default 0]
#   bash scripts/eval_streaming.sh stream3d_vlm_4b stream3d_vlm 0
# Run it in the model's own environment, with the model's repository under third_party/:
#   timechat_online  wyccccc/TimeChatOnline-7B     third_party/TimeChat-Online  (yaolinli/TimeChat-Online @ 25a1578)
#   livecc           chenjoya/LiveCC-7B-Instruct   third_party/livecc           (showlab/livecc @ fdaa91e)
#   dispider         Mar2Ding/Dispider             third_party/Dispider         (Mark12Ding/Dispider @ 803f6b4)
#   stream3d_vlm     JonnyYu828/Stream3D-VLM-4B    third_party/Stream3D-VLM     (hanxunyu/Stream3D-VLM @ 6a7929a,
#                                                  then  git apply ../../patches/stream3d_vlm_eval.patch)
# Output: results/<tag>/<task>.jsonl; then  python scripts/score.py results/<tag>
# EPISODES=<file> runs another episode file (same format), TASKS="<task> ..." only some tasks (to spread tasks over GPUs).
set -eu
cd "$(dirname "$0")/.."
TAG=$1; MODEL=$2; GPU=${3:-0}; OUT=${OUT:-results/$TAG}; EPISODES=${EPISODES:-data/prospace_bench.jsonl}
TASKS=${TASKS:-object_finding object_refinding object_direction ego_rotation origin_return visual_homing place_transition place_entry place_revisit}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PROSPACE_FRAMES_BY_DATASET=1 CUDA_VISIBLE_DEVICES=$GPU
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac   # the runners change directory
mkdir -p "$OUT"
# Stream3D-VLM is asked in its own query style; the request of each task, written once and then kept fixed
declare -A Q=(
  [object_finding]="Please monitor the video stream and report a {label}'s distance and clock direction as soon as one comes into view"
  [object_refinding]="Please report the distance and clock position of the {label} marked with the red box in the first frame. Don't answer immediately; wait until that exact {label} is in view again after leaving the view; another {label} does not count"
  [object_direction]="Please report the {label}'s clock direction relative to me. Don't answer immediately; wait until the {label} has left the view and is {direction_cam}"
  [ego_rotation]="Please report the camera's horizontal rotation angle. Don't answer immediately; wait until I am facing {rot_target} compared with the direction I face now"
  [origin_return]="I am controlling the camera to explore. Remember where I am standing now. Once I am back at this spot after walking away, tell me my distance from it; the direction I face does not matter"
  [visual_homing]="Please report the distance and clock position of this starting spot, the place the first frame is taken from. Don't answer immediately; wait until the spot is in view again after I walked away from it"
  [place_transition]="Please monitor the video stream and report the room's length, width and area as soon as the camera has entered a room different from the starting one; standing in the doorway still counts as the old room"
  [place_entry]="I'm looking for the {room} as I walk. Keep an eye out and report its size when I am standing inside it, not when I only see it through a doorway"
  [place_revisit]="Please estimate the room's dimensions (length, width) and area. Don't answer immediately; wait until I have left the {room} I start in and come back into that same room"
)
for t in $TASKS; do
  task=$t; from=""
  [ $t = ego_rotation ] && task=ego_rotation_lrb       # the harness name of the benchmark's Ego Rotation prompt
  [ $t = object_refinding ] && from=gap_start_i        # responses during the initial showing are not scored
  io="--task $task --episodes $EPISODES --out $OUT/$t.jsonl"
  case $MODEL in
    timechat_online) python runners/streaming/timechat_online.py --ckpt wyccccc/TimeChatOnline-7B --stop_at_first $io ${from:+--score_from_field $from} ;;
    livecc) python runners/streaming/livecc.py --model chenjoya/LiveCC-7B-Instruct --mode ovobench --stop_at_first $io ${from:+--score_from_field $from} ;;
    dispider) python runners/streaming/dispider.py --model Mar2Ding/Dispider $io ${from:+--score_from_field $from} ;;
    stream3d_vlm) python runners/streaming/stream3d_vlm.py --frames-by-dataset --query-text "${Q[$t]}" $io ${from:+--score-from-field $from} ;;
    *) echo "unknown model $MODEL"; exit 1 ;;
  esac > "$OUT/$t.stdout" 2>&1
  echo "$t done: $(wc -l < "$OUT/$t.jsonl") records"
done
partial=""; [ "$EPISODES" != data/prospace_bench.jsonl ] || [ $(echo $TASKS | wc -w) != 9 ] && partial=--partial
python scripts/score.py "$OUT" $partial
