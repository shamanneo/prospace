"""The benchmark protocol pieces shared by every runner: fixed input policy, prompts, task table, episode
loading, first-trigger scoring and the strict one-word parser.  No torch / transformers imports, so a runner
for any model family (and any Python environment) can import this.

Every runner must go through these functions so the evaluation stays identical across model families.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- fixed policy ----------------------------------------------------------------------------
FRAME_W, FRAME_H = 512, 384
SAMPLE_FPS = 1.0
FRAMES = Path(
    os.environ.get("PROSPACE_FRAMES", ROOT / "data" / "frames")
)  # data/frames/<dataset>/<scene_id>/frame_XXXXXX.jpg
DEFAULT_FRAMES = str(FRAMES / "scannetpp")
DATASET_FRAMES = {
    ds: str(FRAMES / ds) for ds in ("scannetpp", "scannet", "ca1m", "mp3d")
}  # frame root per episode `dataset` (--frames-by-dataset)

# --- prompts --------------------------------------------------------------------------------
POLL_PROMPT = (
    "You are monitoring an ongoing indoor video for a standing user request.\n"
    "The request was given at time 0. The supplied video contains only the observations available so far, "
    "sampled at 1 frame per second.\n"
    "You have not sent an alert for this request.\n"
    "Output RESPOND as soon as the observed video provides sufficient evidence that the requested event has "
    "occurred. Otherwise output WAIT.\n"
    "The event may or may not occur. Do not predict a future event or acknowledge the request.\n"
    "Output exactly WAIT or RESPOND, with no explanation or timestamp.\n\n"
    "Standing request: {request}\n"
    "Current observation time: {t} seconds."
)

STATE_PROMPT = (
    "This is a video recorded at 1 frame per second, walking through an indoor scene. "
    "The video shows only what has been observed so far; the last frame is the current moment ({t} seconds).\n"
    "{question}\n"
    "Answer with exactly one word: {pos} or {neg}."
)

ONESHOT_PREFIX = "This is a video recorded at 1 frame per second, walking through an indoor scene. "
ONESHOT_SUFFIX = " Answer with the number of seconds only."

DIRECTION = {"left": "to its left", "right": "to its right", "behind": "behind it"}
ROOM_PHRASE = {"storage": "storage room"}  # Place Entry room label -> how the room is named in questions

TASKS = {
    "object_finding": dict(
        state="Is a {label} visible in the current frame?",
        episodes="out/epsiodes/object_finding_v4_all.json",
        request='Notify me when an object of category "{label}" first comes into view.',
        oneshot="At what second does the {label} first come into view?",
    ),
    "object_refinding": dict(
        state="Is the {label} marked with the red box in the first frame visible in the current frame? Another {label} "
        "does not count.",
        episodes="out/epsiodes/object_refinding.json",
        frame0_bbox=True,
        request=(
            "The red box in the first frame identifies the target {label}. Notify me when that same object "
            "first comes back into view after going out of view. Seeing another object of the same category "
            "does not count."
        ),
        oneshot=(
            "The {label} marked with the red box in the first frame goes out of view and later reappears. "
            "At what second does this exact {label} come back into view?"
        ),
    ),
    "object_direction": dict(
        # P1 grammar fix: {direction} rendered "behind it of the camera"; the camera-referenced phrase
        # reads "behind the camera" / "to the left of the camera". 50-episode study: same timely as the old wording.
        state="Is the {label} you saw earlier now out of view and {direction_cam}?",
        episodes="out/epsiodes/object_direction_eval1000.json",
        request=(
            "Notify me the first time the previously observed {label} is out of view and {direction_cam}, "
            "relative to the camera's current position and viewing direction."
        ),
        oneshot=(
            "Watch for the {label}. After the camera has seen it and it has left the view, at what second is "
            "the {label} {direction_cam}?"
        ),
    ),
    "ego_rotation": dict(
        state=(
            "Since the first frame, have you turned a full half circle, about 180 degrees, so that the direction you "
            "face now "
            "is the reverse of your starting direction? If you have turned less than about 150 degrees, answer No."
        ),
        episodes="out/epsiodes/ego_rotation_eval1000.json",
        request="Notify me when the camera first faces the opposite direction from its viewing direction at the start.",
        oneshot=(
            "Remember the direction the camera is facing in the first frame. At what second is the camera "
            "facing the opposite direction?"
        ),
    ),
    "ego_rotation_lrb": dict(  # Ego Rotation left/right/behind bench (out/epsiodes/bench/ego_rotation_lrb.json): the state question
        # follows the episode direction and reuses the existing ones verbatim (left/right = ego_rotation_lr, behind =
        # ego_rotation)
        state={
            "left": "Remember the direction you were facing in the first frame. Are you now facing the direction that "
            "was on your {side}?",
            "right": "Remember the direction you were facing in the first frame. Are you now facing the direction that "
            "was on your {side}?",
            "behind": (
                "Since the first frame, have you turned a full half circle, about 180 degrees, so that the direction "
                "you face now "
                "is the reverse of your starting direction? If you have turned less than about 150 degrees, answer No."
            ),
        },
        episodes="out/epsiodes/bench/ego_rotation_lrb.json",
        request="{question}",
        oneshot="{question}",
    ),
    "ego_rotation_lr": dict(  # Ego Rotation left/right variant (out/epsiodes/bench/ego_rotation_{left,right}.json); state = bench Yes/No form
        state="Remember the direction you were facing in the first frame. Are you now facing the direction that was on "
        "your {side}?",
        episodes="out/epsiodes/bench/ego_rotation_left.json",
        request="{question}",
        oneshot="{question}",
    ),
    "origin_return": dict(
        state="Is the camera now back at the spot where it was standing in the first frame, after having walked away "
        "from it?",
        episodes="out/epsiodes/origin_return_eval1000.json",
        request=(
            "Notify me when the camera first returns to its starting location after moving away from it. "
            "Returning to the original viewing direction is not required."
        ),
        oneshot=(
            "Remember where the camera is standing in the first frame. At what second is the camera back at "
            "that spot after walking away?"
        ),
    ),
    "place_entry": dict(  # state = bench Yes/No form; request/oneshot = the episode's own sentence verbatim
        state=(
            "You are walking through a house with several rooms connected by doorways. Have you walked through a "
            "doorway and are you now standing inside the {room}? Answer No while you are in any other room or a "
            "hallway, and No if you can only see the {room} through a doorway."
        ),  # MP3D prompt study (E1'), pilot 38.0 vs 28.0
        episodes="out/epsiodes/mp3d_room_bench/place_entry.json",
        request="{question}",
        oneshot="{question}",
    ),
    "place_revisit": dict(  # MP3D room task: the clip starts inside the named room; respond when you re-enter it after leaving
        state=(
            "You started in the {room}. Have you left it and now come back into this same {room}? Answer No while you "
            "are "
            "still in it for the first time, No while you are away from it, and No if you are in a different {room}."
        ),
        episodes="out/epsiodes/mp3d_room_bench/place_revisit.json",
        request="{question}",
        oneshot="At what second do you come back into the {room} you started in, after having left it?",
    ),
    "place_transition": dict(  # MP3D prompt study (T5): 50-episode pilot overall 44.0 vs 8.0 for the old wording
        state=(
            "Are you now inside a different room from the one where the video began? Answer Yes only after you have "
            "crossed a doorway and entered the new room. Seeing another room through a doorway, or being in the "
            "doorway, still counts as No."
        ),
        episodes="out/epsiodes/mp3d_room_bench/place_transition.json",
        request=("Notify me the moment the camera moves out of the room it starts in and enters a different room."),
        oneshot="At what second does the camera move out of the room it starts in and enter a different room?",
    ),
    "visual_homing": dict(
        state="Is the location of the first frame back in view now, after you walked away from it?",
        episodes="out/epsiodes/visual_homing.json",
        request=(
            "The first frame is the view from the starting spot. Notify me when the starting spot first comes "
            "back into view after the camera has walked away from it."
        ),
        oneshot=(
            "The first frame is the view from the starting spot, and the camera then walks away from it. "
            "At what second does the starting spot come back into view?"
        ),
    ),
    "relative_proximity": dict(
        state="Is the {y} now closer to the camera than the {x}?",
        episodes="out/epsiodes/relative_proximity.json",
        request="Watch for the {x} and the {y}. Notify me when the {y} first becomes closer to the camera than the "
        "{x}.",
        oneshot="Watch for the {x} and the {y}. At what second is the {y} closer to the camera than the {x}?",
    ),
}


def fill(template, ep: dict) -> str:
    d = ep.get("direction")
    if isinstance(template, dict):  # per-direction templates (ego_rotation_lrb)
        template = template[d]
    return template.format(
        label=ep.get("label", ""),
        x=ep.get("x_label", ""),
        y=ep.get("y_label", ""),
        question=ep.get("question", ""),
        room=ROOM_PHRASE.get(ep.get("label", ""), ep.get("label", "")),
        side=ep.get("direction", ""),
        direction=DIRECTION.get(d, ""),
        direction_cam={
            "left": "to the left of the camera",
            "right": "to the right of the camera",
            "behind": "behind the camera",
        }.get(d, ""),
    )


def frames_root_for(ep: dict, default) -> Path:
    """Frame root for one episode.  With PROSPACE_FRAMES_BY_DATASET=1 the root follows the episode `dataset`
    (DATASET_FRAMES), for episode files that mix ScanNet++ / ScanNet / CA-1M; otherwise `default`."""
    if os.environ.get("PROSPACE_FRAMES_BY_DATASET") == "1":
        return Path(DATASET_FRAMES.get(ep.get("dataset"), default))
    return Path(default)


def episode_frames(ep: dict, root: Path, burn_bbox: bool, frame0_crop: bool = False) -> list:
    paths = [str(root / ep["scene_id"] / f"frame_{fid:06d}.jpg") for fid in ep["frame_ids"]]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{ep['episode_id']}: {len(missing)} frames missing, e.g. {missing[0]}")
    if burn_bbox:
        bbox = ep.get("frame0_bbox")
        if not bbox:
            raise ValueError(f"{ep['episode_id']} has no frame0_bbox")
        from PIL import Image, ImageDraw

        im = Image.open(paths[0]).convert("RGB")
        sx, sy = im.width / 1920, im.height / 1440
        # line width chosen so the box is ~6 px after the resize to FRAME_W (frames on disk may be 1920 wide)
        lw = max(4, round(6 * im.width / FRAME_W))
        if frame0_crop:
            # ablation: scene context removed from frame 0 - only the target crop (10% margin), scaled to fit
            # and centred on a grey canvas of the original size, red box kept around it so the question still reads
            x0, y0, x1, y1 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
            mx, my = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
            crop = im.crop((max(0, x0 - mx), max(0, y0 - my), min(im.width, x1 + mx), min(im.height, y1 + my)))
            k = min(im.width / crop.width, im.height / crop.height) * 0.9
            crop = crop.resize((max(1, round(crop.width * k)), max(1, round(crop.height * k))), Image.BILINEAR)
            canvas = Image.new("RGB", (im.width, im.height), (128, 128, 128))
            ox, oy = (im.width - crop.width) // 2, (im.height - crop.height) // 2
            canvas.paste(crop, (ox, oy))
            ImageDraw.Draw(canvas).rectangle(
                [ox - lw, oy - lw, ox + crop.width + lw, oy + crop.height + lw], outline=(255, 0, 0), width=lw
            )
            return [canvas] + paths[1:]
        ImageDraw.Draw(im).rectangle(
            [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy], outline=(255, 0, 0), width=lw
        )
        return [im] + paths[1:]
    return paths


def load_episodes(path, task: str) -> list[dict]:
    """Episodes of one task: the lines of that task in the benchmark .jsonl, or the "episodes" of a .json file."""
    path = ROOT / path
    if path.suffix == ".jsonl":
        task = {"ego_rotation_lrb": "ego_rotation"}.get(task, task)  # the harness name of the task
        return [e for e in map(json.loads, open(path)) if e["task"] == task]
    return json.loads(path.read_text())["episodes"]


def select_episodes(args) -> list[dict]:
    eps = load_episodes(args.episodes, args.task)
    if args.episode_id:
        by = {e["episode_id"]: e for e in eps}
        eps = [by[i] for i in args.episode_id]
    if args.limit:
        eps = eps[: args.limit]
    i, n = (int(v) for v in args.shard.split("/"))
    eps = eps[i::n]
    if args.out and Path(args.out).exists():
        done = {json.loads(l)["episode_id"] for l in open(args.out) if l.strip()}
        eps = [e for e in eps if e["episode_id"] not in done]
    return eps


# --- scoring --------------------------------------------------------------------------------
def has_event(ep: dict) -> bool:
    """True for positive episodes (a timely window exists); absence episodes have timely_start_i None."""
    return ep.get("timely_start_i") is not None


def verdict(ep: dict, response_i: int | None, has_event: bool) -> str:
    if response_i is None:
        return "miss" if has_event else "correct_abstention"
    if not has_event:
        return "false_alarm"
    return (
        "premature" if response_i < ep["timely_start_i"] else "delayed" if response_i > ep["timely_end_i"] else "timely"
    )


def parse_action(text: str, style: str = "alert", words: tuple[str, str] = ("Yes", "No")) -> str | None:
    """Exact match after normalisation (whole reply, letters only, upper-case): 'Yes.' -> YES, but
    'Yes and No', 'Nobody', 'WAITING' or a second line are invalid."""
    s = re.sub(r"[^A-Za-z]", "", text).upper()
    mapping = (
        {"RESPOND": "RESPOND", "WAIT": "WAIT"}
        if style == "alert"
        else {words[0].upper(): "RESPOND", words[1].upper(): "WAIT"}
    )
    return mapping.get(s)


def parse_seconds(text: str) -> float | None:
    m = re.search(r"\d+(?:\.\d+)?", text)
    return float(math.floor(float(m.group()))) if m else None


def poll_loop(
    ask,
    frames,
    question: str,
    words: tuple[str, str] = ("Yes", "No"),
    step: int = 1,
    stop_at_first: bool = True,
    start: int = 0,
    score_from: int = 0,
) -> dict:
    """Prefix polling for any backend.  `ask(clip, text) -> (reply, p_pos_restricted, p_pos_full, p_neg_full, n_tokens)`
    is called at every second t with clip = frames[0..t] and the STATE_PROMPT for that second; the first
    trigger word is the response.  Same semantics as runners/qwen3vl.py:run_polling (state style)."""
    trace, response_i, invalid = [], None, None
    n = len(frames)
    for t in range(start, n, step):
        text = STATE_PROMPT.format(question=question, t=t, pos=words[0], neg=words[1])
        reply, p_pos, p_pos_full, p_neg_full, n_tok = ask(frames[: t + 1], text)
        action = parse_action(reply, "state", words)
        trace.append(
            {
                "t": t,
                "action": action,
                "p_respond": round(float(p_pos), 4),
                "p_respond_full": round(float(p_pos_full), 4),
                "p_wait_full": round(float(p_neg_full), 4),
                "reply": reply,
                "tokens": int(n_tok),
            }
        )
        if action is None:
            invalid = {"t": t, "reply": reply}
            break
        if action == "RESPOND" and response_i is None and t >= score_from:  # a Yes before score_from is not a response
            response_i = t
            if stop_at_first:
                break
    return {"response_i": response_i, "invalid": invalid, "n_calls": len(trace), "trace": trace}
