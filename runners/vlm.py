"""Prefix polling for VLM families other than Qwen3-VL, on the same protocol and scoring.

Everything task-side (prompts, first-trigger rule, premature / timely / delayed / miss, absence episodes)
comes from runners/qwen3vl.py; only the model input differs, so a family here is just a way of turning
"text + the frames seen so far" into processor inputs:

  qwen3    Qwen3-VL / Qwen3-VL-MoE  - video path with per-frame metadata (the benchmark's reference runner)
  qwen25   Qwen2.5-VL and its forks - same video path, no video_metadata
  images   anything else HF-native  - the frames as an interleaved image list via the chat template

    python runners/vlm.py --family images --model OpenGVLab/InternVL3_5-8B-HF \
        --task object_finding --mode polling --style state --step 1 \
        --episodes out/epsiodes/final_v1_protocol50.json --frames-by-dataset --all --out out/runs/.../x.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen3vl as R  # noqa: E402
from PIL import Image  # noqa: E402


def _pil(frames) -> list:
    out = []
    for f in frames:
        im = f if isinstance(f, Image.Image) else Image.open(f)
        out.append(im.convert("RGB").resize((R.FRAME_W, R.FRAME_H)))
    return out


def build_qwen25(processor, frames, text: str, indices=None):
    """Qwen2.5-VL: the qwen_vl_utils video path, without the Qwen3 per-frame metadata."""
    from qwen_vl_utils import process_vision_info

    video = list(frames) if len(frames) >= 2 else list(frames) * 2
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "video",
                    "video": video,
                    "resized_height": R.FRAME_H,
                    "resized_width": R.FRAME_W,
                    "sample_fps": R.SAMPLE_FPS,
                },
            ],
        }
    ]  # frame lists are timed by sample_fps (fps only applies to video files)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps = kwargs.pop("fps", None)  # this processor wants a scalar, qwen_vl_utils returns a list
    if isinstance(fps, (list, tuple)):
        fps = fps[0] if fps else R.SAMPLE_FPS
    assert (
        abs(float(fps) - R.SAMPLE_FPS) < 1e-6
    ), f"qwen_vl_utils reported {fps} fps for a {R.SAMPLE_FPS} fps frame list"
    inputs = processor(text=[chat], images=images, videos=videos, fps=fps, **kwargs, return_tensors="pt")
    return inputs.to(R.model_device())


_NO_TILE: dict = {}


def build_images(processor, frames, text: str, indices=None):
    """Generic HF chat template: the question first, then one image per second seen so far.

    Tiling processors (InternVL) would cut every 512x384 frame into up to 13 patches, so a 50 s prefix
    becomes 160k tokens; one tile per frame is what a 1 fps stream needs, hence crop_to_patches=False."""
    imgs = _pil(frames)
    content = [{"type": "text", "text": text}] + [{"type": "image", "image": im} for im in imgs]
    msg = [{"role": "user", "content": content}]
    key = type(processor).__name__
    if key not in _NO_TILE:
        try:
            processor.apply_chat_template(
                msg[:1],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                crop_to_patches=False,
            )
            _NO_TILE[key] = {"crop_to_patches": False}
        except Exception:
            _NO_TILE[key] = {}
    inputs = processor.apply_chat_template(
        msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **_NO_TILE[key]
    )
    return inputs.to(R.model_device())


def build_images_numbered(processor, frames, text: str, indices=None):
    """Like `images`, but every frame is introduced by its second ("Frame 12 (12 s):"), the way VLMEvalKit
    feeds video frames to InternVL - the model then knows the images are an ordered 1 fps sequence."""
    imgs = _pil(frames)
    content = [{"type": "text", "text": text + "\nThe frames follow, one per second, in order."}]
    for i, im in enumerate(imgs):
        content += [{"type": "text", "text": f"Frame {i} ({i} s):"}, {"type": "image", "image": im}]
    msg = [{"role": "user", "content": content}]
    key = type(processor).__name__
    if key not in _NO_TILE:
        try:
            processor.apply_chat_template(
                msg[:1],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                crop_to_patches=False,
            )
            _NO_TILE[key] = {"crop_to_patches": False}
        except Exception:
            _NO_TILE[key] = {}
    inputs = processor.apply_chat_template(
        msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **_NO_TILE[key]
    )
    return inputs.to(R.model_device())


BUILDERS = {"qwen3": None, "qwen25": build_qwen25, "images": build_images, "images_numbered": build_images_numbered}


def main() -> None:
    argv = sys.argv[1:]
    family = "qwen3"
    if "--family" in argv:
        i = argv.index("--family")
        family = argv[i + 1]
        del argv[i : i + 2]
    if family not in BUILDERS:
        raise SystemExit(f"--family must be one of {sorted(BUILDERS)}")
    if BUILDERS[family] is not None:
        R.build_inputs = BUILDERS[family]
    sys.argv = [sys.argv[0]] + argv
    print(f"[vlm] family={family}", flush=True)
    R.main()


if __name__ == "__main__":
    main()
