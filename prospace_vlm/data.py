"""Streams as model inputs.

One stream is one user turn: the instruction, then one frame per second, and after every frame a decision token.
The request is placed before the frame of the second it is asked at (the first frame, or later for a mid-stream
request):

    <instruction> [F0 , F1 , ...] \\n<request> Fk , Fk+1 , ...

The decision token after a frame is "," (keep watching) or "\\n" (respond now).  The model always sees "," in its
context, as it would while it has not responded yet, and is trained to predict "\\n" at every second of the timely
window.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFile

FRAME_W, FRAME_H = 512, 384
INSTRUCTION = (
    "This is a video recorded at 1 frame per second, walking through an indoor scene. "
    "Frames arrive one at a time; the latest frame is the current moment."
)
WAIT, RESPOND = ",", "\n"


# ----------------------------------------------------------------------------- frames
def is_portrait(im) -> bool:
    return im.width < im.height


def fit(im, resample=Image.BICUBIC):
    """PIL image -> 512 x 384 RGB.  Portrait frames (upright CA-1M captures) are scaled to 384 high and centred on a
    black canvas instead of being stretched; the benchmark has landscape frames only."""
    im = im.convert("RGB")
    if not is_portrait(im):
        return im.resize((FRAME_W, FRAME_H), resample)
    w = round(im.width * FRAME_H / im.height)
    canvas = Image.new("RGB", (FRAME_W, FRAME_H), (0, 0, 0))
    canvas.paste(im.resize((w, FRAME_H), resample), ((FRAME_W - w) // 2, 0))
    return canvas


def fit_any(im, resample=Image.BICUBIC):
    """PIL image of any aspect -> 512 x 384 RGB, scaled to fit and centred on black (never stretched)."""
    im = im.convert("RGB")
    s = min(FRAME_W / im.width, FRAME_H / im.height)
    w, h = max(1, round(im.width * s)), max(1, round(im.height * s))
    if (w, h) == (FRAME_W, FRAME_H):
        return im.resize((FRAME_W, FRAME_H), resample)
    canvas = Image.new("RGB", (FRAME_W, FRAME_H), (0, 0, 0))
    canvas.paste(im.resize((w, h), resample), ((FRAME_W - w) // 2, (FRAME_H - h) // 2))
    return canvas


def frame_paths(ep: dict, roots: dict) -> list[str]:
    """Absolute frame paths; a stream stores them as "<dataset>/<scene>/frame_XXXXXX.jpg"."""
    return [str(Path(roots[ep["dataset"]]) / p.split("/", 1)[1]) for p in ep["frames"]]


def load_frames(ep: dict, roots: dict, box_at: int = 0) -> list:
    """The stream's frames at 512 x 384.  Object Re-finding marks its target with a red box on the frame the request
    is asked at; the box is given in 1920 x 1440 units and drawn on the native frame, before any resizing."""
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    ims = []
    for i, p in enumerate(frame_paths(ep, roots)):
        im = Image.open(p).convert("RGB")
        if i == box_at and ep["task"] == "object_refinding" and ep.get("frame0_bbox"):
            bb = ep["frame0_bbox"]
            sx, sy = im.width / 1920, im.height / 1440
            ImageDraw.Draw(im).rectangle(
                [bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy],
                outline=(255, 0, 0),
                width=max(4, round(6 * im.width / 512)),
            )
        if is_portrait(im):
            im = fit(im)
        elif im.size != (FRAME_W, FRAME_H):
            im = im.resize((FRAME_W, FRAME_H), Image.BICUBIC)
        ims.append(im)
    return ims


def fit_to_backbone(ims: list, processor) -> list:
    """512 x 384 frames -> the size the backbone's processor would give them (sides rounded to patch x merge):
    Qwen2.5-VL (28 px) gets 504 x 392 = 18 x 14 = 252 tokens per frame."""
    f = getattr(processor.image_processor, "patch_size", 16) * getattr(processor.image_processor, "merge_size", 2)
    out = []
    for im in ims:
        w, h = max(f, round(im.width / f) * f), max(f, round(im.height / f) * f)
        out.append(im if (w, h) == im.size else im.resize((w, h), Image.BICUBIC))
    return out


# ----------------------------------------------------------------------------- encoding
def request_second(ep: dict) -> int:
    """The second the request is asked at (0 unless the record has `q_at`)."""
    return int(ep.get("q_at") or 0)


def decision_token_ids(tokenizer) -> tuple[int, int]:
    """(wait_id, respond_id)"""
    return (
        tokenizer.encode(WAIT, add_special_tokens=False)[0],
        tokenizer.encode(RESPOND, add_special_tokens=False)[0],
    )


def timely_seconds(ep: dict) -> list[bool]:
    """Per second: is it inside the timely window [w0, w1]?  All False for an absence stream (w0 is None)."""
    n = len(ep["frames"])
    y = [False] * n
    if ep.get("w0") is not None:
        for t in range(int(ep["w0"]), min(int(ep["w1"]) + 1, n)):
            y[t] = True
    return y


def encode(processor, ep: dict, roots: dict, train: bool) -> dict:
    """One stream -> model inputs, the position of every second's decision and its target token.

    train=True cuts a stream with an event after the last timely second (nothing is supervised after the deadline);
    train=False keeps the full stream.  Seconds before a mid-stream request have target -100 (nothing was asked yet).
    """
    tok = processor.tokenizer
    wait_id, respond_id = decision_token_ids(tok)
    P = request_second(ep)
    n = len(ep["frames"])
    # CUT3R is causal, so the geometry of a cut stream is the first n seconds of the full stream's: the full path list
    # is kept as the cache key and sliced in the forward
    geom_paths = frame_paths(ep, roots)
    if train and ep.get("w0") is not None:
        n = min(int(ep["w1"]), n - 1) + 1
        ep = {**ep, "frames": ep["frames"][:n], "frame_ids": ep["frame_ids"][:n]}
    assert P < n, f"{ep['episode_id']}: request at {P} beyond the stream ({n})"
    ims = fit_to_backbone(load_frames(ep, roots, box_at=P), processor)
    content = [{"type": "text", "text": INSTRUCTION}]
    for t in range(n):
        if t == P:
            content.append({"type": "text", "text": "\n" + ep["question"]})
        content.append({"type": "image"})
        content.append({"type": "text", "text": WAIT})
    chat = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=False
    )
    x = processor(text=[chat], images=ims, do_resize=False, return_tensors="pt")
    vend = tok.convert_tokens_to_ids("<|vision_end|>")
    if P:
        # the tokenizer merges the "," after frame P-1 and the "\n" of the request into one token ",\n": split it back,
        # so that frame P-1 keeps its decision token
        ids = x["input_ids"][0]
        q = int(torch.nonzero(ids == vend)[P - 1, 0]) + 1
        merged = tok.encode(",\n", add_special_tokens=False)
        nl = tok.encode("\n", add_special_tokens=False)
        assert (
            len(merged) == 1 and int(ids[q]) == merged[0]
        ), f"{ep['episode_id']}: expected ',\\n' after frame {P - 1}, got {tok.decode([int(ids[q])])!r}"
        L = len(ids)
        for k in list(x.keys()):
            v = x[k]
            if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == L:
                row = v[0]
                ins = torch.tensor([wait_id] + nl, dtype=row.dtype) if k == "input_ids" else row[q].repeat(1 + len(nl))
                x[k] = torch.cat([row[:q], ins, row[q + 1 :]])[None]
    ids = x["input_ids"][0]
    pos = torch.nonzero(ids == vend, as_tuple=False)[:, 0] + 1  # the token right after each frame
    assert len(pos) == n and all(
        int(ids[p]) in (wait_id, respond_id) for p in pos
    ), f"{ep['episode_id']}: decision tokens misplaced"
    targets = torch.tensor([respond_id if y else wait_id for y in timely_seconds(ep)])
    targets[:P] = -100
    return {
        "inputs": {k: v for k, v in x.items()},
        "slots": pos - 1,  # the decision of second t is read at the last token of frame t
        "targets": targets,
        "respond_id": respond_id,
        "wait_id": wait_id,
        "geom_paths": geom_paths,
        "ep": ep,
    }


class Streams(torch.utils.data.Dataset):
    def __init__(self, eps, processor, roots, train):
        self.eps, self.processor, self.roots, self.train = eps, processor, roots, train

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, i):
        return encode(self.processor, self.eps[i], self.roots, self.train)


def loader(eps, processor, roots, workers, train):
    """One stream per item, in the order of `eps`."""
    return torch.utils.data.DataLoader(
        Streams(eps, processor, roots, train),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=2 if workers else None,
        persistent_workers=False,
    )
