"""Causal geometry from CUT3R: for every frame, a camera pose and one 3D point per vision token.

CUT3R is recurrent, so the geometry of second t is estimated from frames 0..t only.  Poses and points live in the
coordinate frame of the stream's first camera (metric).  CUT3R is frozen; its output for a stream is cached on disk.

Setup (README): third_party/CUT3R with patches/cut3r_rope.patch applied (the pure-PyTorch RoPE path, evaluated for
any integer position) and the released weights at third_party/CUT3R/src/cut3r_512_dpt_4_64.pth.  Only `dust3r.*` and
`croco.*` are imported from it.
"""

from __future__ import annotations

import functools
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile

from prospace_vlm.data import FRAME_H, FRAME_W, fit, is_portrait

ROOT = Path(__file__).resolve().parents[1]
# PROSPACE_CUT3R: another CUT3R checkout (with the RoPE patch) than third_party/CUT3R
CUT3R = Path(os.environ.get("PROSPACE_CUT3R", str(ROOT / "third_party" / "CUT3R")))
if str(CUT3R / "src") not in sys.path:
    sys.path.insert(0, str(CUT3R / "src"))

GRID = (12, 16)  # rows x cols of the cached points: one per 32 x 32 pixels of a 512 x 384 frame


class Cut3r(nn.Module):
    def __init__(self, cache_dir: str | None = None, ckpt: str | None = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.load = functools.partial(torch.load, weights_only=False)  # the checkpoint pickles an omegaconf config
        from dust3r.model import ARCroco3DStereo

        self.model = ARCroco3DStereo.from_pretrained(ckpt or str(CUT3R / "src" / "cut3r_512_dpt_4_64.pth"))
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _views(self, paths: list[str], device) -> list[dict]:
        """CUT3R views from the same 512 x 384 frames the VLM sees; normalisation = CUT3R's ImgNorm."""
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        from dust3r.utils.image import ImgNorm

        views = []
        for i, p in enumerate(paths):
            im = Image.open(p).convert("RGB")
            if is_portrait(im):
                im = fit(im)
            elif im.size != (FRAME_W, FRAME_H):
                im = im.resize((FRAME_W, FRAME_H), Image.BICUBIC)
            img = ImgNorm(im)[None]
            views.append(
                {
                    "img": img.to(device),
                    "ray_map": torch.full((img.shape[0], 6, img.shape[-2], img.shape[-1]), torch.nan, device=device),
                    "true_shape": torch.from_numpy(np.int32([im.size[::-1]])).to(device),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.eye(4, dtype=torch.float32, device=device)[None],
                    "img_mask": torch.tensor([True], device=device),
                    "ray_mask": torch.tensor([False], device=device),
                    "update": torch.tensor([True], device=device),
                    "reset": torch.tensor([False], device=device),
                }
            )
        return views

    def _cache_path(self, paths: list[str]):
        if self.cache_dir is None:
            return None
        return self.cache_dir / (hashlib.md5("\n".join(paths).encode()).hexdigest() + ".npz")

    @torch.no_grad()
    def forward(self, paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Frames in order -> camera-to-world poses (T, 4, 4) and points (T, 192, 3): the mean of CUT3R's world-frame
        point map over each cell of the 12 x 16 grid."""
        from dust3r.inference import loss_of_one_batch
        from dust3r.utils.camera import pose_encoding_to_camera

        device = next(self.model.parameters()).device
        cache = self._cache_path(paths)
        if cache is not None and cache.exists():
            z = np.load(cache)
            if "xyz" in z.files:
                return torch.from_numpy(z["c2w"]).to(device), torch.from_numpy(z["xyz"].astype(np.float32)).to(device)
        views = self._views(paths, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            res, _ = loss_of_one_batch(views, self.model, None, None, inference=True)
        preds = res["pred"]
        rows, cols = GRID

        def block(x):  # (H, W, C) -> (rows * cols, C)
            H, W, C = x.shape
            return x.reshape(rows, H // rows, cols, W // cols, C).float().mean((1, 3)).reshape(rows * cols, C)

        c2w = torch.cat([pose_encoding_to_camera(p["camera_pose"].float()) for p in preds], 0)
        xyz = torch.stack([block(p["pts3d_in_other_view"][0]) for p in preds])
        if cache is None:
            return c2w, xyz
        conf = torch.stack([block(p["conf"][0][..., None])[:, 0] for p in preds])
        depth = torch.stack([p["pts3d_in_self_view"][0, ..., 2].float().flatten().median() for p in preds])
        tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")  # written whole, then renamed
        np.savez_compressed(
            tmp,
            c2w=c2w.cpu().numpy(),
            depth=depth.cpu().numpy(),
            xyz=xyz.cpu().numpy().astype(np.float16),
            conf=conf.cpu().numpy().astype(np.float16),
        )
        os.replace(tmp, cache)
        return self.forward(paths)  # read back the cached float16 points: a first run equals every later run


def resample_points(xyz: torch.Tensor, gh: int, gw: int, src=GRID) -> torch.Tensor:
    """Points (T, 12 * 16, 3) -> (T, gh * gw, 3) on the backbone's token grid (Qwen2.5-VL: 14 x 18).  Bilinear, with
    invalid points left out of the average; a token with less than half of its weight on valid points is invalid
    (NaN)."""
    T = xyz.shape[0]
    if (gh, gw) == tuple(src):
        return xyz
    g = xyz.float().reshape(T, src[0], src[1], 3).permute(0, 3, 1, 2)
    ok = torch.isfinite(g).all(1, keepdim=True).float()
    num = torch.nn.functional.interpolate(torch.nan_to_num(g) * ok, size=(gh, gw), mode="bilinear", align_corners=False)
    den = torch.nn.functional.interpolate(ok, size=(gh, gw), mode="bilinear", align_corners=False)
    out = torch.where(den > 0.5, num / den.clamp(min=1e-6), torch.full_like(num, float("nan")))
    return out.permute(0, 2, 3, 1).reshape(T, gh * gw, 3).to(xyz.dtype)
