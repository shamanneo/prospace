"""ProSpace-VLM: Qwen2.5-VL-7B + LoRA with explicit geometric encodings injected into intermediate layers.

For every vision token, CUT3R gives the 3D point it shows and the camera pose of its frame (prospace_vlm/geometry.py).
Both are encoded and, at each of the layers L = {4, 8, 12, 16}, a per-layer MLP maps the encoding to the hidden size
and the result is added to the token's hidden state at that layer's output:

    h_i  <-  h_i + MLP_l( encode(point_i, camera_i) )

The past tokens stay in the context (the KV cache in streaming inference), so every later decision can attend to
what was seen together with where it was and from where it was seen.  Each MLP's last layer starts at zero: at
initialisation the model is exactly Qwen2.5-VL.

Without the injection (`inject=None`) the same class is the Qwen2.5-VL native-streaming baseline.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from prospace_vlm.geometry import Cut3r, resample_points

BACKBONE = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def fourier_features(x: torch.Tensor, n_freq: int, base_wavelength: float = 16.0) -> torch.Tensor:
    """(N, 3) metres -> (N, 6 * n_freq): sin / cos at wavelengths base, base / 2, ... (16 m .. 12.5 cm for 8)."""
    freqs = (2.0 * math.pi / base_wavelength) * (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype))
    ang = x[:, :, None] * freqs[None, None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1).reshape(x.shape[0], -1)


class GeometryInjection(nn.Module):
    """One MLP per injection layer: geometric encoding of a vision token -> an update of its hidden state."""

    def __init__(self, hid: int, layers: tuple = (4, 8, 12, 16), n_freq: int = 8, d: int = 512):
        super().__init__()
        self.layers, self.n_freq = tuple(int(l) for l in layers), n_freq
        dim = (6 * n_freq + 3 + 1) + (6 * n_freq + 3 + 6)  # point: 52, camera: 57
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(dim, d), nn.GELU(), nn.Linear(d, hid)) for _ in self.layers])
        for mlp in self.mlps:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def encode(self, xyz: torch.Tensor, valid: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
        """Per token: its point xyz (N, 3) in metres, whether the point is valid (N,), and the camera-to-world pose
        of its frame (N, 4, 4)  ->  (N, 109).

        point:  Fourier features of xyz, xyz / 4, valid flag (all zero for an invalid point)
        camera: Fourier features of the camera centre, centre / 4, the first two columns of the rotation
        """
        x = torch.where(valid[:, None], xyz.float(), torch.zeros_like(xyz, dtype=torch.float32))
        point = torch.cat([fourier_features(x, self.n_freq), x / 4.0, valid.float()[:, None]], -1)
        point = point * valid.float()[:, None]
        c2w = c2w.float()
        o = c2w[:, :3, 3]
        ok = torch.isfinite(c2w.reshape(-1, 16)).all(-1)[:, None].float()
        o = torch.nan_to_num(o)
        r6 = torch.nan_to_num(c2w[:, :3, :2].reshape(-1, 6))
        camera = torch.cat([fourier_features(o, self.n_freq), o / 4.0, r6], -1) * ok
        return torch.cat([point, camera], -1)

    def delta(self, i: int, code: torch.Tensor, dtype) -> torch.Tensor:
        mlp = self.mlps[i]
        return mlp(code.to(mlp[0].weight.dtype)).to(dtype)


class ProSpaceVLM(nn.Module):
    def __init__(self, qwen, geom: Cut3r | None = None, inject: GeometryInjection | None = None):
        super().__init__()
        self.qwen, self.geom, self.inject = qwen, geom, inject
        self._ctx, self._stats = None, {}  # the current stream's vision-token positions and encodings
        if inject is None:
            return
        # permanent hooks (gradient checkpointing runs the layers again in the backward pass); every forward sets
        # self._ctx for its stream first
        for li, L in enumerate(inject.layers):

            def hook(module, args, output, li=li):
                ctx = self._ctx
                if ctx is None:
                    return output
                hs = output[0] if isinstance(output, tuple) else output
                if hs.shape[1] <= int(ctx["img_pos"][-1]):
                    return output  # a decoding step after the prefill
                d = self.inject.delta(li, ctx["code"], hs.dtype)
                rms = float(d.detach().float().pow(2).mean().sqrt())
                self._stats[f"mem/fusion{li}_rms"] = rms
                self._stats[f"mem/fusion{li}_ratio"] = rms / max(
                    1e-8, float(hs[0, ctx["img_pos"]].detach().float().pow(2).mean().sqrt())
                )
                hs = hs.clone()
                hs[0, ctx["img_pos"]] = hs[0, ctx["img_pos"]] + d
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

            self.core.model.language_model.layers[L].register_forward_hook(hook)

    @property
    def core(self):
        """The Qwen model under PEFT: `.model` (vision + language, returns hidden states) and `.lm_head`."""
        return self.qwen.get_base_model() if hasattr(self.qwen, "get_base_model") else self.qwen

    def _set_geometry(self, item: dict, inputs: dict, device) -> float:
        """Geometry of this stream's vision tokens -> self._ctx, read by the layer hooks.  Returns the seconds spent."""
        t0 = time.time()
        m = self.core.model
        T = int(inputs["image_grid_thw"].shape[0])
        c2w, xyz = self.geom(item["geom_paths"])
        c2w, xyz = c2w[:T], xyz[:T]  # a stream cut for training: the first T seconds (CUT3R is causal)
        mg = m.visual.spatial_merge_size
        gh, gw = int(inputs["image_grid_thw"][0, 1]) // mg, int(inputs["image_grid_thw"][0, 2]) // mg
        xyz = resample_points(xyz, gh, gw)  # 14 x 18 tokens per frame
        torch.cuda.synchronize()
        seconds = time.time() - t0
        valid = torch.isfinite(xyz).all(-1)
        img_pos = torch.nonzero((inputs["input_ids"] == m.config.image_token_id)[0], as_tuple=False)[:, 0]
        assert (
            img_pos.numel() == xyz.shape[0] * xyz.shape[1]
        ), f"{img_pos.numel()} vision tokens vs points {tuple(xyz.shape)}"
        cam = c2w[:, None].expand(-1, xyz.shape[1], -1, -1).reshape(-1, 4, 4)
        self._ctx = {"img_pos": img_pos, "code": self.inject.encode(xyz.reshape(-1, 3), valid.reshape(-1), cam)}
        self._stats = {}
        return seconds

    def forward(self, item: dict, device) -> dict:
        """One stream (prospace_vlm/data.py: encode) -> for every second, `logits` over the vocabulary at its
        decision and `final`, the logit of responding: log p(respond) - log p(wait)."""
        core, m = self.core, self.core.model
        inputs = {k: v.to(device) for k, v in item["inputs"].items()}
        geom_s = self._set_geometry(item, inputs, device) if self.inject is not None else 0.0
        # the vision tower is frozen and nothing trainable sits before it: run it without autograd (enable_input_
        # require_grads would otherwise make the backward pass go through it for nothing)
        orig = m.get_image_features

        def frozen_image_features(*args, **kwargs):
            with torch.no_grad():
                return orig(*args, **kwargs)

        m.get_image_features = frozen_image_features
        try:
            hidden = m(**inputs).last_hidden_state  # (1, L, dim)
        finally:
            del m.get_image_features
        h = hidden[0, item["slots"].to(device)]  # (T, dim)
        logits = core.lm_head(h).float()  # (T, V)
        p = torch.softmax(logits[:, [item["respond_id"], item["wait_id"]]], -1)[:, 0]
        return {
            "logits": logits,
            "final": torch.logit(p.clamp(1e-4, 1 - 1e-4)),
            "geom_s": geom_s,
            "stats": dict(self._stats) if self.inject is not None else {},
        }


def patch_peft() -> None:
    """Environment quirk: peft probes for gptqmodel on every LoRA layer; make that probe a constant."""
    import peft.tuners.lora.model  # noqa: F401

    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("peft") and hasattr(mod, "is_gptqmodel_available"):
            mod.is_gptqmodel_available = lambda *a, **k: False


def load_model(ckpt: str | Path, geom_cache: str | None = None):
    """A trained model for inference -> (model, processor).

    `ckpt` is the released single file (checkpoints/prospace_vlm.pth: LoRA weights + the per-layer MLPs) or a folder
    saved by prospace_vlm/train.py (runs/<name>/best: a PEFT adapter + prospace_vlm.pt).
    """
    from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForImageTextToText, AutoProcessor

    patch_peft()
    ckpt = Path(ckpt)
    one_file = ckpt.is_file()
    extra = torch.load(ckpt if one_file else ckpt / "prospace_vlm.pt", map_location="cpu", weights_only=False)
    args = extra["args"]
    device = torch.device("cuda")
    name = args.get("model") or BACKBONE
    processor = AutoProcessor.from_pretrained(name)
    qwen = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to(device)
    if one_file:
        qwen = get_peft_model(qwen, LoraConfig(inference_mode=True, **extra["lora_config"]))
        res = set_peft_model_state_dict(qwen, extra["lora"])
        assert not res.unexpected_keys, f"unexpected LoRA keys: {res.unexpected_keys[:4]}"
        qwen = qwen.eval()
    else:
        qwen = PeftModel.from_pretrained(qwen, str(ckpt)).eval()
    geom = inject = None
    if "memory" in extra:  # the per-layer MLPs (the key name of the released checkpoint)
        layers = args.get("layers") or args.get("fusion_layers")
        geom = Cut3r(cache_dir=geom_cache).to(device).eval()
        inject = GeometryInjection(
            qwen.config.get_text_config().hidden_size, layers=tuple(int(v) for v in layers.split(","))
        ).to(device)
        inject.load_state_dict(extra["memory"])
        inject.eval()
    return ProSpaceVLM(qwen, geom, inject).eval(), processor
