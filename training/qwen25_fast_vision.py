"""Qwen2.5-VL in the polling trainer: make the FROZEN vision tower cheap.

Two costs of the stock path, both outside the model's maths:

1. `enable_input_require_grads()` (needed for LoRA + gradient checkpointing on the language tower) also hooks the vision
   patch embedding, so the whole frozen tower is traced and re-run in backward although it has no trainable parameter
   and nothing needs a gradient with respect to the pixels.  -> the tower runs under no_grad.
2. Outside flash-attention the tower calls the attention once per window (8 x 8 patches).  -> windows of the same length
   are stacked and go through ONE scaled_dot_product_attention call (the same function, the same arguments, per window
   the same numbers; only the batching differs).

`install(model)` patches the given model instance only; the benchmark runner loads its own, unpatched model.
"""

from __future__ import annotations

import types

import torch


def _attn_forward(self, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

    n = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(n, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)  # (n, heads, d)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    starts, lengths = cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1]
    out = torch.empty_like(v)
    filled = 0
    for L in torch.unique(lengths).tolist():
        if L <= 0:
            continue
        st = starts[lengths == L]
        idx = (st[:, None] + torch.arange(L, device=st.device)[None]).reshape(-1)  # the tokens of these windows
        qq, kk, vv = (
            t[idx].reshape(-1, L, self.num_heads, t.shape[-1]).transpose(1, 2) for t in (q, k, v)
        )  # (m, heads, L, d)
        o = torch.nn.functional.scaled_dot_product_attention(
            qq, kk, vv, attn_mask=None, dropout_p=0.0, scale=self.scaling, is_causal=False
        )
        out[idx] = o.transpose(1, 2).reshape(-1, self.num_heads, o.shape[-1]).to(out.dtype)
        filled += idx.numel()
    assert filled == n, f"window lengths cover {filled} of {n} vision tokens"
    return self.proj(out.reshape(n, -1))


def install(model) -> None:
    """`model` = Qwen2_5_VLForConditionalGeneration (before or after PEFT wrapping its language tower)."""
    core = model
    for _ in range(4):  # peel PEFT / wrapper layers down to the module that owns `.visual`
        if hasattr(core, "visual"):
            break
        core = getattr(core, "model", None) or getattr(core, "base_model")
    vis = core.visual
    assert not any(
        p.requires_grad for p in vis.parameters()
    ), "the vision tower has trainable parameters: do not install"
    for blk in vis.blocks:
        blk.attn.forward = types.MethodType(_attn_forward, blk.attn)
        if hasattr(blk, "gradient_checkpointing"):
            blk.gradient_checkpointing = False
    orig = vis.forward

    def forward(*a, **k):
        with torch.no_grad():
            return orig(*a, **k)

    vis.forward = forward
    print(
        f"[qwen25_fast_vision] installed on {type(vis).__name__}: no_grad tower, {len(vis.blocks)} blocks with batched window attention",
        flush=True,
    )
