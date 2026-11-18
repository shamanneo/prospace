"""Prefix polling for the offline spatial VLMs whose code lives in their own repos and Python envs:
Spatial-MLLM, VLM-3R, Cambrian-S and DeepSeek-VL2 (each runs in its own repository's environment).  Protocol, prompts,
scoring and the parser come from runners/protocol.py (identical to the Qwen runner); only the model call follows each
repo's official inference code.

  Spatial-MLLM  third_party/Spatial-MLLM/src/inference.py
  VLM-3R        third_party/VLM-3R/playground/demo/image_demo.py
  Cambrian-S    third_party/cambrian-s/lmms-eval/.../cambrians.py
  LLaVA-Video   third_party/LLaVA-NeXT (upstream) + lmms-eval llava_vid
  DeepSeek-VL2  third_party/DeepSeek-VL2/inference.py

    python runners/offline_spatial.py --backend spatial_mllm --model Diankun/Spatial-MLLM-v1.1-Instruct-820K \
        --task object_finding --episodes out/epsiodes/object_finding_stream10.json --frames-by-dataset --all --out ...

--max-frames N: models with a hard frame budget get the prefix uniformly subsampled to N frames (the current
frame is always kept); the record says so, because it departs from the 1 fps prefix the other models see.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P  # noqa: E402

THIRD = P.ROOT / "third_party"


def _chdir_repo(repo: Path) -> None:
    """Each repo imports its own package by name (`src.`, `llava.`, `cambrian.`) relative to its root, and
    some resolve checkpoints relative to the working directory, so run the model call from there."""
    os.chdir(repo)
    sys.path.insert(0, str(repo))


def _pil(frames) -> list:
    out = []
    for f in frames:
        im = f if isinstance(f, Image.Image) else Image.open(f)
        out.append(im.convert("RGB").resize((P.FRAME_W, P.FRAME_H)))
    return out


def _subsample(clip, max_frames: int):
    if not max_frames or len(clip) <= max_frames:
        return clip
    idx = np.unique(np.round(np.linspace(0, len(clip) - 1, max_frames)).astype(int))
    if idx[-1] != len(clip) - 1:
        idx[-1] = len(clip) - 1
    return [clip[i] for i in idx]


def _variant_ids(tokenizer, word: str) -> list[int]:
    """First-token ids of every casing / leading-space spelling of a one-word answer ('Yes', 'yes', ' YES', ...)."""
    ids = set()
    for w in {word, word.lower(), word.upper(), word.capitalize()}:
        for s in (w, " " + w):
            t = tokenizer.encode(s, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


def _first_token_probs(scores0: torch.Tensor, pos_ids: list[int], neg_ids: list[int]):
    full = torch.softmax(scores0[0].float(), dim=-1)
    p_pos, p_neg = float(full[pos_ids].sum()), float(full[neg_ids].sum())
    return p_pos / max(p_pos + p_neg, 1e-12), p_pos, p_neg


# ------------------------------------------------------------------------------------------------
class SpatialMLLM:
    """Spatial-MLLM v1.1: Qwen2.5-VL + VGGT spatial encoder.  Mirrors src/inference.py (video as a frame list)."""

    max_frames_default = 16  # the released inference / VSI-Bench evaluation use nframes=16

    def __init__(self, model_path: str):
        _chdir_repo(THIRD / "Spatial-MLLM")
        from transformers import Qwen2_5_VLProcessor
        from src.qwenvl.model.spatial_mllm import SpatialMLLMConfig, SpatialMLLMForConditionalGeneration
        from src.inference import prepare_spatial_mllm_inputs
        from qwen_vl_utils import process_vision_info

        self._prep, self._pvi = prepare_spatial_mllm_inputs, process_vision_info
        config = SpatialMLLMConfig.from_pretrained(model_path)
        self.model = SpatialMLLMForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype="bfloat16",
            device_map="cuda",
            attn_implementation="flash_attention_2",
        ).eval()
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        tok = self.processor.tokenizer
        self.ids = lambda w: _variant_ids(tok, w)

    def ask(self, clip, text: str, words):
        frames = _pil(clip)
        if len(frames) < 2:
            frames = frames * 2
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "sample_fps": P.SAMPLE_FPS},
                    {"type": "text", "text": text},
                ],
            }
        ]  # official order: video first
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._pvi(messages)
        batch = self.processor(
            text=[prompt],
            images=image_inputs or None,
            videos=video_inputs or None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        batch = self._prep(batch, video_inputs, image_inputs)
        batch.to(self.model.device)
        if batch.get("video_tchw") is not None:
            batch["video_tchw"] = [v.to(self.model.device) for v in batch["video_tchw"]]
        if batch.get("image_tchw") is not None:
            batch["image_tchw"] = [v.to(self.model.device) for v in batch["image_tchw"]]
        with torch.no_grad():
            out = self.model.generate(
                **batch,
                max_new_tokens=4,
                do_sample=False,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen = out.sequences[0, batch["input_ids"].shape[1] :]
        reply = self.processor.batch_decode(gen[None], skip_special_tokens=True)[0]
        p = _first_token_probs(out.scores[0], self.ids(words[0]), self.ids(words[1]))
        return reply, *p, int(batch["input_ids"].shape[1])


class VLM3R:
    """VLM-3R (LLaVA-NeXT-Video-7B-Qwen2 + LoRA + CUT3R).  Mirrors playground/demo/image_demo.py: one <image>
    per frame, conv template qwen_1_5, modalities=["image"] * N; CUT3R runs inside the model."""

    max_frames_default = 0
    pool_mode = "average"  # image_demo / video_demo.sh setting; the official VSI-Bench eval (VLM3RVideo) uses bilinear

    def __init__(self, model_path: str, model_base: str = "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"):
        _chdir_repo(THIRD / "VLM-3R")
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

        self.IMAGE_TOKEN_INDEX, self.DEFAULT_IMAGE_TOKEN = IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        self.conv_templates, self.tokenizer_image_token = conv_templates, tokenizer_image_token
        name = get_model_name_from_path(model_path)
        overwrite = {"mm_spatial_pool_mode": self.pool_mode, "mm_spatial_pool_stride": 2, "mm_newline_position": "grid"}
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, model_base, name, overwrite_config=overwrite
        )
        self.model.to("cuda").eval()
        if self.tokenizer.pad_token_id is None and "qwen" in self.tokenizer.name_or_path.lower():
            self.tokenizer.pad_token_id = 151643
        self.ids = lambda w: _variant_ids(self.tokenizer, w)

    def ask(self, clip, text: str, words):
        frames = _pil(clip)
        pix = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        images = [im.half().cuda() for im in pix]
        placeholders = (self.DEFAULT_IMAGE_TOKEN + "\n") * len(frames)
        conv = self.conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], text + "\n" + placeholders.rstrip("\n"))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        attn = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()
        with torch.inference_mode():
            out = self.model.generate(
                inputs=input_ids,
                images=images,
                attention_mask=attn,
                modalities=["image"] * len(frames),
                do_sample=False,
                max_new_tokens=4,
                num_beams=1,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        reply = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        p = _first_token_probs(out.scores[0], self.ids(words[0]), self.ids(words[1]))
        return reply, *p, int(input_ids.shape[1])


class CambrianS:
    """Cambrian-S (standard checkpoints).  Mirrors lmms-eval/lmms_eval/models/simple/cambrians.py: multi-image
    path (process_images with use_pad), <image> * N + question, conv template qwen_2."""

    max_frames_default = 0

    def __init__(self, model_path: str):
        _chdir_repo(THIRD / "cambrian-s")
        from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from cambrian.conversation import conv_templates
        from cambrian.model.builder import load_pretrained_model
        from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

        self.IMAGE_TOKEN_INDEX, self.DEFAULT_IMAGE_TOKEN = IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        self.conv_templates, self.tokenizer_image_token, self.process_images = (
            conv_templates,
            tokenizer_image_token,
            process_images,
        )
        name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, name, device_map="cuda:0"
        )
        self.model.eval()
        self.ids = lambda w: _variant_ids(self.tokenizer, w)

    def ask(self, clip, text: str, words):
        frames = _pil(clip)
        tensors, sizes = self.process_images(frames, self.image_processor, self.model.config, use_pad=True)
        qs = self.DEFAULT_IMAGE_TOKEN * len(frames) + "\n" + text
        conv = self.conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        tensors = [t.half().cuda() for t in tensors]
        with torch.inference_mode():
            out = self.model.generate(
                inputs=input_ids,
                images=tensors,
                image_sizes=sizes,
                use_cache=True,
                do_sample=False,
                num_beams=1,
                max_new_tokens=4,
                output_scores=True,
                return_dict_in_generate=True,
            )
        reply = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        p = _first_token_probs(out.scores[0], self.ids(words[0]), self.ids(words[1]))
        return reply, *p, int(input_ids.shape[1])


class CambrianSVideo(CambrianS):
    """Cambrian-S through its OFFICIAL EVALUATION path (lmms-eval/evaluate_all_in_one.sh ->
    lmms_eval/models/simple/cambrians.py, model_args video_max_frames=64, miv_token_len=64, si_token_len=729; class
    defaults video_fps=1, no force_sample, no time instruction): the clip is one video as in process_videos
    (expand2square with the processor mean, image_sizes=(W, H, T)), one <image> token before the question, 64 tokens per
    frame.  The 1 fps prefix is used as is up to 64 frames and uniformly resampled above that with np.linspace(0, n-1,
    64, dtype=int), which keeps the first and the current frame."""

    max_frames_default = 0  # sampling happens in ask(), like process_video_with_decord
    n_frames = 64

    def __init__(self, model_path: str):
        super().__init__(model_path)
        from cambrian.mm_utils import expand2square

        self._expand2square = expand2square
        cfg = self.model.config
        cfg.video_max_frames, cfg.video_fps, cfg.video_force_sample, cfg.add_time_instruction = (
            self.n_frames,
            1,
            False,
            False,
        )
        cfg.miv_token_len, cfg.si_token_len, cfg.image_aspect_ratio, cfg.anyres_max_subimages = 64, 729, "anyres", 9

    def ask(self, clip, text: str, words):
        n = len(clip)
        idx = np.linspace(0, n - 1, self.n_frames, dtype=int).tolist() if n > self.n_frames else list(range(n))
        frames = _pil([clip[i] for i in idx])
        procs = self.image_processor if isinstance(self.image_processor, (list, tuple)) else [self.image_processor]
        tensors = []
        for proc in procs:
            square = [self._expand2square(im, tuple(int(x * 255) for x in proc.image_mean)) for im in frames]
            tensors.append(torch.stack([proc.preprocess(square, return_tensors="pt")["pixel_values"]]))  # 1, T, C, H, W
        sizes = [(frames[0].width, frames[0].height, len(frames))]  # W, H, T
        if getattr(self.model.config, "mm_use_im_start_end", False):
            from cambrian.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

            qs = DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            qs = self.DEFAULT_IMAGE_TOKEN * len(tensors) + "\n" + text
        conv = self.conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        tensors = [t.half().cuda() for t in tensors]
        with torch.inference_mode():
            out = self.model.generate(
                inputs=input_ids,
                images=tensors,
                image_sizes=sizes,
                use_cache=True,
                do_sample=False,
                num_beams=1,
                max_new_tokens=4,
                output_scores=True,
                return_dict_in_generate=True,
            )
        reply = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        p = _first_token_probs(out.scores[0], self.ids(words[0]), self.ids(words[1]))
        return reply, *p, int(input_ids.shape[1])


class CambrianSVideoSampledText(CambrianSVideo):
    """Frame-cap ablation (not a bench setting): cambrians_video, but once the prefix exceeds the 64-frame cap the
    first STATE_PROMPT sentence states how the frames were sampled instead of claiming 1 frame per second."""

    first_sentence = "This is a video recorded at 1 frame per second, walking through an indoor scene."

    def ask(self, clip, text: str, words):
        n = len(clip)
        if n > self.n_frames:
            if self.first_sentence not in text:
                raise ValueError("STATE_PROMPT first sentence changed; update CambrianSVideoSampledText")
            text = text.replace(
                self.first_sentence,
                f"This is a video walking through an indoor scene, shown as {self.n_frames} frames "
                f"sampled uniformly from the {n} seconds recorded so far.",
                1,
            )
        return super().ask(clip, text, words)


class CambrianSVideoNoCap(CambrianSVideo):
    """Frame-cap ablation (not a bench setting): cambrians_video with every 1 fps frame of the prefix (no 64-frame cap)."""

    n_frames = 10**6


class VLM3RVideo(VLM3R):
    """VLM-3R through its OFFICIAL EVALUATION path (thinking-in-space/eval_vlm_3r_vsibench.sh ->
    lmms_eval/models/vlm_3r.py): the clip is one video tensor (`images=[video]`, modalities="video"), one <image> token
    before the question, mm_spatial_pool_mode=bilinear / stride 2 / grid newline, max_frames_num=32 with pyav-style
    sampling np.linspace(0, n-1, min(n, 32), dtype=int) (short prefixes keep every frame, the last index is the current
    frame), no force_sample and no time instruction.  CUT3R runs inside the model on the same frames."""

    max_frames_default = 0  # sampling happens in ask(), like the official loader
    pool_mode = "bilinear"
    n_frames = 32

    def ask(self, clip, text: str, words):
        n = len(clip)
        idx = np.linspace(0, n - 1, min(n, self.n_frames), dtype=int).tolist()
        frames = _pil([clip[i] for i in idx])
        video = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
        qs = self.DEFAULT_IMAGE_TOKEN + "\n" + text
        conv = self.conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        attn = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()
        with torch.inference_mode():
            out = self.model.generate(
                inputs=input_ids,
                images=[video],
                attention_mask=attn,
                modalities="video",
                do_sample=False,
                max_new_tokens=4,
                num_beams=1,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        reply = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        p = _first_token_probs(out.scores[0], self.ids(words[0]), self.ids(words[1]))
        return reply, *p, int(input_ids.shape[1])


class VLM3RVideoDemo(VLM3RVideo):
    """VLM-3R exactly as playground/demo/video_demo.py with this checkpoint's config (force_sample=True,
    add_time_instruction=True, for_get_frames_num 32): the clip 0..t is always resampled to 32 frame indices with
    np.linspace(0, n-1, 32, dtype=int) (short prefixes repeat frames; the last index is the current frame), and the
    question is preceded by the demo's time instruction built from those indices at 1 fps.  vlm3r_video (above) instead
    caps at 32 without repeating and has no time instruction."""

    max_frames_default = 0  # sampling happens inside ask(), like the demo's load_video
    pool_mode = "average"  # scripts/video/demo/video_demo.sh / README `32 2 average grid True`
    n_frames = 32

    def ask(self, clip, text: str, words):
        n = len(clip)
        cfg = self.model.config
        if n > self.n_frames or getattr(cfg, "force_sample", False):
            idx = np.linspace(0, n - 1, self.n_frames, dtype=int).tolist()
        else:
            idx = list(range(n))
        frames = _pil([clip[i] for i in idx])
        video = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
        qs = text
        if getattr(cfg, "add_time_instruction", False):
            fps = P.SAMPLE_FPS
            frame_time = ",".join(f"{i / fps:.2f}s" for i in idx)
            qs = (
                f"The video lasts for {n / fps:.2f} seconds, and {len(idx)} frames are uniformly sampled from it. "
                f"These frames are located at {frame_time}.Please answer the following questions related to this video.\n{qs}"
            )
        qs = self.DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = self.conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        attn = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()
        with torch.inference_mode():
            out = self.model.generate(
                inputs=input_ids,
                images=[video],
                attention_mask=attn,
                modalities="video",
                do_sample=False,
                max_new_tokens=4,
                num_beams=1,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        reply = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        p = _first_token_probs(out.scores[0], self.ids(words[0]), self.ids(words[1]))
        return reply, *p, int(input_ids.shape[1])


class DeepSeekVL2:
    """DeepSeek-VL2 (deepseek-ai/deepseek-vl2-small, deepseek-ai/deepseek-vl2) through its OFFICIAL multi-image path
    (third_party/DeepSeek-VL2 inference.py + README "Simple Inference Example with Multiple Images"):
    DeepseekVLV2Processor
    with the `deepseek` SFT template and an empty system prompt, one "<image>\n" placeholder per frame before the question,
    bf16, prepare_inputs_embeds -> language.generate, greedy.  The model has no video mode.
    With more than two images the processor turns dynamic tiling off (processing_deepseek_vl_v2.py:273
    `cropping=len(pil_images) <= 2`): every frame is the 384x384 padded global view plus one 384x384 local tile, i.e.
    14*15 + 1 + 14*15 = 421 tokens, and the language model context is max_position_embeddings=4096 with no rope scaling,
    so at most 9 frames fit next to the prompt.  PROTOCOL CAVEAT: a prefix longer than n_frames is resampled with
    np.linspace(0, n-1, n_frames, dtype=int) (first and current frame kept); prefixes up to 9 s are shown in full."""

    max_frames_default = 0  # sampling happens in ask(), like CambrianSVideo
    n_frames = 9
    max_new_tokens = 4

    def __init__(self, model_path: str):
        _chdir_repo(THIRD / "DeepSeek-VL2")
        from transformers import AutoModelForCausalLM
        from deepseek_vl2.models import (
            DeepseekVLV2Processor,
        )  # importing the package registers the config/model with Auto*

        self.processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
            .cuda()
            .eval()
        )
        self.max_len = int(self.model.config.language_config.max_position_embeddings)
        self.ids = lambda w: _variant_ids(self.tokenizer, w)

    def prepare(self, frames: list, text: str):
        """Official conversation format (inference.py:81-91): placeholders first, question after, empty system prompt."""
        conversation = [
            {"role": "<|User|>", "content": "<image>\n" * len(frames) + text},
            {"role": "<|Assistant|>", "content": ""},
        ]
        return self.processor(conversations=conversation, images=frames, force_batchify=True, system_prompt="")

    def ask(self, clip, text: str, words):
        n = len(clip)
        idx = np.linspace(0, n - 1, self.n_frames, dtype=int).tolist() if n > self.n_frames else list(range(n))
        frames = _pil([clip[i] for i in idx])
        inputs = self.prepare(frames, text).to(self.model.device, dtype=torch.bfloat16)
        n_tok = int(inputs.input_ids.shape[1])
        if n_tok + self.max_new_tokens > self.max_len:
            raise ValueError(
                f"{n_tok} input tokens for {len(frames)} frames exceed the {self.max_len}-token context; lower n_frames"
            )
        tok = self.tokenizer
        with torch.no_grad():
            inputs_embeds = self.model.prepare_inputs_embeds(**inputs)
            out = self.model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                pad_token_id=tok.eos_token_id,
                bos_token_id=tok.bos_token_id,
                eos_token_id=tok.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        # generate() from inputs_embeds returns only the new tokens (README decodes outputs[0] as is)
        reply = tok.decode(out.sequences[0].cpu().tolist(), skip_special_tokens=True).strip()
        p = _first_token_probs(out.scores[0], self.ids(words[0]), self.ids(words[1]))
        return reply, *p, n_tok


class LLaVAVideo:
    """LLaVA-Video-7B/72B-Qwen2 (lmms-lab) through its OFFICIAL EVALUATION path: lmms-eval `llava_vid` as run in
    third_party/LLaVA-NeXT/docs/LLaVA_Video_1003.md "Reproducing Evaluation Results" (L122-133):
        --model llava_vid --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average
    plus the class defaults of lmms-eval lmms_eval/models/simple/llava_vid.py (main; same in v0.3.0, the release-era
    file): video_fps=1, mm_spatial_pool_stride=2, mm_newline_position=grid, mm_resampler_location=before, overwrite=True
    (L75-110, L144-158), force_sample=False, add_time_instruction=False, torch_dtype=float16, attn_implementation=sdpa,
    tie_weights, Qwen pad=bos 151643
    (L200-205), prompt "<image>\n" + question (mm_use_im_start_end False), generate(images=[video], modalities=["video"], temperature 0
    -> greedy, num_beams 1, KeywordsStoppingCriteria(conv.sep)) (L434-560).  Frame sampling = load_video(fps=1,
    max_frames_num=64, force_sample=False) (L331-350): every 1 fps frame while n <= 64, else np.linspace(0, n-1, 64,
    dtype=int) (keeps the first and the current frame) - exactly our 1 fps prefix with a 64 cap.  The eval setting has
    NO time instruction (only the model card / playground/demo/video_demo.py add it, with force_sample=True; see
    llava_video_demo).  Model code: the untouched upstream clone third_party/LLaVA-NeXT (llava/model/builder.py L206-228
    qwen path, L288-294 vision tower; llava_arch.py get_2dPool average ->
    13x13 + 13 newline tokens per frame, 64 frames = 11,648 visual tokens), put first on sys.path so it wins over the
    vlm3r env's editable VLM-3R fork.  Env: the VLM-3R environment (torch 2.1.1+cu121, transformers 4.40.0 = the
    LLaVA-NeXT pins) with
    `unset LD_LIBRARY_PATH`.  device_map: cuda:0 with one visible GPU (lmms-eval single process), "auto" with several ->
    the 72B (fp16 ~145 GB) needs 2 H200.  Overrides for smoke tests: LLAVA_VIDEO_DTYPE=float16|bfloat16,
    LLAVA_VIDEO_DEVICE_MAP.
    """

    max_frames_default = 0  # sampling happens in ask(), like load_video
    n_frames = 64
    conv_template = "qwen_1_5"
    overwrite_config = {
        "mm_spatial_pool_stride": 2,
        "mm_spatial_pool_mode": "average",
        "mm_pooling_position": "before",
        "mm_newline_position": "grid",
        "delay_load": False,
    }
    force_sample = False
    add_time_instruction = False
    torch_dtype = "float16"
    attn_implementation = "sdpa"

    def __init__(self, model_path: str):
        repo = THIRD / "LLaVA-NeXT"
        _chdir_repo(repo)
        import llava

        if not str(Path(llava.__file__).resolve()).startswith(str(repo.resolve())):
            raise ImportError(f"llava resolved to {llava.__file__}, not the upstream clone {repo}")
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

        self.IMAGE_TOKEN_INDEX, self.DEFAULT_IMAGE_TOKEN = IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        self.conv_templates, self.tokenizer_image_token, self.KeywordsStoppingCriteria = (
            conv_templates,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        self.torch_dtype = os.environ.get("LLAVA_VIDEO_DTYPE", self.torch_dtype)
        device_map = os.environ.get("LLAVA_VIDEO_DEVICE_MAP") or ("auto" if torch.cuda.device_count() > 1 else "cuda:0")
        name = get_model_name_from_path(model_path)  # "LLaVA-Video-7B-Qwen2" -> builder's llava + qwen branch
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path,
            None,
            name,
            device_map=device_map,
            torch_dtype=self.torch_dtype,
            overwrite_config=dict(self.overwrite_config) if self.overwrite_config else None,
            attn_implementation=self.attn_implementation,
        )
        self.model.eval()
        self.model.tie_weights()
        if self.tokenizer.pad_token_id is None and "qwen" in self.tokenizer.name_or_path.lower():
            self.tokenizer.pad_token_id = 151643
        self.device = self.model.device
        self.ids = lambda w: _variant_ids(self.tokenizer, w)
        print(
            f"LLaVAVideo: device_map {device_map}  dtype {self.torch_dtype}  pool {self.model.config.mm_spatial_pool_mode}/"
            f"{getattr(self.model.config, 'mm_spatial_pool_stride', 2)}  newline {self.model.config.mm_newline_position}  "
            f"force_sample {self.force_sample}  time_instruction {self.add_time_instruction}  gpus {torch.cuda.device_count()}",
            flush=True,
        )

    def sample(self, n: int) -> list[int]:
        if n > self.n_frames or self.force_sample:
            return np.linspace(0, n - 1, self.n_frames, dtype=int).tolist()
        return list(range(n))

    def ask(self, clip, text: str, words):
        n = len(clip)
        idx = self.sample(n)
        frames = _pil([clip[i] for i in idx])
        video = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(
            self.device
        )  # T, 3, 384, 384
        video = video.bfloat16() if self.torch_dtype == "bfloat16" else video.half()
        qs = text
        if self.add_time_instruction:  # video_demo.py L210 / lmms-eval L546-548 wording, times at our 1 fps
            fps = P.SAMPLE_FPS
            frame_time = ",".join(f"{i / fps:.2f}s" for i in idx)
            qs = (
                f"The video lasts for {n / fps:.2f} seconds, and {len(idx)} frames are uniformly sampled from it. "
                f"These frames are located at {frame_time}.Please answer the following questions related to this video.\n{qs}"
            )
        qs = self.DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = self.conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        attn = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.device)
        stop = self.KeywordsStoppingCriteria([conv.sep], self.tokenizer, input_ids)
        with torch.inference_mode():
            out = self.model.generate(
                inputs=input_ids,
                images=[video],
                attention_mask=attn,
                modalities=["video"],
                use_cache=True,
                stopping_criteria=[stop],
                do_sample=False,
                num_beams=1,
                max_new_tokens=4,
                output_scores=True,
                return_dict_in_generate=True,
            )
        reply = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        p = _first_token_probs(out.scores[0], self.ids(words[0]), self.ids(words[1]))
        return reply, *p, int(input_ids.shape[1])


class LLaVAVideoDemo(LLaVAVideo):
    """LLaVA-Video as in the model card / docs/LLaVA_Video_1003.md "Inference" (L55-66) and
    playground/demo/video_demo.py (not the bench setting): no overwrite_config, so the checkpoint's own
    mm_spatial_pool_mode=bilinear (14x14 + 14 tokens per frame), bfloat16, and the checkpoint config's force_sample=True
    (always 64 indices; short prefixes repeat frames) and add_time_instruction=True (time instruction built from those
    indices at 1 fps, before the question)."""

    overwrite_config = None
    force_sample = True
    add_time_instruction = True
    torch_dtype = "bfloat16"


class VST:
    """VST (Visual Spatial Tuning, third_party/VST): plain Qwen2.5-VL weights (rayruiyang/VST-*).  Mirrors the official
    video evaluation path (benchmark/eval_vsibench.py): system "You are a helpful assistant.", video FIRST then the
    question, video pixel budget total_pixels 8192*28*28 with min_pixels 16*28*28 (qwen_vl_utils spreads the budget over
    the frames, so long prefixes get smaller frames), no thinking prompt.  Frames are our 1 fps grid (their fps=4 only
    samples video files).  The official eval decodes with temperature 0.1 / top_p 0.001 (near-greedy); we use greedy
    like every other runner.  flash_attention_2 when installed, else sdpa (same math)."""

    max_frames_default = 0
    TOTAL_PIXELS, MIN_PIXELS = 8192 * 28 * 28, 16 * 28 * 28

    def __init__(self, model_path: str):
        import importlib.util
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info

        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self._pvi = process_vision_info
        tok = self.processor.tokenizer
        self.ids = lambda w: _variant_ids(tok, w)
        print(f"VST backend: attn {attn}, total_pixels {self.TOTAL_PIXELS}, min_pixels {self.MIN_PIXELS}", flush=True)

    def ask(self, clip, text: str, words):
        frames = _pil(clip)
        if len(frames) < 2:
            frames = frames * 2
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "total_pixels": self.TOTAL_PIXELS,
                        "min_pixels": self.MIN_PIXELS,
                        "sample_fps": P.SAMPLE_FPS,
                    },
                    {"type": "text", "text": text},
                ],
            },
        ]  # official order: video first
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = self._pvi(
            messages, return_video_kwargs=True, return_video_metadata=True
        )
        video_inputs, metas = map(list, zip(*video_inputs))
        metas = [dict(m, fps=P.SAMPLE_FPS) for m in metas]
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=metas,
            padding=True,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen = out.sequences[0, inputs["input_ids"].shape[1] :]
        reply = self.processor.batch_decode(gen[None], skip_special_tokens=True)[0]
        p = _first_token_probs(out.scores[0], self.ids(words[0]), self.ids(words[1]))
        return reply, *p, int(inputs["input_ids"].shape[1])


# ------------------------------------------------------------------------------------------------
class SpatialTTT:
    """Spatial-TTT (third_party/Spatial-TTT, THU-SI/Spatial-TTT-nano): Qwen3-VL-2B-Instruct whose attention layers are
    wrapped with LaCT test-time-training layers (fast weights updated online over the video chunks).  Mirrors the
    official
    VSI-Bench evaluation (evaluation/spatial/scripts/eval_spatial_ttt_2b.sh + lmms_eval/models/simple/spatial_ttt.py):
    load_spatial_ttt_model(base Qwen3-VL-2B config, nano safetensors) with num_lact_heads 4, w0_w2_low_rank 0, no fused
    kernel, conv layer on, lact_chunk_size = window_size = 2648, lact_layers 0/1/2/4/.../26, flash_attention_2; greedy
    decoding through generate_with_spatial_ttt (their manual prefill + decode loop).  Inputs follow our Qwen3-VL polling
    runner (runners/qwen3vl.build_inputs: 1 fps frame list as one video at 512x384, odd prefix padded without inventing
    a future second) instead of the adapter's image-list trick (every frame duplicated, which doubles the timestamps).
    The first-token Yes/No probability is captured from the logits their loop samples from (no extra forward)."""

    max_frames_default = 0
    BASE = os.environ.get("SPATIALTTT_BASE", "Qwen/Qwen3-VL-2B-Instruct")
    LACT = dict(
        num_lact_heads=4,
        w0_w2_low_rank=0,
        use_fused_kernel=False,
        use_conv_layer=True,
        lact_chunk_size=2648,
        window_size=2648,
        lact_layers="0/1/2/4/5/6/8/9/10/12/13/14/16/17/18/20/21/22/24/25/26",
    )

    def __init__(self, model_path: str, adapter: str | None = None):
        repo = Path(__file__).resolve().parents[1] / "third_party" / "Spatial-TTT" / "qwen-vl-finetune"
        sys.path.insert(0, str(repo))
        import models.spatial_ttt as ST
        from transformers import AutoProcessor
        import qwen3vl as Q

        _orig_resolve = ST._resolve_safetensors  # their helper returns a PosixPath; this accelerate wants a str
        ST._resolve_safetensors = lambda cp: str(_orig_resolve(cp))
        load_spatial_ttt_model = ST.load_spatial_ttt_model
        self.Q = Q
        self.model = load_spatial_ttt_model(
            model_path=self.BASE, checkpoint_path=model_path, device="cuda", torch_dtype=torch.bfloat16, **self.LACT
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.BASE)
        tok = self.processor.tokenizer
        self.ids = lambda w: _variant_ids(tok, w)
        self.eos = tok.convert_tokens_to_ids("<|im_end|>")
        # Their _set_kv_cache_layer indexes DynamicCache.layers[layer_idx] before the cache has grown to that layer
        # (transformers 4.57 grows the layer list lazily in update()), so hand their loop a cache that is pre-grown.
        from transformers.cache_utils import DynamicCache

        n_layers = int(self.model.config.text_config.num_hidden_layers)

        class _PregrownCache(DynamicCache):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                while len(self.layers) < n_layers:
                    self.layers.append(self.layer_class_to_replicate())

        ST.DynamicCache = _PregrownCache
        if adapter:  # our SFT run: LoRA on the language tower, merged back into the same model the runner uses
            from peft import PeftModel

            inner = PeftModel.from_pretrained(self.model._model, adapter)
            lm = inner.base_model
            if "model" in lm.__dict__ and "model" not in lm._modules:
                lm._modules["model"] = lm.__dict__.pop("model")  # see src/training/train_spatial_ttt._fix_peft
            self.model._model = inner.merge_and_unload()
            self.model.model = self.model._model.model
            self.model.lm_head = self.model._model.lm_head
            print(f"Spatial-TTT adapter merged: {adapter}", flush=True)
        print(
            f"Spatial-TTT backend: base {self.BASE}, checkpoint {model_path}, {self.LACT}, cache pre-grown to {n_layers} layers",
            flush=True,
        )

    def _build_inputs(self, frames, text: str):
        """Same clip handling as runners/qwen3vl.build_inputs (1 fps video, odd prefix padded with the last seen
        second) but with the official adapter's system prompt in front."""
        Q = self.Q
        n_seen = len(frames)
        clip = list(frames) + ([frames[-1]] if n_seen % 2 else [])
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "video",
                        "video": clip,
                        "resized_height": P.FRAME_H,
                        "resized_width": P.FRAME_W,
                        "sample_fps": P.SAMPLE_FPS,
                    },
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True
        )
        video_inputs, metas = map(list, zip(*video_inputs))
        for m in metas:  # the padded copy must not read as a future second (same fix as qwen3vl.build_inputs)
            n_enc = len(m.get("frames_indices", [])) or len(clip)
            m["fps"] = P.SAMPLE_FPS
            m["frames_indices"] = [min(j, n_seen - 1) for j in range(n_enc)]
            m["total_num_frames"] = n_seen
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=metas,
            padding=True,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        return inputs.to("cuda")

    def ask(self, clip, text: str, words):
        frames = _pil(clip)
        inputs = self._build_inputs(frames, text)
        captured = []
        orig = self.model._sample_token

        def hook(logits, *a, **k):
            if not captured:
                captured.append(logits.detach()[:, -1, :].float())
            return orig(logits, *a, **k)

        self.model._sample_token = hook
        try:
            with torch.no_grad():
                out = self.model.generate_with_spatial_ttt(
                    inputs["input_ids"],
                    pixel_values_videos=inputs.get("pixel_values_videos"),
                    video_grid_thw=inputs.get("video_grid_thw"),
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    max_new_tokens=1,
                    do_sample=False,
                    eos_token_id=self.eos,
                )  # the answer word only: the checkpoint
                # keeps generating VSI-Bench style digits after "Yes"/"No" ("No1.1", "Yes010") instead of <|im_end|>
        finally:
            self.model._sample_token = orig
        gen = out[0, inputs["input_ids"].shape[1] :]
        reply = self.processor.batch_decode(gen[None], skip_special_tokens=True)[0]
        p = _first_token_probs(captured[0], self.ids(words[0]), self.ids(words[1]))
        return reply, *p, int(inputs["input_ids"].shape[1])


BACKENDS = {
    "spatial_mllm": SpatialMLLM,
    "vlm3r": VLM3R,
    "vlm3r_video": VLM3RVideo,
    "vlm3r_video_demo": VLM3RVideoDemo,
    "cambrians": CambrianS,
    "cambrians_video": CambrianSVideo,
    "cambrians_video_sampled_text": CambrianSVideoSampledText,
    "cambrians_video_nocap": CambrianSVideoNoCap,
    "deepseek_vl2": DeepSeekVL2,
    "llava_video": LLaVAVideo,
    "llava_video_demo": LLaVAVideoDemo,
    "vst": VST,
    "spatial_ttt": SpatialTTT,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-base", default=None, help="VLM-3R: LoRA base (default lmms-lab/LLaVA-NeXT-Video-7B-Qwen2)")
    ap.add_argument("--adapter", default=None, help="Spatial-TTT: our LoRA adapter directory, merged at load")
    ap.add_argument("--task", required=True, choices=sorted(P.TASKS))
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--frames", default=P.DEFAULT_FRAMES)
    ap.add_argument("--frames-by-dataset", action="store_true")
    ap.add_argument("--episode-id", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--start-field", default=None)
    ap.add_argument(
        "--score-from-field",
        default=None,
        help="episode field giving the first scored second (Re-finding: show_len); earlier Yes answers are recorded, "
        "not scored",
    )
    ap.add_argument("--no-stop", action="store_true")
    ap.add_argument("--state-override", default=None)
    ap.add_argument(
        "--max-frames", type=int, default=-1, help="-1 = backend default (Spatial-MLLM 16, others none); 0 = no cap"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = P.TASKS[args.task]
    args.episodes = args.episodes or cfg["episodes"]
    args.out = str(Path(args.out).resolve())
    eps = P.select_episodes(args)
    words = ("Yes", "No")
    kw = {"model_base": args.model_base} if (args.backend.startswith("vlm3r") and args.model_base) else {}
    if args.backend == "spatial_ttt" and args.adapter:
        kw["adapter"] = args.adapter
    backend = BACKENDS[args.backend](args.model, **kw)
    max_frames = backend.max_frames_default if args.max_frames < 0 else args.max_frames
    print(
        f"{args.task} polling [{args.backend}] {len(eps)} episode(s) shard {args.shard}  {P.FRAME_W}x{P.FRAME_H} @ {P.SAMPLE_FPS} fps"
        f"  step {args.step}  max_frames {max_frames or 'none'}  model {args.model}\ntext: {args.state_override or cfg['state']}\n",
        flush=True,
    )

    def ask(clip, text):
        return backend.ask(_subsample(clip, max_frames), text, words)

    counts: dict[str, int] = {}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for k, ep in enumerate(eps, 1):
        has_event = bool(ep.get("has_event", ep.get("timely_start_i") is not None))
        t0 = time.time()
        rec = {
            "model": args.model,
            "backend": args.backend,
            "task": args.task,
            "mode": "polling",
            "episode_id": ep["episode_id"],
            "label": ep.get("label"),
            "frames": len(ep["frame_ids"]),
            "size": [P.FRAME_W, P.FRAME_H],
            "step": args.step,
            "has_event": has_event,
            "timely": [ep["timely_start_i"], ep["timely_end_i"]] if has_event else None,
            "poll_start": int(ep.get(args.start_field, 0)) if args.start_field else 0,
            "score_from": int(ep.get(args.score_from_field, 0)) if args.score_from_field else 0,
            "max_frames": max_frames,
            "style": "state",
            "answer_words": list(words),
        }
        try:
            root = Path(P.DATASET_FRAMES[ep["dataset"]]) if args.frames_by_dataset else Path(args.frames)
            frames = P.episode_frames(ep, root, cfg.get("frame0_bbox", False))
            question = P.fill(args.state_override or cfg["state"], ep)
            res = P.poll_loop(
                ask,
                frames,
                question,
                words,
                args.step,
                stop_at_first=not args.no_stop,
                start=rec["poll_start"],
                score_from=rec["score_from"],
            )
            rec.update(question=question, **res)
            v = "invalid" if res["invalid"] else P.verdict(ep, res["response_i"], has_event)
        except Exception as e:  # keep going; the row records the failure
            import traceback

            rec.update(error=traceback.format_exc()[-1500:])
            v = "error"
        rec["verdict"] = v
        rec["seconds"] = round(time.time() - t0, 1)
        counts[v] = counts.get(v, 0) + 1
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"[{k}/{len(eps)}] {ep['episode_id']:26s} {str(ep.get('label'))[:14]:14s} {len(ep['frame_ids']):4d}f  "
            f"timely {rec['timely']} -> response {rec.get('response_i')}  {v:18s} ({rec.get('n_calls', 0)} calls, {rec['seconds']:.0f}s)",
            flush=True,
        )
    n = sum(counts.values())
    print(
        "\nRESULT "
        + "  ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in sorted(counts.items()))
        + f"  -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
