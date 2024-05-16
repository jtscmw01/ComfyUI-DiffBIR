import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
great_grandparent_dir = os.path.dirname(grandparent_dir)
sys.path.append(great_grandparent_dir)

from typing import overload, Tuple, Optional
import time

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from PIL import Image
from einops import rearrange


import comfy
from comfy import model_management

from ..model.cldm import ControlLDM
from ..model.gaussian_diffusion import Diffusion
from ..model.bsrnet import RRDBNet
from ..model.swinir import SwinIR
from ..model.scunet import SCUNet
from ..utils.sampler import SpacedSampler
from ..utils.cond_fn import Guidance
from ..utils.common import wavelet_decomposition, wavelet_reconstruction, count_vram_usage


def bicubic_resize(img: np.ndarray, scale: float) -> np.ndarray:
    pil = Image.fromarray(img)
    res = pil.resize(tuple(int(x * scale) for x in pil.size), Image.BICUBIC)
    return np.array(res)


def resize_short_edge_to(imgs: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h == w:
        new_h, new_w = size, size
    elif h < w:
        new_h, new_w = size, int(w * (size / h))
    else:
        new_h, new_w = int(h * (size / w)), size
    return F.interpolate(imgs, size=(new_h, new_w), mode="bicubic", antialias=True)


def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    # get_pad = lambda x: (x // multiple + 1) * multiple - x
    get_pad = lambda x: (x // multiple + int(x % multiple != 0)) * multiple - x
    ph, pw = get_pad(h), get_pad(w)
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)


class Pipeline:

    def __init__(self, stage1_model: nn.Module, cldm: ControlLDM, diffusion: Diffusion, cond_fn: Optional[Guidance], device: str) -> None:
        self.stage1_model = stage1_model
        self.cldm = cldm
        self.diffusion = diffusion
        self.cond_fn = cond_fn
        self.device = device
        self.final_size: Tuple[int] = None

    def set_final_size(self, lq: torch.Tensor) -> None:
        h, w = lq.shape[1:3]
        self.final_size = (h, w)

    @overload
    def run_stage1(self, lq: torch.Tensor) -> torch.Tensor:
        ...

    @count_vram_usage
    def run_stage2(
        self,
        clean: torch.Tensor,
        steps: int,
        strength: float,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        better_start: float
    ) -> torch.Tensor:
        ### preprocess
        bs, _, ori_h, ori_w = clean.shape
        # pad: ensure that height & width are multiples of 64
        pad_clean = pad_to_multiples_of(clean, multiple=64)
        h, w = pad_clean.shape[2:]
        # prepare conditon
        if not tiled:
            cond = self.cldm.prepare_condition(pad_clean, [pos_prompt] * bs)
            uncond = self.cldm.prepare_condition(pad_clean, [neg_prompt] * bs)
        else:
            cond = self.cldm.prepare_condition_tiled(pad_clean, [pos_prompt] * bs, tile_size, tile_stride)
            uncond = self.cldm.prepare_condition_tiled(pad_clean, [neg_prompt] * bs, tile_size, tile_stride)
        if self.cond_fn:
            self.cond_fn.load_target(pad_clean * 2 - 1)
        old_control_scales = self.cldm.control_scales
        self.cldm.control_scales = [strength] * 13
        if better_start:
            # using noised low frequency part of condition as a better start point of 
            # reverse sampling, which can prevent our model from generating noise in 
            # image background.
            _, low_freq = wavelet_decomposition(pad_clean)
            if not tiled:
                x_0 = self.cldm.vae_encode(low_freq)
            else:
                x_0 = self.cldm.vae_encode_tiled(low_freq, tile_size, tile_stride)
            x_T = self.diffusion.q_sample(
                x_0,
                torch.full((bs, ), self.diffusion.num_timesteps - 1, dtype=torch.long, device=self.device),
                torch.randn(x_0.shape, dtype=torch.float32, device=self.device)
            )
            # print(f"diffusion sqrt_alphas_cumprod: {self.diffusion.sqrt_alphas_cumprod[-1]}")
        else:
            x_T = torch.randn((bs, 4, h // 8, w // 8), dtype=torch.float32, device=self.device)
        ### run sampler
        sampler = SpacedSampler(self.diffusion.betas)
        z = sampler.sample(
            model=self.cldm, device=self.device, steps=steps, batch_size=bs, x_size=(4, h // 8, w // 8),
            cond=cond, uncond=uncond, cfg_scale=cfg_scale, x_T=x_T, progress=True,
            progress_leave=True, cond_fn=self.cond_fn, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        if not tiled:
            x = self.cldm.vae_decode(z)
        else:
            x = self.cldm.vae_decode_tiled(z, tile_size // 8, tile_stride // 8)
        ### postprocess
        self.cldm.control_scales = old_control_scales
        sample = x[:, :, :ori_h, :ori_w]
        return sample

    @torch.no_grad()
    def run(
        self,
        lq: np.ndarray,
        steps: int,
        strength: float,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        stage1_tile: bool,
        stage1_tile_size: int,
        stage1_tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        better_start: bool
    ) -> np.ndarray:
        # image to tensor
        # lq = torch.tensor((lq / 255.).clip(0, 1), dtype=torch.float32, device=self.device)
        lq = torch.tensor(lq, dtype=torch.float32, device=self.device)
        lq = rearrange(lq, "n h w c -> n c h w").contiguous()
        # set pipeline output size
        
        t_s = time.time()
        self.set_final_size(lq)
        clean = self.run_stage1(lq, stage1_tile, stage1_tile_size, stage1_tile_stride)
        t_e = time.time()
        print(f'stage1 time: {t_e - t_s}')

        t_s = time.time()
        sample = self.run_stage2(
            clean, steps, strength, tiled, tile_size, tile_stride,
            pos_prompt, neg_prompt, cfg_scale, better_start
        )
        t_e = time.time()
        print(f'stage2 time: {t_e - t_s}')

        # colorfix (borrowed from StableSR, thanks for their work)
        sample = (sample + 1) / 2
        sample = wavelet_reconstruction(sample, clean)
        # resize to desired output size
        sample = F.interpolate(sample, size=self.final_size, mode="bicubic", antialias=True)
        # tensor to image
        sample = rearrange(sample, "n c h w -> n h w c")

        return sample


class BSRNetPipeline(Pipeline):

    def __init__(self, bsrnet: RRDBNet, cldm: ControlLDM, diffusion: Diffusion, cond_fn: Optional[Guidance], device: str, upscale: float) -> None:
        super().__init__(bsrnet, cldm, diffusion, cond_fn, device)
        self.upscale = upscale
        self.stage1_scale = 4

    def set_final_size(self, lq: torch.Tensor) -> None:
        h, w = lq.shape[2:]
        self.final_size = (int(h * self.upscale), int(w * self.upscale))

    def tile_process(self, image, tile_size, tile_stride, upscale_model):
        device = model_management.get_torch_device()

        memory_required = model_management.module_size(upscale_model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(self.stage1_scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.to(device)

        tile = tile_size
        overlap = tile_size - tile_stride

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=self.stage1_scale, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        # upscale_model.cpu()
        s = torch.clamp(s, min=0, max=1.0)
        s = s.to(device)

        return s

    @count_vram_usage
    def run_stage1(self, lq: torch.Tensor, stage1_tile, tile_size=512, tile_stride=256) -> torch.Tensor:
        # NOTE: default upscale 4x in stage1
        if stage1_tile:
            clean = self.tile_process(lq, tile_size, tile_stride, self.stage1_model)
        else:
            clean = self.stage1_model(lq)

        if min(self.final_size) < 512:
            clean = resize_short_edge_to(clean, size=512)
        else:
            clean = F.interpolate(clean, size=self.final_size, mode="bicubic", antialias=True)

        return clean


class SwinIRPipeline(Pipeline):

    def __init__(self, swinir: SwinIR, cldm: ControlLDM, diffusion: Diffusion, cond_fn: Optional[Guidance], device: str, upscale: float) -> None:
        super().__init__(swinir, cldm, diffusion, cond_fn, device)
        self.upscale = upscale
        self.stage1_scale = 1

    def set_final_size(self, lq: torch.Tensor) -> None:
        h, w = lq.shape[2:]
        self.final_size = (int(h * self.upscale), int(w * self.upscale))

    def tile_process(self, image, tile_size, tile_stride, upscale_model):
        device = model_management.get_torch_device()

        memory_required = model_management.module_size(upscale_model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(self.stage1_scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.to(device)

        tile = tile_size
        overlap = tile_size - tile_stride

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=self.stage1_scale, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        # upscale_model.cpu()
        s = torch.clamp(s, min=0, max=1.0)
        s = s.to(device)

        return s

    @count_vram_usage
    def run_stage1(self, lq: torch.Tensor, stage1_tile, tile_size=512, tile_stride=256) -> torch.Tensor:
        # NOTE: default upscale 4x in stage1
        if stage1_tile:
            clean = self.tile_process(lq, tile_size, tile_stride, self.stage1_model)
        else:
            clean = self.stage1_model(lq)

        if min(self.final_size) < 512:
            clean = resize_short_edge_to(clean, size=512)
        else:
            clean = F.interpolate(clean, size=self.final_size, mode="bicubic", antialias=True)

        return clean


class SCUNetPipeline(Pipeline):

    def __init__(self, scunet: SCUNet, cldm: ControlLDM, diffusion: Diffusion, cond_fn: Optional[Guidance], device: str, upscale: float) -> None:
        super().__init__(scunet, cldm, diffusion, cond_fn, device)
        self.upscale = upscale
        self.stage1_scale = 1

    def set_final_size(self, lq: torch.Tensor) -> None:
        h, w = lq.shape[2:]
        self.final_size = (int(h * self.upscale), int(w * self.upscale))

    def tile_process(self, image, tile_size, tile_stride, upscale_model):
        device = model_management.get_torch_device()

        memory_required = model_management.module_size(upscale_model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(self.stage1_scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.to(device)

        tile = tile_size
        overlap = tile_size - tile_stride

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=self.stage1_scale, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        # upscale_model.cpu()
        s = torch.clamp(s, min=0, max=1.0)
        s = s.to(device)

        return s

    @count_vram_usage
    def run_stage1(self, lq: torch.Tensor, stage1_tile, tile_size=512, tile_stride=256) -> torch.Tensor:
        # NOTE: default upscale 4x in stage1
        if stage1_tile:
            clean = self.tile_process(lq, tile_size, tile_stride, self.stage1_model)
        else:
            clean = self.stage1_model(lq)

        if min(self.final_size) < 512:
            clean = resize_short_edge_to(clean, size=512)
        else:
            clean = F.interpolate(clean, size=self.final_size, mode="bicubic", antialias=True)

        return clean