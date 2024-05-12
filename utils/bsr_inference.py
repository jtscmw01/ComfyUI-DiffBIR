import os
from typing import overload, Generator, Dict
from argparse import Namespace

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from ..utils.common import load_file_from_url, count_vram_usage
from ..utils.helpers import (
    Pipeline,
    BSRNetPipeline
)
from ..utils.cond_fn import MSEGuidance, WeightedMSEGuidance


MODELS = {
    ### stage_1 model weights
    "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
    # the following checkpoint is up-to-date, but we use the old version in our paper
    # "swinir_face": "https://github.com/zsyOAOA/DifFace/releases/download/V1.0/General_Face_ffhq512.pth",
    "swinir_face": "https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt",
    "scunet_psnr": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    "swinir_general": "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
    ### stage_2 model weights
    "sd_v21": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
    "v1_face": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth",
    "v1_general": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth",
    "v2": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth"
}


def load_model_from_url(url: str) -> Dict[str, torch.Tensor]:
    sd_path = load_file_from_url(url, model_dir="models/diffbir")
    sd = torch.load(sd_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if list(sd.keys())[0].startswith("module"):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


class InferenceLoop:

    def __init__(self, args: Namespace) -> "InferenceLoop":
        self.args = args
        self.loop_ctx = {}
        self.pipeline: Pipeline = None
        self.init_cond_fn()

    @overload
    def init_stage1_model(self) -> None:
        ...

    @count_vram_usage
    def init_stage2_model(self, cldm, diffusion) -> None:
        self.cldm = cldm
        self.diffusion = diffusion

    def init_cond_fn(self) -> None:
        if not self.args.guidance:
            self.cond_fn = None
            return
        if self.args.g_loss == "mse":
            cond_fn_cls = MSEGuidance
        elif self.args.g_loss == "w_mse":
            cond_fn_cls = WeightedMSEGuidance
        else:
            raise ValueError(self.args.g_loss)
        self.cond_fn = cond_fn_cls(
            scale=self.args.g_scale, t_start=self.args.g_start, t_stop=self.args.g_stop,
            space=self.args.g_space, repeat=self.args.g_repeat
        )

    @overload
    def init_pipeline(self) -> None:
        ...

    @torch.no_grad()
    def run(self) -> None:
        # We don't support batch processing since input images may have different size
        loader = [self.args.input[0]]

        for lq in loader:
            sample = self.pipeline.run(
                lq[None], self.args.steps, 1.0, self.args.tiled,
                self.args.tile_size, self.args.tile_stride,
                self.args.pos_prompt, self.args.neg_prompt, self.args.cfg_scale,
                self.args.better_start
            )[0]
            print(sample.shape)
            return sample.unsqueeze(0)


class BSRInferenceLoop(InferenceLoop):

    @count_vram_usage
    def init_stage1_model(self, stage1_model) -> None:
        self.bsrnet = stage1_model

    def init_pipeline(self) -> None:
        self.pipeline = BSRNetPipeline(self.bsrnet, self.cldm, self.diffusion, self.cond_fn, self.args.device, self.args.upscale)