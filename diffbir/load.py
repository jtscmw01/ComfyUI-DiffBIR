from typing import overload, Generator, Dict

import torch
from omegaconf import OmegaConf

from ..model.cldm import ControlLDM
from ..model.gaussian_diffusion import Diffusion
from ..model.bsrnet import RRDBNet
from ..utils.common import instantiate_from_config, load_file_from_url, count_vram_usage

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


class DiffBIR_Load:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "device": (
                    [
                        'cuda',
                        'cpu',
                    ], {
                        "default": 'cuda'
                    }),

            }
        }

    RETURN_TYPES = ("CLDM", "DIFFUSION")
    RETURN_NAMES = ("cldm", "diffusion")
    FUNCTION = "init_stage2"
    CATEGORY = "DiffBIR"
    DESCRIPTION = """"""

    def init_stage2(self, device):
        cldm: ControlLDM = instantiate_from_config(OmegaConf.load("custom_nodes/ComfyUI-DiffBIR/configs/inference/cldm.yaml"))
        sd = load_model_from_url(MODELS["sd_v21"])
        unused = cldm.load_pretrained_sd(sd)
        print(f"strictly load pretrained sd_v2.1, unused weights: {unused}")
        ### load controlnet
        # if args.version == "v1":
        #     if args.task == "fr":
        #         control_sd = load_model_from_url(MODELS["v1_face"])
        #     elif args.task == "sr":
        #         control_sd = load_model_from_url(MODELS["v1_general"])
        #     else:
        #         raise ValueError(f"DiffBIR v1 doesn't support task: {args.task}, please use v2 by passsing '--version v2'")
        # else:
        control_sd = load_model_from_url(MODELS["v2"])

        cldm.load_controlnet_from_ckpt(control_sd)
        cldm.eval().to(device)
        diffusion: Diffusion = instantiate_from_config(OmegaConf.load("custom_nodes/ComfyUI-DiffBIR/configs/inference/diffusion.yaml"))
        diffusion.to(device)

        return cldm, diffusion