from typing import overload, Generator, Dict

import torch
from omegaconf import OmegaConf

from ..model.cldm import ControlLDM
from ..model.gaussian_diffusion import Diffusion
from ..model.bsrnet import RRDBNet
from ..model.scunet import SCUNet
from ..model.swinir import SwinIR
from ..utils.common import instantiate_from_config, load_file_from_url, count_vram_usage

MODELS = {
    ### stage_1 model weights
    "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
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


class Stage2_load:

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
        control_sd = load_model_from_url(MODELS["v2"])

        cldm.load_controlnet_from_ckpt(control_sd)
        cldm.eval().to(device)
        diffusion: Diffusion = instantiate_from_config(OmegaConf.load("custom_nodes/ComfyUI-DiffBIR/configs/inference/diffusion.yaml"))
        diffusion.to(device)

        return cldm, diffusion
    

class Stage1_load:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "task": (
                    [
                        'bsr',
                        'bfr',
                        'bid',
                    ], {
                        "default": 'bsr'
                    }),
            "device": (
                    [
                        'cuda',
                        'cpu',
                    ], {
                        "default": 'cuda'
                    }),

            }
        }

    RETURN_TYPES = ("STAGE1", "TASK")
    RETURN_NAMES = ("stage1_model", "task")
    FUNCTION = "init_stage1"
    CATEGORY = "DiffBIR"
    DESCRIPTION = """"""

    def init_stage1(self, task, device):
        if task == 'bsr':
            bsrnet: RRDBNet = instantiate_from_config(OmegaConf.load("custom_nodes/ComfyUI-DiffBIR/configs/inference/bsrnet.yaml"))
            sd = load_model_from_url(MODELS["bsrnet"])
            bsrnet.load_state_dict(sd, strict=True)
            bsrnet.eval().to(device)
            stage1_model = bsrnet
        elif task == 'bfr':
            swinir_face: SwinIR = instantiate_from_config(OmegaConf.load("custom_nodes/ComfyUI-DiffBIR/configs/inference/swinir.yaml"))
            sd = load_model_from_url(MODELS["swinir_face"])
            swinir_face.load_state_dict(sd, strict=True)
            swinir_face.eval().to(device)
            stage1_model = swinir_face
        elif task == 'bid':
            scunet_psnr: SCUNet = instantiate_from_config(OmegaConf.load("custom_nodes/ComfyUI-DiffBIR/configs/inference/scunet.yaml"))
            sd = load_model_from_url(MODELS["scunet_psnr"])
            scunet_psnr.load_state_dict(sd, strict=True)
            scunet_psnr.eval().to(device)
            stage1_model = scunet_psnr
        
        return (stage1_model, task, )
    

class Simple_load:

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

    RETURN_TYPES = ("STAGE1", "CLDM", "DIFFUSION")
    RETURN_NAMES = ("stage1_model", "cldm", "diffusion")
    FUNCTION = "simple_load"
    CATEGORY = "DiffBIR"
    DESCRIPTION = """"""

    def simple_load(self, device):
        bsrnet: RRDBNet = instantiate_from_config(OmegaConf.load("custom_nodes/ComfyUI-DiffBIR/configs/inference/bsrnet.yaml"))
        sd = load_model_from_url(MODELS["bsrnet"])
        bsrnet.load_state_dict(sd, strict=True)
        bsrnet.eval().to(device)

        cldm: ControlLDM = instantiate_from_config(OmegaConf.load("custom_nodes/ComfyUI-DiffBIR/configs/inference/cldm.yaml"))
        sd = load_model_from_url(MODELS["sd_v21"])
        unused = cldm.load_pretrained_sd(sd)
        print(f"strictly load pretrained sd_v2.1, unused weights: {unused}")
        ### load controlnet
        control_sd = load_model_from_url(MODELS["v2"])

        cldm.load_controlnet_from_ckpt(control_sd)
        cldm.eval().to(device)
        diffusion: Diffusion = instantiate_from_config(OmegaConf.load("custom_nodes/ComfyUI-DiffBIR/configs/inference/diffusion.yaml"))
        diffusion.to(device)

        return (bsrnet, cldm, diffusion)