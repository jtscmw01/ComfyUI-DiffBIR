from typing import overload, Generator, Dict
import os

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



def find_file(file_name, root_dir='.'):
    for root, _, files in os.walk(root_dir):
        if file_name in files:
            return os.path.join(root, file_name)
    return None


def load_model_from_url(url: str) -> Dict[str, torch.Tensor]:
    current_directory = os.getcwd()
    current_directory_contents = os.listdir(current_directory)

    if "ComfyUI" in current_directory_contents and "custom_nodes" not in current_directory_contents:
        model_dir = os.path.join(current_directory, "ComfyUI", "models", "diffbir")
    else:
        model_dir = os.path.join(current_directory, "models", "diffbir")

    sd_path = load_file_from_url(url, model_dir=model_dir)
    sd = torch.load(sd_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if list(sd.keys())[0].startswith("module"):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


class Stage2_load:

    def __init__(self):
        current_directory = os.getcwd()
        current_directory_contents = os.listdir(current_directory)

        # if "ComfyUI" in current_directory_contents and "custom_nodes" not in current_directory_contents:
        #     self.pre_path = os.path.join(current_directory, "ComfyUI", "custom_nodes", "ComfyUI-DiffBIR", "configs", "inference")
        # else:
        self.pre_path = os.path.join(current_directory, "custom_nodes", "ComfyUI-DiffBIR", "configs", "inference")
        
    
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

        config_path = os.path.join(self.pre_path, "cldm.yaml")
        if not os.path.isfile(config_path):
            config_path = find_file("cldm.yaml")
        cldm: ControlLDM = instantiate_from_config(OmegaConf.load(config_path))
        sd = load_model_from_url(MODELS["sd_v21"])
        unused = cldm.load_pretrained_sd(sd)
        print(f"strictly load pretrained sd_v2.1, unused weights: {unused}")
        ### load controlnet
        control_sd = load_model_from_url(MODELS["v2"])

        cldm.load_controlnet_from_ckpt(control_sd)
        cldm.eval().to(device)
        config_path = os.path.join(self.pre_path, "diffusion.yaml")
        if not os.path.isfile(config_path):
            config_path = find_file("diffusion.yaml")
        diffusion: Diffusion = instantiate_from_config(OmegaConf.load(config_path))
        diffusion.to(device)

        return cldm, diffusion
    

class Stage1_load:

    def __init__(self):
        current_directory = os.getcwd()
        current_directory_contents = os.listdir(current_directory)

        # if "ComfyUI" in current_directory_contents and "custom_nodes" not in current_directory_contents:
        #     self.pre_path = os.path.join(current_directory, "ComfyUI", "custom_nodes", "ComfyUI-DiffBIR", "configs", "inference")
        # else:
        self.pre_path = os.path.join(current_directory, "custom_nodes", "ComfyUI-DiffBIR", "configs", "inference")
        
    
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
            config_path = os.path.join(self.pre_path, "bsrnet.yaml")
            if not os.path.isfile(config_path):
                config_path = find_file("bsrnet.yaml")
            bsrnet: RRDBNet = instantiate_from_config(OmegaConf.load(config_path))
            sd = load_model_from_url(MODELS["bsrnet"])
            bsrnet.load_state_dict(sd, strict=True)
            bsrnet.eval().to(device)
            stage1_model = bsrnet
        elif task == 'bfr':
            config_path = os.path.join(self.pre_path, "swinir.yaml")
            if not os.path.isfile(config_path):
                config_path = find_file("swinir.yaml")
            swinir_face: SwinIR = instantiate_from_config(OmegaConf.load(config_path))
            sd = load_model_from_url(MODELS["swinir_face"])
            swinir_face.load_state_dict(sd, strict=True)
            swinir_face.eval().to(device)
            stage1_model = swinir_face
        elif task == 'bid':
            config_path = os.path.join(self.pre_path, "scunet.yaml")
            if not os.path.isfile(config_path):
                config_path = find_file("scunet.yaml")
            scunet_psnr: SCUNet = instantiate_from_config(OmegaConf.load(config_path))
            sd = load_model_from_url(MODELS["scunet_psnr"])
            scunet_psnr.load_state_dict(sd, strict=True)
            scunet_psnr.eval().to(device)
            stage1_model = scunet_psnr
        
        return (stage1_model, task, )
    

class Simple_load:

    def __init__(self):
        current_directory = os.getcwd()
        current_directory_contents = os.listdir(current_directory)

        # if "ComfyUI" in current_directory_contents and "custom_nodes" not in current_directory_contents:
        #     self.pre_path = os.path.join(current_directory, "ComfyUI", "custom_nodes", "ComfyUI-DiffBIR", "configs", "inference")
        # else:
        self.pre_path = os.path.join(current_directory, "custom_nodes", "ComfyUI-DiffBIR", "configs", "inference")
    
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

        config_path = os.path.join(self.pre_path, "bsrnet.yaml")
        if not os.path.isfile(config_path):
            config_path = find_file("bsrnet.yaml")
        print('1:', config_path)
        bsrnet: RRDBNet = instantiate_from_config(OmegaConf.load(config_path))
        sd = load_model_from_url(MODELS["bsrnet"])
        bsrnet.load_state_dict(sd, strict=True)
        bsrnet.eval().to(device)

        config_path = os.path.join(self.pre_path, "cldm.yaml")
        if not os.path.isfile(config_path):
            config_path = find_file("cldm.yaml")
        print('2:', config_path)
        cldm: ControlLDM = instantiate_from_config(OmegaConf.load(config_path))
        sd = load_model_from_url(MODELS["sd_v21"])
        unused = cldm.load_pretrained_sd(sd)
        print(f"strictly load pretrained sd_v2.1, unused weights: {unused}")
        ### load controlnet
        control_sd = load_model_from_url(MODELS["v2"])

        cldm.load_controlnet_from_ckpt(control_sd)
        cldm.eval().to(device)
        config_path = os.path.join(self.pre_path, "diffusion.yaml")
        if not os.path.isfile(config_path):
            config_path = find_file("diffusion.yaml")
        print('3:', config_path)
        diffusion: Diffusion = instantiate_from_config(OmegaConf.load(config_path))
        diffusion.to(device)

        return (bsrnet, cldm, diffusion)