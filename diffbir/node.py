import cv2
import torch
import numpy as np



class DiffBIR_sample:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "upscale_ratio": ("FLOAT", {"default": 2, "min": 0.1, "max": 8.0, "step": 0.1}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 0xffffffffffffffff, "step": 1}),
            "better_start": ("BOOLEAN", {"default": True}),
            "tiled": ("BOOLEAN", {"default": True}),
            "tile_size": ("INT", {"default": 512, "min": 1, "max": 0xffffffffffffffff, "step": 1}),
            "tile_stride": ("INT", {"default": 256, "min": 1, "max": 0xffffffffffffffff, "step": 1}),
            "pos_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "neg_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "seed": ("INT", {"default": 123, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "device": (
                    [
                        'cuda',
                        'cpu',
                        'mps',
                    ], {
                        "default": 'cuda'
                    }),

        }, "optional": {

            "guidance": ("BOOLEAN", {"default": False}),
            "g_loss": (
                    [
                        'mse',
                        'w_mse',
                    ], {
                        "default": 'w_mse'
                    }),
            "g_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "g_start": ("INT", {"default": 1001, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "g_stop": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "step": 1}),
            "g_space": (
                    [
                        'latent',
                        'rgb',
                    ], {
                        "default": 'latent'
                    }),
            "g_repeat": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "DiffBIR"
    DESCRIPTION = """"""

    def sample(self, image, upscale_ratio, steps, better_start, tiled, tile_size, tile_stride, pos_prompt, neg_prompt, 
               seed, device, guidance, g_loss, g_scale, g_start, g_stop, g_space, g_repeat):
        
        return (image,)