import cv2
import torch
import numpy as np



class DiffBIR_sample:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
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

    def sample(self, image):
        return (image)

    # def sample(self, SUPIR_model, latents, steps, seed, cfg_scale_end, EDM_s_churn, s_noise, positive, negative,
    #             cfg_scale_start, control_scale_start, control_scale_end, restore_cfg, keep_model_loaded, DPMPP_eta,
    #             sampler, sampler_tile_size=1024, sampler_tile_stride=512):
        
    #     torch.manual_seed(seed)
    #     device = mm.get_torch_device()
    #     mm.unload_all_models()
    #     mm.soft_empty_cache()

    #     self.sampler_config = {
    #         'target': f'.sgm.modules.diffusionmodules.sampling.{sampler}',
    #         'params': {
    #             'num_steps': steps,
    #             'restore_cfg': restore_cfg,
    #             's_churn': EDM_s_churn,
    #             's_noise': s_noise,
    #             'discretization_config': {
    #                 'target': '.sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization'
    #             },
    #             'guider_config': {
    #                 'target': '.sgm.modules.diffusionmodules.guiders.LinearCFG',
    #                 'params': {
    #                     'scale': cfg_scale_start,
    #                     'scale_min': cfg_scale_end
    #                 }
    #             }
    #         }
    #     }
    #     if 'Tiled' in sampler:
    #         self.sampler_config['params']['tile_size'] = sampler_tile_size // 8
    #         self.sampler_config['params']['tile_stride'] = sampler_tile_stride // 8
    #     if 'DPMPP' in sampler:
    #         self.sampler_config['params']['eta'] = DPMPP_eta
    #         self.sampler_config['params']['restore_cfg'] = -1
    #     if not hasattr (self,'sampler') or self.sampler_config != self.current_sampler_config: 
    #         self.sampler = instantiate_from_config(self.sampler_config)
    #         self.current_sampler_config = self.sampler_config
 
    #     print("sampler_config: ", self.sampler_config)
        
    #     SUPIR_model.denoiser.to(device)
    #     SUPIR_model.model.diffusion_model.to(device)
    #     SUPIR_model.model.control_model.to(device)
        
    #     use_linear_control_scale = control_scale_start != control_scale_end

    #     denoiser = lambda input, sigma, c, control_scale: SUPIR_model.denoiser(SUPIR_model.model, input, sigma, c, control_scale)

    #     original_size = positive['original_size']
    #     positive = positive['cond']
    #     negative = negative['uncond']
    #     samples = latents["samples"]
    #     samples = samples.to(device)
    #     #print("positives: ", len(positive))
    #     #print("negatives: ", len(negative))
    #     out = []
    #     pbar = comfy.utils.ProgressBar(samples.shape[0])
    #     for i, sample in enumerate(samples):
    #         try:
    #             if 'original_size' in latents:
    #                 print("Using random noise")
    #                 noised_z = torch.randn_like(sample.unsqueeze(0), device=samples.device)
    #             else:
    #                 print("Using latent from input")
    #                 noised_z = sample.unsqueeze(0) * 0.13025
    #             if len(positive) != len(samples):
    #                 print("Tiled sampling")
    #                 _samples = self.sampler(denoiser, noised_z, cond=positive, uc=negative, x_center=sample.unsqueeze(0), control_scale=control_scale_end,
    #                                 use_linear_control_scale=use_linear_control_scale, control_scale_start=control_scale_start)
    #             else:
    #                 #print("positives[i]: ", len(positive[i]))
    #                 #print("negatives[i]: ", len(negative[i]))
    #                 _samples = self.sampler(denoiser, noised_z, cond=positive[i], uc=negative[i], x_center=sample.unsqueeze(0), control_scale=control_scale_end,
    #                                         use_linear_control_scale=use_linear_control_scale, control_scale_start=control_scale_start)

                
    #         except torch.cuda.OutOfMemoryError as e:
    #             mm.free_memory(mm.get_total_memory(mm.get_torch_device()), mm.get_torch_device())
    #             SUPIR_model = None
    #             mm.soft_empty_cache()
    #             print("It's likely that too large of an image or batch_size for SUPIR was used,"
    #                   " and it has devoured all of the memory it had reserved, you may need to restart ComfyUI. Make sure you are using tiled_vae, "
    #                   " you can also try using fp8 for reduced memory usage if your system supports it.")
    #             raise e
    #         out.append(_samples)
    #         print("Sampled ", i+1, " of ", samples.shape[0])
    #         pbar.update(1)

    #     if not keep_model_loaded:
    #         SUPIR_model.denoiser.to('cpu')
    #         SUPIR_model.model.diffusion_model.to('cpu')
    #         SUPIR_model.model.control_model.to('cpu')
    #         mm.soft_empty_cache()

    #     if len(out[0].shape) == 4:
    #         samples_out_stacked = torch.cat(out, dim=0)
    #     else:
    #         samples_out_stacked = torch.stack(out, dim=0)

    #     return ({"samples":samples_out_stacked, "original_size": original_size},)