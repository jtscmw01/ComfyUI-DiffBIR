from .diffbir.sampler import DiffBIR_sample
from .diffbir.load import DiffBIR_load

NODE_CLASS_MAPPINGS = {
    "DiffBIR_sample": DiffBIR_sample,
    "DiffBIR_load": DiffBIR_load,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffBIR_sample": "DiffBIR Sampler",
    "DiffBIR_load": "DiffBIR loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]