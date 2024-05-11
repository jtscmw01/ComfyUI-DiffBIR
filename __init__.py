from .diffbir.node import DiffBIR_sample

NODE_CLASS_MAPPINGS = {
    "DiffBIR_sample": DiffBIR_sample,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffBIR_sample": "DiffBIR Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]