import os

# FlashAttention-2 (trellis default) requires Ampere+ (sm_80+). Turing and older GPUs
# (e.g. RTX 2080 Ti) must use PyTorch SDPA for dense attention and xformers for sparse.
try:
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        os.environ.setdefault("ATTN_BACKEND", "sdpa")
        os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
except Exception:
    pass

from . import models
from . import modules
from . import pipelines
from . import renderers
from . import representations
from . import utils
