import os

import torch
import torch.backends.cudnn as torch_backends_cudnn

import numpy as np


def fix_seed(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_backends_cudnn.benchmark = False
