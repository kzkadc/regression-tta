import torch
from torch import Tensor

import numpy as np


@torch.no_grad()
def get_pca_basis(stat_file: str, contrib_top_k: int) -> tuple[Tensor, Tensor, Tensor]:
    stat_dict = torch.load(stat_file)
    print(f"load {stat_file!r}")

    mean: Tensor
    basis: Tensor
    eigvals: Tensor
    mean, basis, eigvals = stat_dict["mean"], stat_dict["basis"], stat_dict["eigvals"]
    print(f"mean: {mean.shape}, basis: {basis.shape}, eigvals: {eigvals.shape}")

    eigval_topk_set = np.argsort(eigvals.numpy())[-contrib_top_k:]
    indices = torch.from_numpy(eigval_topk_set).long()

    pca_basis = basis[:, indices].float()   # (dim, idx)
    pc_vars = eigvals[indices].float()

    print(f"basis: {pca_basis.shape}, vars: {pc_vars.shape}")

    return mean, pca_basis, pc_vars
