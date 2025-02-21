from torch import Tensor


def diagonal_gaussian_kl_loss(m1: Tensor, v1: Tensor,
                              m2: Tensor, v2: Tensor,
                              eps: float = 0.0,
                              dim_reduction: str = "sum") -> Tensor:
    loss = (v2.log() - v1.log() + (v1 + (m2 - m1).square()) / (v2 + eps) - 1) / 2
    match dim_reduction:
        case "sum":
            return loss.sum()
        case "mean":
            return loss.mean()
        case "none":
            return loss

        case _:
            raise ValueError(f"Invalid dim_reduction: {dim_reduction!r}")
