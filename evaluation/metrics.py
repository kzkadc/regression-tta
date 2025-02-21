from typing import Any
from collections.abc import Callable
import copy

import torch
from torch import nn, Tensor
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

import numpy as np

from utils.loss import diagonal_gaussian_kl_loss


class ModelDistanceMetric(Metric):
    def __init__(self,
                 source_model: nn.Module,
                 device: torch.device = torch.device("cpu")):
        super().__init__(lambda x: x, device)

        self.cur_model = source_model
        self.source_model = copy.deepcopy(source_model)
        self.source_model.cpu()

    def reset(self):
        super().reset()
        self._distance = -1.0

    @torch.no_grad()
    def update(self, *_):
        if self._distance >= 0:
            return

        cur_params: dict[str, Tensor] = dict(self.cur_model.named_parameters())
        source_params: dict[str, Tensor] = dict(
            self.source_model.named_parameters())
        dist = torch.tensor(0).float()
        for k, p in cur_params.items():
            sp = source_params[k]
            dist += (p.cpu() - sp).square().sum()

        self._distance = float(dist.sqrt().item())

    def compute(self) -> float:
        if self._distance < 0:
            raise NotComputableError("distance has not been computed yet.")
        return self._distance


class PearsonCorrelation(Metric):
    def reset(self):
        self._labels: list[np.ndarray] = []
        self._predictions: list[np.ndarray] = []

    def update(self, output: tuple[Tensor, Tensor]):
        y_pred, y = output
        self._labels.append(y.flatten().cpu().numpy())
        self._predictions.append(y_pred.cpu().numpy())

    def compute(self) -> float:
        labels = np.concatenate(self._labels)
        predictions = np.concatenate(self._predictions)

        r = np.corrcoef(labels, predictions)[0, 1]
        return float(r)


class FeatureKLMetric(Metric):
    def __init__(self, output_transform: Callable,
                 target_mean: Tensor, target_var: Tensor,
                 reverse_kl: bool = False):
        self.target_mean = target_mean.cpu()
        self.target_var = target_var.cpu()

        self.reverse_kl = reverse_kl

        super().__init__(output_transform)

    def reset(self):
        self._features = []

    def update(self, output: Tensor):
        self._features.append(output.cpu())

    def compute(self) -> float:
        assert len(self._features) >= 1, "No features accumulated"

        features = torch.cat(self._features)
        mean = features.mean(dim=0)
        var = features.var(dim=0)

        if self.reverse_kl:
            loss = diagonal_gaussian_kl_loss(self.target_mean, self.target_var,
                                             mean, var)
        else:
            loss = diagonal_gaussian_kl_loss(
                mean, var, self.target_mean, self.target_var)
        return float(loss.item())


class OptimizerLastState(Metric):
    def __init__(self, opt: torch.optim.Optimizer, key: str):
        super().__init__(lambda x: (None, None))

        self.opt = opt
        self.key = key

    def reset(self):
        self._last_state = None

    def update(self, *_):
        self._last_state = self.opt.param_groups[0][self.key]

    def compute(self) -> Any:
        return self._last_state
