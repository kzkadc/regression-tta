from dataclasses import dataclass, InitVar

import torch
from torch import Tensor
import torch.nn.functional as F
from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics.regression.r2_score import R2Score

from model import Regressor
from evaluation.metrics import PearsonCorrelation, OptimizerLastState


@dataclass
class RegressionTrainer(Engine):
    net: Regressor
    opt: torch.optim.Optimizer
    compile_model: InitVar[dict | None]

    def __post_init__(self, compile_model: dict | None):
        super().__init__(self.update)

        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse")
        MeanAbsoluteError(y_ot).attach(self, "mae")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")

        OptimizerLastState(self.opt, "lr").attach(self, "lr")

        if compile_model is not None:
            try:
                self.net = torch.compile(
                    self.net, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")

    def update(self,
               engine: Engine,
               batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        self.net.train()
        self.opt.zero_grad()

        x, y = batch
        x = x.cuda()
        y = y.float().flatten().cuda()

        y_pred = self.net(x)

        loss = F.mse_loss(y_pred, y)

        loss.backward()
        self.opt.step()

        return {
            "y_pred": y_pred,
            "y": y
        }
