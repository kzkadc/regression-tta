from dataclasses import dataclass, InitVar

import torch
from torch import Tensor
from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics.regression.r2_score import R2Score

from model import Regressor
from utils.pca_basis import get_pca_basis
from evaluation.metrics import ModelDistanceMetric, PearsonCorrelation, FeatureKLMetric


@dataclass
class RegressionEvaluator(Engine):
    net: Regressor
    pc_config: InitVar[dict | None] = None
    raw_feat_pt: InitVar[str | None] = None
    compile_model: InitVar[dict | None] = None

    def __post_init__(self,
                      pc_config: dict | None,
                      raw_feat_pt: str | None,
                      compile_model: dict | None):
        super().__init__(self.inference)

        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")

        ModelDistanceMetric(self.net).attach(self, "model_dist")

        if pc_config is not None:
            self.mean, self.pc_basis, self.pc_var = get_pca_basis(
                regress_weight=self.net.regressor.weight, **pc_config)

            self.mean = self.mean.cuda()
            self.pc_basis = self.pc_basis.cuda()
            self.pc_var = self.pc_var.cuda()

            raw_features = torch.load(raw_feat_pt)
            self.raw_var = torch.var(raw_features, dim=0).cuda()

            pc_ot = lambda d: d["feat_pc"]
            zeros = torch.zeros_like(self.pc_var)
            FeatureKLMetric(pc_ot, zeros, self.pc_var, reverse_kl=False) \
                .attach(self, "pc_kl")
            FeatureKLMetric(pc_ot, zeros, self.pc_var, reverse_kl=True) \
                .attach(self, "rev_pc_kl")

            raw_ot = lambda d: d["feat_raw"]
            FeatureKLMetric(raw_ot, self.mean, self.raw_var, reverse_kl=False) \
                .attach(self, "raw_kl")
            FeatureKLMetric(raw_ot, self.mean, self.raw_var, reverse_kl=True) \
                .attach(self, "rev_raw_kl")
        else:
            self.mean = None

        if compile_model is None:
            self.feature_extractor = self.net.feature
        else:
            try:
                self.feature_extractor = torch.compile(
                    self.net.feature, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")
                self.feature_extractor = self.net.feature

    @torch.no_grad()
    def inference(self,
                  engine: Engine,
                  batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        self.net.eval()

        x, y = batch
        x = x.cuda()
        y = y.float().flatten().cuda()

        feature = self.feature_extractor(x)
        y_pred = self.net.predict_from_feature(feature)

        output = {
            "y_pred": y_pred,
            "y": y,
            "feat_raw": feature
        }
        if self.mean is not None:
            output["feat_pc"] = (feature - self.mean) @ self.pc_basis
        return output
