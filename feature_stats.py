from dataclasses import dataclass, InitVar
from pprint import pprint
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from ignite.engine import Engine

import numpy as np
import yaml

from dataset import get_datasets
from model import create_regressor, Regressor


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=True,
                        help="config file")
    parser.add_argument("-o", required=True, help="output directory")
    parser.add_argument("--save_feature", action="store_true")
    parser.add_argument("--validation", action="store_true")

    args = parser.parse_args()
    pprint(vars(args))
    main(args)


def main(args):
    Path(args.o).mkdir(parents=True, exist_ok=True)

    with open(args.c, "r", encoding="utf-8") as f:
        if args.c.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    config["dataset"]["train_aug"] = False
    pprint(config)

    ds = get_datasets(config)[1 if args.validation else 0]
    dl = DataLoader(ds, **config["dataloader"])

    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    print(f"load regressor: {p}")

    calc = FeatureStatCalculator(
        regressor, **config["calculator"])
    print("run calculator", flush=True)
    calc.run(dl)

    mean, cov, features = calc.compute_stats()
    cov_np = cov.numpy()
    eigvals, eigvecs = np.linalg.eigh(cov_np)

    print(f"rank of cov: {np.linalg.matrix_rank(cov_np)}")

    p = Path(args.o, "feature_stats.pt")
    torch.save({
        "mean": mean,
        "basis": torch.from_numpy(eigvecs),
        "eigvals": torch.from_numpy(eigvals)
    }, str(p))

    if args.save_feature:
        p = Path(args.o, "raw_features.pt")
        torch.save(features, str(p))
        print("feature saved")


@dataclass
class FeatureStatCalculator(Engine):
    regressor: Regressor
    compile_model: InitVar[dict | None]

    def __post_init__(self, compile_model: dict | None):
        super().__init__(self.inference)

        if compile_model is None:
            self.regressor_feature = self.regressor.feature
        else:
            try:
                self.regressor_feature = torch.compile(
                    self.regressor.feature, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")
                self.regressor_feature = self.regressor.feature

        self.reset()

    def reset(self):
        self._features = []

    @torch.no_grad()
    def inference(self, engine: Engine, batch: tuple[Tensor, Tensor]):
        self.regressor.eval()

        x, _ = batch
        x = x.cuda()

        feat = self.regressor_feature(x)    # (B,D)
        self._features.append(feat.cpu())

    def compute_stats(self, ddof: int = 1) -> tuple[Tensor, Tensor, Tensor]:
        assert len(self._features) >= 1, "no data accumulated"

        features = torch.cat(self._features)
        mean = features.mean(dim=0)
        features_c = features - mean    # (B,D)
        cov = features_c.T @ features_c / (features.shape[0] - ddof)

        return mean, cov, features


if __name__ == "__main__":
    parse_args()
