from typing import Any
from collections.abc import Iterator

from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import timm


class Regressor(nn.Module):
    regressor: nn.Linear

    def feature(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def predict_from_feature(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def get_regressor(self) -> nn.Module:
        raise NotImplementedError

    def get_feature_extractor(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        z = self.feature(x)
        y_pred = self.predict_from_feature(z)
        return y_pred


class CNNRegressor(Regressor):
    def __init__(self, backbone: str, pretrained: bool, in_channels: int):
        super().__init__()

        match backbone:
            case "resnet26":
                base_net = timm.create_model("resnet26", pretrained=pretrained)
                if in_channels != 3:
                    base_net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                               stride=2, padding=3, bias=False)
                self.feature_extractor = create_feature_extractor(
                    base_net, {"global_pool": "feature"})

            case "resnet50":
                weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                base_net = resnet50(weights=weights)
                if in_channels != 3:
                    base_net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                               stride=2, padding=3, bias=False)
                self.feature_extractor = create_feature_extractor(
                    base_net, {"avgpool": "feature"})

            case _:
                raise ValueError(f"Invalid backbone: {backbone!r}")

        self.regressor = nn.Linear(2048, 1)

    def feature(self, x: Tensor) -> Tensor:
        z: Tensor = self.feature_extractor(x)["feature"]
        return z.flatten(start_dim=1)

    def predict_from_feature(self, z: Tensor) -> Tensor:
        y_pred: Tensor = self.regressor(z)
        return y_pred.flatten()

    def get_regressor(self) -> nn.Module:
        return self.regressor

    def get_feature_extractor(self) -> nn.Module:
        return self.feature_extractor


class MLPRegressor(Regressor):
    def __init__(self, in_dims: int, h_dims: int, n_rep: int):
        super().__init__()

        self.fe = nn.Sequential(
            nn.Linear(in_dims, h_dims, bias=False),
            nn.BatchNorm1d(h_dims),
            nn.ReLU(),
            *(
                nn.Sequential(
                    nn.Linear(h_dims, h_dims, bias=False),
                    nn.BatchNorm1d(h_dims),
                    nn.ReLU()
                )
                for _ in range(n_rep)
            )
        )

        self.regressor = nn.Linear(h_dims, 1)

    def feature(self, x: Tensor) -> Tensor:
        if x.ndim >= 3:
            x = x.flatten(start_dim=1)
        return self.fe(x)

    def predict_from_feature(self, z: Tensor) -> Tensor:
        y_pred: Tensor = self.regressor(z)
        return y_pred.flatten()

    def get_regressor(self) -> nn.Linear:
        return self.regressor

    def get_feature_extractor(self) -> nn.Module:
        return self.fe


def create_regressor(config: dict[str, Any]) -> Regressor:
    match config["regressor"]["type"]:
        case "image":
            net = CNNRegressor(**config["regressor"]["config"])

        case "table":
            net = MLPRegressor(**config["regressor"]["config"])

        case _ as t:
            raise ValueError(f"Invalid type: {t!r}")

    return net


def extract_bn_layers(mod: nn.Module) -> Iterator[_BatchNorm]:
    for m in mod.children():
        if isinstance(m, _BatchNorm):
            yield m
        else:
            yield from extract_bn_layers(m)
