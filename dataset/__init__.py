from typing import Any

from torch.utils.data import Dataset

from .raw_datasets import get_mnist, get_svhn, get_transforms
from .utkface import get_utkface
from .biwi_kinect import get_biwi_kinect
from .california_housing import get_calfornia_housing


def get_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    train_aug = config["dataset"].get("train_aug", True)
    print("train_aug:", train_aug)

    match name := config["dataset"]["name"]:
        case "svhn":
            train_transform, val_transform = get_transforms(name)
            if not train_aug:
                train_transform = val_transform
            train_ds, val_ds = get_svhn(train_transform, val_transform)
        case "mnist":
            train_transform, val_transform = get_transforms(name)
            if not train_aug:
                train_transform = val_transform
            train_ds, val_ds = get_mnist(train_transform, val_transform)
        case "utkface":
            train_ds, val_ds = get_utkface(config)
        case "biwi_kinect":
            aug_config = config["dataset"].get("aug_config", {})
            train_ds, val_ds = get_biwi_kinect(
                config, classification=False, **aug_config)
        case "calfornia_housing":
            train_ds, val_ds = get_calfornia_housing(
                config, classification=False)
        case _:
            raise ValueError(f"Invalid dataset: {name!r}")

    print(f"dataset: {name}")

    return train_ds, val_ds
