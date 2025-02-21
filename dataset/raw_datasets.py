from collections.abc import Callable

from torchvision import datasets
from torchvision.transforms import v2 as transforms

from .dataset_config import SVHN_PATH, MNIST_PATH


def get_svhn(train_transform, val_transform) -> tuple[datasets.SVHN, datasets.SVHN]:
    train_ds = datasets.SVHN(SVHN_PATH, split="train",
                             transform=train_transform,)
    val_ds = datasets.SVHN(SVHN_PATH, split="test", transform=val_transform)
    return train_ds, val_ds


def get_mnist(train_transform, val_transform) -> tuple[datasets.MNIST, datasets.MNIST]:
    train_ds = datasets.MNIST(MNIST_PATH, train=True,
                              transform=train_transform)
    val_ds = datasets.MNIST(MNIST_PATH, train=False, transform=val_transform)
    return train_ds, val_ds


def get_transforms(name: str) -> tuple[Callable, Callable]:
    match name:
        case "svhn":
            return (
                transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor()
                ]),
                transforms.ToTensor()
            )
        case "mnist":
            return (
                transforms.Compose([
                    transforms.Grayscale(3),
                    transforms.Resize(32),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor()
                ]),
                transforms.Compose([
                    transforms.Grayscale(3),
                    transforms.Resize(32),
                    transforms.ToTensor()
                ])
            )
        case _:
            raise ValueError(f"Invalid dataset: {name!r}")
