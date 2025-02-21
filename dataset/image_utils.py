from typing import Any
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset, Subset


def load_image(p: Path) -> Image.Image:
    with p.open("rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageDataset(Dataset):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int) -> tuple[Any, float]:
        raise NotImplementedError


class ImageTransformDataset(ImageDataset):
    def __init__(self,
                 dataset: ImageDataset,
                 transform: Callable[[Image.Image], Tensor]):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> tuple[Tensor, float]:
        img, label = self.dataset[i]
        img = self.transform(img)
        return img, label


class ImageSubset(ImageDataset):
    def __init__(self, dataset: ImageDataset, indices: Sequence[int]):
        self.dataset = Subset(dataset, indices)

    def __getitem__(self, i: int) -> tuple[Any, float]:
        return self.dataset[i]

    def __len__(self) -> int:
        return len(self.dataset)


def random_split(dataset: ImageDataset, n1: int) -> tuple[ImageSubset, ImageSubset]:
    perm = np.random.permutation(len(dataset)).tolist()
    return ImageSubset(dataset, perm[:n1]), ImageSubset(dataset, perm[n1:])
