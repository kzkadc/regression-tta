from typing import Any
from pathlib import Path

from PIL import Image
import numpy as np
import imagenet_c

from torchvision.transforms import v2 as transforms

from .image_utils import (load_image, random_split, ImageDataset,
                          ImageTransformDataset, ImageSubset)
from .dataset_config import UTKFACE_PATH


class UTKFace(ImageDataset):
    def __init__(self,
                 filter_gender: int | None = None):
        root = Path(UTKFACE_PATH)
        self.path_list = list(root.glob("*.jpg"))

        self.ages: np.ndarray = np.array([
            float(p.name.split("_")[0])
            for p in self.path_list
        ], dtype=np.float32)

        if filter_gender is not None:
            genders = np.array([
                int(p.name.split("_")[1])
                for p in self.path_list
            ])
            gender_mask: np.ndarray = genders == filter_gender
            self.path_list = [
                p
                for p, m in zip(self.path_list, gender_mask)
                if m
            ]
            self.ages = self.ages[gender_mask]

    def __len__(self) -> int:
        return len(self.path_list)

    def __getitem__(self, i: int) -> tuple[Image.Image, float]:
        p = self.path_list[i]
        img = load_image(p)
        return img, self.ages[i]


def get_utkface(config: dict[str, Any]) -> tuple[ImageDataset, ImageDataset]:
    if (corruption := config["dataset"].get("val_corruption", None)) is not None:
        corrupt_func = lambda x: Image.fromarray(
            imagenet_c.corrupt(np.asarray(x), **corruption))
        print(f"Corruption: {corruption}")
    else:
        corrupt_func = lambda x: x

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        corrupt_func,
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    if config["dataset"].get("train_aug", True):
        train_aug = transforms.Compose([
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip()
        ])
    else:
        train_aug = transforms.CenterCrop((224, 224))

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        train_aug,
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = UTKFace(**config["dataset"]["config"])

    if (val_ind_file := config["dataset"].get("val_indices", None)) is None:
        print("UTKFace: split randomly")
        train_num = int(len(dataset) * config["dataset"]["train_ratio"])
        train_ds, val_ds = random_split(dataset, train_num)
    else:
        print(f"UTKFace: load val indices from {val_ind_file!r}")
        val_indices: np.ndarray = np.load(val_ind_file)

        train_mask = np.ones(len(dataset), dtype=np.bool_)
        train_mask[val_indices] = False
        train_indices = np.arange(len(dataset))[train_mask]

        train_ds = ImageSubset(dataset, train_indices.tolist())
        val_ds = ImageSubset(dataset, val_indices.tolist())

    train_ds = ImageTransformDataset(train_ds, train_transform)
    val_ds = ImageTransformDataset(val_ds, val_transform)

    return train_ds, val_ds
