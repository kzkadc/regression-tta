from pathlib import Path
import json
import tarfile
import io
import re

from torchvision.transforms import v2 as transforms

import pandas as pd
import numpy as np
from PIL import Image

from .dataset_config import BIWI_PATH
from .image_utils import (ImageDataset, ImageTransformDataset, ImageSubset,
                          random_split)


_ROOT = "biwi-kinect-head-pose-database"


class BiwiKinect(ImageDataset):
    def __init__(self, gender: str, target: str):
        assert target in ("yaw", "pitch", "roll")
        self.target = target

        tar_path = Path(BIWI_PATH, f"{_ROOT}.tar")
        self.tar = tarfile.open(str(tar_path), mode="r")

        with Path(BIWI_PATH, "gender.json").open("r", encoding="utf-8") as f:
            person_dirs: str = json.load(f)[gender]

        df = {
            "person": [],
            "frame": [],
            "yaw": [],
            "roll": [],
            "pitch": []
        }

        tar_members = [
            m.name
            for m in self.tar.getmembers()
            if m.isfile()
        ]
        for person in person_dirs:
            metadata_paths = (
                m
                for m in tar_members
                if re.search(rf"faces_0/{person}/frame_\d+_pose.txt", m)
            )

            for p in metadata_paths:
                fp = self.tar.extractfile(p)
                assert fp is not None

                lines = fp.read().decode(encoding="utf-8").strip().split("\n")

                rot_matrix = np.array([
                    [float(x) for x in l.strip().split(" ")]
                    for l in lines[:3]
                ])

                frame = p.split("/")[-1].split("_")[1]

                y, r, p = matrix_to_angles(rot_matrix)
                df["person"].append(person)
                df["frame"].append(frame)
                df["yaw"].append(y)
                df["roll"].append(r)
                df["pitch"].append(p)

        self.metadata = pd.DataFrame(df)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, i: int) -> tuple[Image.Image, float]:
        metadata = self.metadata.iloc[i].to_dict()
        y = metadata[self.target]

        img_path = f"{_ROOT}/faces_0/{metadata['person']}/frame_{metadata['frame']}_rgb.png"
        fp = self.tar.extractfile(img_path)
        assert fp is not None

        with io.BytesIO(fp.read()) as bio:
            img = Image.open(bio).convert("RGB")

        w, h = img.width, img.height
        c = (w - h) // 2
        img = img.crop((c, 0, w - c, h))

        return img, y

    def close(self):
        self.tar.close()


def matrix_to_angles(m: np.ndarray) -> tuple[float, float, float]:
    y = np.arctan2(m[1, 0], m[0, 0])
    r = np.arctan2(-m[2, 0], np.sqrt(m[2, 1] * m[2, 1] + m[2, 2] * m[2, 2]))
    p = np.arctan2(m[2, 1], m[2, 2])
    return y, r, p


class BiwiKinectClassification(BiwiKinect):
    def __init__(self, n_bins: int, gender: str, target: str):
        super().__init__(gender, target)

        self.n_bins = n_bins

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        x, y = super().__getitem__(i)

        MAX_RAD = 60 * np.pi / 180
        MIN_RAD = -60 * np.pi / 180

        y = np.clip((y - MIN_RAD) * self.n_bins /
                    (MAX_RAD - MIN_RAD), 0, self.n_bins)
        return x, int(y)


def get_biwi_kinect(config: dict, apply_to_tensor: bool = True, classification: bool = False) -> tuple[ImageDataset, ImageDataset]:
    if apply_to_tensor:
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        to_tensor = transforms.ToTensor()

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        to_tensor
    ])

    if config["dataset"].get("train_aug", True):
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            to_tensor
        ])
    else:
        train_transform = val_transform

    if classification:
        ds = BiwiKinectClassification(**config["dataset"]["config"])
    else:
        ds = BiwiKinect(**config["dataset"]["config"])

    if "val_indices" in config["dataset"]:
        val_indices: np.ndarray = np.load(config["dataset"]["val_indices"])
        print(
            f"BiwiKinect: load val indices from {config['dataset']['val_indices']}")

        train_mask = np.ones(len(ds), dtype=np.bool_)
        train_mask[val_indices] = False
        train_indices = np.arange(len(ds))[train_mask]

        train_ds = ImageSubset(ds, train_indices.tolist())
        val_ds = ImageSubset(ds, val_indices.tolist())
    else:
        print("BiwiKinect: split randomly")
        n = int(len(ds) * config["dataset"]["train_ratio"])
        train_ds, val_ds = random_split(ds, n)

    train_ds = ImageTransformDataset(train_ds, train_transform)
    val_ds = ImageTransformDataset(val_ds, val_transform)
    return train_ds, val_ds
