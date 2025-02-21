from pathlib import Path

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, random_split

from dataset.dataset_config import CALIFORNIA_HOUSING_PATH


class CaliforniaHousing(Dataset):
    def __init__(self, source_domain: bool, standardize: bool):
        DOMAIN_COL = "ocean_proximity"
        TARGET_COL = "median_house_value"

        p = Path(CALIFORNIA_HOUSING_PATH, "housing.csv")
        df = pd.read_csv(str(p)).dropna()

        source_data = df.query(f"{DOMAIN_COL} != 'NEAR BAY'").drop(
            columns=[DOMAIN_COL])
        target_data = df.query(f"{DOMAIN_COL} == 'NEAR BAY'").drop(
            columns=[DOMAIN_COL])

        if standardize:
            source_mean = source_data.mean()
            source_std = source_data.std(ddof=0)

            source_data = (source_data - source_mean) / (source_std + 1e-8)
            target_data = (target_data - source_mean) / (source_std + 1e-8)

        data = source_data if source_domain else target_data

        self.labels = data[TARGET_COL].to_numpy()
        self.data = data.drop(
            columns=[TARGET_COL]).to_numpy().astype(np.float32)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, i: int) -> tuple[np.ndarray, float]:
        return self.data[i], self.labels[i]


class CaliforniaHousingClassification(CaliforniaHousing):
    def __init__(self, n_bins: int, source_domain: bool):
        super().__init__(source_domain, standardize=True)

        self.n_bins = n_bins

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        x, y = super().__getitem__(i)

        MIN, MAX = -2, 2
        y = np.clip((y - MIN) / (MAX - MIN), 0, 1)
        y = int(y * self.n_bins)

        return x, y


def get_calfornia_housing(config: dict, classification: bool = False) -> tuple[Dataset, Dataset]:
    if classification:
        ds = CaliforniaHousingClassification(**config["dataset"]["config"])
    else:
        ds = CaliforniaHousing(**config["dataset"]["config"])

    if r := config["dataset"].get("train_ratio"):
        train_num = int(len(ds) * r)
        val_num = len(ds) - train_num
        train_ds, val_ds = random_split(ds, [train_num, val_num])
    else:
        train_ds = ds
        val_ds = ds

    return train_ds, val_ds
