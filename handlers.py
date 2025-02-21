from typing import Any
from dataclasses import dataclass
import copy

from torch.utils.data.dataloader import DataLoader
from ignite.engine.engine import Engine

import pandas as pd


class EvaluationAccumulator:
    def __init__(self):
        self.df = {
            "timestamp": [],
            "dataset": [],
            "epoch": [],
            "iteration": [],
        }

    def append_metrics(self,
                       engine: Engine,
                       metrics: dict[str, Any],
                       name: str):
        self.df["timestamp"].append(pd.Timestamp.now())
        self.df["dataset"].append(name)
        self.df["epoch"].append(engine.state.epoch)
        self.df["iteration"].append(engine.state.iteration)

        for key, value in metrics.items():
            if key in self.df:
                self.df[key].append(value)
            else:
                self.df[key] = [value]

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.df)


@dataclass
class EvaluationRunner:
    evaluator: Engine
    dataloader: DataLoader
    name: str
    logger: EvaluationAccumulator
    print_log: bool = True
    run_evaluator: bool = True

    def __call__(self, engine: Engine):
        if self.run_evaluator:
            self.evaluator.run(self.dataloader)

        d: dict[str, Any] = copy.copy(self.evaluator.state.metrics)
        self.logger.append_metrics(engine, d, self.name)

        if self.print_log:
            print(self.logger.get_dataframe().iloc[-1].to_dict(), flush=True)
