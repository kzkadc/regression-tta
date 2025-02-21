from typing import Any
from pprint import pprint
import json
from pathlib import Path
import itertools

import yaml

import torch
from torch.utils.data import DataLoader
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint

from utils.seed import fix_seed
from model import create_regressor, Regressor, extract_bn_layers
from dataset import get_datasets
from evaluation.evaluator import RegressionEvaluator
from tta import TTAEngine


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=True, help="config")
    parser.add_argument("-o", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="save model")

    args = parser.parse_args()
    pprint(vars(args))
    main(args)


def main(args):
    fix_seed(args.seed)

    with open(args.c, "r", encoding="utf-8") as f:
        if args.c.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    pprint(config)

    Path(args.o).mkdir(parents=True, exist_ok=True)
    with Path(args.o, "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    print(f"load {p}")

    _, val_ds = get_datasets(config)
    val_dl = DataLoader(val_ds, **config["adapt_dataloader"])

    opt = create_optimizer(regressor, config)
    engine = TTAEngine(regressor, opt, **config["tta"]["config"])

    if args.save:
        engine.add_event_handler(Events.COMPLETED,
                                 ModelCheckpoint(
                                     args.o, "adapted", require_empty=False),
                                 {"regressor": regressor})
    reg_evaluator = RegressionEvaluator(regressor, **config["evaluator"])

    engine.run(val_dl)
    reg_evaluator.run(val_dl)

    metrics = {
        "iteration": engine.state.iteration,
        "online": engine.state.metrics,
        "offline": reg_evaluator.state.metrics
    }
    pprint(metrics)
    with Path(args.o, "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


def create_optimizer(net: Regressor, config: dict[str, Any]) -> torch.optim.Optimizer:
    match config["optimizer"]["param"]:
        case "all":
            params = net.parameters()
        case "fe":
            params = net.get_feature_extractor().parameters()
        case "fe_bn":
            bn_layers = extract_bn_layers(net.get_feature_extractor())
            params = itertools.chain.from_iterable(
                l.parameters() for l in bn_layers
            )
        case _ as p:
            raise ValueError(f"Invalid param: {p!r}")

    opt = eval(f"torch.optim.{config['optimizer']['name']}")(
        params, **config["optimizer"]["config"])
    return opt


if __name__ == "__main__":
    parse_args()
