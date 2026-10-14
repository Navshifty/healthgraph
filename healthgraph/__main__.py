"""HealthGraph command-line interface.

    python -m healthgraph tune        --config configs/uci.yaml [--what gat|boosting|all]
    python -m healthgraph experiment  --config configs/uci.yaml
    python -m healthgraph ablation    --config configs/uci.yaml
    python -m healthgraph benchmark   --config configs/uci.yaml
    python -m healthgraph analyze     --config configs/uci.yaml
    python -m healthgraph all         --config configs/uci.yaml   # tune -> experiment -> ablation -> analyze -> benchmark

Override any config value with --set, e.g. --set train.epochs=50 --set seeds=[0].
"""
import argparse
import os
import warnings

import torch

from .config import load_config
from .datasets import load_dataset
from .tracking import Tracker

warnings.filterwarnings("ignore", category=UserWarning)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="healthgraph")
    ap.add_argument("command", choices=["tune", "experiment", "ablation", "benchmark", "analyze", "all"])
    ap.add_argument("--config", default="configs/uci.yaml")
    ap.add_argument("--what", default="all", choices=["gat", "boosting", "all"])
    ap.add_argument("--trials", type=int, default=None, help="override number of GAT trials")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, dict(kv.split("=", 1) for kv in a.set))
    os.makedirs(cfg["output_dir"], exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    tracker = Tracker(cfg)
    dataset = None if a.command == "benchmark" else load_dataset(cfg)

    if a.command in ("tune", "all"):
        from .hpo import tune_boosting, tune_gat
        if a.what in ("boosting", "all"):
            tune_boosting(cfg, dataset, tracker)
        if a.what in ("gat", "all"):
            tune_gat(cfg, dataset, tracker, a.trials)
    if a.command in ("experiment", "all"):
        from .pipeline import run_experiment
        run_experiment(cfg, dataset, tracker)
    if a.command in ("ablation", "all"):
        from .pipeline import run_ablation
        run_ablation(cfg, dataset, tracker)
    if a.command in ("analyze", "all"):
        from .analysis import run_analysis
        run_analysis(cfg, dataset)
    if a.command in ("benchmark", "all") and cfg["data"]["source"] == "uci":
        from .benchmark import run_benchmark
        run_benchmark(cfg, tracker, seeds=tuple(cfg["seeds"]))


if __name__ == "__main__":
    main()
