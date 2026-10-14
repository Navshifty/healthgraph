"""YAML configuration loading with tuned-hyperparameter overlay."""
import copy
import json
import os

import yaml


def load_config(path, overrides=None):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = path
    for key, value in (overrides or {}).items():         # "train.lr=0.003" style
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(str(value))
    return cfg


def hpo_dir(cfg):
    return os.path.join(cfg["output_dir"], "hpo")


def tuned(cfg, name):
    """Best hyperparameters found by `healthgraph tune`, or None."""
    path = os.path.join(hpo_dir(cfg), f"best_{name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def with_tuned_gnn(cfg):
    """Return a copy of cfg with the tuned GAT/graph/optimiser settings applied."""
    cfg = copy.deepcopy(cfg)
    best = tuned(cfg, "gat")
    if best:
        for k, v in best.items():
            section = "graph" if k.startswith("k_") else "train" if k in ("lr", "weight_decay") else "model"
            cfg[section][k] = v
        cfg["tuned"] = True
    return cfg
