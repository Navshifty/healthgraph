"""Hyperparameter optimisation with Optuna (TPE sampler + median pruning), logged to MLflow.

Objective for every model: validation AUPRC on the seed-0 split (the primary metric
for a 9 %-prevalence outcome). The test split is never touched during tuning.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.metrics import average_precision_score

from . import baselines, graph
from .config import hpo_dir
from .train import stratified_split, train_gnn

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _save(cfg, name, study, extra=None):
    out = hpo_dir(cfg)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"best_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration"))
    df.to_csv(os.path.join(out, f"trials_{name}.csv"), index=False)
    try:
        ax = optuna.visualization.matplotlib.plot_optimization_history(study)
        ax.set_title(f"Optuna optimisation history: {name}")
        plt.gcf().set_size_inches(7, 4)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"history_{name}.png"), dpi=140)
        plt.close("all")
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 5:
            ax = optuna.visualization.matplotlib.plot_param_importances(study)
            ax.set_title(f"Hyperparameter importance (fANOVA): {name}")
            plt.gcf().set_size_inches(7, 4)
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"importance_{name}.png"), dpi=140)
            plt.close("all")
    except Exception as e:                               # plotting must never kill a long study
        print(f"   (optuna plot skipped: {e})")
    print(f"[hpo] {name}: best val AUPRC {study.best_value:.4f} with {study.best_params}")


def _study(cfg, name, sampler=None, pruner=None):
    os.makedirs(hpo_dir(cfg), exist_ok=True)
    storage = f"sqlite:///{os.path.join(hpo_dir(cfg), 'optuna.db')}"
    return optuna.create_study(study_name=name, direction="maximize", storage=storage,
                               load_if_exists=True, sampler=sampler or optuna.samplers.TPESampler(seed=0),
                               pruner=pruner or optuna.pruners.NopPruner())


def tune_gat(cfg, dataset, tracker, n_trials=None):
    h = cfg["hpo"]
    split = stratified_split(dataset.y, 0, cfg["split"]["val"], cfg["split"]["test"])
    data, _ = dataset.prepare(split["train"], seed=0)
    R = len(graph.RELATIONS)
    graphs = {}

    def objective(trial):
        p = {
            "hidden": trial.suggest_categorical("hidden", [16, 32]),
            "heads": trial.suggest_categorical("heads", [2, 4]),
            "layers": trial.suggest_int("layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.2, 0.6),
            "edge_dropout": trial.suggest_float("edge_dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 3e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 2e-2, log=True),
            "k_sim": trial.suggest_categorical("k_sim", [5, 10, 20]),
            "k_code": trial.suggest_categorical("k_code", [3, 5, 10]),
            "k_group": trial.suggest_categorical("k_group", [0, 4, 8]),
        }
        key = (p["k_sim"], p["k_code"], p["k_group"])
        if key not in graphs:
            graphs[key] = graph.build_graph(data, *key)
        ei, et = graphs[key]
        best = [0.0]

        def on_epoch(epoch, score):
            best[0] = max(best[0], score)
            trial.report(best[0], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        with tracker.run(f"gat-trial-{trial.number}", nested=True, tags={"stage": "hpo"}):
            tracker.params(p)
            try:
                _, _, info, _ = train_gnn("gat", data.x, data.y, ei, et, R, split, 0,
                                          epochs=h["max_epochs"], patience=h["patience"], on_epoch=on_epoch,
                                          **{k: v for k, v in p.items() if not k.startswith("k_")})
            except optuna.TrialPruned:
                tracker.metrics({"val_auprc": best[0], "pruned": 1})
                raise
            tracker.metrics({"val_auprc": info["best_val_auprc"], "epochs": info["epochs"],
                             "seconds": info["seconds"], "pruned": 0})
        print(f"   trial {trial.number:2d}: val AUPRC {info['best_val_auprc']:.4f} ({info['seconds']:.0f}s) {p}")
        return info["best_val_auprc"]

    study = _study(cfg, "gat", pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30))
    with tracker.run("hpo-gat", tags={"stage": "hpo"}):
        remaining = (n_trials or h["n_trials"]) - len(study.trials)
        if remaining > 0:
            study.optimize(objective, n_trials=remaining)
        tracker.params({f"best_{k}": v for k, v in study.best_params.items()})
        tracker.metrics({"best_val_auprc": study.best_value})
    _save(cfg, "gat", study)
    return study.best_params


def tune_boosting(cfg, dataset, tracker):
    split = stratified_split(dataset.y, 0, cfg["split"]["val"], cfg["split"]["test"])
    data, _ = dataset.prepare(split["train"], seed=0)
    x, y, va = data.x, data.y, split["val"]
    n = cfg["hpo"]["boosting_trials"]

    spaces = {
        "lightgbm": lambda t: {
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": t.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 10, 200, log=True),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30, log=True)},
        "xgboost": lambda t: {
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 8),
            "min_child_weight": t.suggest_float("min_child_weight", 1, 50, log=True),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30, log=True)},
        "rf": lambda t: {
            "max_depth": t.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": t.suggest_float("max_features", 0.1, 0.8)},
    }
    best = {}
    for name, space in spaces.items():
        def objective(trial, name=name, space=space):
            params = space(trial)
            _, p = baselines.fit_predict(name, x, y, split, 0, params)
            return average_precision_score(y[va], p[va])

        study = _study(cfg, name)
        with tracker.run(f"hpo-{name}", tags={"stage": "hpo"}):
            remaining = (n if name != "rf" else max(10, n // 3)) - len(study.trials)
            if remaining > 0:
                study.optimize(objective, n_trials=remaining)
            tracker.params(study.best_params)
            tracker.metrics({"best_val_auprc": study.best_value})
        _save(cfg, name, study)
        best[name] = study.best_params

    # Logistic regression: exhaustive grid search over the regularisation strength.
    grid = {"C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}
    study = _study(cfg, "logreg", sampler=optuna.samplers.GridSampler(grid))
    with tracker.run("hpo-logreg", tags={"stage": "hpo"}):
        remaining = len(grid["C"]) - len(study.trials)
        if remaining > 0:
            study.optimize(lambda t: average_precision_score(
                y[va], baselines.fit_predict("logreg", x, y, split, 0,
                                             {"C": t.suggest_categorical("C", grid["C"])})[1][va]),
                n_trials=remaining)
        tracker.metrics({"best_val_auprc": study.best_value})
    _save(cfg, "logreg", study)
    best["logreg"] = study.best_params
    return best
