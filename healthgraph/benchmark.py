"""Comparison with published benchmarks on the UCI Diabetes 130-US Hospitals dataset.

Published numbers depend heavily on the evaluation protocol, so we re-run our tuned
tabular models under each protocol before comparing:

  P1  all 101,766 encounters, random encounter-level split   (protocol of the published papers)
  P2  all encounters, patient-grouped split                   (removes same-patient leakage)
  P3  strict cohort: first encounter per patient, no death/hospice, full size
  P4  strict cohort, 20k stratified subsample                 (the cohort used for the graph models)
"""
import copy
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import baselines
from .config import tuned
from .datasets import UCIDataset
from .train import evaluate, stratified_split, to_logit

PUBLISHED = [
    # source, model, AUROC, protocol note
    ("Emi-Johnson & Nkrumah 2025 [Cureus]", "XGBoost", 0.667, "all encounters, 80/20 random split"),
    ("Emi-Johnson & Nkrumah 2025 [Cureus]", "Logistic Regression", 0.642, "all encounters, 80/20 random split"),
    ("Emi-Johnson & Nkrumah 2025 [Cureus]", "Random Forest", 0.630, "all encounters, 80/20 random split"),
    ("Emi-Johnson & Nkrumah 2025 [Cureus]", "Deep neural network", 0.579, "all encounters, 80/20 random split"),
    ("Salim & Ibrahim 2026 [Healthcare]", "XGBoost (tuned)", 0.664, "all encounters, nested 5-fold CV"),
    ("Salim & Ibrahim 2026 [Healthcare]", "LightGBM", 0.660, "all encounters, nested 5-fold CV"),
    ("Salim & Ibrahim 2026 [Healthcare]", "Logistic Regression", 0.657, "all encounters, nested 5-fold CV"),
    ("Salim & Ibrahim 2026 [Healthcare]", "Random Forest", 0.650, "all encounters, nested 5-fold CV"),
    ("Salim & Ibrahim 2026 [Healthcare]", "Stacking ensemble", 0.665, "all encounters, nested 5-fold CV"),
]
MODELS = ["logreg", "xgboost", "lightgbm"]


def grouped_split(groups, y, seed, val=0.15, test=0.15):
    idx = np.arange(len(y))
    g1 = GroupShuffleSplit(1, test_size=val + test, random_state=seed)
    train, rest = next(g1.split(idx, y, groups))
    g2 = GroupShuffleSplit(1, test_size=test / (val + test), random_state=seed)
    v, t = next(g2.split(rest, y[rest], groups[rest]))
    return {"train": np.sort(train), "val": np.sort(rest[v]), "test": np.sort(rest[t])}


def run_benchmark(cfg, tracker, seeds=(0, 1, 2)):
    protocols = {
        "P1 all encounters, random split (published protocol)": ("all", None, "random"),
        "P2 all encounters, patient-grouped split": ("all", None, "grouped"),
        "P3 strict cohort (first encounter), full": ("first", None, "random"),
        "P4 strict cohort, 20k subsample (graph cohort)": ("first", cfg["data"].get("max_patients"), "random"),
    }
    out = cfg["output_dir"]
    ckpt = os.path.join(out, "benchmark_runs.csv")          # one row per finished run; enables resuming
    done = set()
    if os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        done = set(zip(prev["protocol"], prev["model"], prev["seed"]))
        print(f"[benchmark] resuming, {len(done)} runs already done")
    with tracker.run("benchmark-protocols", tags={"stage": "benchmark"}):
        for pname, (cohort, maxp, how) in protocols.items():
            if all((pname, baselines.LABELS[n], s) in done for n in MODELS for s in seeds):
                continue
            c = copy.deepcopy(cfg)
            c["data"]["max_patients"] = maxp
            ds = UCIDataset(c, cohort=cohort)
            groups = ds.df["patient_nbr"].to_numpy()
            print(f"[benchmark] {pname}: {len(ds.y)} encounters, {len(np.unique(groups))} patients, "
                  f"prevalence {ds.y.mean():.3f}")
            for seed in seeds:
                todo = [n for n in MODELS if (pname, baselines.LABELS[n], seed) not in done]
                if not todo:
                    continue
                split = (grouped_split(groups, ds.y, seed) if how == "grouped"
                         else stratified_split(ds.y, seed, cfg["split"]["val"], cfg["split"]["test"]))
                overlap = len(np.intersect1d(groups[split["train"]], groups[split["test"]]))
                data, _ = ds.prepare(split["train"], seed)
                for name in todo:
                    _, p = baselines.fit_predict(name, data.x, data.y, split, seed, tuned(cfg, name))
                    m, _, _ = evaluate(to_logit(p), data.y, split)
                    row = {"protocol": pname, "model": baselines.LABELS[name], "seed": seed,
                           "n": len(ds.y), "patients_in_train_and_test": overlap, **m}
                    pd.DataFrame([row]).to_csv(ckpt, mode="a", header=not os.path.exists(ckpt), index=False)
                    print(f"   {name:8s} seed {seed}: AUROC {m['AUROC']:.3f} AUPRC {m['AUPRC']:.3f} "
                          f"(train/test patient overlap {overlap})")
        df = pd.read_csv(ckpt)
        df = df[df["protocol"].isin(protocols) & df["seed"].isin(seeds)]
        df["protocol"] = pd.Categorical(df["protocol"], list(protocols), ordered=True)
        df = df.sort_values(["protocol", "seed"], kind="stable")
        df["protocol"] = df["protocol"].astype(str)
        df.to_csv(os.path.join(out, "benchmark_per_seed.csv"), index=False)
        summary = df.groupby(["protocol", "model"], sort=False).agg(
            n=("n", "first"), overlap=("patients_in_train_and_test", "mean"),
            AUROC_mean=("AUROC", "mean"), AUROC_std=("AUROC", "std"),
            AUPRC_mean=("AUPRC", "mean"), AUPRC_std=("AUPRC", "std"))
        summary.to_csv(os.path.join(out, "benchmark.csv"))
        pd.DataFrame(PUBLISHED, columns=["source", "model", "AUROC", "protocol"]).to_csv(
            os.path.join(out, "published_benchmarks.csv"), index=False)
        for (p, mdl), r in summary.iterrows():
            tracker.metrics({f"{p[:2]}.{mdl}.AUROC": r["AUROC_mean"]})
        print(summary.round(4).to_string())
    return summary
