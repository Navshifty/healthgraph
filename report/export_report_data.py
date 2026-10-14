"""Export report numbers and tables to JSON for the Word (.docx) report generators.

    python report/export_report_data.py <out.json>      (run from the repository root)
"""
import json
import re
import sys

import pandas as pd

U, Y = "results/uci", "results/synthetic"
ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "MLP (no graph)", "GCN", "HealthGAT",
         "Oracle (true risk)"]


def summary(path, cols):
    df = pd.read_csv(path)
    g = df.groupby("model")
    m, s = g.mean(numeric_only=True), g.std(numeric_only=True)
    rows = []
    for mod in [o for o in ORDER if o in m.index]:
        rows.append([mod] + [f"{m.loc[mod, c]:.3f}±{s.loc[mod, c]:.3f}" if c in ("AUROC", "AUPRC")
                             else f"{m.loc[mod, c]:.3f}" for c in cols])
    return rows


def main(out):
    text = open("report/tables/numbers.tex", encoding="utf-8").read()
    macros = {k: v.replace("{,}", ",") for k, v in re.findall(r"newcommand\{\\([A-Za-z]+)\}\{(.*)\}", text)}
    data = {"macros": macros,
            "main": summary(f"{U}/metrics_per_seed.csv", ["AUROC", "AUPRC", "LogLoss", "Brier", "ECE", "F1",
                                                           "Sensitivity", "PPV", "Top10%_capture"]),
            "syn": summary(f"{Y}/metrics_per_seed.csv", ["AUROC", "AUPRC", "Brier", "ECE", "Top10%_capture"])}
    sig = pd.read_csv(f"{U}/significance.csv")
    dcol = [c for c in sig.columns if c.startswith("ΔAUROC")][0]
    data["sig"] = [[r["model"], f"{r['AUROC']:.3f}", r["AUROC 95% CI"], r["AUPRC 95% CI"],
                    "" if pd.isna(r[dcol]) else f"{float(r[dcol]):+.4f}",
                    "" if pd.isna(r["DeLong p per seed"]) else r["DeLong p per seed"]]
                   for _, r in sig.iterrows() if r["model"] != "Oracle (true risk)"]
    pub = pd.read_csv(f"{U}/published_benchmarks.csv")
    bench = pd.read_csv(f"{U}/benchmark.csv")
    names = {"P1": "P1: all encounters, random split", "P2": "P2: all encounters, patient-grouped",
             "P3": "P3: strict cohort (70k)", "P4": "P4: strict cohort (20k)"}
    data["bench"] = ([[r.source.split(" [")[0], r.model, f"{r.AUROC:.3f}"] for r in pub.itertuples()]
                     + [[names[r.protocol[:2]] + " (ours)", r.model, f"{r.AUROC_mean:.3f}±{r.AUROC_std:.3f}"]
                        for r in bench.itertuples()])
    ab = pd.read_csv(f"{U}/ablation.csv")
    data["abl"] = [[r.variant, f"{r.AUROC_mean:.3f}±{r.AUROC_std:.3f}", f"{r.AUPRC_mean:.3f}±{r.AUPRC_std:.3f}"]
                   for r in ab.itertuples()]
    sab = pd.read_csv(f"{Y}/ablation.csv")
    data["synabl"] = [[r.variant, f"{r.AUROC_mean:.3f}±{r.AUROC_std:.3f}"] for r in sab.itertuples()]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(len(macros), "macros;", {k: len(v) for k, v in data.items() if k != "macros"})


if __name__ == "__main__":
    main(sys.argv[1])
