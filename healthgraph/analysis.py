"""Post-hoc analysis of saved predictions: tables, significance tests, error analysis, figures."""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import stats, viz

HL = "HealthGAT"


def _load_preds(out):
    runs = []
    for path in sorted(glob.glob(os.path.join(out, "preds_seed*.npz"))):
        z = np.load(path)
        seed = int(path.split("seed")[-1].split(".")[0])
        probs = {k[3:]: z[k] for k in z.files if k.startswith("p::")}
        thr = {k[5:]: float(z[k]) for k in z.files if k.startswith("thr::")}
        runs.append({"seed": seed, "y": z["y"], "test": z["test"], "probs": probs, "thr": thr})
    return runs


def _fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def run_analysis(cfg, dataset=None):
    out = cfg["output_dir"]
    fig = os.path.join(out, "figures")
    os.makedirs(fig, exist_ok=True)
    report = [f"# Analysis: {cfg['experiment']}\n"]

    # ---- 1. main table ----
    per_seed = pd.read_csv(os.path.join(out, "metrics_per_seed.csv"))
    cols = ["AUROC", "AUPRC", "LogLoss", "Brier", "ECE", "F1", "Sensitivity", "Specificity", "PPV",
            "Top10%_capture"]
    g = per_seed.groupby("model", sort=False)[cols]
    mean, std = g.mean(), g.std().fillna(0)
    table = pd.DataFrame({c: [_fmt(m, s) for m, s in zip(mean[c], std[c])] for c in cols}, index=mean.index)
    table.to_csv(os.path.join(out, "metrics_summary.csv"))
    mean.to_csv(os.path.join(out, "metrics_mean.csv"))
    report += ["## Test-set metrics (mean ± sd over seeds)\n", table.to_markdown(), ""]

    runs = _load_preds(out)
    r0 = runs[0]
    y0, t0 = r0["y"], r0["test"]

    # ---- 2. bootstrap CIs + DeLong + McNemar ----
    rows = []
    for name, p in r0["probs"].items():
        ci = stats.bootstrap_ci(y0[t0], p[t0])
        row = {"model": name, "AUROC": roc_auc_score(y0[t0], p[t0]),
               "AUROC 95% CI": f"[{ci['AUROC_CI'][0]:.3f}, {ci['AUROC_CI'][1]:.3f}]",
               "AUPRC 95% CI": f"[{ci['AUPRC_CI'][0]:.3f}, {ci['AUPRC_CI'][1]:.3f}]"}
        if name != HL and HL in r0["probs"]:
            deltas, ps = [], []
            for r in runs:
                yy, tt = r["y"], r["test"]
                a1, a2, _, pv = stats.delong(yy[tt], r["probs"][HL][tt], r["probs"][name][tt])
                deltas.append(a1 - a2)
                ps.append(pv)
            row["ΔAUROC (HealthGAT − model), mean over seeds"] = f"{np.mean(deltas):+.4f}"
            row["DeLong p per seed"] = ", ".join(f"{v:.3f}" for v in ps)
        rows.append(row)
    sig = pd.DataFrame(rows).set_index("model")
    sig.to_csv(os.path.join(out, "significance.csv"))
    report += ["## Uncertainty and significance (seed-0 bootstrap CIs; paired DeLong tests per seed)\n",
               sig.fillna("").to_markdown(), ""]

    best_tab = max([m for m in r0["probs"] if m in ("LightGBM", "XGBoost", "Logistic Regression")],
                   key=lambda m: roc_auc_score(y0[t0], r0["probs"][m][t0]))
    if HL in r0["probs"]:
        mc = stats.mcnemar(y0[t0], r0["probs"][HL][t0] >= r0["thr"][HL],
                           r0["probs"][best_tab][t0] >= r0["thr"][best_tab])
        report += [f"McNemar (decisions at validation-tuned thresholds), HealthGAT vs {best_tab}: "
                   f"{json.dumps(mc)}\n"]

    # ---- 3. figures from seed 0 ----
    curves = {k: (y0[t0], v[t0]) for k, v in r0["probs"].items()}
    viz.curves(curves, os.path.join(fig, "roc_pr_calibration.png"))
    conf = [(m, y0[t0], r0["probs"][m][t0], r0["thr"][m]) for m in [HL, best_tab, "Logistic Regression"]
            if m in r0["probs"]]
    viz.confusion(conf, os.path.join(fig, "confusion_matrices.png"))
    viz.decision_curve({k: v for k, v in curves.items() if k in (HL, best_tab, "Logistic Regression", "GCN")},
                       os.path.join(fig, "decision_curve.png"))

    # ---- 4. subgroup + error analysis (UCI), pooled over seeds ----
    if dataset is not None and dataset.name == "uci_diabetes" and HL in r0["probs"]:
        df = dataset.df
        pooled = []
        for r in runs:
            t = r["test"]
            d = df.iloc[t][["age", "gender", "race", "admission_type_id", "number_inpatient", "number_emergency",
                            "time_in_hospital", "num_medications", "number_diagnoses", "num_lab_procedures",
                            "diag_1", "A1Cresult"]].copy()
            d["y"], d["p"], d["pred"] = r["y"][t], r["probs"][HL][t], r["probs"][HL][t] >= r["thr"][HL]
            pooled.append(d)
        d = pd.concat(pooled, ignore_index=True)
        from .uci import icd9_group
        age_num = d["age"].astype(str).str.extract(r"\[(\d+)-")[0].astype(int)
        groups = {
            "age band": pd.cut(age_num, [0, 50, 70, 100], right=False, labels=["<50", "50-69", "70+"]),
            "gender": d["gender"].where(d["gender"].isin(["Female", "Male"]), "other"),
            "race": d["race"].fillna("Unknown").where(d["race"].isin(["Caucasian", "AfricanAmerican"]), "other"),
            "admission type": d["admission_type_id"].map({1: "emergency", 2: "urgent", 3: "elective"}).fillna("other"),
            "prior inpatient visits": pd.cut(d["number_inpatient"], [-1, 0, 1, 100], labels=["0", "1", "2+"]),
            "primary diagnosis": d["diag_1"].map(icd9_group),
        }
        srows = []
        for var, grp in groups.items():
            for lv in grp.dropna().unique():
                m = (grp == lv).to_numpy()
                if m.sum() < 100 or d["y"][m].nunique() < 2:
                    continue
                srows.append({"variable": var, "group": lv, "n": int(m.sum()), "prevalence": d["y"][m].mean(),
                              "auroc": roc_auc_score(d["y"][m], d["p"][m]), "mean_pred": d["p"][m].mean(),
                              "observed": d["y"][m].mean(),
                              "sensitivity": d["pred"][m & (d["y"] == 1).to_numpy()].mean()})
        sub = pd.DataFrame(srows)
        sub["calibration_gap"] = sub["mean_pred"] - sub["observed"]
        sub.to_csv(os.path.join(out, "subgroups.csv"), index=False)
        viz.subgroups(sub, os.path.join(fig, "subgroups.png"))
        report += ["## Subgroup analysis (HealthGAT, test sets pooled over seeds)\n",
                   sub.round(3).to_markdown(index=False), ""]

        num = ["number_inpatient", "number_emergency", "time_in_hospital", "num_medications",
               "number_diagnoses", "num_lab_procedures"]
        d["age_num"] = age_num
        cat = {"TP": (d["y"] == 1) & d["pred"], "FN": (d["y"] == 1) & ~d["pred"],
               "FP": (d["y"] == 0) & d["pred"], "TN": (d["y"] == 0) & ~d["pred"]}
        err = pd.DataFrame({k: d.loc[v, num + ["age_num"]].mean() for k, v in cat.items()}).T
        err["n"] = [int(v.sum()) for v in cat.values()]
        err["mean_predicted_risk"] = [d.loc[v, "p"].mean() for v in cat.values()]
        err.to_csv(os.path.join(out, "error_profile.csv"))
        report += ["## Error analysis: mean characteristics by confusion-matrix cell\n",
                   err.round(3).to_markdown(), ""]

    with open(os.path.join(out, "analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n".join(report[:3]))
    return table
