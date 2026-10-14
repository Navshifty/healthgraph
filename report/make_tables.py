"""Generate LaTeX tables and number macros for the report from the saved results.

    python report/make_tables.py          (run from the repository root)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                       # the bundle unpickles healthgraph classes
UCI = os.path.join(ROOT, "results", "uci")
SYN = os.path.join(ROOT, "results", "synthetic")
OUT = os.path.join(ROOT, "report", "tables")
os.makedirs(OUT, exist_ok=True)
macros = {}

ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "MLP (no graph)", "GCN", "HealthGAT",
         "Oracle (true risk)"]
SHORT = {"Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGB", "LightGBM": "LGBM",
         "MLP (no graph)": "MLP", "GCN": "GCN", "HealthGAT": "GAT", "Oracle (true risk)": "Oracle"}


def m(name, value):
    macros[name] = value


def tex_escape(s):
    return str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def results_table(path_dir, label, caption, metrics):
    df = pd.read_csv(os.path.join(path_dir, "metrics_per_seed.csv"))
    g = df.groupby("model")
    mean, std = g.mean(numeric_only=True), g.std(numeric_only=True).fillna(0)
    models = [o for o in ORDER if o in mean.index]
    best = {c: (mean.loc[[x for x in models if not x.startswith("Oracle")], c].idxmin()
                if c in ("Brier", "ECE", "LogLoss") else
                mean.loc[[x for x in models if not x.startswith("Oracle")], c].idxmax()) for c in metrics}
    head = " & ".join(["Model"] + [c.replace("Top10%_capture", "Top-10\\%").replace("Sensitivity", "Sens.")
                                   for c in metrics])
    rows = []
    for mod in models:
        cells = []
        for c in metrics:
            s = f"{mean.loc[mod, c]:.3f}" + (f"$\\pm${std.loc[mod, c]:.3f}" if c in ("AUROC", "AUPRC") else "")
            cells.append(f"\\textbf{{{s}}}" if best[c] == mod else s)
        name = mod if mod != "HealthGAT" else "\\textbf{HealthGAT (ours)}"
        if mod.startswith("Oracle"):
            rows.append("\\midrule")
            name = "\\textit{Oracle (true risk)}"
        rows.append(" & ".join([name] + cells) + r" \\")
        if mod == "LightGBM":
            rows.append("\\midrule")
    body = "\n".join(rows)
    tex = (f"\\begin{{table*}}[t]\\centering\\caption{{{caption}}}\\label{{{label}}}\n"
           f"\\begin{{tabular}}{{l{'c' * len(metrics)}}}\\toprule\n{head} \\\\\\midrule\n{body}\n"
           f"\\bottomrule\\end{{tabular}}\\end{{table*}}\n")
    return tex, mean, std


def main():
    # ---------------- UCI main results ----------------
    try:
        metrics = ["AUROC", "AUPRC", "LogLoss", "Brier", "ECE", "F1", "Sensitivity", "PPV", "Top10%_capture"]
        tex, mean, std = results_table(UCI, "tab:main", "Test-set results on UCI Diabetes-130 (strict cohort, "
                                       "20k patients; mean$\\pm$sd over 3 seeds; best per column in bold).", metrics)
        open(os.path.join(OUT, "main_results.tex"), "w").write(tex)
        for mod in mean.index:
            k = SHORT.get(mod, mod)
            m(f"auroc{k}", f"{mean.loc[mod, 'AUROC']:.3f}")
            m(f"aurocsd{k}", f"{std.loc[mod, 'AUROC']:.3f}")
            m(f"auprc{k}", f"{mean.loc[mod, 'AUPRC']:.3f}")
            m(f"ece{k}", f"{mean.loc[mod, 'ECE']:.3f}")
            m(f"brier{k}", f"{mean.loc[mod, 'Brier']:.3f}")
            m(f"sens{k}", f"{100 * mean.loc[mod, 'Sensitivity']:.1f}")
            m(f"ppv{k}", f"{100 * mean.loc[mod, 'PPV']:.1f}")
            m(f"topdec{k}", f"{100 * mean.loc[mod, 'Top10%_capture']:.1f}")
    except FileNotFoundError as e:
        print(f'   skipped [UCI main results]: missing {e.filename}')

    # ---------------- significance ----------------
    try:
        sig = pd.read_csv(os.path.join(UCI, "significance.csv")).set_index("model")
        rows = []
        for mod in [o for o in ORDER if o in sig.index]:
            r = sig.loc[mod]
            d = r.get("ΔAUROC (HealthGAT − model), mean over seeds", "")
            p = r.get("DeLong p per seed", "")
            rows.append(f"{mod} & {r['AUROC']:.3f} & {r['AUROC 95% CI']} & {r['AUPRC 95% CI']} & "
                        f"{'' if pd.isna(d) else d} & {'' if pd.isna(p) else p} \\\\")
        open(os.path.join(OUT, "significance.tex"), "w").write(
            "\\begin{table*}[t]\\centering\\caption{Uncertainty and paired significance tests. 95\\% CIs from 1{,}000 "
            "bootstrap resamples of the seed-0 test set; $\\Delta$AUROC = HealthGAT minus model (mean over seeds); "
            "DeLong $p$-values per seed.}\\label{tab:sig}\n\\begin{tabular}{lccccc}\\toprule\n"
            "Model & AUROC (seed 0) & AUROC 95\\% CI & AUPRC 95\\% CI & $\\Delta$AUROC & DeLong $p$ (seeds 0,1,2) \\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    except FileNotFoundError as e:
        print(f'   skipped [significance]: missing {e.filename}')

    # ---------------- benchmark vs published ----------------
    try:
        bench = pd.read_csv(os.path.join(UCI, "benchmark.csv"))
        pub = pd.read_csv(os.path.join(UCI, "published_benchmarks.csv"))
        lines = ["\\begin{table}[t]\\centering\\caption{Comparison with published results. Our tuned models are re-run "
                 "under each evaluation protocol (mean$\\pm$sd AUROC, 3 seeds). P1 reproduces the published protocol.}"
                 "\\label{tab:bench}\\footnotesize\\setlength{\\tabcolsep}{3pt}",
                 "\\begin{tabular}{llc}\\toprule Source / protocol & Model & AUROC \\\\\\midrule"]
        for src, grp in pub.groupby("source", sort=False):
            for i, r in enumerate(grp.itertuples()):
                first = f"\\multirow{{{len(grp)}}}{{*}}{{\\parbox{{2.6cm}}{{{tex_escape(src)}~\\cite{{{'emijohnson2025' if 'Emi' in src else 'salim2026'}}}}}}}" if i == 0 else ""
                lines.append(f"{first} & {r.model} & {r.AUROC:.3f} \\\\")
            lines.append("\\midrule")
        for p, grp in bench.groupby("protocol", sort=False):
            label = {"P1": "P1: all enc., random split", "P2": "P2: all enc., patient-grouped",
                     "P3": "P3: strict cohort (full)", "P4": "P4: strict cohort (20k)"}[p[:2]]
            for i, r in enumerate(grp.itertuples()):
                first = f"\\multirow{{{len(grp)}}}{{*}}{{\\parbox{{2.6cm}}{{{label} (ours)}}}}" if i == 0 else ""
                lines.append(f"{first} & {r.model} & {r.AUROC_mean:.3f}$\\pm${r.AUROC_std:.3f} \\\\")
                m(f"bench{p[:2]}{SHORT[r.model]}", f"{r.AUROC_mean:.3f}")
                m(f"benchapr{p[:2]}{SHORT[r.model]}", f"{r.AUPRC_mean:.3f}")
            if p[:2] != "P4":
                lines.append("\\midrule")
            m(f"benchN{p[:2]}", f"{int(grp['n'].iloc[0]):,}".replace(",", "{,}"))
            m(f"benchOverlap{p[:2]}", f"{int(round(grp['overlap'].iloc[0])):,}".replace(",", "{,}"))
        lines.append("\\bottomrule\\end{tabular}\\end{table}")
        open(os.path.join(OUT, "benchmark.tex"), "w").write("\n".join(lines) + "\n")
        m("aurocP4GAT", macros.get("aurocGAT"))
    except FileNotFoundError as e:
        print(f'   skipped [benchmark vs published]: missing {e.filename}')

    # ---------------- ablation ----------------
    try:
        ab = pd.read_csv(os.path.join(UCI, "ablation.csv")).set_index("variant")
        rows = [f"{tex_escape(v)} & {r.AUROC_mean:.3f}$\\pm${r.AUROC_std:.3f} & {r.AUPRC_mean:.3f}$\\pm${r.AUPRC_std:.3f} \\\\"
                for v, r in ab.iterrows()]
        open(os.path.join(OUT, "ablation.tex"), "w").write(
            "\\begin{table}[t]\\centering\\caption{Relation ablation for HealthGAT on UCI (retrained per variant, "
            "3 seeds).}\\label{tab:abl}\\footnotesize\n\\begin{tabular}{lcc}\\toprule Variant & AUROC & AUPRC \\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
        for v, r in ab.iterrows():
            key = {"all relations": "All", "without clinical-similarity": "NoSim", "without shared-diagnoses": "NoDx",
                   "without shared-provider": "NoSpec", "no edges (self-loops only)": "NoEdges"}[v]
            m(f"abl{key}", f"{r.AUROC_mean:.3f}")
    except FileNotFoundError as e:
        print(f'   skipped [ablation]: missing {e.filename}')

    # ---------------- graph / data facts ----------------
    try:
        homo = json.load(open(os.path.join(UCI, "homophily.json")))
        prev = pd.read_csv(os.path.join(UCI, "metrics_per_seed.csv"))
        b = torch.load(os.path.join(ROOT, "artifacts", "healthgat_uci.pt"), weights_only=False, map_location="cpu")
        base = b["prevalence"]
        for name, k in [("clinical-similarity", "Sim"), ("shared-diagnoses", "Dx"), ("shared-provider", "Spec")]:
            h = homo[name]
            m(f"hom{k}", f"{h['pos_neighbour_rate_for_positives']:.3f}")
            m(f"homLift{k}", f"{h['pos_neighbour_rate_for_positives'] / base:.2f}")
            m(f"edges{k}", f"{h['edges']:,}".replace(",", "{,}"))
        m("nEdges", f"{int(b['edge_index'].shape[1]):,}".replace(",", "{,}"))
        m("nFeatures", str(b["in_dim"]))
        m("nRef", f"{len(b['group']):,}".replace(",", "{,}"))
        m("prevalence", f"{100 * base:.1f}")
        m("pcaVar", f"{100 * b['preprocessor'].pca_.explained_variance_ratio_.sum():.0f}")
        m("threshold", f"{100 * b['calibration']['threshold']:.1f}")
        best = json.load(open(os.path.join(UCI, "hpo", "best_gat.json")))
        for k, v in best.items():
            m("best" + k.replace("_", "").replace("k", "K", 1) if k.startswith("k_") else "best" + k.replace("_", ""),
              f"{v:.2g}" if isinstance(v, float) else str(v))
        trials = pd.read_csv(os.path.join(UCI, "hpo", "trials_gat.csv"))
        m("nTrials", str(len(trials)))
        m("nPruned", str(int((trials["state"] == "PRUNED").sum())))
        m("nComplete", str(int((trials["state"] == "COMPLETE").sum())))
        m("nInterrupted", str(int((~trials["state"].isin(["COMPLETE", "PRUNED"])).sum())))
        m("bestValAuprc", f"{trials.loc[trials['state'] == 'COMPLETE', 'value'].max():.3f}")

        att = json.load(open(os.path.join(UCI, "attention.json")))
        for name, k in [("clinical-similarity", "Sim"), ("shared-diagnoses", "Dx"), ("shared-provider", "Spec"),
                        ("self", "Self")]:
            m(f"att{k}", f"{100 * att['share']['all'][name]:.0f}")
        m("attLiftPos", f"{att['lift']['to_positive_neighbours']:.2f}")
        m("attLiftNeg", f"{att['lift']['to_negative_neighbours']:.2f}")
        m("shapModel", open(os.path.join(UCI, "shap_model.txt")).read().strip())
    except FileNotFoundError as e:
        print(f'   skipped [graph / data facts]: missing {e.filename}')

    # ---------------- synthetic ----------------
    try:
        if os.path.exists(os.path.join(SYN, "metrics_per_seed.csv")):
            tex, smean, sstd = results_table(SYN, "tab:syn", "Controlled synthetic study (6{,}000 patients, known "
                                             "ground-truth risk; mean$\\pm$sd over 3 seeds).",
                                             ["AUROC", "AUPRC", "Brier", "ECE", "Top10%_capture"])
            tex = tex.replace("table*", "table").replace("\\begin{tabular}", "\\footnotesize\\setlength{\\tabcolsep}{2.5pt}\\begin{tabular}")
            open(os.path.join(OUT, "synthetic.tex"), "w").write(tex)
            for mod in smean.index:
                m(f"syn{SHORT.get(mod, mod)}", f"{smean.loc[mod, 'AUROC']:.3f}")
                m(f"synapr{SHORT.get(mod, mod)}", f"{smean.loc[mod, 'AUPRC']:.3f}")
            if os.path.exists(os.path.join(SYN, "ablation.csv")):
                sab = pd.read_csv(os.path.join(SYN, "ablation.csv")).set_index("variant")
                for v, key in [("all relations", "All"), ("without shared-provider", "NoSpec"),
                               ("without clinical-similarity", "NoSim"), ("without shared-diagnoses", "NoDx"),
                               ("no edges (self-loops only)", "NoEdges")]:
                    if v in sab.index:
                        m(f"synabl{key}", f"{sab.loc[v, 'AUROC_mean']:.3f}")
            sh = json.load(open(os.path.join(SYN, "homophily.json")))
            m("synHomLiftSpec", f"{sh['shared-provider']['pos_neighbour_rate_for_positives'] / 0.155:.2f}")
    except FileNotFoundError as e:
        print(f'   skipped [synthetic]: missing {e.filename}')

    # ---------------- subgroups / error profile ----------------
    try:
        sub = pd.read_csv(os.path.join(UCI, "subgroups.csv"))
        worst = sub.loc[sub["auroc"].idxmin()]
        bestg = sub.loc[sub["auroc"].idxmax()]
        m("subWorst", f"{worst['variable']} = {worst['group']}")
        m("subWorstAuc", f"{worst['auroc']:.3f}")
        m("subBest", f"{bestg['variable']} = {bestg['group']}")
        m("subBestAuc", f"{bestg['auroc']:.3f}")
        m("subMaxCalGap", f"{100 * sub['calibration_gap'].abs().max():.1f}")
        err = pd.read_csv(os.path.join(UCI, "error_profile.csv"), index_col=0)
        for cell in ["TP", "FN", "FP", "TN"]:
            m(f"err{cell}Inp", f"{err.loc[cell, 'number_inpatient']:.2f}")
            m(f"err{cell}Risk", f"{100 * err.loc[cell, 'mean_predicted_risk']:.1f}")
            m(f"err{cell}N", f"{int(err.loc[cell, 'n']):,}".replace(",", "{,}"))
    except FileNotFoundError as e:
        print(f'   skipped [subgroups / error profile]: missing {e.filename}')

    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        words = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
                                        "Nine"]))
        for k, v in macros.items():                                       # LaTeX macro names: letters only
            k = "".join(words.get(ch, ch) for ch in k if ch.isalnum())
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(macros)} macros and tables to {OUT}")


if __name__ == "__main__":
    main()
