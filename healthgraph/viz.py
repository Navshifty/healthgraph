"""All figures used in the report. Every function writes one PNG and returns its path."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

STYLE = {"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
         "axes.titleweight": "bold", "font.size": 9}


def reset_style():
    """Some libraries (e.g. SHAP) change matplotlib's global style on import; restore ours."""
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update(STYLE)


reset_style()
HL = "HealthGAT"
PALETTE = {"Logistic Regression": "#8c8c8c", "Random Forest": "#b07aa1", "XGBoost": "#f28e2b",
           "LightGBM": "#e15759", "MLP (no graph)": "#76b7b2", "GCN": "#59a14f", HL: "#4e79a7",
           "Oracle (true risk)": "#333333"}


def _style(name):
    s = {"color": PALETTE.get(name), "lw": 1.3, "alpha": 0.9}
    if name == HL:
        s.update(lw=2.6, alpha=1)
    if name.startswith("Oracle"):
        s.update(ls="--")
    return s


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def eda(df, y, path):
    """Class balance, readmission rate by age band and by prior inpatient visits, and missingness."""
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    counts = np.bincount(y)
    ax[0].bar(["not readmitted", "readmitted <30d"], counts, color=["#bab0ac", "#e15759"])
    for i, c in enumerate(counts):
        ax[0].text(i, c, f"{c:,}\n({c / len(y):.1%})", ha="center", va="bottom", fontsize=8)
    ax[0].set_title("Class balance")
    ax[0].set_ylim(0, counts.max() * 1.25)
    t = pd.DataFrame({"age": df["age"].astype(str), "y": y}).groupby("age")["y"].agg(["mean", "size"])
    ax[1].bar(t.index, t["mean"], color="#4e79a7")
    ax[1].set_title("30-day readmission rate by age")
    ax[1].tick_params(axis="x", rotation=45)
    inp = np.minimum(df["number_inpatient"].astype(int), 5)
    t = pd.DataFrame({"inp": inp, "y": y}).groupby("inp")["y"].mean()
    ax[2].bar([str(i) if i < 5 else "5+" for i in t.index], t.values, color="#f28e2b")
    ax[2].set_title("Readmission rate by prior inpatient visits")
    ax[2].set_xlabel("inpatient visits in prior year")
    miss = {c: df[c].isna().mean() for c in ["medical_specialty", "diag_1", "diag_2", "diag_3", "race"]}
    miss["A1C not measured"] = (df["A1Cresult"].astype(str) == "None").mean()
    s = pd.Series(miss).sort_values()
    ax[3].barh(s.index, s.values, color="#76b7b2")
    ax[3].set_title("Missingness / not measured")
    ax[3].set_xlim(0, 1)
    return _save(fig, path)


def homophily(homo, base_rate, path):
    names = list(homo)
    v = [homo[n]["pos_neighbour_rate_for_positives"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(names, v, color=["#4e79a7", "#59a14f", "#f28e2b"][:len(names)])
    ax.axhline(base_rate, ls="--", color="gray", label=f"base rate {base_rate:.3f}")
    for i, val in enumerate(v):
        ax.text(i, val, f"{val:.3f}\n(x{val / base_rate:.2f})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("P(neighbour readmitted | patient readmitted)")
    ax.set_title("Outcome homophily per relation")
    ax.set_ylim(0, max(v) * 1.35)
    ax.legend()
    return _save(fig, path)


def curves(curves_, path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, (y, p) in curves_.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, label=name, **_style(name))
        ax[1].plot(rec, prec, label=name, **_style(name))
        if not name.startswith("Oracle"):
            fy, fp = calibration_curve(y, p, n_bins=10, strategy="quantile")
            ax[2].plot(fp, fy, marker="o", ms=3, label=name, **_style(name))
    y0 = next(iter(curves_.values()))[0]
    ax[0].plot([0, 1], [0, 1], ":", color="lightgray")
    ax[1].axhline(y0.mean(), ls=":", color="lightgray")
    lim = max(0.05, max(p.max() for _, p in curves_.values()))
    ax[2].plot([0, lim], [0, lim], ":", color="lightgray")
    ax[0].set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].set(title="Precision-recall curve (test)", xlabel="Recall", ylabel="Precision")
    ax[2].set(title="Reliability diagram (after Platt scaling)", xlabel="Predicted risk",
              ylabel="Observed readmission rate")
    ax[1].legend(fontsize=7)
    return _save(fig, path)


def confusion(items, path):
    """items: list of (name, y, p, threshold)."""
    fig, ax = plt.subplots(1, len(items), figsize=(3.6 * len(items), 3.3))
    ax = np.atleast_1d(ax)
    for a, (name, y, p, thr) in zip(ax, items):
        cm = confusion_matrix(y, p >= thr)
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, f"{v:,}\n{v / cm[i].sum():.1%}", ha="center", va="center",
                   color="white" if v > cm.max() / 2 else "black", fontsize=8)
        a.set_xticks([0, 1], ["pred 0", "pred 1"])
        a.set_yticks([0, 1], ["true 0", "true 1"])
        a.set_title(f"{name}\n(threshold {thr:.3f})", fontsize=9)
    return _save(fig, path)


def training_curves(histories, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for name, h in histories.items():
        h = pd.DataFrame(h)
        s = _style(name)
        ax[0].plot(h["epoch"], h["train_loss"], label=name, **s)
        ax[1].plot(h["epoch"], h["val_loss"], label=name, **s)
        ax[2].plot(h["epoch"], h["val_auprc"], label=name, **s)
    ax[0].set(title="Training loss (weighted BCE)", xlabel="epoch")
    ax[1].set(title="Validation loss", xlabel="epoch")
    ax[2].set(title="Validation AUPRC (early-stopping criterion)", xlabel="epoch")
    ax[2].legend()
    return _save(fig, path)


def embeddings(projections, y, path):
    """projections: dict title -> [n, 2] array."""
    fig, ax = plt.subplots(1, len(projections), figsize=(4.6 * len(projections), 4.2))
    order = np.argsort(y)                                # draw positives on top
    for a, (title, z) in zip(np.atleast_1d(ax), projections.items()):
        a.scatter(z[order, 0], z[order, 1], c=np.where(y[order] == 1, "#e15759", "#bab0ac"),
                  s=np.where(y[order] == 1, 6, 3), alpha=0.7, linewidths=0)
        a.set_title(title, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    np.atleast_1d(ax)[0].scatter([], [], c="#e15759", label="readmitted <30d")
    np.atleast_1d(ax)[0].scatter([], [], c="#bab0ac", label="not readmitted")
    np.atleast_1d(ax)[0].legend(loc="lower left", fontsize=7)
    return _save(fig, path)


def pca_variance(pca, path):
    r = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(range(1, len(r) + 1), r, color="#4e79a7", label="per component")
    ax.plot(range(1, len(r) + 1), np.cumsum(r), "o-", color="#e15759", ms=3, label="cumulative")
    ax.set(title="PCA of clinical features (similarity view)", xlabel="principal component",
           ylabel="explained variance ratio")
    ax.legend()
    return _save(fig, path)


def bar_importance(names, values, title, path, top=20, xlabel="mean |attribution|"):
    order = np.argsort(values)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(order) + 1))
    ax.barh(np.array(names)[order], np.array(values)[order], color="#4e79a7")
    ax.set(title=title, xlabel=xlabel)
    return _save(fig, path)


def attention(pos, neg, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = np.arange(len(pos))
    ax.bar(xs - 0.2, pos.values, 0.4, label="readmitted", color="#e15759")
    ax.bar(xs + 0.2, neg.values, 0.4, label="not readmitted", color="#bab0ac")
    ax.set_xticks(xs, pos.index, rotation=10)
    ax.set_ylabel("share of attention (last layer)")
    ax.set_title("HealthGAT attention budget by relation")
    ax.legend()
    return _save(fig, path)


def ablation(table, path, metric="AUROC"):
    mean = table[f"{metric}_mean"]
    std = table[f"{metric}_std"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    colors = ["#4e79a7"] + ["#bab0ac"] * (len(mean) - 1)
    ax.barh(mean.index[::-1], mean.values[::-1], xerr=std.values[::-1], color=colors[::-1], capsize=3)
    ax.set_xlim(mean.min() - 3 * std.max() - 0.01, mean.max() + 3 * std.max() + 0.01)
    ax.set(title=f"Relation ablation (test {metric}, mean ± sd over seeds)", xlabel=metric)
    return _save(fig, path)


def subgroups(df, path):
    """df columns: variable, group, n, auroc, mean_pred, observed."""
    vars_ = df["variable"].unique()
    fig, ax = plt.subplots(1, len(vars_), figsize=(3.6 * len(vars_), 3.6), sharey=True)
    for a, v in zip(np.atleast_1d(ax), vars_):
        d = df[df["variable"] == v]
        a.bar(d["group"].astype(str), d["auroc"], color="#4e79a7")
        for i, (au, n) in enumerate(zip(d["auroc"], d["n"])):
            a.text(i, au, f"{au:.2f}\nn={n}", ha="center", va="bottom", fontsize=7)
        a.axhline(0.5, ls=":", color="gray")
        a.set_title(v)
        a.tick_params(axis="x", rotation=40)
        a.set_ylim(0.4, 0.9)
    np.atleast_1d(ax)[0].set_ylabel("HealthGAT test AUROC")
    return _save(fig, path)


def decision_curve(items, path):
    """Net benefit (Vickers & Elkin) for threshold probabilities 1-40 %. items: name -> (y, p)."""
    ts = np.linspace(0.01, 0.4, 60)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    y0 = next(iter(items.values()))[0]
    prev = y0.mean()
    ax.plot(ts, prev - (1 - prev) * ts / (1 - ts), color="gray", lw=1, label="treat all")
    ax.axhline(0, color="black", lw=0.8, label="treat none")
    for name, (y, p) in items.items():
        nb = [((p >= t) & (y == 1)).mean() - ((p >= t) & (y == 0)).mean() * t / (1 - t) for t in ts]
        ax.plot(ts, nb, label=name, **_style(name))
    ax.set_ylim(-0.01, prev * 1.1)
    ax.set(title="Decision curve analysis (test)", xlabel="threshold probability", ylabel="net benefit")
    ax.legend(fontsize=7)
    return _save(fig, path)
