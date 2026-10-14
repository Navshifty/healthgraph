"""End-to-end experiment: per-seed preprocessing -> graph -> baselines + GNNs -> evaluation.

Writes to cfg["output_dir"]:
  metrics_per_seed.csv, preds_seed{s}.npz (calibrated test-set probabilities of every model),
  histories.json, seed-0 interpretability figures, ablation.csv, and (UCI) the deployment bundle.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from . import baselines, explain, graph, viz
from .config import tuned, with_tuned_gnn
from .train import evaluate, clinical_metrics, stratified_split, to_logit, train_gnn

GNN_LABELS = {"mlp": "MLP (no graph)", "gcn": "GCN", "gat": "HealthGAT"}
TABULAR = ["logreg", "rf", "xgboost", "lightgbm"]


def _gnn_kwargs(cfg):
    m, t = cfg["model"], cfg["train"]
    return dict(hidden=m["hidden"], heads=m["heads"], layers=m["layers"], dropout=m["dropout"],
                edge_dropout=m["edge_dropout"], lr=t["lr"], weight_decay=t["weight_decay"],
                epochs=t["epochs"], patience=t["patience"])


def _graph(cfg, data):
    g = cfg["graph"]
    return graph.build_graph(data, g["k_sim"], g["k_code"], g["k_group"])


def run_experiment(cfg, dataset, tracker):
    cfg = with_tuned_gnn(cfg)
    out = cfg["output_dir"]
    fig = os.path.join(out, "figures")
    os.makedirs(fig, exist_ok=True)
    R = len(graph.RELATIONS)
    kw = _gnn_kwargs(cfg)
    rows, histories = [], {}
    print(f"[experiment] {dataset.name}: {len(dataset.y)} patients, prevalence {dataset.y.mean():.1%}, "
          f"tuned={cfg.get('tuned', False)}")

    with tracker.run(f"experiment-{cfg['experiment']}", tags={"stage": "experiment"}):
        tracker.params({**{f"graph.{k}": v for k, v in cfg["graph"].items()},
                        **{f"gnn.{k}": v for k, v in kw.items()}, "n_patients": len(dataset.y)})
        for seed in cfg["seeds"]:
            split = stratified_split(dataset.y, seed, cfg["split"]["val"], cfg["split"]["test"])
            data, pre = dataset.prepare(split["train"], seed)
            ei, et = _graph(cfg, data)
            y, test = data.y, split["test"]
            probs, cals, keep = {}, {}, {}
            if seed == cfg["seeds"][0]:
                homo = graph.homophily(ei, et, y)
                with open(os.path.join(out, "homophily.json"), "w") as f:
                    json.dump(homo, f, indent=2)
                viz.homophily(homo, y.mean(), os.path.join(fig, "homophily.png"))
                print(f"[graph] {ei.shape[1]} edges; " + ", ".join(
                    f"{k}: x{v['pos_neighbour_rate_for_positives'] / y.mean():.2f}" for k, v in homo.items()))

            for name in TABULAR:
                label = baselines.LABELS[name]
                with tracker.run(f"{name}-seed{seed}", nested=True, tags={"model": label, "seed": str(seed)}):
                    params = tuned(cfg, name)
                    model, p = baselines.fit_predict(name, data.x, y, split, seed, params)
                    m, probs[label], cals[label] = evaluate(to_logit(p), y, split)
                    tracker.params({"model": label, "seed": seed, **(params or {})})
                    tracker.metrics(m)
                keep[label] = model
                rows.append({"model": label, "seed": seed, **m})

            for kind in ["mlp", "gcn", "gat"]:
                label = GNN_LABELS[kind]
                with tracker.run(f"{kind}-seed{seed}", nested=True, tags={"model": label, "seed": str(seed)}):
                    model, logits, info, hist = train_gnn(kind, data.x, y, ei, et, R, split, seed, **kw)
                    m, probs[label], cals[label] = evaluate(logits, y, split)
                    tracker.params({"model": label, "seed": seed, **kw})
                    for h in hist:
                        tracker.metrics({"train_loss": h["train_loss"], "val_loss": h["val_loss"],
                                         "val_auprc": h["val_auprc"]}, step=h["epoch"])
                    tracker.metrics({**m, "epochs": info["epochs"], "train_seconds": info["seconds"]})
                keep[label] = model
                rows.append({"model": label, "seed": seed, **m})
                if seed == cfg["seeds"][0]:
                    histories[label] = hist
                print(f"   seed {seed} {kind}: {info['epochs']} epochs ({info['seconds']:.0f}s) "
                      f"test AUROC {m['AUROC']:.3f} AUPRC {m['AUPRC']:.3f}")

            if dataset.true_risk is not None:
                label = "Oracle (true risk)"
                probs[label] = dataset.true_risk
                cals[label] = {"threshold": 0.5}
                rows.append({"model": label, "seed": seed,
                             **clinical_metrics(y[test], dataset.true_risk[test], 0.5)})

            np.savez_compressed(os.path.join(out, f"preds_seed{seed}.npz"), y=y, **split,
                                **{f"p::{k}": v for k, v in probs.items()},
                                **{f"thr::{k}": v["threshold"] for k, v in cals.items()})
            print(f"[seed {seed}] " + "  ".join(f"{r['model']}={r['AUROC']:.3f}" for r in rows if r["seed"] == seed))

            if seed == cfg["seeds"][0]:
                _interpret(cfg, dataset, data, pre, split, ei, et, keep, probs, fig, tracker)
                if dataset.name == "uci_diabetes":
                    from .inference import export_bundle
                    path = export_bundle(cfg, pre, data, split, ei, et, keep["HealthGAT"], cals["HealthGAT"],
                                         dataset.df)
                    print(f"[export] deployment bundle -> {path}")

        pd.DataFrame(rows).to_csv(os.path.join(out, "metrics_per_seed.csv"), index=False)
        with open(os.path.join(out, "histories.json"), "w") as f:
            json.dump(histories, f)
        viz.training_curves(histories, os.path.join(fig, "training_curves.png"))
        summary = pd.DataFrame(rows).groupby("model", sort=False).mean(numeric_only=True)
        tracker.metrics({f"{m}.{k}": v for m, r in summary.iterrows() for k, v in r.items()
                         if k in ("AUROC", "AUPRC") and m == "HealthGAT"})
        for f_ in sorted(os.listdir(fig)):
            tracker.artifact(os.path.join(fig, f_))
    return rows


def _interpret(cfg, dataset, data, pre, split, ei, et, models, probs, fig, tracker):
    """Seed-0 interpretability: SHAP, gradient attributions, attention, embeddings, PCA."""
    import shap
    from sklearn.manifold import TSNE
    viz.reset_style()
    test, y, R = split["test"], data.y, len(graph.RELATIONS)
    names = data.feature_names

    if dataset.name == "uci_diabetes":
        viz.eda(dataset.df, dataset.y, os.path.join(fig, "eda.png"))
        viz.pca_variance(pre.pca_, os.path.join(fig, "pca_variance.png"))

    # SHAP (TreeExplainer) for the tree model with the best *validation* AUPRC.
    from sklearn.metrics import average_precision_score
    va = split["val"]
    tree = max(["XGBoost", "LightGBM", "Random Forest"],
               key=lambda m: average_precision_score(y[va], probs[m][va]))
    rng = np.random.default_rng(0)
    sample = rng.choice(test, size=min(2000 if tree != "Random Forest" else 300, len(test)), replace=False)
    sv = shap.TreeExplainer(models[tree]).shap_values(data.x[sample])
    sv = sv[1] if isinstance(sv, list) else sv
    sv = sv[..., 1] if sv.ndim == 3 else sv
    import matplotlib.pyplot as plt
    shap.summary_plot(sv, data.x[sample], feature_names=names, max_display=20, show=False)
    plt.title(f"SHAP summary: {tree} (test sample)")
    plt.gcf().set_size_inches(8, 7)
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "shap_tree_model.png"), dpi=140, bbox_inches="tight")
    plt.close("all")
    viz.reset_style()
    pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False).to_csv(
        os.path.join(cfg["output_dir"], "shap_importance.csv"))
    with open(os.path.join(cfg["output_dir"], "shap_model.txt"), "w") as f:
        f.write(tree)

    # HealthGAT: gradient x input on a test sample (own features only).
    gat = models["HealthGAT"]
    nodes = rng.choice(test, size=min(150, len(test)), replace=False)
    attr = explain.global_attribution(gat, data.x, ei, et, nodes)
    viz.bar_importance(names, attr, "HealthGAT feature attribution (|gradient x input|)",
                       os.path.join(fig, "gat_attribution.png"))
    pd.Series(attr, index=names).sort_values(ascending=False).to_csv(
        os.path.join(cfg["output_dir"], "gat_attribution.csv"))

    # Attention analysis.
    alpha, aei, aet = explain.last_layer_attention(gat, data.x, ei, et)
    n = len(y)
    rel = {k: explain.attention_by_relation(alpha, aei, aet, n, graph.RELATIONS, nodes_)
           for k, nodes_ in {"all": test, "event=1": test[y[test] == 1], "event=0": test[y[test] == 0]}.items()}
    lift = explain.attention_lift_by_neighbour_outcome(alpha, aei, aet, y, R)
    viz.attention(rel["event=1"], rel["event=0"], os.path.join(fig, "attention_by_relation.png"))
    with open(os.path.join(cfg["output_dir"], "attention.json"), "w") as f:
        json.dump({"share": {k: v.round(4).to_dict() for k, v in rel.items()}, "lift": lift}, f, indent=2)

    # Case studies: highest-risk true positives.
    meta_cols = (["age", "gender", "medical_specialty", "diag_1", "primary_dx_group", "time_in_hospital",
                  "number_inpatient", "number_emergency", "num_medications"]
                 if dataset.name == "uci_diabetes" else
                 ["age", "phenotype", "provider", "provider_quality", "prior_admissions_1y", "albumin"])
    p = probs["HealthGAT"]
    tp = test[y[test] == 1]
    report = ["# HealthGAT case studies (seed 0, test set)\n"]
    for node in tp[np.argsort(-p[tp])][:3]:
        report.append(explain.explain_patient(node, p, alpha, aei, aet, y, graph.RELATIONS, data.meta, gat,
                                              data.x, ei, et, names, meta_cols))
        report.append("")
    with open(os.path.join(cfg["output_dir"], "case_studies.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    # Embedding projections: raw features vs learned HealthGAT representation.
    sub = rng.choice(test, size=min(2500, len(test)), replace=False)
    z = gat.embed(torch.as_tensor(data.x), torch.as_tensor(ei), torch.as_tensor(et)).numpy()
    proj = {"t-SNE: raw input features": TSNE(2, random_state=0, init="pca").fit_transform(data.x[sub]),
            "t-SNE: HealthGAT embedding": TSNE(2, random_state=0, init="pca").fit_transform(z[sub])}
    try:
        import umap
        proj["UMAP: HealthGAT embedding"] = umap.UMAP(random_state=0).fit_transform(z[sub])
    except Exception as e:
        print(f"   (UMAP skipped: {e})")
    viz.embeddings(proj, y[sub], os.path.join(fig, "embeddings.png"))


def run_ablation(cfg, dataset, tracker):
    """Retrain HealthGAT without each relation (tuned config, all seeds)."""
    cfg = with_tuned_gnn(cfg)
    R = len(graph.RELATIONS)
    kw = _gnn_kwargs(cfg)
    variants = {"all relations": list(range(R)),
                **{f"without {r}": [t for t in range(R) if t != i] for i, r in enumerate(graph.RELATIONS)},
                "no edges (self-loops only)": []}
    prev = os.path.join(cfg["output_dir"], "metrics_per_seed.csv")
    rows = []
    if os.path.exists(prev):
        full = pd.read_csv(prev)
        full = full[(full["model"] == "HealthGAT") & full["seed"].isin(cfg["seeds"])]
        if set(full["seed"]) == set(cfg["seeds"]):
            rows += [{"variant": "all relations", "seed": r.seed, "AUROC": r.AUROC, "AUPRC": r.AUPRC}
                     for r in full.itertuples()]
            del variants["all relations"]
            print("   ablation all relations: reused from experiment")
    # Checkpoint: every finished (variant, seed) is appended immediately, so an interrupted run resumes.
    ckpt = os.path.join(cfg["output_dir"], "ablation_runs.csv")
    done = pd.read_csv(ckpt) if os.path.exists(ckpt) else pd.DataFrame(columns=["variant", "seed", "AUROC", "AUPRC"])
    finished = set(zip(done["variant"], done["seed"]))
    if finished:
        print(f"   ablation: resuming, {len(finished)} runs already done")
    with tracker.run(f"ablation-{cfg['experiment']}", tags={"stage": "ablation"}):
        for seed in cfg["seeds"]:
            todo = [n for n in variants if (n, seed) not in finished]
            if not todo:
                continue
            split = stratified_split(dataset.y, seed, cfg["split"]["val"], cfg["split"]["test"])
            data, _ = dataset.prepare(split["train"], seed)
            ei, et = _graph(cfg, data)
            for name in todo:
                m = np.isin(et, variants[name])
                with tracker.run(f"ablation-{name}-seed{seed}", nested=True):
                    _, logits, _, _ = train_gnn("gat", data.x, data.y, ei[:, m], et[m], R, split, seed, **kw)
                    met, _, _ = evaluate(logits, data.y, split)
                    tracker.metrics(met)
                row = {"variant": name, "seed": seed, "AUROC": met["AUROC"], "AUPRC": met["AUPRC"]}
                pd.DataFrame([row]).to_csv(ckpt, mode="a", header=not os.path.exists(ckpt), index=False)
                print(f"   ablation seed {seed} {name}: AUROC {met['AUROC']:.3f} AUPRC {met['AUPRC']:.3f}")
    done = pd.read_csv(ckpt)
    rows += done[done["seed"].isin(cfg["seeds"]) & done["variant"].isin(variants)].to_dict("records")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg["output_dir"], "ablation_per_seed.csv"), index=False)
    order = ["all relations"] + [f"without {r}" for r in graph.RELATIONS] + ["no edges (self-loops only)"]
    table = df.groupby("variant").agg(AUROC_mean=("AUROC", "mean"), AUROC_std=("AUROC", "std"),
                                      AUPRC_mean=("AUPRC", "mean"), AUPRC_std=("AUPRC", "std")).reindex(order)
    table.to_csv(os.path.join(cfg["output_dir"], "ablation.csv"))
    viz.ablation(table, os.path.join(cfg["output_dir"], "figures", "ablation.png"))
    print(table.round(4).to_string())
    return table
