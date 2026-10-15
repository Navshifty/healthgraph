# CAT II guideline → code evidence map

Walk-through for the evaluation: every guideline item, the file and line that implements it, and the
output it produced. Paths are relative to the repository root.

## A. Advanced dataset selection & preprocessing (rubric 1, 8 marks)

| Guideline item | Code | Output / proof |
|---|---|---|
| Complex dataset from UCI | [healthgraph/uci.py:74](healthgraph/uci.py#L74) `load_raw` (cohort definition), exclusions at [uci.py:34](healthgraph/uci.py#L34) | [results/uci/figures/eda.png](results/uci/figures/eda.png) |
| Handling missingness | [uci.py:39](healthgraph/uci.py#L39) `_missing`; "Unknown"/"None" kept as categories in `_hc_values` | missingness panel in `eda.png` |
| High-cardinality encoding | [uci.py:89](healthgraph/uci.py#L89) `UCIPreprocessor`; smoothed target encoding [uci.py:126](healthgraph/uci.py#L126); out-of-fold (leakage-safe) [uci.py:151](healthgraph/uci.py#L151); frequency encoding [uci.py:186](healthgraph/uci.py#L186); ICD-9 grouping [uci.py:43](healthgraph/uci.py#L43) | [results/uci/shap_importance.csv](results/uci/shap_importance.csv) (encodings rank in the top features) |
| Transforms, rare-level pooling | log1p [uci.py:112](healthgraph/uci.py#L112); rare levels [uci.py:144](healthgraph/uci.py#L144) | |
| Dimensionality reduction (PCA) | [uci.py:165](healthgraph/uci.py#L165) | [figures/pca_variance.png](results/uci/figures/pca_variance.png) |
| t-SNE / UMAP embeddings | [healthgraph/pipeline.py:194](healthgraph/pipeline.py#L194) (t-SNE), [pipeline.py:198](healthgraph/pipeline.py#L198) (UMAP) | [figures/embeddings.png](results/uci/figures/embeddings.png) |
| ≥ 5 advanced visualizations | [healthgraph/viz.py](healthgraph/viz.py) (14 plot functions from line 43); SHAP [pipeline.py:141](healthgraph/pipeline.py#L141) | [results/uci/figures/](results/uci/figures) (13 figures) + [results/uci/hpo/](results/uci/hpo) (Optuna plots) |

Guideline examples → figure: SHAP `shap_tree_model.png` · confusion matrices `confusion_matrices.png` ·
ROC/PR `roc_pr_calibration.png` · loss curves `training_curves.png` · embedding projections `embeddings.png`.

## B. Model architecture, pipelines & optimization (rubric 2, 10 marks)

| Guideline item | Code | Output / proof |
|---|---|---|
| Advanced DL model (from scratch) | [healthgraph/models.py:43](healthgraph/models.py#L43) `RelGATv2Conv`; edge softmax [models.py:19](healthgraph/models.py#L19); full model [models.py:87](healthgraph/models.py#L87) | [tests/test_models.py](tests/test_models.py) |
| XGBoost / LightGBM / RF / LR | [healthgraph/baselines.py:20](healthgraph/baselines.py#L20) | [results/uci/metrics_summary.csv](results/uci/metrics_summary.csv) |
| Graph construction | [healthgraph/graph.py:45](healthgraph/graph.py#L45); homophily [graph.py:67](healthgraph/graph.py#L67) | [results/uci/homophily.json](results/uci/homophily.json) |
| Training loop, calibration | [healthgraph/train.py:94](healthgraph/train.py#L94); Platt scaling [train.py:54](healthgraph/train.py#L54) | `figures/training_curves.png` |
| Modular end-to-end pipeline | CLI [healthgraph/\_\_main\_\_.py:25](healthgraph/__main__.py#L25); stages [pipeline.py:34](healthgraph/pipeline.py#L34) | `python -m healthgraph all --config configs/uci.yaml` |
| Configuration files (.yaml) | [configs/uci.yaml](configs/uci.yaml) (data, graph, model, train, `hpo:` at line 35), [configs/synthetic.yaml](configs/synthetic.yaml) | tuned params as JSON: [results/uci/hpo/best_gat.json](results/uci/hpo/best_gat.json) |
| Hyperparameter optimization: Optuna | [healthgraph/hpo.py:57](healthgraph/hpo.py#L57) `tune_gat`; search space [hpo.py:66](healthgraph/hpo.py#L66); pruning [hpo.py:103](healthgraph/hpo.py#L103); boosting studies [hpo.py:114](healthgraph/hpo.py#L114) | [hpo/history_gat.png](results/uci/hpo/history_gat.png), [hpo/importance_gat.png](results/uci/hpo/importance_gat.png), [hpo/trials_gat.csv](results/uci/hpo/trials_gat.csv) |
| Hyperparameter optimization: Grid search | [hpo.py:159](healthgraph/hpo.py#L159) (`GridSampler`, logistic regression) | [hpo/trials_logreg.csv](results/uci/hpo/trials_logreg.csv) |
| Experiment tracking: MLflow | [healthgraph/tracking.py:19](healthgraph/tracking.py#L19) `Tracker`; [tracking.py:26](healthgraph/tracking.py#L26) | `mlflow ui --backend-store-uri sqlite:///mlflow.db`; JSONL copy [results/uci/runs.jsonl](results/uci/runs.jsonl) |

## C. Benchmarking & comparative analysis (rubric 3, 10 marks)

| Guideline item | Code | Output / proof |
|---|---|---|
| Comparison with baseline papers | published numbers [healthgraph/benchmark.py:23](healthgraph/benchmark.py#L23); protocols [benchmark.py:47](healthgraph/benchmark.py#L47); patient-grouped split [benchmark.py:38](healthgraph/benchmark.py#L38) | [results/uci/benchmark.csv](results/uci/benchmark.csv), [published_benchmarks.csv](results/uci/published_benchmarks.csv) |
| Metrics (ROC-AUC, F1, Log-Loss, …) | [train.py:34](healthgraph/train.py#L34) `clinical_metrics` | [results/uci/metrics_per_seed.csv](results/uci/metrics_per_seed.csv) |
| Significance tests | DeLong [healthgraph/stats.py:24](healthgraph/stats.py#L24); bootstrap CIs [stats.py:42](healthgraph/stats.py#L42); McNemar [stats.py:55](healthgraph/stats.py#L55) | [results/uci/significance.csv](results/uci/significance.csv) |
| Rigorous error analysis | [healthgraph/analysis.py:30](healthgraph/analysis.py#L30); subgroups [analysis.py:126](healthgraph/analysis.py#L126); error profile [analysis.py:138](healthgraph/analysis.py#L138) | [results/uci/analysis.md](results/uci/analysis.md), [subgroups.csv](results/uci/subgroups.csv), [error_profile.csv](results/uci/error_profile.csv) |
| Relation ablation | [pipeline.py:204](healthgraph/pipeline.py#L204) `run_ablation` | [results/uci/ablation.csv](results/uci/ablation.csv), `figures/ablation.png` |
| Controlled study | [healthgraph/synthetic.py](healthgraph/synthetic.py) | [results/synthetic/analysis.md](results/synthetic/analysis.md) |

## D. Scalable UI & production deployment (rubric 4, 8 marks)

| Guideline item | Code | Output / proof |
|---|---|---|
| FastAPI real-time inference | [app/main.py:79](app/main.py#L79) `POST /predict`; batch [main.py:85](app/main.py#L85); health [main.py:69](app/main.py#L69) | live docs http://3.7.153.70/docs |
| Input validation | [app/schemas.py:28](app/schemas.py#L28) `Patient`; ranges e.g. [schemas.py:38](app/schemas.py#L38); ICD-9 regex [schemas.py:6](app/schemas.py#L6); unknown fields rejected [schemas.py:30](app/schemas.py#L30); domain rule [schemas.py:60](app/schemas.py#L60) | tests [tests/test_api.py:46](tests/test_api.py#L46) (HTTP 422) |
| Inference engine | [healthgraph/inference.py:59](healthgraph/inference.py#L59); L-hop subgraph [inference.py:102](healthgraph/inference.py#L102) | exactness test [tests/test_inference.py:20](tests/test_inference.py#L20) |
| Interactive web UI | [app/ui.py](app/ui.py); mounted at [main.py:91](app/main.py#L91) | live UI http://3.7.153.70/ui · [report/figures/ui_screenshot.png](report/figures/ui_screenshot.png) |
| Docker container | [Dockerfile](Dockerfile) (base line 3, non-root line 22, health check line 25), [requirements-serve.txt](requirements-serve.txt) | `docker build -t healthgraph .` |
| Cloud hosting (AWS) | [deploy/deploy_aws_ec2.sh:76](deploy/deploy_aws_ec2.sh#L76) (EC2), HTTP-only firewall [line 55](deploy/deploy_aws_ec2.sh#L55); on/off [deploy/aws_power.ps1](deploy/aws_power.ps1) | http://3.7.153.70/ui |

## E. Documentation & submission (rubric 5, 4 marks)

| Guideline item | File |
|---|---|
| IEEE report (≤ 10 pages, Sections 1–8) | [report/HealthGraph_IEEE_Report.docx](report/HealthGraph_IEEE_Report.docx) (7 pp) · [report/healthgraph_report.pdf](report/healthgraph_report.pdf) (9 pp, LaTeX source [healthgraph_report.tex](report/healthgraph_report.tex)) |
| Source code (modular, documented) | [healthgraph/](healthgraph), [app/](app), [tests/](tests) (22 tests: `pytest -q`) |
| requirements.txt, Dockerfile, config | [requirements.txt](requirements.txt), [Dockerfile](Dockerfile), [configs/](configs) |
| Repository organisation | [README.md](README.md) |
| Live links | UI http://3.7.153.70/ui · GitHub https://github.com/Navshifty/healthgraph |
