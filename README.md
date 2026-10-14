# HealthGraph: Patient Risk Prediction Using Graph Attention Networks

**19MAM56 Advanced Machine Learning Lab · CAT II project**

HealthGraph predicts **30-day hospital readmission** for diabetic inpatients by treating
the patient cohort as a **multi-relational graph** and learning on it with a
**relation-aware GATv2** network implemented from scratch in PyTorch. It ships with
Optuna tuning, MLflow tracking, a benchmark against published results, a validated
FastAPI service with a Gradio UI, and a Docker image for Hugging Face Spaces.

| | |
|---|---|
| Dataset | UCI *Diabetes 130-US Hospitals 1999–2008* (101,766 encounters, 71,518 patients) |
| Task | binary: readmitted within 30 days (`readmitted == "<30"`) |
| Model | HealthGAT: relation-aware GATv2, 3 edge types, DropEdge, Platt calibration |
| Baselines | Logistic Regression, Random Forest, XGBoost, LightGBM, MLP, GCN |
| Tuning / tracking | Optuna (TPE + median pruning, grid search for LR) · MLflow (SQLite) + JSONL |
| Serving | FastAPI (`/predict`, `/predict/batch`, `/model/info`, `/docs`) + Gradio UI (`/ui`) |
| Deployment | Docker (port 7860, non-root) → Hugging Face Spaces (Docker SDK) |

## Repository layout

```
configs/                 YAML experiment configs (uci.yaml, synthetic.yaml)
healthgraph/             library
  uci.py                 cohort definition + split-aware preprocessing (target/frequency encoding,
                         ICD-9 grouping, PCA similarity view)
  synthetic.py           ground-truth synthetic EHR generator (controlled study)
  datasets.py            dataset adapters (preprocessing fitted on train rows only)
  graph.py               multi-relational graph construction + homophily diagnostics
  models.py              RelGATv2Conv, GCNConv, PatientGNN (gat / gcn / mlp)
  baselines.py           LR, RF, XGBoost, LightGBM
  train.py               training loop, Platt calibration, clinical metrics
  hpo.py                 Optuna studies (logged to MLflow)
  pipeline.py            multi-seed experiment, interpretability, relation ablation, bundle export
  analysis.py            significance tests, subgroup + error analysis, figures
  benchmark.py           re-runs published evaluation protocols for a fair comparison
  stats.py               DeLong test, bootstrap CIs, McNemar
  explain.py, viz.py     attention / attribution analysis, all figures
  inference.py           inductive real-time scoring on the L-hop subgraph
  tracking.py            MLflow + JSONL tracker
app/                     FastAPI service (main.py), Pydantic schemas, Gradio UI
tests/                   pytest: layers, inductive-inference exactness, API contract
deploy/                  Hugging Face Space README + deploy script
report/                  IEEE-format project report (LaTeX + PDF)
Dockerfile, requirements*.txt
```

## Reproduce

```bash
pip install -r requirements.txt
# dataset (3 MB zip) -> data/uci/diabetic_data.csv
#   https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

python -m healthgraph all --config configs/uci.yaml        # tune -> experiment -> ablation -> analyze -> benchmark
python -m healthgraph experiment --config configs/synthetic.yaml   # controlled synthetic study
mlflow ui --backend-store-uri sqlite:///mlflow.db          # browse every run, curve and artifact
pytest -q                                                  # 22 tests
```

Single stages: `tune [--what gat|boosting]`, `experiment`, `ablation`, `analyze`, `benchmark`.
Override any config value with `--set train.epochs=50 seeds=[0]`.

## Method in one paragraph

Each encounter is a node. Three relations connect nodes: **clinical similarity** (k-NN in a
16-dimensional PCA projection of the clinical features), **shared diagnoses** (cosine k-NN on
multi-hot ICD-9 codes) and **shared admitting specialty** (a care-team proxy). Edges come from
features only, never labels. HealthGAT computes
`e_ij = aᵀ LeakyReLU(W_s x_j + W_t x_i + r_type(j→i))`, a softmax over each node's
neighbourhood, and multi-head aggregation, with residual connections, LayerNorm, DropEdge and a
jumping-knowledge risk head. It is trained full-batch with a class-weighted BCE loss and early
stopping on validation AUPRC, then Platt-calibrated on the validation split. All preprocessing
(scaling, rare-level pooling, smoothed out-of-fold target encoding of the 72 specialties and 700+
ICD-9 codes, PCA) is fitted on the training rows only.

## Deployment

**Live (AWS EC2, free-tier t3.micro, ap-south-1, Elastic IP):** UI http://3.7.153.70/ui · API docs http://3.7.153.70/docs
· switch on/off with `bash deploy/aws_power.sh start|stop|status` (the address stays the same)

```bash
# local API + UI
uvicorn app.main:app --port 7860          # http://localhost:7860/ui  ·  /docs
curl -X POST localhost:7860/predict -H "content-type: application/json" -d @tests/example_patient.json

# Docker
docker build -t healthgraph .
docker run -p 7860:7860 healthgraph

# AWS EC2 (one-time: aws configure). Private S3 build context, HTTP-only security group,
# t3.micro + 2 GB swap; the image builds on first boot. --destroy removes everything.
bash deploy/deploy_aws_ec2.sh            # or: bash deploy/deploy_aws_ec2.sh --destroy

# Hugging Face Spaces (Docker Spaces now need a PRO plan): hf auth login, then
bash deploy/deploy_hf_space.sh <your-hf-username>/healthgraph
```

A new patient is linked to the stored reference cohort with the same three edge rules, and
HealthGAT runs only on the patient's L-hop computational subgraph (L = 3 for the tuned model,
about 10k of the 20k reference patients; about 0.3 s per request on CPU, 0.5 s with explanations). A test checks
that this gives the same output as a full-graph forward pass.

## Results

See `results/uci/analysis.md`, `results/uci/benchmark.csv` and the report in `report/`.

## Disclaimer

Research prototype trained on de-identified 1999–2008 data. Not a medical device and not for
clinical decision-making.
