#!/usr/bin/env bash
# Full reproducible run (Linux/macOS). Each stage resumes safely: Optuna studies are persisted in SQLite.
set -e
cd "$(dirname "$0")/.."
mkdir -p results/uci results/synthetic
python -u -m healthgraph all --config configs/uci.yaml --trials 16 2>&1 | tee results/uci/log.txt
for stage in experiment ablation analyze; do
  python -u -m healthgraph $stage --config configs/synthetic.yaml 2>&1 | tee -a results/synthetic/log.txt
done
