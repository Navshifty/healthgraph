#!/usr/bin/env bash
# Publish the Docker app to a Hugging Face Space.
# Prerequisites (run once yourself):  pip install -U huggingface_hub  &&  hf auth login
# Usage: bash deploy/deploy_hf_space.sh <hf-username>/<space-name>
set -euo pipefail
SPACE="${1:?usage: deploy_hf_space.sh <user>/<space-name>}"
cd "$(dirname "$0")/.."
STAGE="$(mktemp -d)"
cp -r Dockerfile requirements-serve.txt healthgraph app "$STAGE"/
mkdir -p "$STAGE/artifacts" && cp artifacts/healthgat_uci.pt "$STAGE/artifacts/"
cp deploy/hf_space/README.md "$STAGE/README.md"
printf '*\n!healthgraph/\n!app/\n!artifacts/healthgat_uci.pt\n!requirements-serve.txt\n**/__pycache__\n' > "$STAGE/.dockerignore"
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
hf repos create "$SPACE" --repo-type space --space-sdk docker --public --exist-ok
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy HealthGraph"
echo "Deployed: https://huggingface.co/spaces/$SPACE"
