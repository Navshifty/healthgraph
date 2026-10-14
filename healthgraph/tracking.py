"""Experiment tracking: MLflow (SQLite backend) plus an always-on JSONL log.

Every run's params, per-epoch curves, final metrics and artifacts go to MLflow
(`mlflow ui --backend-store-uri sqlite:///mlflow.db`). The JSONL file keeps a
dependency-free copy that the report scripts read.
"""
import contextlib
import json
import os
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
try:
    import mlflow
except ImportError:                                     # tracking degrades to JSONL only
    mlflow = None


class Tracker:
    def __init__(self, cfg, jsonl=None):
        self.enabled = mlflow is not None
        self.jsonl = jsonl or os.path.join(cfg["output_dir"], "runs.jsonl")
        os.makedirs(os.path.dirname(self.jsonl), exist_ok=True)
        self._record = None
        if self.enabled:
            mlflow.set_tracking_uri(cfg.get("tracking", {}).get("mlflow_uri", "sqlite:///mlflow.db"))
            mlflow.set_experiment(f"healthgraph-{cfg['experiment']}")

    @contextlib.contextmanager
    def run(self, name, nested=False, tags=None):
        outer = self._record
        self._record = {"run": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "tags": tags or {},
                        "params": {}, "metrics": {}}
        ctx = mlflow.start_run(run_name=name, nested=nested, tags=tags) if self.enabled \
            else contextlib.nullcontext()
        try:
            with ctx:
                yield self
        finally:
            with open(self.jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._record, default=float) + "\n")
            self._record = outer

    def params(self, d):
        d = {k: v for k, v in d.items() if not isinstance(v, (dict, list))}
        self._record["params"].update(d)
        if self.enabled:
            mlflow.log_params({k: str(v)[:250] for k, v in d.items()})

    def metrics(self, d, step=None):
        d = {k.replace("%", "pct"): float(v) for k, v in d.items()}
        if step is None:
            self._record["metrics"].update(d)
        if self.enabled:
            mlflow.log_metrics(d, step=step)

    def artifact(self, path):
        if self.enabled and os.path.exists(path):
            mlflow.log_artifact(path)
