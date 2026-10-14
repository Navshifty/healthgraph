"""Inductive inference must match an explicit full-graph forward pass."""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import BUNDLE

pytestmark = pytest.mark.skipif(not os.path.exists(BUNDLE), reason="deployment bundle not exported")


@pytest.fixture(scope="module")
def predictor():
    from healthgraph.inference import HealthGraphPredictor
    return HealthGraphPredictor(BUNDLE)


def test_subgraph_equals_full_graph(predictor):
    from app.schemas import EXAMPLE
    rec = predictor._complete(EXAMPLE)
    x_new, sim, codes, group = predictor.pre.transform(pd.DataFrame([rec]))
    rng = np.random.default_rng(0)
    nbrs, ntypes = predictor._attach(sim, codes, group, rng)
    new = predictor.n_ref
    # full graph + new node
    ei = np.hstack([predictor.b["edge_index"].astype(np.int64),
                    np.vstack([nbrs, np.full(len(nbrs), new)]), np.vstack([np.full(len(nbrs), new), nbrs])])
    et = np.concatenate([predictor.b["edge_type"].astype(np.int64), ntypes, ntypes])
    key = np.unique(np.vstack([ei, et[None]]), axis=1)
    X = torch.cat([predictor.x, torch.as_tensor(x_new)])
    with torch.no_grad():
        full = predictor.model(X, torch.as_tensor(key[:2]), torch.as_tensor(key[2]))[new].item()
        nodes, sei, set_, me = predictor._subgraph(nbrs, ntypes)
        sub = predictor.model(X[torch.as_tensor(nodes)], torch.as_tensor(sei), torch.as_tensor(set_))[me].item()
    assert abs(full - sub) < 1e-4


def test_predict_output(predictor):
    from app.schemas import EXAMPLE
    r = predictor.predict(EXAMPLE)
    assert 0 <= r["readmission_risk"] <= 1
    assert r["risk_band"] in {"low", "elevated", "high"}
    assert len(r["top_risk_drivers"]) == 8 and r["most_attended_similar_patients"]
    assert predictor.predict(EXAMPLE)["readmission_risk"] == r["readmission_risk"]   # deterministic
