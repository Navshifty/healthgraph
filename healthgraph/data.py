"""Patient data container shared by the synthetic generator and real-data loaders."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PatientData:
    x: np.ndarray                 # [N, F] model input features (standardized)
    y: np.ndarray                 # [N] binary outcome (1 = adverse event / readmission)
    feature_names: list
    sim_view: np.ndarray          # [N, d] clinical vector used for similarity edges
    codes: np.ndarray             # [N, C] multi-hot diagnosis codes
    code_names: list
    group: np.ndarray             # [N] care-team / provider id (-1 = unknown)
    meta: pd.DataFrame            # human-readable patient attributes for explanations
    true_risk: np.ndarray = None  # ground-truth probability (synthetic data only)
    name: str = "dataset"
    extras: dict = field(default_factory=dict)


def standardize(df: pd.DataFrame) -> np.ndarray:
    v = df.to_numpy(dtype=np.float64)
    mu, sd = v.mean(0), v.std(0)
    sd[sd == 0] = 1.0
    return ((v - mu) / sd).astype(np.float32)
