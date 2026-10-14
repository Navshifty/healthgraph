"""Dataset adapters: a fixed cohort plus a per-split `prepare` that fits preprocessing on train only."""
import numpy as np

from . import synthetic, uci
from .data import PatientData


class UCIDataset:
    name = "uci_diabetes"

    def __init__(self, cfg, cohort=None):
        d = cfg["data"]
        self.d = d
        self.df = uci.load_raw(d["path"], cohort=cohort or d.get("cohort", "first"),
                               max_patients=d.get("max_patients"), seed=0)
        self.y = self.df["y"].to_numpy()
        self.true_risk = None
        self.meta = self.df[["encounter_id", "patient_nbr", "age", "gender", "race", "medical_specialty",
                             "admission_type_id", "diag_1", "diag_2", "diag_3"] + uci.COUNTS
                            + ["A1Cresult", "insulin", "readmitted"]].copy()
        self.meta["primary_dx_group"] = self.df["diag_1"].map(uci.icd9_group)

    def prepare(self, train_idx, seed=0):
        pre = uci.UCIPreprocessor(n_codes=self.d.get("n_codes", 100),
                                  pca_components=self.d.get("pca_components", 16),
                                  te_smoothing=self.d.get("te_smoothing", 30)).fit(self.df, train_idx, seed)
        x, sim, codes, group = pre.transform(self.df, rows=np.arange(len(self.df)))
        data = PatientData(x=x, y=self.y, feature_names=pre.all_feature_names, sim_view=sim,
                           codes=codes, code_names=[f"ICD9 {c}" for c in pre.vocab_], group=group,
                           meta=self.meta, name=self.name)
        return data, pre


class SyntheticDataset:
    name = "synthetic"

    def __init__(self, cfg):
        d = cfg["data"]
        self.data = synthetic.generate(n_patients=d.get("n_patients", 6000),
                                       n_providers=d.get("n_providers", 120),
                                       prevalence=d.get("prevalence", 0.15))
        self.y, self.meta, self.true_risk = self.data.y, self.data.meta, self.data.true_risk

    def prepare(self, train_idx, seed=0):
        return self.data, None


def load_dataset(cfg, **kw):
    return UCIDataset(cfg, **kw) if cfg["data"]["source"] == "uci" else SyntheticDataset(cfg)
