"""UCI "Diabetes 130-US Hospitals (1999-2008)" dataset: cohort definition and preprocessing.

Download `diabetic_data.csv` from
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Task: 30-day readmission (`readmitted == "<30"`).

Cohort ("first", default): first encounter per patient, encounters ending in death or
hospice removed. A patient therefore never appears in both train and test.
Cohort ("all"): every encounter, as in most published benchmarks (used only to
reproduce their protocol in `benchmark.py`).

Every statistic (scaling, category levels, target encodings, ICD vocabulary, PCA)
is fitted on the training rows only, and the same fitted object is reused by
the deployed API, so training and serving see an identical transform.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split

COUNTS = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
          "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
LOG_COUNTS = ["number_outpatient", "number_emergency", "number_inpatient"]
CATEGORICAL = ["race", "gender", "admission_type_id", "discharge_disposition_id",
               "admission_source_id", "A1Cresult", "max_glu_serum", "insulin", "change", "diabetesMed"]
HIGH_CARDINALITY = ["medical_specialty", "diag_1", "diag_2", "diag_3"]
MEDICATIONS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
               "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
               "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
               "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
               "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
# Died or discharged to hospice: cannot be readmitted (standard exclusion).
EXCLUDED_DISPOSITIONS = {11, 13, 14, 19, 20, 21}
ICD9_GROUPS = ["circulatory", "respiratory", "digestive", "diabetes", "injury",
               "musculoskeletal", "genitourinary", "neoplasms", "other"]


def _missing(code):
    return code is None or code == "" or (isinstance(code, float) and np.isnan(code))


def icd9_group(code) -> str:
    """ICD-9 chapter grouping used by Strack et al. (2014)."""
    if _missing(code) or str(code).startswith(("V", "E")):
        return "other"
    try:
        v = float(code)
    except ValueError:
        return "other"
    if 390 <= v <= 459 or int(v) == 785:
        return "circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "digestive"
    if int(v) == 250:
        return "diabetes"
    if 800 <= v <= 999:
        return "injury"
    if 710 <= v <= 739:
        return "musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "genitourinary"
    if 140 <= v <= 239:
        return "neoplasms"
    return "other"


def icd9_3char(code) -> str:
    return "none" if _missing(code) else str(code).split(".")[0]


def load_raw(path, cohort="first", max_patients=None, seed=0) -> pd.DataFrame:
    df = pd.read_csv(path, na_values="?", keep_default_na=False, low_memory=False)
    df = df.sort_values("encounter_id")
    if cohort == "first":
        df = df.drop_duplicates("patient_nbr", keep="first")
        df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISPOSITIONS)]
    if max_patients and len(df) > max_patients:
        df, _ = train_test_split(df, train_size=max_patients, stratify=df["readmitted"] == "<30",
                                 random_state=seed)
        df = df.sort_values("encounter_id")
    df = df.reset_index(drop=True)
    df["y"] = (df["readmitted"] == "<30").astype(np.int64)
    return df


class UCIPreprocessor:
    """Split-aware feature engineering.

    * counts: log1p for heavy-tailed visit counts, z-scored with training statistics
    * engineered: total prior visits, number of diabetes drugs prescribed / dose-changed
    * low-cardinality categoricals: one-hot, rare levels (<min_count in train) dropped
    * high-cardinality (specialty: 72 levels, ICD-9 codes: 700+): frequency encoding plus
      smoothed target encoding (out-of-fold on training rows so a row never sees its own label)
    * ICD-9 chapter groups (primary diagnosis one-hot + any-position counts)
    * top-n 3-character ICD-9 codes as multi-hot (also used for shared-diagnosis edges)
    * PCA on the label-free block -> the clinical-similarity view used to build k-NN edges
    """

    def __init__(self, n_codes=100, pca_components=16, te_smoothing=30, te_folds=5, min_count=20):
        self.n_codes, self.pca_components = n_codes, pca_components
        self.te_smoothing, self.te_folds, self.min_count = te_smoothing, te_folds, min_count

    # ---------- building blocks ----------
    @staticmethod
    def _base_numeric(df):
        num = df[COUNTS].astype(float).copy()
        num["age"] = df["age"].astype(str).str.extract(r"\[(\d+)-")[0].astype(float) + 5
        for c in LOG_COUNTS:
            num[c] = np.log1p(num[c])
        num["total_prior_visits"] = np.log1p(df[["number_outpatient", "number_emergency",
                                                 "number_inpatient"]].astype(float).sum(1))
        meds = df[[m for m in MEDICATIONS if m in df]].astype(str)
        num["n_diabetes_meds"] = (meds != "No").sum(1)
        num["n_dose_changes"] = meds.isin(["Up", "Down"]).sum(1)
        return num

    @staticmethod
    def _hc_values(df, col):
        if col.startswith("diag"):
            return df[col].map(icd9_3char)
        return df[col].map(lambda v: "Unknown" if _missing(v) else str(v))

    def _target_encode(self, keys, y, prior):
        stats = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
        enc = (stats["sum"] + self.te_smoothing * prior) / (stats["count"] + self.te_smoothing)
        return enc.to_dict()

    # ---------- fit / transform ----------
    def fit(self, df, train_idx, seed=0):
        train_idx = np.asarray(train_idx)
        tr = df.iloc[train_idx]
        y = tr["y"].to_numpy()
        self.prior_ = float(y.mean())
        num = self._base_numeric(tr)
        self.num_cols_ = list(num.columns)
        self.mu_, self.sd_ = num.mean(), num.std().replace(0, 1.0)

        self.levels_ = {}
        for c in CATEGORICAL:
            vc = tr[c].astype(str).value_counts()
            self.levels_[c] = list(vc[vc >= self.min_count].index)

        self.freq_, self.te_ = {}, {}
        for c in HIGH_CARDINALITY:
            keys = self._hc_values(tr, c)
            self.freq_[c] = (keys.value_counts() / len(keys)).to_dict()
            self.te_[c] = self._target_encode(keys.to_numpy(), y, self.prior_)
        # Out-of-fold encodings for the training rows themselves.
        self.oof_ = {c: np.empty(len(tr)) for c in HIGH_CARDINALITY}
        skf = StratifiedKFold(self.te_folds, shuffle=True, random_state=seed)
        for fit_i, enc_i in skf.split(np.zeros(len(tr)), y):
            for c in HIGH_CARDINALITY:
                keys = self._hc_values(tr, c).to_numpy()
                m = self._target_encode(keys[fit_i], y[fit_i], self.prior_)
                self.oof_[c][enc_i] = [m.get(k, self.prior_) for k in keys[enc_i]]
        self.train_pos_ = {int(i): p for p, i in enumerate(train_idx)}

        codes = pd.concat([self._hc_values(tr, c) for c in ["diag_1", "diag_2", "diag_3"]])
        self.vocab_ = list(codes[codes != "none"].value_counts().index[:self.n_codes])
        self.groups_ = sorted(set(self._hc_values(tr, "medical_specialty")) - {"Unknown"})

        self.pca_ = PCA(self.pca_components, random_state=seed).fit(self._label_free(tr))
        self.feature_names_ = list(self._features(tr.iloc[:1], rows=None).columns)
        return self

    def _label_free(self, df):
        num = (self._base_numeric(df)[self.num_cols_] - self.mu_) / self.sd_
        onehot = [pd.DataFrame({f"{c}={lv}": (df[c].astype(str) == lv).astype(float)
                                for lv in self.levels_[c]}, index=df.index) for c in CATEGORICAL]
        primary = df["diag_1"].map(icd9_group)
        grp = pd.DataFrame({f"primary_dx={g}": (primary == g).astype(float) for g in ICD9_GROUPS},
                           index=df.index)
        groups = [df[c].map(icd9_group) for c in ["diag_1", "diag_2", "diag_3"]]
        anyg = pd.DataFrame({f"n_dx_{g}": sum((s == g).astype(float) for s in groups)
                             for g in ICD9_GROUPS}, index=df.index)
        return pd.concat([num] + onehot + [grp, anyg], axis=1)

    def _features(self, df, rows):
        base = self._label_free(df)
        enc = {}
        for c in HIGH_CARDINALITY:
            keys = self._hc_values(df, c).to_numpy()
            enc[f"{c}_freq"] = [np.log(self.freq_[c].get(k, 1e-5)) for k in keys]
            te = np.array([self.te_[c].get(k, self.prior_) for k in keys])
            if rows is not None:                          # training rows get out-of-fold values
                for j, r in enumerate(rows):
                    p = self.train_pos_.get(int(r))
                    if p is not None:
                        te[j] = self.oof_[c][p]
            enc[f"{c}_target_enc"] = (te - self.prior_) / max(self.prior_, 1e-6)
        return pd.concat([base, pd.DataFrame(enc, index=df.index)], axis=1)

    def codes(self, df):
        idx = {c: i for i, c in enumerate(self.vocab_)}
        out = np.zeros((len(df), len(self.vocab_)), dtype=np.float32)
        for c in ["diag_1", "diag_2", "diag_3"]:
            for r, k in enumerate(self._hc_values(df, c)):
                j = idx.get(k)
                if j is not None:
                    out[r, j] = 1.0
        return out

    def group(self, df):
        idx = {g: i for i, g in enumerate(self.groups_)}
        return np.array([idx.get(k, -1) for k in self._hc_values(df, "medical_specialty")])

    def transform(self, df, rows=None):
        """rows: positional indices of df in the fitted frame (enables out-of-fold encodings)."""
        feats = self._features(df, rows)
        codes = self.codes(df)
        x = np.hstack([feats.to_numpy(np.float32), codes])
        sim = self.pca_.transform(self._label_free(df)).astype(np.float32)
        return x, sim, codes, self.group(df)

    @property
    def all_feature_names(self):
        return self.feature_names_ + [f"ICD9 {c}" for c in self.vocab_]
