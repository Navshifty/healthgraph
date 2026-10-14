"""Tabular baselines: the models the published benchmarks on this dataset use."""
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DEFAULTS = {
    "lightgbm": {"n_estimators": 600, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 40,
                 "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 1.0},
    "xgboost": {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 5,
                "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 1.0},
    "logreg": {"C": 0.1},
    "rf": {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 10},
}
LABELS = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgboost": "XGBoost",
          "lightgbm": "LightGBM"}


def make(name, params, y_train, seed):
    p = {**DEFAULTS[name], **(params or {})}
    spw = float((1 - y_train.mean()) / y_train.mean())
    if name == "lightgbm":
        # metric must be set explicitly: otherwise early stopping also watches the default
        # binary_logloss, which rises at once under scale_pos_weight and stops training after ~3 trees.
        return lgb.LGBMClassifier(**p, scale_pos_weight=spw, metric="average_precision", random_state=seed,
                                  verbose=-1, n_jobs=-1)
    if name == "xgboost":
        return xgb.XGBClassifier(**p, scale_pos_weight=spw, random_state=seed, n_jobs=-1,
                                 eval_metric="aucpr", tree_method="hist", early_stopping_rounds=50)
    if name == "logreg":
        return LogisticRegression(**p, class_weight="balanced", max_iter=5000)
    if name == "rf":
        return RandomForestClassifier(**p, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    raise ValueError(name)


def fit_predict(name, x, y, split, seed, params=None):
    """Fit on train (validation used for early stopping where supported); return P(y=1) for all rows."""
    tr, va = split["train"], split["val"]
    model = make(name, params, y[tr], seed)
    if name == "xgboost":
        model.fit(x[tr], y[tr], eval_set=[(x[va], y[va])], verbose=False)
    elif name == "lightgbm":
        model.fit(x[tr], y[tr], eval_set=[(x[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(x[tr], y[tr])
    return model, model.predict_proba(x)[:, 1].astype(np.float64)
