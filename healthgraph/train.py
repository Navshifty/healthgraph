"""GNN training loop, post-hoc calibration and clinical evaluation metrics."""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from .models import PatientGNN


def stratified_split(y, seed, val=0.15, test=0.15):
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=val + test, stratify=y, random_state=seed)
    v, t = train_test_split(rest, test_size=test / (val + test), stratify=y[rest], random_state=seed)
    return {"train": np.sort(train), "val": np.sort(v), "test": np.sort(t)}


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return sum(abs(p[which == b].mean() - y[which == b].mean()) * (which == b).mean()
               for b in range(bins) if (which == b).any())


def best_f1_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 100)))
    return float(max(grid, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def clinical_metrics(y, p, thr):
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    k = max(1, int(round(0.1 * len(y))))                     # top-decile capture
    top = np.argsort(-p)[:k]
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "LogLoss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "Brier": brier_score_loss(y, p),
        "ECE": ece(y, p),
        "F1": f1_score(y, pred, zero_division=0),
        "Sensitivity": tp / max(tp + fn, 1),
        "Specificity": tn / max(tn + fp, 1),
        "PPV": tp / max(tp + fp, 1),
        "Top10%_capture": y[top].sum() / max(y.sum(), 1),
    }


def fit_platt(logits, y):
    """Post-hoc Platt scaling (p = sigmoid(a * logit + b)) fitted on validation logits.

    Needed because the pos_weighted loss shifts every logit upward; a temperature alone
    cannot undo a shift, Platt's bias term can.
    """
    w = torch.tensor([1.0, 0.0], requires_grad=True)
    lg, yt = torch.as_tensor(logits, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(w[0].clamp(0.01, 20) * lg + w[1], yt)
        loss.backward()
        return loss

    opt.step(closure)
    a, b = float(w[0].detach().clamp(0.01, 20)), float(w[1].detach())
    return (a, b) if np.isfinite(a) and np.isfinite(b) else (1.0, 0.0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def evaluate(logits, y, split):
    """Calibrate on val (Platt), pick threshold on val (max F1), report test metrics.

    Returns (test metrics, calibrated probabilities for all nodes, calibration dict)."""
    a, b = fit_platt(logits[split["val"]], y[split["val"]])
    p = sigmoid(a * logits + b)
    thr = best_f1_threshold(y[split["val"]], p[split["val"]])
    return clinical_metrics(y[split["test"]], p[split["test"]], thr), p, {"a": a, "b": b, "threshold": thr}


def train_gnn(kind, x, y, edge_index, edge_type, num_relations, split, seed,
              epochs=300, patience=60, lr=1e-3, weight_decay=5e-3, on_epoch=None, verbose=False,
              **model_kw):
    """Full-batch transductive training with early stopping on validation AUPRC.

    on_epoch(epoch, val_auprc) is called every epoch (used for Optuna pruning)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.as_tensor(x)
    Y = torch.as_tensor(y, dtype=torch.float32)
    EI, ET = torch.as_tensor(edge_index), torch.as_tensor(edge_type)
    tr, va = torch.as_tensor(split["train"]), split["val"]
    model = PatientGNN(x.shape[1], num_relations, kind=kind, **model_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=20)
    pos_weight = torch.tensor((1 - y[split["train"]].mean()) / y[split["train"]].mean())

    best, best_state, bad, t0, history = -1.0, None, 0, time.time(), []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logit = model(X, EI, ET)
        loss = F.binary_cross_entropy_with_logits(logit[tr], Y[tr], pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            p = model(X, EI, ET)
            val_loss = F.binary_cross_entropy_with_logits(p[va], Y[va], pos_weight=pos_weight).item()
            p = p.numpy()
        score = average_precision_score(y[va], p[va])
        history.append({"epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss,
                        "val_auprc": score, "val_auroc": roc_auc_score(y[va], p[va])})
        sched.step(score)
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if on_epoch is not None:
            on_epoch(epoch, score)
        if verbose and epoch % 25 == 0:
            print(f"  [{kind}] epoch {epoch:3d} loss {loss.item():.4f} val AUPRC {score:.4f}")
        if bad >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X, EI, ET).numpy()
    info = {"epochs": epoch + 1, "best_epoch": epoch + 1 - bad, "seconds": time.time() - t0,
            "best_val_auprc": best}
    return model, logits, info, history
