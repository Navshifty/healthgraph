"""Statistical tests for model comparison: DeLong AUROC test, bootstrap CIs, McNemar."""
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(y, p1, p2):
    """Paired DeLong test (fast version, Sun & Xu 2014). Returns (auc1, auc2, z, two-sided p)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v01, v10 = [], [], []
    for p in (p1, p2):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.concatenate([p[pos], p[neg]]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)
    s01, s10 = np.cov(np.vstack(v01)), np.cov(np.vstack(v10))
    var = (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / m + (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-12))
    return aucs[0], aucs[1], float(z), float(2 * stats.norm.sf(abs(z)))


def bootstrap_ci(y, p, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    au, ap = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        au.append(roc_auc_score(y[i], p[i]))
        ap.append(average_precision_score(y[i], p[i]))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"AUROC_CI": tuple(np.percentile(au, q)), "AUPRC_CI": tuple(np.percentile(ap, q))}


def mcnemar(y, pred1, pred2):
    """Exact McNemar test on paired correct/incorrect decisions."""
    c1, c2 = pred1 == y, pred2 == y
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"only_model1_correct": b, "only_model2_correct": c, "p_value": float(p)}
