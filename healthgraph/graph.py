"""Multi-relational patient graph construction.

Relations (edge types):
  0 clinical-similarity : k nearest neighbours on standardized vitals/labs/utilisation
  1 shared-diagnoses    : k nearest neighbours by cosine similarity of diagnosis-code sets
  2 shared-provider     : random k peers treated by the same provider / care team
The graph is built from features only, never labels, so it adds no label leakage.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

RELATIONS = ["clinical-similarity", "shared-diagnoses", "shared-provider"]


def _knn(Z, k, metric):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Z)), metric=metric).fit(Z)
    _, idx = nn.kneighbors(Z)
    src = idx[:, 1:].ravel()
    dst = np.repeat(np.arange(len(Z)), idx.shape[1] - 1)
    return src, dst


def _group_edges(group, k, rng):
    src, dst = [], []
    for g in np.unique(group[group >= 0]):
        members = np.flatnonzero(group == g)
        if len(members) < 2:
            continue
        for i in members:
            peers = rng.choice(members[members != i], size=min(k, len(members) - 1), replace=False)
            src.append(peers)
            dst.append(np.full(len(peers), i))
    if not src:
        return np.array([], int), np.array([], int)
    return np.concatenate(src), np.concatenate(dst)


def _undirected_unique(src, dst, n):
    s, d = np.concatenate([src, dst]), np.concatenate([dst, src])
    keep = s != d
    key = np.unique(s[keep].astype(np.int64) * n + d[keep])
    return key // n, key % n


def build_graph(data, k_sim=10, k_code=5, k_group=8, seed=0):
    """Return edge_index [2, E] (row 0 = source, row 1 = target) and edge_type [E]."""
    rng = np.random.default_rng(seed)
    n = len(data.y)
    parts = []

    parts.append(_knn(data.sim_view, k_sim, "euclidean"))

    has_code = np.flatnonzero(data.codes.sum(1) > 0)
    s, d = _knn(data.codes[has_code], k_code, "cosine")
    parts.append((has_code[s], has_code[d]))

    parts.append(_group_edges(data.group, k_group, rng))

    srcs, dsts, types = [], [], []
    for t, (s, d) in enumerate(parts):
        s, d = _undirected_unique(s, d, n)
        srcs.append(s); dsts.append(d); types.append(np.full(len(s), t))
    edge_index = np.vstack([np.concatenate(srcs), np.concatenate(dsts)]).astype(np.int64)
    return edge_index, np.concatenate(types).astype(np.int64)


def homophily(edge_index, edge_type, y):
    """Fraction of edges joining two patients with the same outcome, per relation,
    and the same quantity among positive patients (how often a positive's neighbour is positive)."""
    out = {}
    for t, name in enumerate(RELATIONS):
        m = edge_type == t
        if m.sum() == 0:
            continue
        s, d = edge_index[0, m], edge_index[1, m]
        pos = y[d] == 1
        out[name] = {"edges": int(m.sum()),
                     "same_label": float((y[s] == y[d]).mean()),
                     "pos_neighbour_rate_for_positives": float(y[s][pos].mean()) if pos.any() else float("nan")}
    return out
