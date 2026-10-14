"""Deployment: bundle export and inductive real-time scoring of new patients.

A new patient is attached to the stored reference cohort with exactly the rules used
in training (k-NN in the PCA similarity space, cosine k-NN on ICD codes, random peers
from the same admitting specialty). HealthGAT is then run on the patient's L-hop
computational subgraph only, which gives the same output as a full-graph pass
(an L-layer GNN only sees L hops) at a fraction of the cost.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from . import graph, uci
from .models import PatientGNN

BUNDLE_VERSION = 1


def export_bundle(cfg, pre, data, split, ei, et, model, cal, raw_df, path=None):
    path = path or os.path.join("artifacts", "healthgat_uci.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    train_mask = np.zeros(len(data.y), bool)
    train_mask[split["train"]] = True
    ref_info = pd.DataFrame({
        "age": raw_df["age"].astype(str), "gender": raw_df["gender"].astype(str),
        "primary_dx": raw_df["diag_1"].astype(str),
        "primary_dx_group": raw_df["diag_1"].map(uci.icd9_group),
        "specialty": raw_df["medical_specialty"].fillna("Unknown").astype(str),
        "prior_inpatient": raw_df["number_inpatient"].astype(int),
        # outcome shown for training-set reference patients only
        "readmitted_30d": np.where(train_mask, data.y, -1),
    })
    bundle = {
        "version": BUNDLE_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_config": {k: cfg["model"][k] for k in ("hidden", "heads", "layers", "dropout", "edge_dropout")},
        "graph_config": dict(cfg["graph"]),
        "in_dim": int(data.x.shape[1]),
        "state_dict": model.state_dict(),
        "calibration": cal,
        "prevalence": float(data.y[split["train"]].mean()),
        "preprocessor": pre,
        "x": data.x.astype(np.float32), "sim_view": data.sim_view.astype(np.float32),
        "codes": data.codes.astype(np.float32), "group": data.group.astype(np.int32),
        "edge_index": ei.astype(np.int32), "edge_type": et.astype(np.int8),
        "feature_names": data.feature_names, "ref_info": ref_info,
        "test_metrics_seed0": None,
    }
    torch.save(bundle, path)
    return path


class HealthGraphPredictor:
    def __init__(self, path=os.path.join("artifacts", "healthgat_uci.pt")):
        b = torch.load(path, weights_only=False, map_location="cpu")
        self.b = b
        self.pre = b["preprocessor"]
        self.R = len(graph.RELATIONS)
        self.model = PatientGNN(b["in_dim"], self.R, kind="gat", **b["model_config"])
        self.model.load_state_dict(b["state_dict"])
        self.model.eval()
        self.x = torch.as_tensor(b["x"])
        g = b["graph_config"]
        self.k_sim, self.k_code, self.k_group = g["k_sim"], g["k_code"], g["k_group"]
        self.nn_sim = NearestNeighbors(n_neighbors=self.k_sim).fit(b["sim_view"])
        self.has_code = np.flatnonzero(b["codes"].sum(1) > 0)
        self.nn_code = NearestNeighbors(n_neighbors=self.k_code, metric="cosine").fit(b["codes"][self.has_code])
        self.members = pd.Series(np.arange(len(b["group"]))).groupby(b["group"]).apply(np.asarray).to_dict()
        ei = b["edge_index"].astype(np.int64)
        order = np.argsort(ei[1], kind="stable")          # CSR by destination for subgraph extraction
        self.src_sorted, self.type_sorted = ei[0, order], b["edge_type"].astype(np.int64)[order]
        self.ptr = np.concatenate([[0], np.cumsum(np.bincount(ei[1], minlength=len(b["group"])))])
        self.cal = b["calibration"]
        self.n_ref = len(b["group"])

    # ---------- graph attachment ----------
    def _attach(self, sim, codes, group, rng):
        nbrs, types = [], []
        _, idx = self.nn_sim.kneighbors(sim)
        nbrs.append(idx[0]); types.append(np.zeros(len(idx[0]), np.int64))
        if codes.sum() > 0:
            _, idx = self.nn_code.kneighbors(codes)
            nbrs.append(self.has_code[idx[0]]); types.append(np.ones(len(idx[0]), np.int64))
        if group[0] >= 0 and self.k_group > 0 and group[0] in self.members:
            m = self.members[group[0]]
            pick = rng.choice(m, size=min(self.k_group, len(m)), replace=False)
            nbrs.append(pick); types.append(np.full(len(pick), 2, np.int64))
        return np.concatenate(nbrs), np.concatenate(types)

    def _in_edges(self, nodes):
        s = [self.src_sorted[self.ptr[n]:self.ptr[n + 1]] for n in nodes]
        t = [self.type_sorted[self.ptr[n]:self.ptr[n + 1]] for n in nodes]
        d = [np.full(self.ptr[n + 1] - self.ptr[n], n) for n in nodes]
        return np.concatenate(s), np.concatenate(d), np.concatenate(t)

    def _subgraph(self, nbrs, ntypes):
        """L-hop computational subgraph around the new node (global id = n_ref)."""
        L = self.b["model_config"]["layers"]
        new = self.n_ref
        frontier = np.unique(nbrs)
        inner = [frontier]                                # nodes at depth 1..L-1 need full in-edges
        for _ in range(L - 2):
            s, _, _ = self._in_edges(frontier)
            frontier = np.setdiff1d(np.unique(s), np.concatenate(inner))
            inner.append(frontier)
        inner = np.unique(np.concatenate(inner)) if L >= 2 else np.array([], np.int64)
        src, dst, typ = self._in_edges(inner) if len(inner) else (np.array([], np.int64),) * 3
        # new node <-> its neighbours, both directions
        src = np.concatenate([src, nbrs, np.full(len(nbrs), new)])
        dst = np.concatenate([dst, np.full(len(nbrs), new), nbrs])
        typ = np.concatenate([typ, ntypes, ntypes])
        nodes = np.unique(np.concatenate([[new], src, dst]))
        local = {g: i for i, g in enumerate(nodes)}
        ei = np.vstack([[local[s] for s in src], [local[d] for d in dst]])
        # dedupe identical (src, dst, type) triples
        key = np.unique(np.vstack([ei, typ[None]]), axis=1)
        return nodes, key[:2], key[2], local[new]

    # ---------- public API ----------
    def predict(self, record: dict, explain=True, top_k=5):
        t0 = time.perf_counter()
        df = pd.DataFrame([self._complete(record)])
        x_new, sim, codes, group = self.pre.transform(df)
        digest = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:4], "little"))   # same patient -> same peers
        nbrs, ntypes = self._attach(sim, codes, group, rng)
        nodes, ei, et, me = self._subgraph(nbrs, ntypes)
        X = torch.cat([self.x, torch.as_tensor(x_new)])[torch.as_tensor(nodes)].clone()
        EI, ET = torch.as_tensor(ei, dtype=torch.long), torch.as_tensor(et, dtype=torch.long)

        X.requires_grad_(explain)
        logit, atts, (aei, aet) = self.model(X, EI, ET, return_attention=True)
        z = float(logit[me].detach())
        prob = float(1 / (1 + np.exp(-(self.cal["a"] * z + self.cal["b"]))))
        thr = self.cal["threshold"]
        prev = self.b["prevalence"]
        out = {
            "readmission_risk": round(prob, 4),
            "risk_band": "high" if prob >= thr else "elevated" if prob >= prev else "low",
            "decision_threshold": round(thr, 4),
            "flag_for_follow_up": bool(prob >= thr),
            "cohort_baseline_risk": round(prev, 4),
            "graph_neighbours": {r: int((ntypes == i).sum()) for i, r in enumerate(graph.RELATIONS)},
            "subgraph_nodes": int(len(nodes)),
        }
        if explain:
            logit[me].backward()
            attr = (X.grad[me] * X[me]).detach().numpy()
            order = np.argsort(-np.abs(attr))[:8]
            names = self.b["feature_names"]
            out["top_risk_drivers"] = [{"feature": names[j], "contribution": round(float(attr[j]), 4)}
                                       for j in order]
            alpha = atts[-1].detach().mean(1).numpy()
            aei, aet = aei.numpy(), aet.numpy()
            m = (aei[1] == me) & (aei[0] != me)
            top = np.argsort(-alpha[m])[:top_k]
            info = self.b["ref_info"]
            rels = list(graph.RELATIONS) + ["self"]
            out["most_attended_similar_patients"] = [
                {"relation": rels[aet[m][j]], "attention": round(float(alpha[m][j]), 4),
                 **info.iloc[int(nodes[aei[0][m][j]])].to_dict()} for j in top]
            out["self_attention"] = round(float(alpha[(aei[1] == me) & (aei[0] == me)].sum()), 4)
        out["latency_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return out

    @staticmethod
    def _complete(record):
        r = dict(record)
        for med in uci.MEDICATIONS:
            r.setdefault(med, "No")
        meds = r.pop("medications", None) or {}
        r.update(meds)
        for c in ["diag_2", "diag_3", "medical_specialty"]:
            if r.get(c) in (None, ""):
                r[c] = np.nan
        return r

    def info(self):
        b = self.b
        return {"model": "HealthGAT (relation-aware GATv2)", "bundle_version": b["version"],
                "created": b["created"], "reference_patients": self.n_ref,
                "reference_edges": int(b["edge_index"].shape[1]), "relations": list(graph.RELATIONS),
                "model_config": b["model_config"], "graph_config": b["graph_config"],
                "decision_threshold": self.cal["threshold"], "cohort_baseline_risk": b["prevalence"],
                "n_features": b["in_dim"]}
