"""Interpretability for HealthGAT: where attention goes, and why a patient is flagged."""
import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def last_layer_attention(model, x, edge_index, edge_type):
    model.eval()
    _, atts, (ei, et) = model(torch.as_tensor(x), torch.as_tensor(edge_index),
                              torch.as_tensor(edge_type), return_attention=True)
    return atts[-1].mean(1).numpy(), ei.numpy(), et.numpy()     # head-averaged [E]


def attention_by_relation(alpha, ei, et, n, relation_names, nodes=None):
    """Share of each patient's attention budget spent on each relation (self loop included).

    Each row of the result sums to 1, so it can be read as "what fraction of the
    representation update came from each kind of neighbour".
    """
    names = list(relation_names) + ["self"]
    mass = np.zeros((n, len(names)))
    np.add.at(mass, (ei[1], et), alpha)
    if nodes is not None:
        mass = mass[nodes]
    return pd.Series(mass.mean(0), index=names)


def attention_lift_by_neighbour_outcome(alpha, ei, et, y, num_relations):
    """Mean attention x in-degree on edges whose source patient had / did not have the event.

    1.0 = uniform attention. Values > 1 for positive neighbours mean the model
    learned (without seeing labels) to focus on neighbours who turned out to be high risk.
    """
    deg = np.bincount(ei[1], minlength=len(y))
    lift = alpha * deg[ei[1]]
    real = et < num_relations
    return {"to_positive_neighbours": float(lift[real & (y[ei[0]] == 1)].mean()),
            "to_negative_neighbours": float(lift[real & (y[ei[0]] == 0)].mean())}


def feature_attribution(model, x, edge_index, edge_type, node, feature_names, top=8):
    """Gradient x input on the patient's own features for their risk logit."""
    model.eval()
    X = torch.as_tensor(x).clone().requires_grad_(True)
    logit = model(X, torch.as_tensor(edge_index), torch.as_tensor(edge_type))[node]
    logit.backward()
    attr = (X.grad[node] * X[node]).detach().numpy()
    order = np.argsort(-np.abs(attr))[:top]
    return [(feature_names[j], float(attr[j]), float(x[node, j])) for j in order]


def global_attribution(model, x, edge_index, edge_type, nodes):
    """Mean |gradient x input| of each patient's own features over a sample of nodes."""
    model.eval()
    X = torch.as_tensor(x).clone().requires_grad_(True)
    EI, ET = torch.as_tensor(edge_index), torch.as_tensor(edge_type)
    total = np.zeros(x.shape[1])
    for node in nodes:
        X.grad = None
        model(X, EI, ET)[node].backward()
        total += np.abs((X.grad[node] * X[node]).detach().numpy())
    return total / len(nodes)


def explain_patient(node, prob, alpha, ei, et, y, relation_names, meta, model, x, edge_index,
                    edge_type, feature_names, meta_cols, top_k=5):
    names = list(relation_names) + ["self"]
    m = (ei[1] == node) & (ei[0] != node)
    nb, a, t = ei[0, m], alpha[m], et[m]
    order = np.argsort(-a)[:top_k]
    lines = [f"### Patient {node}: predicted risk {prob[node]:.1%}, actual outcome = {int(y[node])}"]
    lines.append("Profile: " + ", ".join(f"{c}={_fmt(meta.iloc[node][c])}" for c in meta_cols))
    lines.append(f"\nTop-{top_k} attended neighbours (of {m.sum()}; self-loop weight "
                 f"{alpha[(ei[1] == node) & (ei[0] == node)].sum():.3f}):\n")
    lines.append("| neighbour | relation | attention | neighbour risk | neighbour outcome |")
    lines.append("|---|---|---|---|---|")
    for j in order:
        lines.append(f"| {nb[j]} | {names[t[j]]} | {a[j]:.3f} | {prob[nb[j]]:.1%} | {int(y[nb[j]])} |")
    lines.append("\nOwn-feature drivers (gradient x input, + raises risk):\n")
    for name, val, xv in feature_attribution(model, x, edge_index, edge_type, node, feature_names):
        lines.append(f"- {name}: {val:+.3f}")
    return "\n".join(lines)


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.2f}"
    return str(v)
