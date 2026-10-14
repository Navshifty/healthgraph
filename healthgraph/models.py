"""Graph neural network models implemented in plain PyTorch (no PyG dependency).

HealthGAT uses a relation-aware GATv2 attention:

    e_ij^h = a_h^T LeakyReLU(W_s x_j + W_t x_i + r_{type(j->i)})
    alpha_ij^h = softmax_j(e_ij^h)   over in-neighbours j of i (incl. self loop)
    x_i'^h = sum_j alpha_ij^h W_s x_j

GATv2 (Brody et al., 2022) applies the non-linearity before the attention
vector, giving "dynamic" attention whose neighbour ranking can depend on the
query node. The relation embedding r lets each head learn how much to trust
clinical-similarity vs shared-diagnosis vs shared-provider edges.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_softmax(scores, dst, num_nodes):
    """Softmax of edge scores [E, H] grouped by destination node."""
    idx = dst.unsqueeze(-1).expand_as(scores)
    mx = torch.full((num_nodes, scores.size(1)), float("-inf"), device=scores.device)
    mx = mx.scatter_reduce(0, idx, scores, reduce="amax", include_self=True).detach()
    ex = (scores - mx[dst]).exp()
    den = torch.zeros(num_nodes, scores.size(1), device=scores.device).index_add_(0, dst, ex)
    return ex / (den[dst] + 1e-16)


def add_self_loops(edge_index, edge_type, num_nodes, self_type):
    loop = torch.arange(num_nodes, device=edge_index.device)
    edge_index = torch.cat([edge_index, torch.stack([loop, loop])], dim=1)
    edge_type = torch.cat([edge_type, torch.full((num_nodes,), self_type, device=edge_type.device)])
    return edge_index, edge_type


def drop_edges(edge_index, edge_type, p, training):
    if not training or p == 0:
        return edge_index, edge_type
    keep = torch.rand(edge_index.size(1), device=edge_index.device) >= p
    return edge_index[:, keep], edge_type[keep]


class RelGATv2Conv(nn.Module):
    def __init__(self, in_dim, out_dim, heads, num_relations, attn_dropout=0.2, negative_slope=0.2):
        super().__init__()
        self.h, self.c = heads, out_dim
        self.lin_src = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.rel = nn.Embedding(num_relations, heads * out_dim)
        self.att = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim))
        self.attn_dropout, self.slope = attn_dropout, negative_slope
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.normal_(self.rel.weight, std=0.1)

    def forward(self, x, edge_index, edge_type):
        n = x.size(0)
        src, dst = edge_index
        xs = self.lin_src(x).view(n, self.h, self.c)
        xd = self.lin_dst(x).view(n, self.h, self.c)
        xs_j = xs[src]
        g = F.leaky_relu(xs_j + xd[dst] + self.rel(edge_type).view(-1, self.h, self.c), self.slope)
        alpha = edge_softmax((g * self.att).sum(-1), dst, n)            # [E, H]
        a = F.dropout(alpha, self.attn_dropout, self.training)
        out = torch.zeros_like(xs).index_add_(0, dst, xs_j * a.unsqueeze(-1))
        return out.reshape(n, -1) + self.bias, alpha


class GCNConv(nn.Module):
    """Kipf & Welling graph convolution: relation-agnostic, fixed symmetric weights."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_type=None):
        n = x.size(0)
        src, dst = edge_index
        deg = torch.zeros(n, device=x.device).index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        w = (deg[src] * deg[dst]).rsqrt()
        h = self.lin(x)
        return torch.zeros_like(h).index_add_(0, dst, h[src] * w.unsqueeze(-1)), None


class PatientGNN(nn.Module):
    """Encoder -> L message-passing blocks (residual + LayerNorm) -> jumping-knowledge risk head.

    kind: "gat" (relation-aware GATv2), "gcn", or "mlp" (no message passing).
    """

    def __init__(self, in_dim, num_relations, kind="gat", hidden=32, heads=4, layers=2,
                 dropout=0.3, edge_dropout=0.1):
        super().__init__()
        self.kind, self.num_relations = kind, num_relations
        d = hidden * heads
        self.encoder = nn.Sequential(nn.Linear(in_dim, d), nn.ELU(), nn.Dropout(dropout))
        if kind == "gat":
            self.convs = nn.ModuleList(RelGATv2Conv(d, hidden, heads, num_relations + 1) for _ in range(layers))
        elif kind == "gcn":
            self.convs = nn.ModuleList(GCNConv(d, d) for _ in range(layers))
        elif kind == "mlp":
            self.convs = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        else:
            raise ValueError(kind)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.dropout, self.edge_dropout = dropout, edge_dropout
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    @torch.no_grad()
    def embed(self, x, edge_index, edge_type):
        """Penultimate patient representation (input to the risk head's output layer)."""
        self.eval()
        _, _, _, z = self.forward(x, edge_index, edge_type, return_attention=True, _with_embedding=True)
        return z

    def forward(self, x, edge_index, edge_type, return_attention=False, _with_embedding=False):
        h0 = h = self.encoder(x)
        attentions = []
        if self.kind != "mlp":
            edge_index, edge_type = drop_edges(edge_index, edge_type, self.edge_dropout, self.training)
            edge_index, edge_type = add_self_loops(edge_index, edge_type, x.size(0), self.num_relations)
        for conv, norm in zip(self.convs, self.norms):
            if self.kind == "mlp":
                m, att = conv(h), None
            else:
                m, att = conv(h, edge_index, edge_type)
            attentions.append(att)
            h = norm(h + F.dropout(F.elu(m), self.dropout, self.training))
        z = torch.cat([h0, h], dim=-1)
        if _with_embedding:
            hidden = self.head[:-1](z)
            return self.head[-1](hidden).squeeze(-1), attentions, (edge_index, edge_type), hidden
        logit = self.head(z).squeeze(-1)
        if return_attention:
            return logit, attentions, (edge_index, edge_type)
        return logit
