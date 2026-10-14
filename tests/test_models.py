"""Unit tests for the from-scratch graph layers."""
import torch

from healthgraph.models import PatientGNN, RelGATv2Conv, add_self_loops, edge_softmax


def test_edge_softmax_normalises_per_destination():
    scores = torch.randn(7, 3)
    dst = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    alpha = edge_softmax(scores, dst, 3)
    sums = torch.zeros(3, 3).index_add_(0, dst, alpha)
    assert torch.allclose(sums, torch.ones(3, 3), atol=1e-6)


def test_gat_layer_shapes_and_attention():
    x = torch.randn(5, 8)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    et = torch.tensor([0, 1, 2, 0])
    ei, et = add_self_loops(ei, et, 5, 3)
    out, alpha = RelGATv2Conv(8, 4, heads=2, num_relations=4)(x, ei, et)
    assert out.shape == (5, 8) and alpha.shape == (ei.shape[1], 2)


def test_message_passing_respects_graph():
    """A node's output must not change when an unconnected node's features change."""
    torch.manual_seed(0)
    model = PatientGNN(6, 3, kind="gat", hidden=4, heads=2, layers=2).eval()
    x = torch.randn(4, 6)
    ei = torch.tensor([[0, 1], [1, 0]])                 # nodes 2, 3 isolated
    et = torch.tensor([0, 0])
    a = model(x, ei, et)
    x2 = x.clone()
    x2[3] += 5.0
    b = model(x2, ei, et)
    assert torch.allclose(a[:3], b[:3], atol=1e-6) and not torch.allclose(a[3], b[3])


def test_all_model_kinds_run():
    x = torch.randn(10, 5)
    ei = torch.randint(0, 10, (2, 30))
    et = torch.randint(0, 3, (30,))
    for kind in ["gat", "gcn", "mlp"]:
        assert PatientGNN(5, 3, kind=kind).eval()(x, ei, et).shape == (10,)
