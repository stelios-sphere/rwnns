"""Correctness tests for the new node kinds (product, softmax-aggregator).

Manually constructs a small graph with hand-picked node kinds, then
verifies forward and backward against a pure-PyTorch reference that
implements each kind in plain Python tensor ops.
"""

from __future__ import annotations

import sys

import torch

# Build a tiny RandomDAG with custom node_kinds and run forward/backward.
from rwnn.graph import RandomDAG, _assemble_dag
from rwnn.model import RWNN, RWNNFunction
import numpy as np


def make_test_graph(parents_lists, node_kinds, n_in, n_bias, n_out):
    """Build a RandomDAG from explicit parent lists + per-node kinds.

    parents_lists[i] is a sorted list of int parent IDs for node i.
    node_kinds[i] is 0 (linear) | 1 (bilinear) | 2 (product) | 3 (softmax_agg).
    """
    n_total = len(parents_lists)
    n_hidden = n_total - n_in - n_bias - n_out
    return _assemble_dag(
        n_in=n_in, n_bias=n_bias, n_hidden=n_hidden, n_out=n_out,
        parents=parents_lists,
        node_kinds=np.array(node_kinds, dtype=np.int8),
    )


def pytorch_reference(graph, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch forward over arbitrary node kinds — matches the kernels."""
    B = x.shape[0]
    N = graph.n_nodes
    device = x.device
    dtype = weights.dtype

    input_pos = {idx: k for k, idx in enumerate(graph.input_ids.tolist())}
    bias_ids = set(graph.bias_ids.tolist())
    output_ids = set(graph.output_ids.tolist())
    parent_offsets = graph.parent_offsets.tolist()
    parent_ids = graph.parent_ids.tolist()
    kinds = graph.node_kinds.tolist()

    rows = []
    for i in range(N):
        if i in input_pos:
            rows.append(x[:, input_pos[i]])
            continue
        if i in bias_ids:
            rows.append(torch.ones(B, device=device, dtype=dtype))
            continue
        s, e = parent_offsets[i], parent_offsets[i + 1]
        kind = kinds[i]

        if kind == 0:  # LINEAR
            pre = torch.zeros(B, device=device, dtype=dtype)
            for k in range(s, e):
                pre = pre + weights[k] * rows[parent_ids[k]]
        elif kind == 1:  # BILINEAR
            pre = torch.zeros(B, device=device, dtype=dtype)
            for k in range(s, e, 2):
                jg, jv = parent_ids[k], parent_ids[k + 1]
                z = weights[k] * rows[jg]
                g = torch.sigmoid(z)
                v = weights[k + 1] * rows[jv]
                pre = pre + g * v
        elif kind == 2:  # PRODUCT
            pre = torch.zeros(B, device=device, dtype=dtype)
            for k in range(s, e, 2):
                pre = pre + rows[parent_ids[k]] * rows[parent_ids[k + 1]]
        else:  # SOFTMAX_AGG
            score_ids = [parent_ids[k] for k in range(s, e, 2)]
            value_ids = [parent_ids[k + 1] for k in range(s, e, 2)]
            scores = torch.stack([rows[j] for j in score_ids], dim=0)  # [K, B]
            values = torch.stack([rows[j] for j in value_ids], dim=0)  # [K, B]
            w = torch.softmax(scores, dim=0)
            pre = (w * values).sum(dim=0)

        rows.append(pre if i in output_ids else torch.tanh(pre))

    out = torch.stack([rows[i] for i in graph.output_ids.tolist()],
                      dim=0).T.contiguous()
    return out


def test_product_only():
    """A graph with one product node feeding the output."""
    # Node layout: 2 inputs, 1 bias, 1 hidden (product), 1 output (linear).
    parents_lists = [
        [], [], [],            # 0,1 inputs ; 2 bias
        [0, 1],                # 3 = product(input0, input1)
        [3, 2],                # 4 = output, linear from hidden + bias
    ]
    node_kinds = [0, 0, 0, 2, 0]  # input/bias=0 (irrelevant), 3=product, 4=linear
    graph = make_test_graph(parents_lists, node_kinds, n_in=2, n_bias=1, n_out=1)

    torch.manual_seed(0)
    net = RWNN(graph, device="cuda")
    x = torch.randn(8, 2, device="cuda")

    y = net(x)
    y_ref = pytorch_reference(graph.to("cuda"), x, net.weights.detach())
    err = (y - y_ref).abs().max().item()
    print(f"[product fwd]   max abs err: {err:.3e}")
    assert err < 1e-5, f"forward mismatch: {err}"

    # Backward via gradcheck-style comparison
    target = torch.randn(8, 1, device="cuda")
    loss = ((net(x) - target) ** 2).mean()
    loss.backward()
    g_triton = net.weights.grad.detach().clone()

    w_ref = net.weights.detach().clone().requires_grad_(True)
    y_ref = pytorch_reference(graph.to("cuda"), x, w_ref)
    loss_ref = ((y_ref - target) ** 2).mean()
    loss_ref.backward()
    g_ref = w_ref.grad.detach()
    err_w = (g_triton - g_ref).abs().max().item()
    print(f"[product wgrad] max abs err: {err_w:.3e}  "
          f"(weights for product/linear edges only)")
    assert err_w < 1e-5, f"weight-grad mismatch: {err_w}"


def test_softmax_agg():
    """A graph with one softmax-aggregator node feeding the output."""
    # Node layout: 3 inputs, 1 bias, 1 hidden (softmax-agg), 1 output (linear).
    # Softmax-agg takes 2 (score, value) pairs. Scores from input0 & input1,
    # values from input2 & bias.
    parents_lists = [
        [], [], [], [],                # 0,1,2 inputs ; 3 bias
        [0, 2, 1, 3],                  # 4 = softmax_agg over (in0, in2), (in1, bias)
        [4, 3],                        # 5 = output, linear from hidden + bias
    ]
    node_kinds = [0, 0, 0, 0, 3, 0]
    graph = make_test_graph(parents_lists, node_kinds, n_in=3, n_bias=1, n_out=1)

    torch.manual_seed(1)
    net = RWNN(graph, device="cuda")
    x = torch.randn(8, 3, device="cuda")

    y = net(x)
    y_ref = pytorch_reference(graph.to("cuda"), x, net.weights.detach())
    err = (y - y_ref).abs().max().item()
    print(f"[softmax fwd]   max abs err: {err:.3e}")
    assert err < 1e-5, f"forward mismatch: {err}"

    target = torch.randn(8, 1, device="cuda")
    loss = ((net(x) - target) ** 2).mean()
    loss.backward()
    g_triton = net.weights.grad.detach().clone()

    w_ref = net.weights.detach().clone().requires_grad_(True)
    y_ref = pytorch_reference(graph.to("cuda"), x, w_ref)
    loss_ref = ((y_ref - target) ** 2).mean()
    loss_ref.backward()
    g_ref = w_ref.grad.detach()
    err_w = (g_triton - g_ref).abs().max().item()
    print(f"[softmax wgrad] max abs err: {err_w:.3e}")
    assert err_w < 1e-5, f"weight-grad mismatch: {err_w}"


def test_input_grad_softmax():
    """Verify backward into the input tensor for softmax-agg paths."""
    parents_lists = [
        [], [], [], [],
        [0, 2, 1, 3],     # softmax-agg
        [4, 3],
    ]
    node_kinds = [0, 0, 0, 0, 3, 0]
    graph = make_test_graph(parents_lists, node_kinds, n_in=3, n_bias=1, n_out=1)

    torch.manual_seed(2)
    net = RWNN(graph, device="cuda")
    x = torch.randn(8, 3, device="cuda", requires_grad=True)
    y = net(x); loss = (y ** 2).mean(); loss.backward()
    g_x = x.grad.detach().clone()

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = net.weights.detach().clone()
    y_ref = pytorch_reference(graph.to("cuda"), x_ref, w_ref)
    loss_ref = (y_ref ** 2).mean(); loss_ref.backward()
    g_x_ref = x_ref.grad.detach()
    err = (g_x - g_x_ref).abs().max().item()
    print(f"[softmax xgrad] max abs err: {err:.3e}")
    assert err < 1e-5


def test_mixed_graph():
    """Linear + bilinear + product + softmax-agg in the same graph."""
    # 3 inputs, 1 bias, 4 hidden of different kinds, 1 output.
    parents_lists = [
        [], [], [], [],            # 0,1,2 inputs ; 3 bias
        [0, 1, 2, 3],              # 4 linear
        [0, 1, 2, 3],              # 5 bilinear (4 parents = 2 pairs)
        [0, 1, 2, 3],              # 6 product (4 parents = 2 pairs)
        [4, 5, 6, 3],              # 7 softmax-agg over (4,5), (6,bias)
        [7, 4, 5, 6],              # 8 output, linear sum of all
    ]
    node_kinds = [0, 0, 0, 0, 0, 1, 2, 3, 0]
    graph = make_test_graph(parents_lists, node_kinds, n_in=3, n_bias=1, n_out=1)

    torch.manual_seed(3)
    net = RWNN(graph, device="cuda")
    x = torch.randn(16, 3, device="cuda")
    target = torch.randn(16, 1, device="cuda")

    y = net(x)
    y_ref = pytorch_reference(graph.to("cuda"), x, net.weights.detach())
    err = (y - y_ref).abs().max().item()
    print(f"[mixed fwd]     max abs err: {err:.3e}")
    assert err < 1e-5, f"forward mismatch: {err}"

    loss = ((net(x) - target) ** 2).mean(); loss.backward()
    g_triton = net.weights.grad.detach().clone()

    w_ref = net.weights.detach().clone().requires_grad_(True)
    y_ref = pytorch_reference(graph.to("cuda"), x, w_ref)
    loss_ref = ((y_ref - target) ** 2).mean(); loss_ref.backward()
    g_ref = w_ref.grad.detach()
    err_w = (g_triton - g_ref).abs().max().item()
    print(f"[mixed wgrad]   max abs err: {err_w:.3e}")
    assert err_w < 5e-5, f"weight-grad mismatch: {err_w}"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        sys.exit(0)
    test_product_only()
    test_softmax_agg()
    test_input_grad_softmax()
    test_mixed_graph()
    print("ALL TESTS PASSED")
