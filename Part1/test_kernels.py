"""Correctness tests: Triton kernels vs. a pure-PyTorch reference.

We refuse to trust the GPU kernels until they agree with a slow,
obviously-correct Python implementation on the same random graph.
"""

from __future__ import annotations

import sys

import torch

from rwnn import build_random_dag
from rwnn.model import RWNN, RWNNFunction


def pytorch_reference(graph, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Naïve reference: iterate nodes in index order, do the math in Python.

    Built as a list of per-node row tensors (no in-place writes) so autograd
    can track gradients through it.
    """
    B = x.shape[0]
    N = graph.n_nodes
    device = x.device
    dtype = weights.dtype

    input_ids = graph.input_ids.tolist()
    bias_ids = set(graph.bias_ids.tolist())
    output_ids = set(graph.output_ids.tolist())
    parent_offsets = graph.parent_offsets.tolist()
    parent_ids = graph.parent_ids.tolist()

    input_pos = {idx: k for k, idx in enumerate(input_ids)}

    rows: list[torch.Tensor] = []
    for i in range(N):
        if i in input_pos:
            rows.append(x[:, input_pos[i]])
            continue
        if i in bias_ids:
            rows.append(torch.ones(B, device=device, dtype=dtype))
            continue
        start, end = parent_offsets[i], parent_offsets[i + 1]
        pre = torch.zeros(B, device=device, dtype=dtype)
        for k in range(start, end):
            j = parent_ids[k]
            pre = pre + weights[k] * rows[j]
        rows.append(pre if i in output_ids else torch.tanh(pre))

    out_rows = [rows[i] for i in graph.output_ids.tolist()]
    return torch.stack(out_rows, dim=0).transpose(0, 1).contiguous()


def test_forward_matches_reference():
    torch.manual_seed(0)
    device = "cuda"
    graph = build_random_dag(n_in=3, n_bias=2, n_hidden=8, n_out=2,
                             edge_prob=0.7, seed=42)
    net = RWNN(graph, device=device)

    x = torch.randn(16, 3, device=device)
    y_triton = net(x)
    y_ref = pytorch_reference(graph.to(device), x, net.weights.detach())

    err = (y_triton - y_ref).abs().max().item()
    print(f"[forward]      max abs err vs reference: {err:.3e}")
    assert err < 1e-5, f"Forward mismatch: {err}"


def test_weight_grad_matches_reference():
    torch.manual_seed(1)
    device = "cuda"
    graph = build_random_dag(n_in=3, n_bias=2, n_hidden=6, n_out=1,
                             edge_prob=0.8, seed=7)
    net = RWNN(graph, device=device)

    x = torch.randn(8, 3, device=device)
    target = torch.randn(8, 1, device=device)

    # Triton path.
    y = net(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()
    g_triton = net.weights.grad.detach().clone()

    # Reference path: rebuild graph on CPU of weights with requires_grad.
    w_ref = net.weights.detach().clone().requires_grad_(True)
    y_ref = pytorch_reference(graph.to(device), x, w_ref)
    loss_ref = ((y_ref - target) ** 2).mean()
    loss_ref.backward()
    g_ref = w_ref.grad.detach()

    err = (g_triton - g_ref).abs().max().item()
    rel = err / (g_ref.abs().max().item() + 1e-12)
    print(f"[weight grad]  max abs err: {err:.3e}  max rel: {rel:.3e}")
    assert err < 1e-4, f"Weight grad mismatch: {err}"


def test_input_grad_matches_reference():
    torch.manual_seed(2)
    device = "cuda"
    graph = build_random_dag(n_in=2, n_bias=1, n_hidden=5, n_out=1,
                             edge_prob=0.8, seed=13)
    net = RWNN(graph, device=device)

    x = torch.randn(4, 2, device=device, requires_grad=True)
    y = net(x)
    loss = (y ** 2).mean()
    loss.backward()
    g_x_triton = x.grad.detach().clone()

    w_ref = net.weights.detach().clone()
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = pytorch_reference(graph.to(device), x_ref, w_ref)
    loss_ref = (y_ref ** 2).mean()
    loss_ref.backward()
    g_x_ref = x_ref.grad.detach()

    err = (g_x_triton - g_x_ref).abs().max().item()
    print(f"[input grad]   max abs err: {err:.3e}")
    assert err < 1e-5, f"Input grad mismatch: {err}"


def test_various_shapes():
    torch.manual_seed(3)
    device = "cuda"
    for n_hidden, B in [(1, 7), (20, 200), (50, 1)]:
        graph = build_random_dag(n_in=2, n_bias=2, n_hidden=n_hidden, n_out=1,
                                 edge_prob=0.6, seed=n_hidden)
        net = RWNN(graph, device=device)
        x = torch.randn(B, 2, device=device)
        y_triton = net(x)
        y_ref = pytorch_reference(graph.to(device), x, net.weights.detach())
        err = (y_triton - y_ref).abs().max().item()
        print(f"[shape test]   n_hidden={n_hidden:3d} B={B:4d}  err={err:.3e}")
        assert err < 1e-5


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        sys.exit(0)
    test_forward_matches_reference()
    test_weight_grad_matches_reference()
    test_input_grad_matches_reference()
    test_various_shapes()
    print("ALL TESTS PASSED")
