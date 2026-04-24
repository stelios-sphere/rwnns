"""Demo: build a large layered RWNN and exercise it.

Matches the request: 1000 nodes, 5 topological layers, user-tunable
connection density. Reports graph stats, runs a forward + backward
pass through the Triton kernels to confirm everything actually works
at that scale, and prints a small timing summary.
"""

from __future__ import annotations

import argparse
import time

import torch

from rwnn import RWNN, build_layered_rwnn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-nodes", type=int, default=1000)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--edge-prob", type=float, default=0.02,
                   help="Keep small for large networks; 0.02 ≈ a few dozen parents per node.")
    p.add_argument("--n-in", type=int, default=2)
    p.add_argument("--n-bias", type=int, default=2)
    p.add_argument("--n-out", type=int, default=1)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    graph = build_layered_rwnn(
        n_nodes=args.n_nodes,
        edge_prob=args.edge_prob,
        n_layers=args.n_layers,
        n_in=args.n_in,
        n_bias=args.n_bias,
        n_out=args.n_out,
        seed=args.seed,
    )

    print("=== graph ===")
    print(f"  n_nodes    : {graph.n_nodes}")
    print(f"  n_edges    : {graph.n_edges}")
    print(f"  n_levels   : {graph.n_levels}  (requested n_layers={args.n_layers})")

    level_starts = graph.level_starts.tolist()
    per_level = [level_starts[ℓ + 1] - level_starts[ℓ] for ℓ in range(graph.n_levels)]
    print(f"  per level  : {per_level}")
    print(f"  avg fan-in : {graph.fan_in.float().mean().item():.1f}")
    print(f"  max fan-in : {graph.fan_in.max().item()}")

    device = "cuda"
    net = RWNN(graph, device=device)
    n_params = net.weights.numel()
    print(f"  parameters : {n_params:,} (one per edge)")

    # Sanity forward + backward.
    x = torch.randn(args.batch, args.n_in, device=device)
    target = torch.randn(args.batch, args.n_out, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Warm up (first call triggers Triton JIT compile).
    pred = net(x)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    print(f"\n=== timing (batch={args.batch}, {args.steps} steps) ===")
    t0 = time.time()
    for _ in range(args.steps):
        pred = net(x)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  total      : {dt:.3f} s")
    print(f"  per step   : {dt / args.steps * 1e3:.3f} ms")
    print(f"  final loss : {loss.item():.4f}")


if __name__ == "__main__":
    main()
