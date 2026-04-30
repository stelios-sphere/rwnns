"""Generate comparison plots from runs/*.csv."""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def read_run(path: str):
    steps, walls, lrs, train, val = [], [], [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            walls.append(float(row["wall_seconds"]))
            lrs.append(float(row["lr"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return dict(step=steps, wall=walls, lr=lrs, train=train, val=val)


def plot_pair(config_name: str):
    nano_path = os.path.join(HERE, "runs", f"nanogpt_{config_name}.csv")
    rwnn_path = os.path.join(HERE, "runs", f"rwnn_{config_name}.csv")
    if not (os.path.exists(nano_path) and os.path.exists(rwnn_path)):
        print(f"missing CSV for {config_name}; have:")
        for p in (nano_path, rwnn_path):
            print(f"  {p}: exists={os.path.exists(p)}")
        return
    nano = read_run(nano_path)
    rwnn = read_run(rwnn_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- loss vs step ----
    ax = axes[0]
    ax.plot(nano["step"], nano["train"], color="#2E6FB7", linestyle="--",
            alpha=0.5, label="nanoGPT train")
    ax.plot(nano["step"], nano["val"], color="#2E6FB7", linewidth=2,
            label="nanoGPT val")
    ax.plot(rwnn["step"], rwnn["train"], color="#C0392B", linestyle="--",
            alpha=0.5, label="RWNN train")
    ax.plot(rwnn["step"], rwnn["val"], color="#C0392B", linewidth=2,
            label="RWNN val")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss (nats / token)")
    ax.set_title(f"{config_name} — loss vs. step")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    # ---- loss vs wall ----
    ax = axes[1]
    ax.plot(nano["wall"], nano["train"], color="#2E6FB7", linestyle="--",
            alpha=0.5, label="nanoGPT train")
    ax.plot(nano["wall"], nano["val"], color="#2E6FB7", linewidth=2,
            label="nanoGPT val")
    ax.plot(rwnn["wall"], rwnn["train"], color="#C0392B", linestyle="--",
            alpha=0.5, label="RWNN train")
    ax.plot(rwnn["wall"], rwnn["val"], color="#C0392B", linewidth=2,
            label="RWNN val")
    ax.set_xlabel("wall-clock seconds")
    ax.set_ylabel("cross-entropy loss (nats / token)")
    ax.set_title(f"{config_name} — loss vs. wall time")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out = os.path.join(HERE, "results", f"compare_{config_name}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def summary(config_name: str):
    """Print a small summary table."""
    nano = read_run(os.path.join(HERE, "runs", f"nanogpt_{config_name}.csv"))
    rwnn = read_run(os.path.join(HERE, "runs", f"rwnn_{config_name}.csv"))
    print(f"\n=== {config_name} summary ===")
    print(f"{'metric':<22} {'nanoGPT':>10} {'RWNN':>10}")
    print(f"{'best val':<22} {min(nano['val']):>10.4f} {min(rwnn['val']):>10.4f}")
    print(f"{'final train':<22} {nano['train'][-1]:>10.4f} {rwnn['train'][-1]:>10.4f}")
    print(f"{'final val':<22} {nano['val'][-1]:>10.4f} {rwnn['val'][-1]:>10.4f}")
    print(f"{'wall sec total':<22} {nano['wall'][-1]:>10.1f} {rwnn['wall'][-1]:>10.1f}")
    print(f"{'sec / 1000 step':<22} "
          f"{nano['wall'][-1]/max(1, nano['step'][-1])*1000:>10.1f} "
          f"{rwnn['wall'][-1]/max(1, rwnn['step'][-1])*1000:>10.1f}")


def main():
    for cfg in ("char_shakespeare", "wikitext_bpe"):
        plot_pair(cfg)
        nano = os.path.join(HERE, "runs", f"nanogpt_{cfg}.csv")
        if os.path.exists(nano):
            summary(cfg)


if __name__ == "__main__":
    main()
