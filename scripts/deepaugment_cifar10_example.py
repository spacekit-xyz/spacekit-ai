#!/usr/bin/env python3
"""Optional DeepAugment policy search on CIFAR-10 (out-of-band from growformer).

Growformer Phase 4e certifies promote–freeze on exported pixels; it does **not**
run DeepAugment. Use this when you want augmentation policies first.

Requires: torch, torchvision, scikit-learn, deepaugment
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-size", type=int, default=10000)
    ap.add_argument("--val-size", type=int, default=2000)
    args = ap.parse_args()

    import numpy as np
    from sklearn.model_selection import train_test_split
    from torchvision.datasets import CIFAR10
    from deepaugment import DeepAugment

    train_data = CIFAR10(root=args.root, train=True, download=True)
    x_train = np.array(train_data.data)
    y_train = np.array(train_data.targets)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

    aug = DeepAugment(
        x_train,
        y_train,
        x_val,
        y_val,
        train_size=args.train_size,
        val_size=args.val_size,
        save_history=True,
        experiment_name="cifar10_full",
    )
    aug.optimize(iterations=args.iterations, epochs=args.epochs, verbose=True)
    aug.show_best(n=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
