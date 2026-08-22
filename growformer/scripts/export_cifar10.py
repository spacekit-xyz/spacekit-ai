#!/usr/bin/env python3
"""Download CIFAR-10 via torchvision and export a flat binary for growformer Phase 4e.

Record layout (3073 bytes): <label u8><3072 RGB pixels> (R,G,B planes, uint8).
Matches the torchvision CIFAR-10 path used with DeepAugment; DeepAugment policy
search stays out-of-band — this script only materializes train/test arrays.

Usage:
  python3 scripts/export_cifar10.py
  CIFAR_ROOT=data/cifar10_export python3 scripts/export_cifar10.py
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=None,
        help="torchvision download root (default: <repo>/data)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="export dir for train.bin/test.bin (default: <repo>/data/cifar10_export)",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    tv_root = Path(args.root) if args.root else repo / "data"
    out_dir = Path(args.out) if args.out else repo / "data" / "cifar10_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torchvision.datasets import CIFAR10
    except ImportError:
        print("torchvision required: pip install torch torchvision", file=sys.stderr)
        return 1

    train = CIFAR10(root=str(tv_root), train=True, download=True)
    test = CIFAR10(root=str(tv_root), train=False, download=True)

    def write_bin(path: Path, data: np.ndarray, labels: np.ndarray) -> None:
        # torchvision: HWC uint8; CIFAR binary convention: CHW planar R then G then B
        assert data.ndim == 4 and data.shape[1:] == (32, 32, 3), data.shape
        n = data.shape[0]
        with path.open("wb") as f:
            for i in range(n):
                lab = int(labels[i]) & 0xFF
                hwc = data[i]
                planar = np.concatenate(
                    [hwc[:, :, 0].reshape(-1), hwc[:, :, 1].reshape(-1), hwc[:, :, 2].reshape(-1)]
                ).astype(np.uint8)
                f.write(struct.pack("B", lab))
                f.write(planar.tobytes())
        print(f"wrote {path} ({n} records × 3073 bytes)")

    x_tr = np.asarray(train.data)
    y_tr = np.asarray(train.targets, dtype=np.int64)
    x_te = np.asarray(test.data)
    y_te = np.asarray(test.targets, dtype=np.int64)

    write_bin(out_dir / "train.bin", x_tr, y_tr)
    write_bin(out_dir / "test.bin", x_te, y_te)
    (out_dir / "README.txt").write_text(
        "CIFAR-10 export for growformer --phase4e-split-cifar-lite\n"
        "Record: 1 byte label + 3072 RGB planar uint8\n"
        "Source: torchvision.datasets.CIFAR10\n"
        "DeepAugment policy search is optional and separate.\n"
    )
    print(f"OK: CIFAR-10 export at {out_dir}")
    print("Next: cargo run --release --bin growformer-demos -- --phase4e-split-cifar-lite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
