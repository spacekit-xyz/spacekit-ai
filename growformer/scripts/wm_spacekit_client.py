#!/usr/bin/env python3
"""Thin SpaceKit-facing client: pipe JSONL ops to growformer-demos --wm-host-stdio.

Usage:
  # Scene host (default)
  python3 scripts/wm_spacekit_client.py scene --bundle /tmp/scene_bundle_42.json

  # Acting / deploy hosts
  python3 scripts/wm_spacekit_client.py acting --bundle /tmp/acting.json
  python3 scripts/wm_spacekit_client.py deploy --bundle /tmp/composed.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="SpaceKit WM host JSONL client")
    p.add_argument("mode", choices=["scene", "acting", "deploy"])
    p.add_argument("--bundle", required=True, help="Path to pinned WM JSON bundle")
    p.add_argument(
        "--bin",
        default="cargo",
        help="cargo or path to growformer-demos binary",
    )
    args = p.parse_args()
    bundle = str(Path(args.bundle).resolve())

    if args.bin == "cargo":
        cmd = [
            "cargo",
            "run",
            "--release",
            "--bin",
            "growformer-demos",
            "--",
            "--wm-host-stdio",
            args.mode,
        ]
    else:
        cmd = [args.bin, "--wm-host-stdio", args.mode]

    if args.mode == "scene":
        ops = [
            {"op": "load_scene", "path": bundle},
            {"op": "fingerprint"},
            {"op": "status"},
        ]
    elif args.mode == "acting":
        ops = [
            {"op": "load_acting", "path": bundle},
            {"op": "act", "obs": [0.2, 0.1, 0.05, 0.0]},
            {"op": "fingerprint"},
        ]
    else:
        ops = [
            {"op": "load_bundle", "path": bundle},
            {"op": "step", "obs": [0.1, -0.2, 0.0, 0.05]},
            {"op": "fingerprint"},
        ]

    payload = "\n".join(json.dumps(o) for o in ops) + "\n"
    print(f"# spawning: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(
        cmd,
        input=payload,
        text=True,
        capture_output=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    # demos print a banner; keep stdout lines that look like JSON
    for line in (proc.stdout or "").splitlines():
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            print(s)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
