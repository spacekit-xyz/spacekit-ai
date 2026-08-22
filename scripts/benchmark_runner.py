#!/usr/bin/env python3
"""Run one benchmark command and emit portable timing data."""

from __future__ import annotations

import platform
import resource
import subprocess
import sys
import time


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "usage: benchmark_runner.py STDOUT_FILE STDERR_FILE COMMAND [ARG ...]",
            file=sys.stderr,
        )
        return 2

    stdout_path, stderr_path = sys.argv[1:3]
    command = sys.argv[3:]
    started = time.perf_counter()
    with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.perf_counter() - started

    max_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if platform.system() == "Darwin":
        max_rss /= 1024
    rss_kib = f"{max_rss:.0f}" if max_rss > 0 else "NA"

    print(f"{elapsed:.9f}\t{rss_kib}\t{completed.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
