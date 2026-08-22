# Growformer Benchmark Protocol

This document separates three different questions that must not share one
headline number:

1. **Performance:** How much wall time and memory does an executable use?
2. **Internal scientific reproduction:** Do the committed routing verdicts
   reproduce under their fixed seeds and gates?
3. **External comparison:** How does Growformer compare with accepted
   continual-learning and routing baselines under one matched protocol?

Only the first two are automated in this repository today. The third remains a
publication requirement; internal phase gates are not substitutes for matched
competitor runs.

## Performance benchmarks

Both shell harnesses build the release binary, run one warmup by default, keep
every sample, and record environment metadata:

```bash
# Five measured runs per task.
scripts/benchmark_core_tasks.sh reports/benchmarks/core 5
scripts/benchmark_language.sh 5 reports/benchmarks/language

# Change warmup count without changing measured iterations.
GROWFORMER_BENCH_WARMUPS=2 scripts/benchmark_core_tasks.sh
```

Each output directory contains:

- `metadata.env`: timestamp, OS, architecture, Rust/Cargo versions, revision,
  profile, and binary path;
- `samples.tsv`: one row per measured invocation;
- `summary.tsv`: arithmetic mean wall time and peak RSS;
- one stdout and timing file per invocation.

The Python runner uses a monotonic clock and normalizes `max_rss_kib` to KiB
across macOS and Linux. Restricted environments that do not expose process RSS
record `NA` instead of reporting a fabricated zero.

To benchmark a prebuilt binary, set `GROWFORMER_BIN` to an executable path. The
harness never evaluates a shell command string, so prompts and paths are passed
as ordinary arguments.

## Criterion microbenchmarks

The Criterion suite measures similarity and routing hot paths:

```bash
cargo bench --bench cosine_similarity
```

Criterion results answer local implementation questions. They do not establish
end-to-end product latency or superiority over another framework.

## Internal scientific reproduction

The routing suite rebuilds `growformer-demos` in release mode and preserves raw
stdout, stderr, environment metadata, and the final verdict:

```bash
# Task E supervised and pseudo-label routing studies.
scripts/benchmark_science.sh routing

# Also run the bounded multi-seed context-free MNIST study.
scripts/benchmark_science.sh full reports/benchmarks/science-full
```

The script fails if a command exits non-zero or does not emit an `OVERALL:`
verdict. It does not reinterpret or loosen a gate.

## Required external comparison matrix

A publication-grade continual-learning comparison still needs one runner that
uses identical datasets, splits, task identity assumptions, seeds, and metrics
for every method. At minimum:

| Family | Required baseline |
| --- | --- |
| No continual-learning protection | Sequential fine-tuning |
| Regularization | EWC or SI |
| Replay | Experience Replay or DER++ |
| Parameter isolation | Progressive Networks and PackNet |
| Growformer | Promote–freeze with both oracle-task and task-free routing |

Recommended datasets are Split MNIST only as a smoke test, then
Split-CIFAR-100 plus CORe50 or DomainNet. Report:

- final average accuracy and average forgetting;
- task-free router agreement, collapse rate, and abstention coverage;
- parameter and artifact growth per task;
- train time, inference latency, and peak memory;
- mean, standard deviation, confidence interval, run count, and fixed seeds.

Avalanche or Mammoth should own the shared dataset and baseline protocol.
Growformer should be integrated as another strategy rather than copying
literature numbers into a table.

## Reporting rules

- Never compare a release Growformer timing with a debug baseline.
- Never combine cold-start, training, and steady-state inference latency.
- Keep raw per-run output; a summary without samples is not reproducible.
- Record failures and under-resolved results unchanged.
- Do not call a single-run gate clearance a supported market comparison.
- Pin the source revision and toolchain before publishing numbers.
