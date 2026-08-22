# Growformer Ledger

Append-only, hash-chained ledger of eval results, plus a paired-SE verdict
table for `PRE_REGISTRATION.md` §1.2.

## Purpose

Make the pre-registration self-enforcing. Every completed eval appends one
immutable record; the §1.2 gate table is a **query** over those records, not a
hand-edited block. A SHA-256 chain makes any post-hoc edit, insert, or delete
detectable, so gates can't be quietly moved after the number lands.

It also fixes the measurement problem: comparing two mean-bpt numbers can't
resolve a 0.05 bpt gap. This stores the full per-window array and does a
**paired** comparison (differences cancel per-window difficulty), reports the
paired SE, and **refuses** to compare across different held-out splits.

Plain hash chain, not a Verkle — that's deliberate. A Verkle only pays off for a
remote verifier who doesn't hold the log (marketplace, §5 fork-provenance).
`record_hashes()` is the hook for that day; it's off the hot path.

## Dependencies

```toml
sha2 = "0.10"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

## Use

Integrated into `growformer-llm` (`tinystories eval` appends when `--train-bin` is set).
Ledger file: `growformer-llm/agent-data/results.jsonl`.

**Write** — automatic on held-out eval, or manual:

```rust
use results_ledger as L;
let split = L::compute_split_hash(
    Path::new("data/tinystories-heldout.bin"), 128, 64, "first")?; // see caveat
L::append_eval_record(
    Path::new("results.jsonl"), "row1b-v2", 'B', &config_hash, 1000,
    "agent-data/tinystories-row1b-v2.json", &split, 128,
    per_window_bpt, "1b-v2 held-out", &git_sha)?;
```

**Read the §1.2 table:**

```bash
cd growformer-llm
cargo run --release --bin tinystories -- ledger-table \
  --baseline row1b-v2 --candidates row3b
```

Or in Rust:

```rust
print!("{}", L::render_bet_b_table(
    Path::new("agent-data/results.jsonl"), "row1b-v2", &["row3b"], 0.05)?);
```

**Verify integrity** (CI check or before trusting the table):

```bash
cargo run --release --bin tinystories -- ledger-verify
```

Or in Rust:

```rust
match L::verify_chain(Path::new("agent-data/results.jsonl"))? {
    None => {}                                   // intact
    Some(i) => panic!("ledger tampered at record {i}"),
}
```

## Output

```
| candidate | mean bpt | Δ vs base | paired 95% CI | verdict |
|-----------|----------|-----------|---------------|---------|
| row1b     | 9.648    | —         | —             | —       |
| row3b     | 9.725    | +0.077    | ±0.228        | indistinguishable |
```

...followed by caveats that fire automatically: *gate finer than the measured
SE*, and *single-seed — a sub-2·SE gap can't separate geometry from seed noise*.
Verdict is driven by the paired CI, not the bare 0.05 gate.

## Two things that will bite you

- **Pre-fix checkpoints** (`row1b`, etc.) evaluated under post-fix forward read
  ~10 bpt — use **`row1b-v2`** as baseline for post-fix comparisons.
- **`selection_tag` must honestly describe how the 64 windows are chosen**
  (`"first"`, `"stride:37"`, `"seed:1000"`). If window selection isn't
  deterministic, `split_hash` can collide across genuinely different window
  sets. Fix determinism first, or no paired number is trustworthy.
- **If you add a field to `EvalRecord`, add it to `canonical_bytes` too.** The
  hash is computed over that fixed-order byte string, never over the JSON
  (serde doesn't guarantee key order). Miss it and old records stop verifying.

## API

| fn | does |
|----|------|
| `append_eval_record(...)` | seal + append one record (computes mean, links chain) |
| `verify_chain(path)` | `Ok(None)` intact, `Ok(Some(i))` first tampered index |
| `compare(base, cand)` | paired SE; `Err` on split/length mismatch |
| `render_bet_b_table(path, base, cands, gate)` | §1.2 Markdown table |
| `compute_split_hash(...)` / `compute_config_hash(...)` | comparability keys |
| `record_hashes(path)` | leaves for an optional Verkle commitment layer |

Independent of the 1b-v2 outcome; landable before step 4000.

## License

MIT. See [`LICENSE`](LICENSE).