//! results_ledger.rs — append-only, hash-chained eval results ledger (in-crate).
//!
//! Source of truth for Bet A/B/C eval results. One record per completed eval;
//! nothing mutates a past record. The SHA-256 chain makes any post-hoc edit,
//! insert, or delete detectable, so the PRE_REGISTRATION.md §1.2 gate table can
//! be a *query* over this file rather than a hand-edited block — that is what
//! makes the pre-registration self-enforcing.
//!
//! Crypto note (deliberate): plain SHA-256 hash chain, NOT a Verkle/Merkle tree.
//! For a single-author local ledger the requirement is tamper-evidence +
//! pre-commitment, which a hash chain gives with one dependency (`sha2`). A
//! Verkle only pays off when a *remote* verifier who does not hold the full log
//! needs a compact membership proof (marketplace, or §5 fork-provenance). If
//! that day comes, fold `record_hashes()` into your Verkle — it is a hook, not
//! a dependency taken on now.
//!
//! Deps (Cargo.toml):
//!   sha2 = "0.10"
//!   serde = { version = "1", features = ["derive"] }
//!   serde_json = "1"

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub const GENESIS: &str = "GENESIS";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalRecord {
    pub run_id: String,
    pub bet: char, // 'A' | 'B' | 'C'
    pub config_hash: String,
    pub seed: u64,
    pub checkpoint_path: String,
    pub split_hash: String,
    pub n_windows: usize,
    pub seq_len: usize,
    pub per_window_bpt: Vec<f64>, // the array — enables paired stats downstream
    pub mean_bpt: f64,
    pub git_sha: String,
    pub timestamp: String, // RFC3339
    pub notes: String,
    pub prev_hash: String,
    pub record_hash: String,
}

fn sha_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex(&h.finalize())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Canonical hash input: an EXPLICIT, fixed-order byte string over the fields.
/// We do NOT hash serde_json output — serde_json does not guarantee key order,
/// so hashing its bytes would make cross-reader verification flaky. This is the
/// one subtlety that bites in Rust; handling it by hand removes all ambiguity.
fn canonical_bytes(r: &EvalRecord) -> Vec<u8> {
    let mut s = String::new();
    s.push_str(&r.run_id);
    s.push('\x1f');
    s.push(r.bet);
    s.push('\x1f');
    s.push_str(&r.config_hash);
    s.push('\x1f');
    s.push_str(&r.seed.to_string());
    s.push('\x1f');
    s.push_str(&r.checkpoint_path);
    s.push('\x1f');
    s.push_str(&r.split_hash);
    s.push('\x1f');
    s.push_str(&r.n_windows.to_string());
    s.push('\x1f');
    s.push_str(&r.seq_len.to_string());
    s.push('\x1f');
    for x in &r.per_window_bpt {
        // fixed formatting so the same value always hashes the same way
        s.push_str(&format!("{:.9}", x));
        s.push(',');
    }
    s.push('\x1f');
    s.push_str(&format!("{:.9}", r.mean_bpt));
    s.push('\x1f');
    s.push_str(&r.git_sha);
    s.push('\x1f');
    s.push_str(&r.timestamp);
    s.push('\x1f');
    s.push_str(&r.notes);
    s.push('\x1f');
    s.push_str(&r.prev_hash);
    s.into_bytes()
}

pub fn sha_file(path: &Path) -> std::io::Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h)?;
    Ok(hex(&h.finalize()))
}

pub fn compute_config_hash(canonical_config: &str) -> String {
    sha_hex(canonical_config.as_bytes())
}

/// Pins the EXACT ordered window set. Paired comparison is valid only across
/// records with the same split_hash. `selection_tag` must capture HOW the
/// n_windows were chosen ("first", "stride:37", "seed:1000"). If window
/// selection is not deterministic, fix that first — it is the real bug.
pub fn compute_split_hash(
    heldout: &Path,
    seq_len: usize,
    n_windows: usize,
    selection_tag: &str,
) -> std::io::Result<String> {
    let payload = format!(
        "{}\x1f{}\x1f{}\x1f{}",
        sha_file(heldout)?,
        seq_len,
        n_windows,
        selection_tag
    );
    Ok(sha_hex(payload.as_bytes()))
}

fn last_record_hash(path: &Path) -> String {
    let Ok(f) = File::open(path) else {
        return GENESIS.into();
    };
    let mut last = None;
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        if !line.trim().is_empty() {
            last = Some(line);
        }
    }
    match last {
        Some(l) => serde_json::from_str::<EvalRecord>(&l)
            .map(|r| r.record_hash)
            .unwrap_or_else(|_| GENESIS.into()),
        None => GENESIS.into(),
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Append one eval record. Computes mean_bpt, links prev_hash, seals record_hash.
#[allow(clippy::too_many_arguments)]
pub fn append_eval_record(
    path: &Path,
    run_id: &str,
    bet: char,
    config_hash: &str,
    seed: u64,
    checkpoint_path: &str,
    split_hash: &str,
    seq_len: usize,
    per_window_bpt: Vec<f64>,
    notes: &str,
    git_sha: &str,
) -> std::io::Result<EvalRecord> {
    assert!(matches!(bet, 'A' | 'B' | 'C'), "bet must be A/B/C");
    assert!(!per_window_bpt.is_empty(), "per_window_bpt is empty");
    let mut rec = EvalRecord {
        run_id: run_id.into(),
        bet,
        config_hash: config_hash.into(),
        seed,
        checkpoint_path: checkpoint_path.into(),
        split_hash: split_hash.into(),
        n_windows: per_window_bpt.len(),
        seq_len,
        mean_bpt: mean(&per_window_bpt),
        per_window_bpt,
        git_sha: git_sha.into(),
        timestamp: rfc3339_now(),
        notes: notes.into(),
        prev_hash: last_record_hash(path),
        record_hash: String::new(),
    };
    rec.record_hash = sha_hex(&canonical_bytes(&rec));
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{}", serde_json::to_string(&rec).unwrap())?;
    Ok(rec)
}

pub fn load_records(path: &Path) -> std::io::Result<Vec<EvalRecord>> {
    let f = File::open(path)?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        if !line.trim().is_empty() {
            out.push(serde_json::from_str(&line).expect("corrupt ledger line"));
        }
    }
    Ok(out)
}

/// Recompute every record_hash and check prev_hash linkage.
/// Ok(None) == chain intact; Ok(Some(i)) == first tampered index.
pub fn verify_chain(path: &Path) -> std::io::Result<Option<usize>> {
    let mut prev = GENESIS.to_string();
    for (i, rec) in load_records(path)?.iter().enumerate() {
        if sha_hex(&canonical_bytes(rec)) != rec.record_hash {
            return Ok(Some(i)); // body altered
        }
        if rec.prev_hash != prev {
            return Ok(Some(i)); // linkage broken (insert/delete)
        }
        prev = rec.record_hash.clone();
    }
    Ok(None)
}

/// Leaves you would fold into a Verkle IF/WHEN a remote verifier needs compact
/// membership proofs (marketplace, or §5 fork-provenance). Not used by the gate
/// table — a hook so the ledger is Verkle-ready without paying for it now.
pub fn record_hashes(path: &Path) -> std::io::Result<Vec<String>> {
    Ok(load_records(path)?
        .into_iter()
        .map(|r| r.record_hash)
        .collect())
}

fn rfc3339_now() -> String {
    // Avoid a chrono dependency: seconds since epoch is enough to order records;
    // swap for chrono::Utc::now().to_rfc3339() if the crate already links it.
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    format!("epoch:{}", d.as_secs())
}

// ── §1.2 paired-SE verdict table ────────────────────────────────────────────
// Why paired: per-window difficulty is the dominant noise term. Two means of
// ~64 windows cannot resolve a 0.05 bpt gap; per-window DIFFERENCES cancel that
// variance and can. compare() REFUSES on split_hash mismatch — an unpaired
// 9.69-vs-9.62 comparison across different held-out sets is meaningless.

const T95: f64 = 2.0; // t_{.975,~63} ≈ 2.00; normal approx, no extra dep

pub struct Paired {
    pub n: usize,
    pub mean_diff: f64, // candidate - baseline; < 0 means candidate better
    pub se: f64,
    pub ci95: f64,
    pub single_seed: bool,
}

fn sample_stdev(xs: &[f64]) -> f64 {
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0);
    var.sqrt()
}

/// Paired comparison. Errors (Err) rather than returning a meaningless number
/// when the two records are not comparable.
pub fn compare(base: &EvalRecord, cand: &EvalRecord) -> Result<Paired, String> {
    if base.split_hash != cand.split_hash {
        return Err(format!(
            "REFUSING paired compare: split_hash mismatch\n  {}: {}...\n  {}: {}...\n\
             Evaluated on different held-out sets; not comparable. Re-run on the same split.",
            base.run_id,
            &base.split_hash[..12.min(base.split_hash.len())],
            cand.run_id,
            &cand.split_hash[..12.min(cand.split_hash.len())],
        ));
    }
    if base.per_window_bpt.len() != cand.per_window_bpt.len() {
        return Err(
            "REFUSING: per_window length mismatch despite equal split_hash \
                    — window ordering unstable; fix eval."
                .into(),
        );
    }
    let diffs: Vec<f64> = base
        .per_window_bpt
        .iter()
        .zip(&cand.per_window_bpt)
        .map(|(b, c)| c - b)
        .collect();
    let n = diffs.len();
    let se = if n > 1 {
        sample_stdev(&diffs) / (n as f64).sqrt()
    } else {
        f64::NAN
    };
    Ok(Paired {
        n,
        mean_diff: mean(&diffs),
        se,
        ci95: T95 * se,
        single_seed: base.seed == cand.seed,
    })
}

/// Render the §1.2 table (Markdown) from the ledger. `gate` is the pre-reg gate
/// (e.g. 0.05); a SE larger than gate/2 is flagged as "gate finer than
/// measurement resolution", which is the honest verdict, not a fixable bug.
pub fn render_bet_b_table(
    path: &Path,
    baseline: &str,
    candidates: &[&str],
    gate: f64,
) -> std::io::Result<String> {
    let recs = load_records(path)?;
    let find = |id: &str| recs.iter().rfind(|r| r.run_id == id);
    let base = find(baseline).unwrap_or_else(|| panic!("no run_id {baseline}"));

    let mut out = String::new();
    out.push_str(&format!(
        "\n## §1.2 — Bet B held-out verdict (baseline {}, n={} windows, split {}...)\n\n",
        baseline,
        base.n_windows,
        &base.split_hash[..12.min(base.split_hash.len())]
    ));
    out.push_str("| candidate | mean bpt | Δ vs base | paired 95% CI | verdict |\n");
    out.push_str("|---|---|---|---|---|\n");
    out.push_str(&format!(
        "| {} (base) | {:.3} | — | — | — |\n",
        baseline, base.mean_bpt
    ));

    let mut caveats = Vec::new();
    for &cid in candidates {
        let cand = find(cid).unwrap_or_else(|| panic!("no run_id {cid}"));
        match compare(base, cand) {
            Err(e) => out.push_str(&format!(
                "| {} | — | — | — | {} |\n",
                cid,
                e.replace('\n', " ")
            )),
            Ok(p) => {
                let verdict = if p.mean_diff.abs() <= p.ci95 {
                    format!(
                        "indistinguishable (Δ {:+.3} within ±{:.3})",
                        p.mean_diff, p.ci95
                    )
                } else if p.mean_diff < 0.0 {
                    format!("candidate BETTER by {:.3} ± {:.3}", -p.mean_diff, p.ci95)
                } else {
                    format!("candidate WORSE by {:.3} ± {:.3}", p.mean_diff, p.ci95)
                };
                out.push_str(&format!(
                    "| {} | {:.3} | {:+.3} | ±{:.3} | {} |\n",
                    cid, cand.mean_bpt, p.mean_diff, p.ci95, verdict
                ));
                if p.se > gate / 2.0 {
                    caveats.push(format!(
                        "[{} vs {}] SE={:.3} → 2·SE={:.3} exceeds gate {}; the \
                         pre-registered gate is finer than measurement resolution",
                        cid,
                        baseline,
                        p.se,
                        2.0 * p.se,
                        gate
                    ));
                }
                if p.single_seed {
                    caveats.push(format!(
                        "[{} vs {}] single-seed: seed variance not bounded — a \
                         sub-2·SE gap cannot separate geometry-effect from seed-effect",
                        cid, baseline
                    ));
                }
            }
        }
    }
    if !caveats.is_empty() {
        out.push_str("\n**Measurement caveats (pre-registration honesty):**\n");
        for c in caveats {
            out.push_str(&format!("- {}\n", c));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "growformer-ledger-{name}-{}.jsonl",
            std::process::id()
        ))
    }

    fn sample_bpt(seed: u64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 9.5 + (i as f64 * 0.01) + (seed as f64 * 1e-6))
            .collect()
    }

    #[test]
    fn chain_append_and_verify() {
        let path = tmp_path("chain");
        let _ = std::fs::remove_file(&path);
        let split = "testsplit";
        append_eval_record(
            &path,
            "a",
            'B',
            "cfg",
            1,
            "ckpt-a.json",
            split,
            128,
            sample_bpt(1, 4),
            "test",
            "deadbeef",
        )
        .unwrap();
        append_eval_record(
            &path,
            "b",
            'B',
            "cfg",
            1,
            "ckpt-b.json",
            split,
            128,
            sample_bpt(2, 4),
            "test",
            "deadbeef",
        )
        .unwrap();
        assert!(verify_chain(&path).unwrap().is_none());
        let recs = load_records(&path).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].prev_hash, GENESIS);
        assert_eq!(recs[1].prev_hash, recs[0].record_hash);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn compare_refuses_split_mismatch() {
        let base = EvalRecord {
            run_id: "base".into(),
            bet: 'B',
            config_hash: "c".into(),
            seed: 1,
            checkpoint_path: "a.json".into(),
            split_hash: "split-a".into(),
            n_windows: 2,
            seq_len: 128,
            per_window_bpt: vec![9.0, 9.1],
            mean_bpt: 9.05,
            git_sha: "x".into(),
            timestamp: "t".into(),
            notes: String::new(),
            prev_hash: GENESIS.into(),
            record_hash: String::new(),
        };
        let mut cand = base.clone();
        cand.run_id = "cand".into();
        cand.split_hash = "split-b".into();
        assert!(compare(&base, &cand).is_err());
    }

    #[test]
    fn tamper_detected() {
        let path = tmp_path("tamper");
        let _ = std::fs::remove_file(&path);
        append_eval_record(
            &path,
            "a",
            'B',
            "cfg",
            1,
            "ckpt.json",
            "split",
            128,
            sample_bpt(0, 3),
            "test",
            "sha",
        )
        .unwrap();
        let mut rec: EvalRecord = load_records(&path).unwrap().remove(0);
        rec.mean_bpt += 0.001;
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{}", serde_json::to_string(&rec).unwrap()).unwrap();
        assert_eq!(verify_chain(&path).unwrap(), Some(0));
        let _ = std::fs::remove_file(&path);
    }
}
