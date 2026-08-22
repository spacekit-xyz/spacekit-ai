//! CL-1: route between two frozen LM specialists with an adjustable cone router (Bet A).

use std::path::Path;

use growformer_ledger::results_ledger as ledger;

use crate::lm_cone_router::{lm_cone_features, LmAdjustableConeRouter, LmConeConfig, LmConeSample};
use crate::tinystories::PackedDataset;
use crate::v2::checkpoint::load_lm_state;
use crate::v2::data::N_SPECIAL;
use crate::v2::sample::softmax as logits_softmax;
use crate::v2::tape::model_forward_logits;
use crate::v2::train_v2::{ModelStateV2, TrainConfigV2};
use crate::v2::vanilla_checkpoint::load_vanilla_state;
use crate::v2::vanilla_train::VanillaModelState;
use crate::vanilla_llm::vanilla_forward_logits;

pub enum FrozenSpecialist {
    Clifford(ModelStateV2),
    Vanilla(VanillaModelState),
}

pub struct WindowStat {
    pub bpt: f64,
    pub mean_top1: f32,
    pub n_pred: usize,
}

/// Max |Δbpt| between specialists to treat them as routing peers (§2.2 preflight).
pub const CL1_SPECIALIST_PARITY_BPT: f64 = 0.20;

pub struct Cl1Result {
    pub mean_bpt_a: f64,
    pub mean_bpt_b: f64,
    pub mean_bpt_oracle: f64,
    pub mean_bpt_routed: f64,
    pub mean_bpt_best_single: f64,
    pub per_window_routed_bpt: Vec<f64>,
    pub route_a_frac: f64,
    pub wins_a: usize,
    pub wins_b: usize,
    pub specialist_gap_bpt: f64,
    pub oracle_gap_bpt: f64,
    /// |mean A − mean B| ≤ parity threshold — specialists are comparable LM peers.
    pub peer_specialists: bool,
    /// At least one specialist wins ≥1 held-out window (complementarity exists in principle).
    pub complementarity_possible: bool,
    /// Large standalone gap — setup cannot test routing (e.g. row2 vs row3b).
    pub imbalanced_specialists: bool,
    /// Oracle ≈ best single because one model wins every window — no router can help.
    pub no_complementarity: bool,
    pub degenerate: bool,
    pub cal_n: usize,
    pub eval_n: usize,
}

fn peek_cfg(path: &Path) -> Result<TrainConfigV2, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    #[derive(serde::Deserialize)]
    struct Peek {
        cfg: TrainConfigV2,
    }
    let p: Peek = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    Ok(p.cfg)
}

pub fn load_frozen_specialist(path: &Path) -> Result<FrozenSpecialist, String> {
    let cfg = peek_cfg(path)?;
    if cfg.vanilla {
        Ok(FrozenSpecialist::Vanilla(load_vanilla_state(path)?))
    } else {
        Ok(FrozenSpecialist::Clifford(load_lm_state(path)?))
    }
}

fn eval_window(spec: &FrozenSpecialist, window: &[usize]) -> WindowStat {
    let mut bits = 0.0f64;
    let mut top1_sum = 0.0f32;
    let mut n_pred = 0usize;
    if window.len() < 2 {
        return WindowStat {
            bpt: 0.0,
            mean_top1: 0.0,
            n_pred: 0,
        };
    }
    let logits_rows = match spec {
        FrozenSpecialist::Clifford(st) => {
            model_forward_logits(&st.alg, &st.model, window, true, st.cfg.dot_attention)
        }
        FrozenSpecialist::Vanilla(st) => vanilla_forward_logits(&st.model, window, true),
    };
    for p in 0..window.len().saturating_sub(1) {
        let target = window[p + 1];
        if target < N_SPECIAL {
            continue;
        }
        let probs = logits_softmax(&logits_rows[p]);
        let pr = (probs[target] as f64).max(1e-12);
        bits += -pr.log2();
        top1_sum += probs.iter().copied().fold(0.0f32, f32::max);
        n_pred += 1;
    }
    if n_pred == 0 {
        WindowStat {
            bpt: 0.0,
            mean_top1: 0.0,
            n_pred: 0,
        }
    } else {
        WindowStat {
            bpt: bits / n_pred as f64,
            mean_top1: top1_sum / n_pred as f32,
            n_pred,
        }
    }
}

/// Run CL-1: train cone on first `cal_windows`, evaluate routed composite on all `eval_windows`.
pub fn run_cl1(
    spec_a: &FrozenSpecialist,
    spec_b: &FrozenSpecialist,
    tokens: &[u32],
    seq_len: usize,
    cal_windows: usize,
    eval_windows: usize,
    cone_seed: u64,
) -> Cl1Result {
    let mut stats_a = Vec::with_capacity(eval_windows);
    let mut stats_b = Vec::with_capacity(eval_windows);
    for w in 0..eval_windows {
        let start = w * seq_len;
        if start + 2 > tokens.len() {
            break;
        }
        let end = (start + seq_len).min(tokens.len());
        let window: Vec<usize> = tokens[start..end].iter().map(|&x| x as usize).collect();
        stats_a.push(eval_window(spec_a, &window));
        stats_b.push(eval_window(spec_b, &window));
    }
    let n = stats_a.len().min(stats_b.len());
    let cal_n = cal_windows.min(n);
    let mut train_samples = Vec::with_capacity(cal_n);
    for i in 0..cal_n {
        let sa = stats_a[i].mean_top1.clamp(0.02, 0.98);
        let sb = stats_b[i].mean_top1.clamp(0.02, 0.98);
        let amb = 0.5 - (sa - sb).abs();
        train_samples.push(LmConeSample {
            features: lm_cone_features(sa, sb),
            route_a: stats_a[i].bpt <= stats_b[i].bpt,
            ambiguity: amb.clamp(0.0, 1.0),
        });
    }
    let router = LmAdjustableConeRouter::train(
        &train_samples,
        LmConeConfig {
            seed: cone_seed,
            ..LmConeConfig::default()
        },
    );

    let mut per_window_routed_bpt = Vec::with_capacity(n);
    let mut route_a_count = 0usize;
    let mut wins_a = 0usize;
    let mut wins_b = 0usize;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_oracle = 0.0f64;
    let mut sum_routed = 0.0f64;
    for i in 0..n {
        sum_a += stats_a[i].bpt;
        sum_b += stats_b[i].bpt;
        sum_oracle += stats_a[i].bpt.min(stats_b[i].bpt);
        if stats_a[i].bpt < stats_b[i].bpt {
            wins_a += 1;
        } else if stats_b[i].bpt < stats_a[i].bpt {
            wins_b += 1;
        }
        let sa = stats_a[i].mean_top1.clamp(0.02, 0.98);
        let sb = stats_b[i].mean_top1.clamp(0.02, 0.98);
        let feats = lm_cone_features(sa, sb);
        let idx = router.route_index(&feats);
        let routed = if idx == 0 {
            route_a_count += 1;
            stats_a[i].bpt
        } else {
            stats_b[i].bpt
        };
        per_window_routed_bpt.push(routed);
        sum_routed += routed;
    }
    let nf = n as f64;
    let mean_a = sum_a / nf;
    let mean_b = sum_b / nf;
    let mean_oracle = sum_oracle / nf;
    let mean_best = mean_a.min(mean_b);
    let route_a_frac = route_a_count as f64 / nf;
    let degenerate = route_a_count == 0 || route_a_count == n;
    let specialist_gap_bpt = (mean_a - mean_b).abs();
    let oracle_gap_bpt = mean_oracle - mean_best;
    let peer_specialists = specialist_gap_bpt <= CL1_SPECIALIST_PARITY_BPT;
    let complementarity_possible = wins_a > 0 && wins_b > 0;
    let imbalanced_specialists = !peer_specialists;
    let no_complementarity = !complementarity_possible;

    Cl1Result {
        mean_bpt_a: mean_a,
        mean_bpt_b: mean_b,
        mean_bpt_oracle: mean_oracle,
        mean_bpt_routed: sum_routed / nf,
        mean_bpt_best_single: mean_best,
        per_window_routed_bpt: per_window_routed_bpt.clone(),
        route_a_frac,
        wins_a,
        wins_b,
        specialist_gap_bpt,
        oracle_gap_bpt,
        peer_specialists,
        complementarity_possible,
        imbalanced_specialists,
        no_complementarity,
        degenerate,
        cal_n,
        eval_n: n,
    }
}

pub fn append_cl1_ledger(
    ledger_path: &Path,
    run_id: &str,
    split_hash: &str,
    seq_len: usize,
    per_window_bpt: &[f64],
    notes: &str,
    git_sha: &str,
) -> Result<(), String> {
    ledger::append_eval_record(
        ledger_path,
        run_id,
        'A',
        "cl1-routed",
        0,
        "cl1-composite",
        split_hash,
        seq_len,
        per_window_bpt.to_vec(),
        notes,
        git_sha,
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn load_heldout_tokens(path: &Path) -> Result<Vec<u32>, String> {
    let ds = PackedDataset::load(path).map_err(|e| e.to_string())?;
    Ok(ds.tokens)
}
