//! Parameter-budget helpers for row 2 (param-matched vanilla vs Clifford LM).

/// Learnable scalars in one `CliffordLinear(in → out)`.
pub fn clifford_linear_scalars(in_dim: usize, out_dim: usize) -> usize {
    16 * (in_dim * out_dim + out_dim)
}

pub fn clifford_block_scalars(d_model: usize, d_ff: usize) -> usize {
    let attn = 4 * clifford_linear_scalars(d_model, d_model);
    let ln = 2 * 2 * 16 * d_model;
    let ffn = 16 * (2 * d_model * d_ff + d_ff + d_model);
    attn + ln + ffn
}

/// Full Clifford LM (matches construction in `train_v2::ModelStateV2::new`).
pub fn clifford_lm_scalars(
    vocab: usize,
    d_model: usize,
    d_ff: usize,
    n_blocks: usize,
    tie_embeddings: bool,
) -> usize {
    let blocks = n_blocks * clifford_block_scalars(d_model, d_ff);
    let embed = vocab * d_model * 16;
    let final_norm = 2 * 16 * d_model;
    let head = if tie_embeddings {
        vocab
    } else {
        vocab * (16 * d_model + 1)
    };
    blocks + embed + final_norm + head
}

pub fn vanilla_block_scalars(d_model: usize, d_ff: usize) -> usize {
    let attn = 4 * (d_model * d_model + d_model);
    let ln = 4 * d_model;
    let ffn = 2 * d_model * d_ff + d_ff + d_model;
    attn + ln + ffn
}

pub fn vanilla_lm_scalars(
    vocab: usize,
    d_model: usize,
    d_ff: usize,
    n_blocks: usize,
    tie_embeddings: bool,
) -> usize {
    let blocks = n_blocks * vanilla_block_scalars(d_model, d_ff);
    let embed = vocab * d_model;
    let final_norm = 2 * d_model;
    let head = if tie_embeddings {
        vocab
    } else {
        vocab * (d_model + 1)
    };
    blocks + embed + final_norm + head
}

/// Smallest `d_model` (≥ `n_heads`, divisible) whose vanilla total is within `tol` of Clifford ref.
pub fn matched_vanilla_d_model(
    vocab: usize,
    clifford_d_model: usize,
    d_ff: usize,
    n_blocks: usize,
    n_heads: usize,
    tie_embeddings: bool,
    tol: usize,
) -> usize {
    let target = clifford_lm_scalars(vocab, clifford_d_model, d_ff, n_blocks, tie_embeddings);
    let lo = n_heads.max(8);
    let hi = 512;
    let mut best_d = lo;
    let mut best_diff = usize::MAX;
    for d in lo..=hi {
        if d % n_heads != 0 {
            continue;
        }
        let diff = target.abs_diff(vanilla_lm_scalars(vocab, d, d_ff, n_blocks, tie_embeddings));
        if diff < best_diff {
            best_diff = diff;
            best_d = d;
        }
        if diff <= tol {
            return d;
        }
    }
    best_d
}

pub fn log_param_match(
    vocab: usize,
    clifford_d: usize,
    vanilla_d: usize,
    d_ff: usize,
    n_blocks: usize,
    tie: bool,
) {
    let c = clifford_lm_scalars(vocab, clifford_d, d_ff, n_blocks, tie);
    let v = vanilla_lm_scalars(vocab, vanilla_d, d_ff, n_blocks, tie);
    eprintln!(
        "[row2] param budget: clifford d={clifford_d} → {c} scalars; \
         vanilla d={vanilla_d} → {v} scalars (Δ={})",
        c as i64 - v as i64
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clifford_default_near_737k() {
        let n = clifford_lm_scalars(2048, 16, 64, 4, true);
        assert!((736_000..=738_000).contains(&n), "got {n}");
    }

    #[test]
    fn matched_vanilla_near_clifford() {
        let d = matched_vanilla_d_model(2048, 16, 64, 4, 4, true, 500);
        let c = clifford_lm_scalars(2048, 16, 64, 4, true);
        let v = vanilla_lm_scalars(2048, d, 64, 4, true);
        assert!(c.abs_diff(v) <= 600, "d={d} c={c} v={v}");
        assert_eq!(d % 4, 0);
    }
}
