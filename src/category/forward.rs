// ── forward.rs ────────────────────────────────────────────────────────────────
// Shape-aligned Pythagoras forward: Hadamard kernels with per-node weight length
// equal to `node.dimension`, input aligned (pad/truncate) before each apply.
// Bifunctor outputs are projected to a common `branch_dim` for heads / disentanglement.

use crate::category::pythagoras::PythagorasNode;
use crate::category::training::TrainingRecord;

/// Deterministic embedding from text (character folding hash → `dim` floats in [-1, 1]).
pub fn char_hash_embed(text: &str, dim: usize) -> Vec<f32> {
    let hash: u32 = text.chars().fold(0u32, |acc, c| acc.wrapping_add(c as u32));
    (0..dim)
        .map(|i| ((hash.wrapping_add(i as u32)) as f32 / u32::MAX as f32) * 2.0 - 1.0)
        .collect()
}

/// Use `record.embedding` if present and compatible, else hash embedding.
pub fn record_embedding(record: &TrainingRecord, embed_dim: usize) -> Vec<f32> {
    match &record.embedding {
        Some(e) if !e.is_empty() => align_to_dim(e, embed_dim),
        _ => char_hash_embed(&record.input, embed_dim),
    }
}

/// Pad with zeros or truncate to exactly `dim` elements.
pub fn align_to_dim(x: &[f32], dim: usize) -> Vec<f32> {
    if x.len() == dim {
        return x.to_vec();
    }
    if x.len() >= dim {
        return x[..dim].to_vec();
    }
    let mut v = x.to_vec();
    v.resize(dim, 0.0);
    v
}

/// Element-wise product; `w` and `x` must match after alignment.
pub fn hadamard(w: &[f32], x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(w.len(), x.len(), "hadamard: weight / input length mismatch");
    x.iter().zip(w.iter()).map(|(xi, wi)| xi * wi).collect()
}

/// Sequential composition through the tree (single output), same semantics as
/// `PythagorasNode::compose` but with explicit alignment.
pub fn compose_aligned(node: &PythagorasNode<Vec<f32>>, input: &[f32]) -> Vec<f32> {
    let x0 = align_to_dim(input, node.dimension);
    node.compose(x0, &|w: &Vec<f32>, v: Vec<f32>| {
        let v = align_to_dim(&v, w.len());
        hadamard(w, &v)
    })
}

/// Left/right branch vectors for disentanglement and task heads.
/// - If the tree has two children: each subtree gets `out` aligned to child dimension.
/// - If leaf: split parent output into first / second half (feature separation).
pub fn bifunctor_branch_vectors(
    node: &PythagorasNode<Vec<f32>>,
    input: &[f32],
    branch_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let x0 = align_to_dim(input, node.dimension);
    let out = hadamard(&node.weights, &x0);

    match (&node.left, &node.right) {
        (Some(l), Some(r)) => {
            let li = align_to_dim(&out, l.dimension);
            let ri = align_to_dim(&out, r.dimension);
            let s = compose_aligned(l, &li);
            let e = compose_aligned(r, &ri);
            (align_to_dim(&s, branch_dim), align_to_dim(&e, branch_dim))
        }
        (Some(l), None) => {
            let li = align_to_dim(&out, l.dimension);
            let s = compose_aligned(l, &li);
            (align_to_dim(&s, branch_dim), align_to_dim(&out, branch_dim))
        }
        (None, Some(r)) => {
            let ri = align_to_dim(&out, r.dimension);
            let e = compose_aligned(r, &ri);
            (align_to_dim(&out, branch_dim), align_to_dim(&e, branch_dim))
        }
        (None, None) => {
            let d = out.len();
            if d < 2 {
                let z = align_to_dim(&out, branch_dim);
                return (z.clone(), z);
            }
            let mid = d / 2;
            let left_part = &out[..mid];
            let right_part = &out[mid..];
            (
                align_to_dim(left_part, branch_dim),
                align_to_dim(right_part, branch_dim),
            )
        }
    }
}

// ── Parse-tree CE backprop (Hadamard + align) ─────────────────────────────────

/// Forward: `out = align_to_dim(source, branch_dim)` with `source.len() == source_len`.
fn grad_align_output_to_source(grad_branch: &[f32], source_len: usize, branch_dim: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; source_len];
    if source_len >= branch_dim {
        for i in 0..branch_dim.min(grad_branch.len()) {
            g[i] += grad_branch[i];
        }
    } else {
        for i in 0..source_len {
            g[i] += grad_branch[i];
        }
    }
    g
}

/// Forward: `b = align_to_dim(a, len_b)` where `a.len() == len_a`.
pub(crate) fn grad_align_b_to_a(grad_b: &[f32], len_a: usize, len_b: usize) -> Vec<f32> {
    let mut ga = vec![0.0f32; len_a];
    if len_a >= len_b {
        for i in 0..len_b.min(grad_b.len()) {
            ga[i] += grad_b[i];
        }
    } else {
        for i in 0..len_a {
            ga[i] += grad_b[i];
        }
    }
    ga
}

fn grad_align_to_dim(g_x0: &[f32], input_len: usize, dim: usize) -> Vec<f32> {
    grad_align_b_to_a(g_x0, input_len, dim)
}

/// Same tree topology as `node`, all weights zero (for gradient accumulation).
pub(crate) fn zero_weight_clone(node: &PythagorasNode<Vec<f32>>) -> PythagorasNode<Vec<f32>> {
    PythagorasNode {
        weights: vec![0.0f32; node.weights.len()],
        dimension: node.dimension,
        left: node.left.as_ref().map(|b| Box::new(zero_weight_clone(b))),
        right: node.right.as_ref().map(|b| Box::new(zero_weight_clone(b))),
    }
}

pub(crate) fn apply_weight_grad_sgd(node: &mut PythagorasNode<Vec<f32>>, grad: &PythagorasNode<Vec<f32>>, scale: f32) {
    debug_assert_eq!(node.weights.len(), grad.weights.len());
    for (w, g) in node.weights.iter_mut().zip(grad.weights.iter()) {
        *w -= scale * g;
    }
    match (node.left.as_mut(), grad.left.as_ref()) {
        (Some(nl), Some(gl)) => apply_weight_grad_sgd(nl, gl, scale),
        (None, None) => {}
        _ => debug_assert!(false, "tree shape mismatch"),
    }
    match (node.right.as_mut(), grad.right.as_ref()) {
        (Some(nr), Some(gr)) => apply_weight_grad_sgd(nr, gr, scale),
        (None, None) => {}
        _ => debug_assert!(false, "tree shape mismatch"),
    }
}

/// Backward through [`compose_aligned`]: accumulates ∂L/∂weights into `acc`, returns ∂L/∂`input`.
fn compose_aligned_backward_acc(
    node: &PythagorasNode<Vec<f32>>,
    input: &[f32],
    grad_out: &[f32],
    acc: &mut PythagorasNode<Vec<f32>>,
) -> Vec<f32> {
    let dim = node.dimension;
    let x0 = align_to_dim(input, dim);
    let out_root = hadamard(&node.weights, &x0);
    debug_assert_eq!(out_root.len(), grad_out.len());

    match (&node.left, &node.right) {
        (None, None) => {
            let gw: Vec<f32> = grad_out.iter().zip(x0.iter()).map(|(g, x)| g * x).collect();
            let gx0: Vec<f32> = grad_out.iter().zip(node.weights.iter()).map(|(g, w)| g * w).collect();
            for (a, g) in acc.weights.iter_mut().zip(gw.iter()) {
                *a += g;
            }
            grad_align_to_dim(&gx0, input.len(), dim)
        }
        (Some(l), None) => {
            let g_mid = compose_aligned_backward_acc(l, &out_root, grad_out, acc.left.as_mut().unwrap());
            let gw: Vec<f32> = g_mid.iter().zip(x0.iter()).map(|(g, x)| g * x).collect();
            let gx0: Vec<f32> = g_mid.iter().zip(node.weights.iter()).map(|(g, w)| g * w).collect();
            for (a, g) in acc.weights.iter_mut().zip(gw.iter()) {
                *a += g;
            }
            grad_align_to_dim(&gx0, input.len(), dim)
        }
        (None, Some(r)) => {
            let g_mid = compose_aligned_backward_acc(r, &out_root, grad_out, acc.right.as_mut().unwrap());
            let gw: Vec<f32> = g_mid.iter().zip(x0.iter()).map(|(g, x)| g * x).collect();
            let gx0: Vec<f32> = g_mid.iter().zip(node.weights.iter()).map(|(g, w)| g * w).collect();
            for (a, g) in acc.weights.iter_mut().zip(gw.iter()) {
                *a += g;
            }
            grad_align_to_dim(&gx0, input.len(), dim)
        }
        (Some(l), Some(r)) => {
            let left_out = compose_aligned(l, &out_root);
            let g_mid = compose_aligned_backward_acc(r, &left_out, grad_out, acc.right.as_mut().unwrap());
            let g_out1 = compose_aligned_backward_acc(l, &out_root, &g_mid, acc.left.as_mut().unwrap());
            let gw: Vec<f32> = g_out1.iter().zip(x0.iter()).map(|(g, x)| g * x).collect();
            let gx0: Vec<f32> = g_out1.iter().zip(node.weights.iter()).map(|(g, w)| g * w).collect();
            for (a, g) in acc.weights.iter_mut().zip(gw.iter()) {
                *a += g;
            }
            grad_align_to_dim(&gx0, input.len(), dim)
        }
    }
}

/// Accumulate ∂(CE_s + λ CE_e)/∂ parse-tree weights for one sample (caller applies mean SGD).
pub(crate) fn bifunctor_branch_vectors_backward_acc(
    node: &PythagorasNode<Vec<f32>>,
    input: &[f32],
    branch_dim: usize,
    grad_s: &[f32],
    grad_e: &[f32],
    acc: &mut PythagorasNode<Vec<f32>>,
) {
    debug_assert_eq!(grad_s.len(), branch_dim);
    debug_assert_eq!(grad_e.len(), branch_dim);

    let x0 = align_to_dim(input, node.dimension);
    let out = hadamard(&node.weights, &x0);

    let g_out = match (&node.left, &node.right) {
        (Some(l), Some(r)) => {
            let li = align_to_dim(&out, l.dimension);
            let ri = align_to_dim(&out, r.dimension);
            let s = compose_aligned(l, &li);
            let e = compose_aligned(r, &ri);
            let gs = grad_align_output_to_source(grad_s, s.len(), branch_dim);
            let ge = grad_align_output_to_source(grad_e, e.len(), branch_dim);
            let g_li = compose_aligned_backward_acc(l, &li, &gs, acc.left.as_mut().unwrap());
            let g_ri = compose_aligned_backward_acc(r, &ri, &ge, acc.right.as_mut().unwrap());
            let ga = grad_align_b_to_a(&g_li, out.len(), l.dimension);
            let gb = grad_align_b_to_a(&g_ri, out.len(), r.dimension);
            let mut sum = vec![0.0f32; out.len()];
            for i in 0..out.len() {
                sum[i] = ga[i] + gb[i];
            }
            sum
        }
        (Some(l), None) => {
            let li = align_to_dim(&out, l.dimension);
            let s = compose_aligned(l, &li);
            let gs = grad_align_output_to_source(grad_s, s.len(), branch_dim);
            let ge_root = grad_align_b_to_a(grad_e, out.len(), branch_dim);
            let g_li = compose_aligned_backward_acc(l, &li, &gs, acc.left.as_mut().unwrap());
            let ga = grad_align_b_to_a(&g_li, out.len(), l.dimension);
            let mut sum = vec![0.0f32; out.len()];
            for i in 0..out.len() {
                sum[i] = ga[i] + ge_root[i];
            }
            sum
        }
        (None, Some(r)) => {
            let ri = align_to_dim(&out, r.dimension);
            let e = compose_aligned(r, &ri);
            let ge_aligned = grad_align_output_to_source(grad_e, e.len(), branch_dim);
            let gs_root = grad_align_b_to_a(grad_s, out.len(), branch_dim);
            let g_ri = compose_aligned_backward_acc(r, &ri, &ge_aligned, acc.right.as_mut().unwrap());
            let gb = grad_align_b_to_a(&g_ri, out.len(), r.dimension);
            let mut sum = vec![0.0f32; out.len()];
            for i in 0..out.len() {
                sum[i] = gs_root[i] + gb[i];
            }
            sum
        }
        (None, None) => {
            let d = out.len();
            if d < 2 {
                let gz: Vec<f32> = grad_s
                    .iter()
                    .zip(grad_e.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                grad_align_b_to_a(&gz, out.len(), branch_dim)
            } else {
                let mid = d / 2;
                let gs_half = grad_align_output_to_source(grad_s, mid, branch_dim);
                let ge_half = grad_align_output_to_source(grad_e, d - mid, branch_dim);
                let mut g = vec![0.0f32; d];
                for i in 0..mid {
                    g[i] += gs_half[i];
                }
                for i in 0..(d - mid) {
                    g[mid + i] += ge_half[i];
                }
                g
            }
        }
    };

    let gw: Vec<f32> = g_out.iter().zip(x0.iter()).map(|(g, x)| g * x).collect();
    for (a, g) in acc.weights.iter_mut().zip(gw.iter()) {
        *a += g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::pythagoras::PythagorasNode;

    #[test]
    fn align_truncates_and_pads() {
        assert_eq!(align_to_dim(&[1.0, 2.0, 3.0], 2), vec![1.0, 2.0]);
        assert_eq!(align_to_dim(&[1.0], 3), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn leaf_bifunctor_halves() {
        let w = vec![1.0f32; 8];
        let node = PythagorasNode::leaf(w, 8);
        let inp = (0..8).map(|i| i as f32).collect::<Vec<_>>();
        let (s, e) = bifunctor_branch_vectors(&node, &inp, 4);
        assert_eq!(s.len(), 4);
        assert_eq!(e.len(), 4);
    }

    #[test]
    fn split_tree_runs() {
        let root = PythagorasNode::auto_split(
            vec![0.5f32; 5],
            5,
            vec![0.1f32; 3],
            vec![0.1f32; 4],
        );
        let inp = vec![1.0f32; 5];
        let (a, b) = bifunctor_branch_vectors(&root, &inp, 8);
        assert_eq!(a.len(), 8);
        assert_eq!(b.len(), 8);
    }
}
