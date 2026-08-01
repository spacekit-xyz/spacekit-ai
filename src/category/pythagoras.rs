// ── pythagoras.rs ─────────────────────────────────────────────────────────────
// Per-node internal composition storage using a Pythagoras tree.
// Root = full morphism, children = bifunctor sub-morphisms (a² + b² ≈ c²).

use serde::{Deserialize, Serialize};

// ── Dimensional budget check ───────────────────────────────────────────────────

/// Verifies the soft Pythagorean constraint: a² + b² ≈ c² within tolerance.
pub fn pythagorean_budget_ok(a: usize, b: usize, c: usize, tolerance: f64) -> bool {
    let (af, bf, cf) = (a as f64, b as f64, c as f64);
    (af * af + bf * bf - cf * cf).abs() < cf * cf * tolerance
}

/// Find the nearest Pythagorean split of dimension `c` into (a, b).
/// Returns the best integer pair satisfying a² + b² ≈ c².
pub fn nearest_pythagorean_split(c: usize) -> (usize, usize) {
    let c2 = (c * c) as f64;
    let mut best = (c / 2, c / 2);
    let mut best_err = f64::MAX;

    for a in 1..c {
        let b2 = c2 - (a * a) as f64;
        if b2 <= 0.0 {
            break;
        }
        let b = b2.sqrt().round() as usize;
        if b == 0 || b >= c {
            continue;
        }
        let err = ((a * a + b * b) as f64 - c2).abs();
        if err < best_err {
            best_err = err;
            best = (a, b);
        }
    }
    best
}

// ── PythagorasNode ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythagorasNode<W> {
    /// Parameters / weights at this level of composition.
    pub weights: W,
    /// The representational dimension at this node (c in a²+b²=c²).
    pub dimension: usize,
    /// Left sub-morphism (e.g. Q/K projection, sentiment branch).
    pub left: Option<Box<PythagorasNode<W>>>,
    /// Right sub-morphism (e.g. V projection, entity branch).
    pub right: Option<Box<PythagorasNode<W>>>,
}

impl<W: Clone> PythagorasNode<W> {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a leaf node (base morphism, no children).
    pub fn leaf(weights: W, dimension: usize) -> Self {
        Self {
            weights,
            dimension,
            left: None,
            right: None,
        }
    }

    /// Create an internal split node, enforcing the Pythagorean budget.
    /// `tolerance` is the fractional allowance, e.g. 0.1 = 10%.
    pub fn split(
        weights: W,
        dimension: usize,
        left: PythagorasNode<W>,
        right: PythagorasNode<W>,
        tolerance: f64,
    ) -> Result<Self, String> {
        if !pythagorean_budget_ok(left.dimension, right.dimension, dimension, tolerance) {
            return Err(format!(
                "Pythagorean budget violated: {}² + {}² ≠ {}² (tolerance {:.0}%)",
                left.dimension,
                right.dimension,
                dimension,
                tolerance * 100.0
            ));
        }
        Ok(Self {
            weights,
            dimension,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        })
    }

    /// Create a split using the nearest valid Pythagorean pair automatically.
    pub fn auto_split(weights: W, dimension: usize, left_weights: W, right_weights: W) -> Self {
        let (a, b) = nearest_pythagorean_split(dimension);
        Self {
            weights,
            dimension,
            left: Some(Box::new(PythagorasNode::leaf(left_weights, a))),
            right: Some(Box::new(PythagorasNode::leaf(right_weights, b))),
        }
    }

    // ── Traversal ─────────────────────────────────────────────────────────────

    /// Depth-first composition: applies kernel at each node, threading state.
    /// This is the categorical sequential composition of the internal morphisms.
    pub fn compose<F, A>(&self, input: A, f: &F) -> A
    where
        F: Fn(&W, A) -> A,
        A: Clone,
    {
        let out = f(&self.weights, input);
        match (&self.left, &self.right) {
            (Some(l), Some(r)) => {
                let left_out = l.compose(out.clone(), f);
                r.compose(left_out, f)
            }
            (Some(l), None) => l.compose(out, f),
            (None, Some(r)) => r.compose(out, f),
            (None, None) => out,
        }
    }

    /// Bifunctor compose: apply left and right branches in parallel, then merge.
    /// Used for the sentiment/entity disentangled split.
    pub fn compose_bifunctor<F, A>(&self, input: A, f: &F) -> (A, A)
    where
        F: Fn(&W, A) -> A,
        A: Clone,
    {
        let out = f(&self.weights, input);
        let left_out = self
            .left
            .as_ref()
            .map(|l| l.compose(out.clone(), f))
            .unwrap_or_else(|| out.clone());
        let right_out = self
            .right
            .as_ref()
            .map(|r| r.compose(out, f))
            .unwrap_or_else(|| left_out.clone());
        (left_out, right_out)
    }

    // ── Introspection ─────────────────────────────────────────────────────────

    pub fn depth(&self) -> usize {
        match (&self.left, &self.right) {
            (None, None) => 1,
            (Some(l), Some(r)) => 1 + l.depth().max(r.depth()),
            (Some(l), None) => 1 + l.depth(),
            (None, Some(r)) => 1 + r.depth(),
        }
    }

    pub fn leaf_count(&self) -> usize {
        match (&self.left, &self.right) {
            (None, None) => 1,
            (Some(l), Some(r)) => l.leaf_count() + r.leaf_count(),
            (Some(l), None) => l.leaf_count(),
            (None, Some(r)) => r.leaf_count(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// Total parameter dimensions summed across all nodes.
    pub fn total_dimension(&self) -> usize {
        let child_sum = self.left.as_ref().map_or(0, |l| l.total_dimension())
            + self.right.as_ref().map_or(0, |r| r.total_dimension());
        self.dimension + child_sum
    }

    // ── Mutation (Growformer grow/prune) ──────────────────────────────────────

    /// Grow: expand a leaf into an internal node.
    /// Core Growformer operation — local to this node, no DAG rewiring needed.
    pub fn grow(self, left_weights: W, right_weights: W, tolerance: f64) -> Result<Self, String> {
        if !self.is_leaf() {
            return Err("Cannot grow a non-leaf node".to_string());
        }
        let (a, b) = nearest_pythagorean_split(self.dimension);
        PythagorasNode::split(
            self.weights,
            self.dimension,
            PythagorasNode::leaf(left_weights, a),
            PythagorasNode::leaf(right_weights, b),
            tolerance,
        )
    }

    /// Prune: collapse an internal node back to a leaf (distillation / compression).
    pub fn prune(self) -> Self {
        PythagorasNode::leaf(self.weights, self.dimension)
    }

    /// Collect all leaf weights in depth-first order.
    pub fn collect_leaves(&self) -> Vec<&W> {
        match (&self.left, &self.right) {
            (None, None) => vec![&self.weights],
            (Some(l), Some(r)) => {
                let mut leaves = l.collect_leaves();
                leaves.extend(r.collect_leaves());
                leaves
            }
            (Some(l), None) => l.collect_leaves(),
            (None, Some(r)) => r.collect_leaves(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pythagorean_budget_typical_split() {
        assert!(pythagorean_budget_ok(3, 4, 5, 0.15));
    }

    #[test]
    fn grow_leaf_then_depth_two() {
        let leaf = PythagorasNode::leaf(vec![0.5f32; 5], 5);
        let grown = leaf
            .grow(vec![0.1f32; 3], vec![0.1f32; 4], 0.15)
            .expect("grow");
        assert_eq!(grown.depth(), 2);
        assert!(!grown.is_leaf());
    }
}
