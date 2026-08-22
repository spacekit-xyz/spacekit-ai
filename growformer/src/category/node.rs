// ── node.rs ───────────────────────────────────────────────────────────────────
// CategoricalNode: the fundamental unit of Growformer.
// External face = DAG node (categorical identity, typed morphisms).
// Internal face = PythagorasTree (composition storage, bifunctor split).

use crate::category::pythagoras::PythagorasNode;
use crate::category::{Layer, NodeId};
use std::marker::PhantomData;

// ── NodeMetadata ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NodeMetadata {
    pub label: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub version: u32,
}

impl NodeMetadata {
    pub fn new(label: impl Into<String>, input_dim: usize, output_dim: usize) -> Self {
        Self {
            label: label.into(),
            input_dim,
            output_dim,
            version: 0,
        }
    }
}

// ── CategoricalNode ───────────────────────────────────────────────────────────

pub struct CategoricalNode<W, A, B> {
    /// Unique identity in the categorical DAG.
    pub id: NodeId,
    pub meta: NodeMetadata,
    /// Internal composition storage — Pythagoras tree.
    pub composition: PythagorasNode<W>,
    /// Phantom types enforce the morphism contract at compile time.
    _phantom: PhantomData<(A, B)>,
}

impl<W: Clone, A: Clone, B> CategoricalNode<W, A, B> {
    pub fn new(id: NodeId, meta: NodeMetadata, composition: PythagorasNode<W>) -> Self {
        Self {
            id,
            meta,
            composition,
            _phantom: PhantomData,
        }
    }

    /// Create a leaf node with a single weight tensor and no internal split.
    pub fn leaf(id: NodeId, label: impl Into<String>, weights: W, dim: usize) -> Self {
        let meta = NodeMetadata::new(label, dim, dim);
        Self::new(id, meta, PythagorasNode::leaf(weights, dim))
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    /// Run the forward pass using the internal Pythagoras composition.
    /// `kernel` is the actual computation applied at each tree node.
    pub fn forward<F>(&self, input: A, kernel: &F) -> A
    where
        F: Fn(&W, A) -> A,
    {
        self.composition.compose(input, kernel)
    }

    /// Bifunctor forward: returns (left_branch, right_branch) outputs.
    /// Used for disentangled sentiment/entity processing.
    pub fn forward_bifunctor<F>(&self, input: A, kernel: &F) -> (A, A)
    where
        F: Fn(&W, A) -> A,
    {
        self.composition.compose_bifunctor(input, kernel)
    }

    // ── Introspection ─────────────────────────────────────────────────────────

    /// Internal Pythagoras tree depth — used by Growformer's depth scheduler.
    pub fn composition_depth(&self) -> usize {
        self.composition.depth()
    }

    pub fn leaf_count(&self) -> usize {
        self.composition.leaf_count()
    }

    pub fn total_dimension(&self) -> usize {
        self.composition.total_dimension()
    }

    pub fn is_leaf_node(&self) -> bool {
        self.composition.is_leaf()
    }

    // ── Mutation (Growformer lifecycle) ───────────────────────────────────────

    /// Grow the internal composition — add one level of bifunctor split.
    /// This is the primary "grow" operation in Growformer.
    pub fn grow(
        &mut self,
        left_weights: W,
        right_weights: W,
        tolerance: f64,
    ) -> Result<(), String> {
        // Take ownership temporarily using a dummy via swap
        let old = std::mem::replace(
            &mut self.composition,
            PythagorasNode::leaf(left_weights.clone(), 1), // temp
        );
        match old.grow(left_weights, right_weights, tolerance) {
            Ok(grown) => {
                self.composition = grown;
                self.meta.version += 1;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Prune the internal composition — collapse to leaf (distillation).
    pub fn prune(&mut self) {
        let dim = self.composition.dimension;
        let w = self.composition.weights.clone();
        let old = std::mem::replace(&mut self.composition, PythagorasNode::leaf(w, dim));
        self.composition = old.prune();
        self.meta.version += 1;
    }

    /// Node summary for logging / debugging.
    pub fn summary(&self) -> String {
        format!(
            "[{}] id={} | in={} out={} | tree_depth={} leaves={} total_dim={} v{}",
            self.meta.label,
            self.id.0,
            self.meta.input_dim,
            self.meta.output_dim,
            self.composition_depth(),
            self.leaf_count(),
            self.total_dimension(),
            self.meta.version,
        )
    }
}

// ── Layer impl for CategoricalNode ───────────────────────────────────────────

/// Default kernel: treats W as Vec<f32> and A as Vec<f32>, applies a dot product.
/// Replace with your actual tensor operation from Growformer.
impl Layer<Vec<f32>, Vec<f32>> for CategoricalNode<Vec<f32>, Vec<f32>, Vec<f32>> {
    fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        crate::category::forward::compose_aligned(&self.composition, &input)
    }
}

// ── Adjunction: Encoder / Decoder pair ────────────────────────────────────────

pub trait Encoder<A, Z> {
    fn encode(&self, input: A) -> Z;
}

pub trait Decoder<Z, A> {
    fn decode(&self, latent: Z) -> A;
}

pub struct Autoencoder<E, D, A, Z> {
    pub encoder: E,
    pub decoder: D,
    _phantom: PhantomData<(A, Z)>,
}

impl<E, D, A: Clone, Z: Clone> Autoencoder<E, D, A, Z>
where
    E: Encoder<A, Z>,
    D: Decoder<Z, A>,
{
    pub fn new(encoder: E, decoder: D) -> Self {
        Self {
            encoder,
            decoder,
            _phantom: PhantomData,
        }
    }

    /// Reconstruct: encode then decode — approximates identity by adjunction law.
    pub fn reconstruct(&self, input: A) -> A {
        self.decoder.decode(self.encoder.encode(input))
    }

    /// Latent representation only.
    pub fn latent(&self, input: A) -> Z {
        self.encoder.encode(input)
    }
}
