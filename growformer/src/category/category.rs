// ── category.rs ───────────────────────────────────────────────────────────────
// Categorical DAG: macro-level topology of the network.
// Objects = tensor shapes, Morphisms = typed layer connections.

use std::collections::HashMap;

// ── NodeId ────────────────────────────────────────────────────────────────────

#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub struct NodeId(pub usize);

impl NodeId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}

// ── MorphismKind ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum MorphismKind {
    /// Standard sequential composition: f . g
    Sequential,
    /// Bifunctor split: A × B — left/right child in Pythagoras tree
    ProductSplit,
    /// Skip/residual connection — impossible in a plain tree, fine in a DAG
    Residual,
    /// Swap one sub-network for another preserving interface contract
    NaturalTransform,
}

// ── Edge ──────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub kind: MorphismKind,
}

// ── CategoricalDAG ────────────────────────────────────────────────────────────

pub struct CategoricalDAG<N> {
    pub nodes: HashMap<NodeId, N>,
    pub edges: Vec<Edge>,
    next_id: usize,
}

impl<N> CategoricalDAG<N> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            next_id: 0,
        }
    }

    /// Insert a node, returning its auto-assigned NodeId.
    pub fn add_node(&mut self, node: N) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;
        self.nodes.insert(id.clone(), node);
        id
    }

    /// Connect two existing nodes with a typed morphism.
    pub fn add_edge(&mut self, source: NodeId, target: NodeId, kind: MorphismKind) {
        self.edges.push(Edge {
            source,
            target,
            kind,
        });
    }

    /// Successors of a node in topological order (direct children only).
    pub fn successors(&self, id: &NodeId) -> Vec<(&NodeId, &MorphismKind)> {
        self.edges
            .iter()
            .filter(|e| &e.source == id)
            .map(|e| (&e.target, &e.kind))
            .collect()
    }

    /// Predecessors of a node (direct parents only).
    pub fn predecessors(&self, id: &NodeId) -> Vec<(&NodeId, &MorphismKind)> {
        self.edges
            .iter()
            .filter(|e| &e.target == id)
            .map(|e| (&e.source, &e.kind))
            .collect()
    }

    /// Simple check: does the DAG contain a cycle?
    pub fn is_acyclic(&self) -> bool {
        let mut visited: HashMap<usize, bool> = HashMap::new();
        let mut rec_stack: HashMap<usize, bool> = HashMap::new();

        for id in self.nodes.keys() {
            if self.dfs_cycle_check(id, &mut visited, &mut rec_stack) {
                return false;
            }
        }
        true
    }

    fn dfs_cycle_check(
        &self,
        node: &NodeId,
        visited: &mut HashMap<usize, bool>,
        rec_stack: &mut HashMap<usize, bool>,
    ) -> bool {
        let id = node.0;
        if *rec_stack.get(&id).unwrap_or(&false) {
            return true;
        }
        if *visited.get(&id).unwrap_or(&false) {
            return false;
        }
        visited.insert(id, true);
        rec_stack.insert(id, true);
        for (next, _) in self.successors(node) {
            if self.dfs_cycle_check(next, visited, rec_stack) {
                return true;
            }
        }
        rec_stack.insert(id, false);
        false
    }
}

impl<N> Default for CategoricalDAG<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Layer trait ───────────────────────────────────────────────────────────────

/// A morphism from A to B: structure-preserving map between objects.
pub trait Layer<A, B> {
    fn forward(&self, input: A) -> B;
}

// ── Functor composition ───────────────────────────────────────────────────────

/// Compose two layers: L1: A→B, L2: B→C  =>  Composed: A→C
/// B is carried as a PhantomData so the impl can constrain it properly.
pub struct Composed<L1, L2, B> {
    pub first: L1,
    pub second: L2,
    _mid: std::marker::PhantomData<B>,
}

impl<L1, L2, B> Composed<L1, L2, B> {
    pub fn new(first: L1, second: L2) -> Self {
        Self {
            first,
            second,
            _mid: std::marker::PhantomData,
        }
    }
}

impl<A, B, C, L1, L2> Layer<A, C> for Composed<L1, L2, B>
where
    L1: Layer<A, B>,
    L2: Layer<B, C>,
{
    fn forward(&self, input: A) -> C {
        self.second.forward(self.first.forward(input))
    }
}

// ── NaturalTransform trait ────────────────────────────────────────────────────

/// A natural transformation maps between two functors (F → G) over A.
/// Used to swap sub-network implementations without changing DAG topology.
pub trait NaturalTransform<F, G> {
    fn transform(fa: F) -> G;
}

// ── Network wrapper ───────────────────────────────────────────────────────────

/// A fully boxed network as a category: convenient end-to-end pipeline.
pub struct Network<Input, Output> {
    pub forward: Box<dyn Fn(Input) -> Output>,
}

impl<I: 'static, O: 'static> Network<I, O> {
    pub fn new<F: Fn(I) -> O + 'static>(f: F) -> Self {
        Self {
            forward: Box::new(f),
        }
    }

    pub fn run(&self, input: I) -> O {
        (self.forward)(input)
    }

    /// Categorical composition: self . next  =>  Network<I, P>
    pub fn then<P: 'static>(self, next: Network<O, P>) -> Network<I, P>
    where
        I: 'static,
        O: 'static,
    {
        Network {
            forward: Box::new(move |x| (next.forward)((self.forward)(x))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dag_acyclic_linear_chain() {
        let mut dag: CategoricalDAG<u32> = CategoricalDAG::new();
        let a = dag.add_node(1);
        let b = dag.add_node(2);
        let c = dag.add_node(3);
        dag.add_edge(a.clone(), b.clone(), MorphismKind::Sequential);
        dag.add_edge(b, c, MorphismKind::Sequential);
        assert!(dag.is_acyclic());
    }

    #[test]
    fn dag_detects_cycle() {
        let mut dag: CategoricalDAG<u32> = CategoricalDAG::new();
        let a = dag.add_node(1);
        let b = dag.add_node(2);
        dag.add_edge(a.clone(), b.clone(), MorphismKind::Sequential);
        dag.add_edge(b, a, MorphismKind::Sequential);
        assert!(!dag.is_acyclic());
    }

    #[test]
    fn network_compose_runs() {
        let n = Network::new(|x: i32| x + 1).then(Network::new(|x: i32| x * 2));
        assert_eq!(n.run(3), 8);
    }
}
