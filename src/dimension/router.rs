//! GroupRouter — maps input to group relevance weights (heuristic: cosine on embedding).

use crate::types::GroupId;

use super::embedding::{retrieve_relevant_groups, GroupEmbedding};

/// Heuristic router: relevance = cosine similarity between input-as-query and group embedding.
/// Input is used as a short "query" vector; we compare to embedding vectors (same-dim via projection or use output layer).
/// For simplicity: use first N embedding dimensions as query if input is 2D we need to project.
/// Spec: "attend(input, &embedding_library)". Embeddings are full hidden-layer mean activations.
/// Input is 2D (e.g. spiral/circles). We don't have input->embedding. So we use a simple heuristic:
/// run input through each group's env to get that group's output (or hidden state), then weight by that.
/// Actually the observer has router.attend(input, &embedding_library). So the router needs to produce
/// attention over groups from input. Without running through envs, we could use embedding vectors as
/// "signatures" and compare a running average of recent inputs to each group's embedding? Or we run
/// input through each group and use output magnitude or similarity to embedding. Simplest: run input
/// through each group, get output; attention = softmax over outputs or 1.0 for max group.
/// For now: return uniform attention over all groups (caller will query all and compose).
/// Or: retrieve_relevant_groups(query_vector, embeddings, k). But query_vector is from where?
/// Spec says "Route: which groups are relevant? let attention = self.router.attend(input, &self.embedding_library)".
/// So attend(input: &[f32], embeddings: &[GroupEmbedding]) -> Vec<(GroupId, f32)>.
/// Input is 2D. Embedding vectors are long (hidden size). We can't directly compare. So we need to
/// either (1) run input through each group's env and get output, then use output as relevance, or
/// (2) have a learned mapping input -> query vector. For heuristic we do (1): in the observer's infer(),
/// we query main with input and get outputs; then we use output magnitude or a fixed rule as attention.
/// So the "router" in the spec might just be "query all groups, then compose by output confidence".
/// Let me provide a stub that returns all groups with equal weight; the manager/observer will do
/// "query main with input" and then compose. We can refine later.
pub fn attend_heuristic(
    _input: &[f32],
    embeddings: &[GroupEmbedding],
    top_k: usize,
) -> Vec<(GroupId, f32)> {
    if embeddings.is_empty() {
        return vec![];
    }
    // No input->embedding mapping; return equal weight for all (or first top_k).
    let n = embeddings.len().min(top_k);
    embeddings
        .iter()
        .take(n)
        .map(|e| (e.group_id, 1.0 / n as f32))
        .collect()
}

/// Given a query vector (e.g. from running input through a reference env), return top-k groups by cosine similarity.
pub fn attend_by_query(
    query: &[f32],
    embeddings: &[GroupEmbedding],
    top_k: usize,
) -> Vec<(GroupId, f32)> {
    retrieve_relevant_groups(query, embeddings, top_k)
}
