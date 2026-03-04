//! Main Dimension — consolidated knowledge store. Frozen groups only; never trains.

use std::collections::HashMap;

use crate::environment::NeuralEnvironment;
use crate::types::GroupId;
use serde::{Deserialize, Serialize};

use super::embedding::GroupEmbedding;

/// One frozen environment per promoted group.
#[derive(Clone, Serialize, Deserialize)]
pub struct FrozenGroupEnv {
    pub group_id: GroupId,
    pub task_name: String,
    pub env: NeuralEnvironment,
    pub embedding: GroupEmbedding,
    pub accuracy: f32,
    pub promoted_at_epoch: u64,
}

/// Main Dimension: only frozen promoted groups. Never trains.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct MainDimension {
    /// One NeuralEnvironment per promoted group; all neurons frozen.
    pub groups: HashMap<GroupId, FrozenGroupEnv>,
    /// Shared embedding library for routing.
    pub embedding_library: Vec<GroupEmbedding>,
    /// Creation order — defines output head indices for multi-task inference.
    pub group_order: Vec<GroupId>,
}

impl MainDimension {
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            embedding_library: Vec::new(),
            group_order: Vec::new(),
        }
    }

    /// Register a frozen env as a new group. Call after freezing all neurons/synapses.
    pub fn register_group(
        &mut self,
        group_id: GroupId,
        task_name: String,
        env: NeuralEnvironment,
        embedding: GroupEmbedding,
        accuracy: f32,
        promoted_at_epoch: u64,
    ) {
        let lib_emb = GroupEmbedding {
            group_id,
            vector: embedding.vector.clone(),
            task_name: task_name.clone(),
            accuracy,
            intrinsic_dim: embedding.intrinsic_dim,
            description: embedding.description.clone(),
            metatags: embedding.metatags.clone(),
            tag_vector: embedding.tag_vector.clone(),
        };
        self.embedding_library.push(lib_emb.clone());
        self.group_order.push(group_id);
        self.groups.insert(
            group_id,
            FrozenGroupEnv {
                group_id,
                task_name,
                env,
                embedding,
                accuracy,
                promoted_at_epoch,
            },
        );
    }

    /// Query selected groups with input; returns (group_id, output activations).
    pub fn query(
        &mut self,
        input: &[f32],
        group_ids: &[GroupId],
    ) -> Vec<(GroupId, Vec<f32>)> {
        group_ids
            .iter()
            .filter_map(|&gid| {
                self.groups.get_mut(&gid).map(|fg| {
                    let out = fg.env.predict(input);
                    (gid, out)
                })
            })
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::NeuralEnvironment;
    use crate::types::EnvironmentConfig;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_main_register_and_query() {
        let config = EnvironmentConfig::default();
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = NeuralEnvironment::new(config);
        env.build_layers(&[2, 16, 16, 1], &mut rng);
        env.freeze_all();
        let calibration = vec![([0.0_f32, 0.0], [0.0]); 10];
        let vector = crate::dimension::embedding::compute_group_embedding(&mut env, &calibration);
        let embedding = GroupEmbedding {
            group_id: 0,
            vector,
            task_name: "test".to_string(),
            accuracy: 0.9,
            intrinsic_dim: None,
            description: None,
            metatags: vec![],
            tag_vector: vec![],
        };
        let mut main = MainDimension::new();
        main.register_group(0, "test".to_string(), env, embedding, 0.9, 0);
        let out = main.query(&[0.5, 0.5], &[0]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 0);
        assert!(!out[0].1.is_empty());
    }
}
