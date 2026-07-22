//! Phase 4a — context-free MNIST routing certifier scaffold (Whitepaper §4.4 / §5.5).
//!
//! Compares **context-guided** (task-tag) dispatch vs **context-free** (hidden↔embedding cosine)
//! on a tiny Split-MNIST subset. Does **not** claim a closed result; Split-CIFAR remains future.

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::Path;

use crate::mnist::{
    filter_digit_pair, load_mnist_normalized, project_dataset, RandomProjection, MNIST_PROJECTED,
};
use crate::types::{EnvironmentConfig, GroupId, Sample};

use super::embedding::{build_tag_vector, cosine_similarity, hidden_activation_vector, TAG_VECTOR_DIM};
use super::manager::{DimensionManager, DimensionManagerConfig};

#[derive(Clone, Debug)]
pub struct ContextFreeMnistResult {
    pub mnist_available: bool,
    pub n_tasks: usize,
    pub context_guided_agree: f32,
    pub context_free_agree: f32,
    pub context_guided_margin: f32,
    pub context_free_margin: f32,
    pub mean_task_acc: f32,
    pub chat_metric_used: bool,
    pub note: String,
}

fn phase2_base_config() -> EnvironmentConfig {
    EnvironmentConfig {
        learning_rate: 0.15,
        weight_decay: 0.0000025,
        lateral_inhibition: 0.12,
        prune_interval: 500,
        ..EnvironmentConfig::default()
    }
}

/// Fast 2-task Split-MNIST scaffold (0v1, 2v3). Returns unavailable if MNIST missing.
pub fn run_phase4a_context_free_mnist(
    seed: u64,
    data_root: &Path,
    train_limit: usize,
    max_epochs: u32,
) -> ContextFreeMnistResult {
    let images = data_root.join("train-images-idx3-ubyte");
    let images_gz = data_root.join("train-images-idx3-ubyte.gz");
    if !images.exists() && !images_gz.exists() {
        return ContextFreeMnistResult {
            mnist_available: false,
            n_tasks: 0,
            context_guided_agree: 0.0,
            context_free_agree: 0.0,
            context_guided_margin: 0.0,
            context_free_margin: 0.0,
            mean_task_acc: 0.0,
            chat_metric_used: false,
            note: "MNIST not found; set MNIST_ROOT or run scripts/download_mnist.sh".into(),
        };
    }

    let (train_imgs, train_lbls, test_imgs, test_lbls) =
        load_mnist_normalized(data_root.to_str().unwrap_or("data"));
    let proj = RandomProjection::new(crate::mnist::MNIST_INPUT, MNIST_PROJECTED, seed);
    const TASKS: [(u8, u8); 2] = [(0, 1), (2, 3)];

    let mut train_per_task: Vec<Vec<Sample>> = Vec::new();
    let mut test_per_task: Vec<Vec<Sample>> = Vec::new();
    for (d1, d2) in TASKS {
        let mut tr = project_dataset(&proj, &filter_digit_pair(&train_imgs, &train_lbls, d1, d2));
        tr.truncate(train_limit);
        train_per_task.push(tr);
        let mut te = project_dataset(&proj, &filter_digit_pair(&test_imgs, &test_lbls, d1, d2));
        te.truncate(200);
        test_per_task.push(te);
    }

    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![MNIST_PROJECTED, 32, 32, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 1,
        calibration_samples: 80,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut group_ids: Vec<GroupId> = Vec::new();

    for (t, (d1, d2)) in TASKS.iter().enumerate() {
        let task_name = format!("cf_task_{t}");
        let train = &train_per_task[t];
        let cal: Vec<Sample> = train.iter().take(80).cloned().collect();
        dm.spawn_mirror(&task_name, 200 + t as u64).expect("spawn");
        for _ in 0..max_epochs {
            let Some(result) = dm.train_mirror_epoch(&task_name, train, &mut rng, Some(32)) else {
                break;
            };
            if result.accuracy >= 0.92 {
                break;
            }
        }
        let gid = dm.force_promote(&task_name, &cal).expect("promote");
        if let Some(emb) = dm
            .main
            .embedding_library
            .iter_mut()
            .find(|e| e.group_id == gid)
        {
            emb.metatags = vec![format!("task_{t}"), format!("digits_{d1}_{d2}")];
            emb.tag_vector = build_tag_vector(&emb.metatags, TAG_VECTOR_DIM);
        }
        group_ids.push(gid);
    }

    let mut guided_ok = 0usize;
    let mut free_ok = 0usize;
    let mut guided_margins = Vec::new();
    let mut free_margins = Vec::new();
    let mut n = 0usize;
    let per_task = 40usize;

    for (t, gid) in group_ids.iter().enumerate() {
        let samples = &test_per_task[t];
        let tag = format!("task_{t}");
        for s in samples.iter().take(per_task) {
            n += 1;
            // Exact task-tag match (context-guided). Hash tag-vectors can collide on
            // short labels; whitepaper §4.4 cares about task-identity availability.
            let mut best_g = *gid;
            let mut best_gs = f32::NEG_INFINITY;
            let mut second_gs = f32::NEG_INFINITY;
            for emb in &dm.main.embedding_library {
                let sc = if emb.metatags.iter().any(|t| t == &tag) {
                    1.0
                } else {
                    0.0
                };
                if sc > best_gs {
                    second_gs = best_gs;
                    best_gs = sc;
                    best_g = emb.group_id;
                } else if sc > second_gs {
                    second_gs = sc;
                }
            }
            if best_g == *gid {
                guided_ok += 1;
            }
            guided_margins.push(best_gs - second_gs);

            let mut best_f = *gid;
            let mut best_fs = f32::NEG_INFINITY;
            let mut second_fs = f32::NEG_INFINITY;
            for &cand in &group_ids {
                let fg = dm.main.groups.get_mut(&cand).unwrap();
                let _ = fg.env.predict(&s.0);
                let hidden = hidden_activation_vector(&fg.env);
                let self_sim = dm
                    .main
                    .embedding_library
                    .iter()
                    .find(|e| e.group_id == cand)
                    .map(|e| cosine_similarity(&hidden, &e.vector))
                    .unwrap_or(-1.0);
                if self_sim > best_fs {
                    second_fs = best_fs;
                    best_fs = self_sim;
                    best_f = cand;
                } else if self_sim > second_fs {
                    second_fs = self_sim;
                }
            }
            if best_f == *gid {
                free_ok += 1;
            }
            free_margins.push(best_fs - second_fs);
        }
    }

    let mean_acc: f32 = group_ids
        .iter()
        .enumerate()
        .map(|(t, &gid)| dm.evaluate_main_group(gid, &test_per_task[t]))
        .sum::<f32>()
        / group_ids.len().max(1) as f32;

    let mean = |v: &[f32]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };

    ContextFreeMnistResult {
        mnist_available: true,
        n_tasks: group_ids.len(),
        context_guided_agree: guided_ok as f32 / n.max(1) as f32,
        context_free_agree: free_ok as f32 / n.max(1) as f32,
        context_guided_margin: mean(&guided_margins),
        context_free_margin: mean(&free_margins),
        mean_task_acc: mean_acc,
        chat_metric_used: false,
        note: "scaffold: 2-task Split-MNIST; context-free not claimed closed".into(),
    }
}
