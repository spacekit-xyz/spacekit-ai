//! Phase 4a/4b — context-free MNIST routing (Whitepaper §4.4 / §5.5).
//!
//! - **4a:** tag-guided vs hidden↔embedding cosine (scaffold).
//! - **4b:** LearnedRouter trained with task labels; **test is context-free** (input only).
//!
//! Split-CIFAR remains a separate scaffold (`split_cifar_scaffold.rs`).

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::Path;

use crate::mnist::{
    filter_digit_pair, load_mnist_normalized, project_dataset, RandomProjection, MNIST_PROJECTED,
};
use crate::types::{EnvironmentConfig, GroupId, Sample};

use super::embedding::{build_tag_vector, cosine_similarity, hidden_activation_vector, TAG_VECTOR_DIM};
use super::manager::{DimensionManager, DimensionManagerConfig};
use super::router::LearnedRouter;

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

#[derive(Clone, Debug)]
pub struct CfMnistRouterResult {
    pub mnist_available: bool,
    pub n_tasks: usize,
    pub context_guided_agree: f32,
    pub cosine_free_agree: f32,
    pub router_free_agree: f32,
    pub router_margin: f32,
    pub router_degenerate: bool,
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

fn mnist_missing(data_root: &Path) -> bool {
    let images = data_root.join("train-images-idx3-ubyte");
    let images_gz = data_root.join("train-images-idx3-ubyte.gz");
    !images.exists() && !images_gz.exists()
}

fn train_split_mnist_dm(
    seed: u64,
    data_root: &Path,
    train_limit: usize,
    max_epochs: u32,
    tasks: &[(u8, u8)],
) -> Option<(DimensionManager, Vec<GroupId>, Vec<Vec<Sample>>, Vec<Vec<Sample>>)> {
    if mnist_missing(data_root) {
        return None;
    }
    let (train_imgs, train_lbls, test_imgs, test_lbls) =
        load_mnist_normalized(data_root.to_str().unwrap_or("data"));
    let proj = RandomProjection::new(crate::mnist::MNIST_INPUT, MNIST_PROJECTED, seed);

    let mut train_per_task: Vec<Vec<Sample>> = Vec::new();
    let mut test_per_task: Vec<Vec<Sample>> = Vec::new();
    for &(d1, d2) in tasks {
        let mut tr = project_dataset(&proj, &filter_digit_pair(&train_imgs, &train_lbls, d1, d2));
        tr.truncate(train_limit);
        train_per_task.push(tr);
        let mut te = project_dataset(&proj, &filter_digit_pair(&test_imgs, &test_lbls, d1, d2));
        te.truncate(240);
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

    for (t, &(d1, d2)) in tasks.iter().enumerate() {
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
    Some((dm, group_ids, train_per_task, test_per_task))
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

/// Fast 2-task Split-MNIST scaffold (0v1, 2v3). Returns unavailable if MNIST missing.
pub fn run_phase4a_context_free_mnist(
    seed: u64,
    data_root: &Path,
    train_limit: usize,
    max_epochs: u32,
) -> ContextFreeMnistResult {
    const TASKS: [(u8, u8); 2] = [(0, 1), (2, 3)];
    let Some((mut dm, group_ids, _, test_per_task)) =
        train_split_mnist_dm(seed, data_root, train_limit, max_epochs, &TASKS)
    else {
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
    };

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
            let mut best_g = *gid;
            let mut best_gs = f32::NEG_INFINITY;
            let mut second_gs = f32::NEG_INFINITY;
            for emb in &dm.main.embedding_library {
                let sc = if emb.metatags.iter().any(|m| m == &tag) {
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

/// Phase 4b — LearnedRouter on projected pixels; **no task tags at test**.
pub fn run_phase4b_cf_mnist_router(
    seed: u64,
    data_root: &Path,
    train_limit: usize,
    max_epochs: u32,
) -> CfMnistRouterResult {
    // Three digit-pair tasks — harder CF routing than binary.
    const TASKS: [(u8, u8); 3] = [(0, 1), (2, 3), (4, 5)];
    let Some((mut dm, group_ids, train_per_task, test_per_task)) =
        train_split_mnist_dm(seed, data_root, train_limit, max_epochs, &TASKS)
    else {
        return CfMnistRouterResult {
            mnist_available: false,
            n_tasks: 0,
            context_guided_agree: 0.0,
            cosine_free_agree: 0.0,
            router_free_agree: 0.0,
            router_margin: 0.0,
            router_degenerate: true,
            mean_task_acc: 0.0,
            chat_metric_used: false,
            note: "MNIST not found; set MNIST_ROOT or run scripts/download_mnist.sh".into(),
        };
    };

    // Train router on held-in train pixels → group index (task identity at train only).
    let mut pairs: Vec<(Vec<f32>, GroupId)> = Vec::new();
    for (t, data) in train_per_task.iter().enumerate() {
        for (x, _) in data.iter().take(200) {
            pairs.push((x.clone(), t as GroupId));
        }
    }
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(99));
    pairs.shuffle(&mut rng);
    let mut router = LearnedRouter::build(MNIST_PROJECTED, group_ids.len(), &pairs);

    let per_task = 50usize;
    let mut guided_ok = 0usize;
    let mut cos_ok = 0usize;
    let mut rout_ok = 0usize;
    let mut margins = Vec::new();
    let mut route_counts = vec![0usize; group_ids.len()];
    let mut n = 0usize;

    for (t, gid) in group_ids.iter().enumerate() {
        let tag = format!("task_{t}");
        for s in test_per_task[t].iter().take(per_task) {
            n += 1;
            // Guided
            let guided = dm
                .main
                .embedding_library
                .iter()
                .find(|e| e.metatags.iter().any(|m| m == &tag))
                .map(|e| e.group_id)
                .unwrap_or(*gid);
            if guided == *gid {
                guided_ok += 1;
            }

            // Cosine free
            let mut best_f = *gid;
            let mut best_fs = f32::NEG_INFINITY;
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
                    best_fs = self_sim;
                    best_f = cand;
                }
            }
            if best_f == *gid {
                cos_ok += 1;
            }

            // Learned router (context-free at test)
            let logits = router.predict_logits(&s.0);
            let mut order: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let pick = order.first().map(|(i, _)| *i).unwrap_or(0);
            let second = order.get(1).map(|(_, v)| *v).unwrap_or(0.0);
            let first = order.first().map(|(_, v)| *v).unwrap_or(0.0);
            margins.push(first - second);
            route_counts[pick] += 1;
            if pick == t {
                rout_ok += 1;
            }
        }
    }

    let max_share = route_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0) as f32
        / n.max(1) as f32;
    let mean_acc: f32 = group_ids
        .iter()
        .enumerate()
        .map(|(t, &gid)| dm.evaluate_main_group(gid, &test_per_task[t]))
        .sum::<f32>()
        / group_ids.len().max(1) as f32;

    CfMnistRouterResult {
        mnist_available: true,
        n_tasks: group_ids.len(),
        context_guided_agree: guided_ok as f32 / n.max(1) as f32,
        cosine_free_agree: cos_ok as f32 / n.max(1) as f32,
        router_free_agree: rout_ok as f32 / n.max(1) as f32,
        router_margin: mean(&margins),
        router_degenerate: max_share > 0.90,
        mean_task_acc: mean_acc,
        chat_metric_used: false,
        note: "4b: LearnedRouter train with task labels; test context-free (input only)".into(),
    }
}
