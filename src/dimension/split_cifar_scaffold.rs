//! Phase 4c/4e/4f — Split-CIFAR protocol (Whitepaper §5.5).
//!
//! - **4c:** synthetic promote–freeze smoke (never a CIFAR claim).
//! - **4e:** CIFAR-10 lite — gray→64d (baseline; often inconclusive).
//! - **4f:** CIFAR-10 with **frozen patch bank** (pinned; adapters only).

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::path::Path;

use crate::cifar10::{
    cifar10_available, filter_class_pair, load_cifar10, CIFAR_GRAY, CIFAR_PROJECTED,
};
use crate::cifar_patch::{filter_class_pair_frozen, FrozenCifarPatchEncoder, FROZEN_FEAT};
use crate::mnist::RandomProjection;
use crate::types::{EnvironmentConfig, GroupId, Sample};

use super::embedding::{build_tag_vector, cosine_similarity, hidden_activation_vector, TAG_VECTOR_DIM};
use super::manager::{DimensionManager, DimensionManagerConfig};
use super::router::LearnedRouter;

pub const SPLIT_CIFAR_OBS: usize = 64;

#[derive(Clone, Debug)]
pub struct SplitCifarScaffoldResult {
    pub cifar_available: bool,
    pub synthetic_smoke_ok: bool,
    pub n_tasks: usize,
    pub mean_task_acc: f32,
    pub retention_zero_forget: bool,
    pub chat_metric_used: bool,
    pub note: String,
}

#[derive(Clone, Debug)]
pub struct SplitCifarLiteResult {
    pub cifar_available: bool,
    pub n_tasks: usize,
    pub mean_task_acc: f32,
    pub retention_zero_forget: bool,
    pub context_guided_agree: f32,
    pub cosine_free_agree: f32,
    pub router_free_agree: f32,
    pub router_margin: f32,
    pub router_degenerate: bool,
    pub chat_metric_used: bool,
    pub note: String,
}

fn phase2_base_config() -> EnvironmentConfig {
    EnvironmentConfig {
        learning_rate: 0.12,
        weight_decay: 0.0000025,
        lateral_inhibition: 0.12,
        prune_interval: 500,
        ..EnvironmentConfig::default()
    }
}

/// Synthetic class-incremental smoke: 5 binary tasks on Gaussian blobs in 64-D.
pub fn run_phase4c_split_cifar_scaffold(seed: u64, data_root: &Path) -> SplitCifarScaffoldResult {
    let cifar = cifar10_available(data_root)
        || data_root.join("cifar-100-binary").exists()
        || data_root.join("cifar-10-batches-py").exists();
    let mut rng = StdRng::seed_from_u64(seed);
    let n_tasks = 5usize;
    let n_train = 80usize;
    let n_test = 40usize;

    let mut centers: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n_tasks * 2 {
        centers.push(
            (0..SPLIT_CIFAR_OBS)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect(),
        );
    }

    let sample_around = |c: &[f32], rng: &mut StdRng, label: f32| -> Sample {
        let input: Vec<f32> = c
            .iter()
            .map(|&v| (v + rng.gen_range(-0.15..0.15)).clamp(-1.5, 1.5))
            .collect();
        (input, [label])
    };

    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![SPLIT_CIFAR_OBS, 24, 24, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 1,
        calibration_samples: 40,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);
    let mut group_ids: Vec<GroupId> = Vec::new();
    let mut test_per_task: Vec<Vec<Sample>> = Vec::new();
    let mut acc_at_promote = Vec::new();

    for t in 0..n_tasks {
        let c0 = &centers[t * 2];
        let c1 = &centers[t * 2 + 1];
        let mut train = Vec::with_capacity(n_train);
        for i in 0..n_train {
            train.push(if i % 2 == 0 {
                sample_around(c0, &mut rng, 0.0)
            } else {
                sample_around(c1, &mut rng, 1.0)
            });
        }
        let mut test = Vec::with_capacity(n_test);
        for i in 0..n_test {
            test.push(if i % 2 == 0 {
                sample_around(c0, &mut rng, 0.0)
            } else {
                sample_around(c1, &mut rng, 1.0)
            });
        }
        let task_name = format!("syn_cifar_{t}");
        dm.spawn_mirror(&task_name, 300 + t as u64).expect("spawn");
        for _ in 0..80 {
            let Some(r) = dm.train_mirror_epoch(&task_name, &train, &mut rng, Some(16)) else {
                break;
            };
            if r.accuracy >= 0.90 {
                break;
            }
        }
        let cal: Vec<Sample> = train.iter().take(40).cloned().collect();
        let gid = dm.force_promote(&task_name, &cal).expect("promote");
        let acc = dm.evaluate_main_group(gid, &test);
        acc_at_promote.push(acc);
        group_ids.push(gid);
        test_per_task.push(test);
    }

    let mut forget = 0.0f32;
    for (t, &gid) in group_ids.iter().enumerate() {
        let acc_now = dm.evaluate_main_group(gid, &test_per_task[t]);
        forget += (acc_at_promote[t] - acc_now).max(0.0);
    }
    let mean_acc = acc_at_promote.iter().sum::<f32>() / n_tasks as f32;
    let retention_zero_forget = forget < 1e-4;

    SplitCifarScaffoldResult {
        cifar_available: cifar,
        synthetic_smoke_ok: mean_acc >= 0.75 && retention_zero_forget,
        n_tasks,
        mean_task_acc: mean_acc,
        retention_zero_forget,
        chat_metric_used: false,
        note: if cifar {
            "CIFAR files detected — use --phase4e-split-cifar-lite (CIFAR-10 export); 4c is synthetic smoke"
                .into()
        } else {
            "CIFAR absent; synthetic promote–freeze smoke only (not a CIFAR claim)".into()
        },
    }
}

/// Phase 4e — Split-CIFAR-10 lite (class pairs, 64-D projected gray).
pub fn run_phase4e_split_cifar_lite(
    seed: u64,
    data_root: &Path,
    train_limit: usize,
    max_epochs: u32,
) -> SplitCifarLiteResult {
    // Standard Split-CIFAR-10 style: five binary tasks over class pairs.
    const TASKS: [(u8, u8); 5] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
    if !cifar10_available(data_root) {
        return SplitCifarLiteResult {
            cifar_available: false,
            n_tasks: 0,
            mean_task_acc: 0.0,
            retention_zero_forget: false,
            context_guided_agree: 0.0,
            cosine_free_agree: 0.0,
            router_free_agree: 0.0,
            router_margin: 0.0,
            router_degenerate: true,
            chat_metric_used: false,
            note: "CIFAR-10 export not found; run: python3 scripts/export_cifar10.py".into(),
        };
    }
    let (train_raw, test_raw) = match load_cifar10(data_root) {
        Ok(v) => v,
        Err(e) => {
            return SplitCifarLiteResult {
                cifar_available: false,
                n_tasks: 0,
                mean_task_acc: 0.0,
                retention_zero_forget: false,
                context_guided_agree: 0.0,
                cosine_free_agree: 0.0,
                router_free_agree: 0.0,
                router_margin: 0.0,
                router_degenerate: true,
                chat_metric_used: false,
                note: e,
            };
        }
    };

    let proj = RandomProjection::new(CIFAR_GRAY, CIFAR_PROJECTED, seed);
    let mut train_per_task = Vec::new();
    let mut test_per_task = Vec::new();
    for &(a, b) in &TASKS {
        train_per_task.push(filter_class_pair(&train_raw, a, b, &proj, train_limit));
        test_per_task.push(filter_class_pair(&test_raw, a, b, &proj, 240));
    }

    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![CIFAR_PROJECTED, 32, 32, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 1,
        calibration_samples: 80,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut group_ids: Vec<GroupId> = Vec::new();
    let mut acc_at_promote = Vec::new();

    for (t, &(a, b)) in TASKS.iter().enumerate() {
        let train = &train_per_task[t];
        let task_name = format!("cifar10_{t}");
        dm.spawn_mirror(&task_name, seed.wrapping_add(t as u64))
            .expect("spawn");
        for _ in 0..max_epochs {
            let Some(r) = dm.train_mirror_epoch(&task_name, train, &mut rng, Some(32)) else {
                break;
            };
            if r.accuracy >= 0.88 {
                break;
            }
        }
        let cal: Vec<Sample> = train.iter().take(80).cloned().collect();
        let gid = dm.force_promote(&task_name, &cal).expect("promote");
        if let Some(emb) = dm
            .main
            .embedding_library
            .iter_mut()
            .find(|e| e.group_id == gid)
        {
            emb.metatags = vec![format!("task_{t}"), format!("cls_{a}_{b}")];
            emb.tag_vector = build_tag_vector(&emb.metatags, TAG_VECTOR_DIM);
        }
        let acc = dm.evaluate_main_group(gid, &test_per_task[t]);
        acc_at_promote.push(acc);
        group_ids.push(gid);
    }

    let mut forget = 0.0f32;
    for (t, &gid) in group_ids.iter().enumerate() {
        let acc_now = dm.evaluate_main_group(gid, &test_per_task[t]);
        forget += (acc_at_promote[t] - acc_now).max(0.0);
    }
    let retention_zero_forget = forget < 1e-4;
    let mean_task_acc = acc_at_promote.iter().sum::<f32>() / group_ids.len().max(1) as f32;

    let mut pairs: Vec<(Vec<f32>, GroupId)> = Vec::new();
    for (t, data) in train_per_task.iter().enumerate() {
        for (x, _) in data.iter().take(200) {
            pairs.push((x.clone(), t as GroupId));
        }
    }
    pairs.shuffle(&mut rng);
    let mut router = LearnedRouter::build(CIFAR_PROJECTED, group_ids.len(), &pairs);

    let per_task = 40usize;
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

    let max_share = route_counts.iter().copied().max().unwrap_or(0) as f32 / n.max(1) as f32;
    let margin_mean = if margins.is_empty() {
        0.0
    } else {
        margins.iter().sum::<f32>() / margins.len() as f32
    };

    SplitCifarLiteResult {
        cifar_available: true,
        n_tasks: group_ids.len(),
        mean_task_acc,
        retention_zero_forget,
        context_guided_agree: guided_ok as f32 / n.max(1) as f32,
        cosine_free_agree: cos_ok as f32 / n.max(1) as f32,
        router_free_agree: rout_ok as f32 / n.max(1) as f32,
        router_margin: margin_mean,
        router_degenerate: max_share > 0.90,
        chat_metric_used: false,
        note: "4e: CIFAR-10 class-pair ×5 (torchvision export), gray→64d promote–freeze + CF LearnedRouter"
            .into(),
    }
}

#[derive(Clone, Debug)]
pub struct SplitCifarFrozenResult {
    pub cifar_available: bool,
    pub encoder_fingerprint: u64,
    pub encoder_pin_stable: bool,
    pub n_tasks: usize,
    pub mean_task_acc: f32,
    pub retention_zero_forget: bool,
    pub context_guided_agree: f32,
    pub cosine_free_agree: f32,
    pub router_free_agree: f32,
    pub router_margin: f32,
    pub router_degenerate: bool,
    pub chat_metric_used: bool,
    pub note: String,
}

/// Phase 4f — Split-CIFAR-10 with frozen contrast-normalized patch bank (128-D).
pub fn run_phase4f_split_cifar_frozen(
    seed: u64,
    data_root: &Path,
    train_limit: usize,
    max_epochs: u32,
) -> SplitCifarFrozenResult {
    const TASKS: [(u8, u8); 5] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
    if !cifar10_available(data_root) {
        return SplitCifarFrozenResult {
            cifar_available: false,
            encoder_fingerprint: 0,
            encoder_pin_stable: false,
            n_tasks: 0,
            mean_task_acc: 0.0,
            retention_zero_forget: false,
            context_guided_agree: 0.0,
            cosine_free_agree: 0.0,
            router_free_agree: 0.0,
            router_margin: 0.0,
            router_degenerate: true,
            chat_metric_used: false,
            note: "CIFAR-10 export not found; run: python3 scripts/export_cifar10.py".into(),
        };
    }
    let (train_raw, test_raw) = match load_cifar10(data_root) {
        Ok(v) => v,
        Err(e) => {
            return SplitCifarFrozenResult {
                cifar_available: false,
                encoder_fingerprint: 0,
                encoder_pin_stable: false,
                n_tasks: 0,
                mean_task_acc: 0.0,
                retention_zero_forget: false,
                context_guided_agree: 0.0,
                cosine_free_agree: 0.0,
                router_free_agree: 0.0,
                router_margin: 0.0,
                router_degenerate: true,
                chat_metric_used: false,
                note: e,
            };
        }
    };

    let enc = FrozenCifarPatchEncoder::new(seed);
    let pin = enc.fingerprint;
    let encoder_pin_stable = enc.verify_pin(pin);

    let mut train_per_task = Vec::new();
    let mut test_per_task = Vec::new();
    for &(a, b) in &TASKS {
        train_per_task.push(filter_class_pair_frozen(
            &train_raw,
            a,
            b,
            &enc,
            train_limit,
        ));
        test_per_task.push(filter_class_pair_frozen(&test_raw, a, b, &enc, 240));
    }

    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![FROZEN_FEAT, 48, 32, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 1,
        calibration_samples: 80,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut group_ids: Vec<GroupId> = Vec::new();
    let mut acc_at_promote = Vec::new();

    for (t, &(a, b)) in TASKS.iter().enumerate() {
        let train = &train_per_task[t];
        let task_name = format!("cifar10_frozen_{t}");
        dm.spawn_mirror(&task_name, seed.wrapping_add(t as u64 + 100))
            .expect("spawn");
        for _ in 0..max_epochs {
            let Some(r) = dm.train_mirror_epoch(&task_name, train, &mut rng, Some(32)) else {
                break;
            };
            if r.accuracy >= 0.90 {
                break;
            }
        }
        let cal: Vec<Sample> = train.iter().take(80).cloned().collect();
        let gid = dm.force_promote(&task_name, &cal).expect("promote");
        if let Some(emb) = dm
            .main
            .embedding_library
            .iter_mut()
            .find(|e| e.group_id == gid)
        {
            emb.metatags = vec![
                format!("task_{t}"),
                format!("cls_{a}_{b}"),
                format!("pin_{:016x}", pin),
            ];
            emb.tag_vector = build_tag_vector(&emb.metatags, TAG_VECTOR_DIM);
        }
        let acc = dm.evaluate_main_group(gid, &test_per_task[t]);
        acc_at_promote.push(acc);
        group_ids.push(gid);
    }

    // Encoder must remain frozen (pin unchanged after all training).
    let encoder_pin_stable = encoder_pin_stable && enc.verify_pin(pin);

    let mut forget = 0.0f32;
    for (t, &gid) in group_ids.iter().enumerate() {
        let acc_now = dm.evaluate_main_group(gid, &test_per_task[t]);
        forget += (acc_at_promote[t] - acc_now).max(0.0);
    }
    let retention_zero_forget = forget < 1e-4;
    let mean_task_acc = acc_at_promote.iter().sum::<f32>() / group_ids.len().max(1) as f32;

    let mut pairs: Vec<(Vec<f32>, GroupId)> = Vec::new();
    for (t, data) in train_per_task.iter().enumerate() {
        for (x, _) in data.iter().take(220) {
            pairs.push((x.clone(), t as GroupId));
        }
    }
    pairs.shuffle(&mut rng);
    let mut router = LearnedRouter::build(FROZEN_FEAT, group_ids.len(), &pairs);

    let per_task = 40usize;
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

    let max_share = route_counts.iter().copied().max().unwrap_or(0) as f32 / n.max(1) as f32;
    let margin_mean = if margins.is_empty() {
        0.0
    } else {
        margins.iter().sum::<f32>() / margins.len() as f32
    };

    SplitCifarFrozenResult {
        cifar_available: true,
        encoder_fingerprint: pin,
        encoder_pin_stable,
        n_tasks: group_ids.len(),
        mean_task_acc,
        retention_zero_forget,
        context_guided_agree: guided_ok as f32 / n.max(1) as f32,
        cosine_free_agree: cos_ok as f32 / n.max(1) as f32,
        router_free_agree: rout_ok as f32 / n.max(1) as f32,
        router_margin: margin_mean,
        router_degenerate: max_share > 0.90,
        chat_metric_used: false,
        note: "4f: CIFAR-10 ×5, frozen patch bank (128-D, pin-stable) + promote–freeze + CF router"
            .into(),
    }
}
