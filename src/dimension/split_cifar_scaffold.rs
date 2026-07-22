//! Phase 4c — Split-CIFAR-100 protocol scaffold (Whitepaper §5.5).
//!
//! Without CIFAR-100 on disk this is an **honest stub**: documents the promote–freeze
//! protocol and runs a synthetic 5-way “class-incremental” smoke (random projected
//! clusters). Real CIFAR download + eval is future work — never claim a green CIFAR
//! result from the synthetic smoke alone.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;

use crate::types::{EnvironmentConfig, GroupId, Sample};

use super::manager::{DimensionManager, DimensionManagerConfig};

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

fn phase2_base_config() -> EnvironmentConfig {
    EnvironmentConfig {
        learning_rate: 0.12,
        weight_decay: 0.0000025,
        lateral_inhibition: 0.12,
        prune_interval: 500,
        ..EnvironmentConfig::default()
    }
}

fn cifar_present(root: &Path) -> bool {
    root.join("cifar-100-binary").exists()
        || root.join("cifar-100-python").exists()
        || root.join("train.bin").exists()
}

/// Synthetic class-incremental smoke: 5 binary tasks on Gaussian blobs in 64-D.
/// Certifies promote–freeze retention shape only — **not** CIFAR-100 accuracy.
pub fn run_phase4c_split_cifar_scaffold(seed: u64, data_root: &Path) -> SplitCifarScaffoldResult {
    let cifar = cifar_present(data_root);
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

    // Retention: re-eval all earlier tasks after all promotions.
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
            "CIFAR files detected — real Split-CIFAR-100 eval not wired yet; synthetic smoke only"
                .into()
        } else {
            "CIFAR-100 absent; synthetic promote–freeze smoke only (not a CIFAR claim)".into()
        },
    }
}
