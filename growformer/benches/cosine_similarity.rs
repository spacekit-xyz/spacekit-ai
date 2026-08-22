//! Micro-benchmarks for embedding-space similarity hot paths.
//!
//! ## Why this exists
//! Several subsystems (`reasoning`, `metacognition`, `dimension::embedding`, …)
//! each implement a small `dot / (||a|| * ||b||)` loop. Before investing in
//! LUT/EML-style fast math or SIMD, **measure** what fraction of wall time lives
//! in these loops vs Clifford `band_coherence` / IO / tokenization.
//!
//! ## Practical workflow
//! 1. `cargo bench --bench cosine_similarity` (release implied by Criterion).
//! 2. For end-to-end hot paths, use a sampling profiler (`samply`, Instruments,
//!    `perf record`) on a representative workload and confirm the flamegraph
//!    actually lands in cosine or `embed_bridge_vector`.
//! 3. If you change the math (rsqrt, fused approximations), compare max
//!    absolute drift against inference gates (`COHERENCE_FLOOR`, System 2
//!    stall thresholds, metacognition blend weights) — not just average error.
//!
//! Note: `dimension::embedding::cosine_similarity` clamps to [-1, 1] and
//! validates lengths; `reasoning`-style cosine below matches the private helper
//! in `reasoning.rs` (no clamp, assumes equal length from caller).

use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use growformer::coherence::band_coherence;
use growformer::dimension::embedding::cosine_similarity;
use growformer::dimension::group_gen::IndexedGenEnv;
use growformer::reasoning::{CognitiveMap, ReasoningEngine};

/// Mirrors `growformer::src/reasoning.rs` `cosine_sim` (no clamp).
fn cosine_sim_reasoning_style(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Mirrors `growformer::src/metacognition.rs` `cosine_sim` (min length, no clamp).
fn cosine_sim_metacognition_style(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let dot: f32 = a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(x, y)| x * y)
        .sum();
    let na = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    dot / (na * nb)
}

fn bench_single_pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_pair");
    for dim in [64usize, 128, 384] {
        let a: Vec<f32> = (0..dim).map(|i| ((i * 17) as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i * 31) as f32 * 0.001).cos()).collect();

        group.bench_with_input(BenchmarkId::new("embedding_cosine", dim), &dim, |ben, _| {
            ben.iter(|| black_box(cosine_similarity(black_box(&a), black_box(&b))));
        });
        group.bench_with_input(
            BenchmarkId::new("reasoning_style_cosine", dim),
            &dim,
            |ben, _| {
                ben.iter(|| black_box(cosine_sim_reasoning_style(black_box(&a), black_box(&b))));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("metacog_style_cosine", dim),
            &dim,
            |ben, _| {
                ben.iter(|| {
                    black_box(cosine_sim_metacognition_style(black_box(&a), black_box(&b)))
                });
            },
        );
    }
    group.finish();
}

fn bench_query_vs_centroids(c: &mut Criterion) {
    const DIM: usize = 128;
    const N: usize = 256;
    let query: Vec<f32> = (0..DIM).map(|i| ((i * 13) as f32 * 0.01).sin()).collect();
    let centroids: Vec<Vec<f32>> = (0..N)
        .map(|j| {
            (0..DIM)
                .map(|i| ((i + j * 7) as f32 * 0.009).cos())
                .collect()
        })
        .collect();

    c.bench_function("scan_256_centroids_reasoning_cosine_dim128", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for cvec in &centroids {
                acc += cosine_sim_reasoning_style(black_box(&query), black_box(cvec));
            }
            black_box(acc)
        });
    });

    c.bench_function("scan_256_centroids_embedding_cosine_dim128", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for cvec in &centroids {
                acc += cosine_similarity(black_box(&query), black_box(cvec));
            }
            black_box(acc)
        });
    });
}

fn bench_band_coherence_pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("band_coherence_pair");
    for dim in [8usize, 32, 128] {
        let a: Vec<f32> = (0..dim).map(|i| ((i * 11) as f32 * 0.02).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i * 19) as f32 * 0.02).cos()).collect();
        group.bench_with_input(BenchmarkId::new("embed_plus_bands", dim), &dim, |ben, _| {
            ben.iter(|| {
                let out = band_coherence(black_box(&a), black_box(&b));
                black_box(out.combined)
            });
        });
    }
    group.finish();
}

/// Two small `IndexedGenEnv` lattices + [`ReasoningEngine::should_reason`], which
/// scans every program centroid per group (same inner cosine as activation scoring).
fn bench_reasoning_should_reason(c: &mut Criterion) {
    // Exercises Hopf / E8 scoring inside `IndexedGenEnv::build` (not a multiple of 64).
    const DIM: usize = 32;
    const PAIRS: usize = 56;

    fn build_env(offset: usize) -> IndexedGenEnv {
        let texts: Vec<String> = (0..PAIRS)
            .map(|i| format!("bench lattice line {} offset {}", i, offset))
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..PAIRS)
            .map(|i| {
                (0..DIM)
                    .map(|j| {
                        (((i * 17 + j * 3 + offset) as f32) * 0.031)
                            .sin()
                            .clamp(-0.9, 0.9)
                    })
                    .collect()
            })
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let emb_refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        IndexedGenEnv::build(&text_refs, &emb_refs, 8, 0.85)
    }

    let env_a = build_env(0);
    let env_b = build_env(1000);
    let mut group_envs = HashMap::new();
    group_envs.insert(0usize, env_a);
    group_envs.insert(1usize, env_b);

    let engine = ReasoningEngine::new(
        CognitiveMap::build(&HashMap::new(), &HashMap::new()),
        HashMap::new(),
    );

    let query: Vec<f32> = (0..DIM).map(|j| ((j * 11) as f32 * 0.041).cos()).collect();

    c.bench_function("reasoning_should_reason_two_groups", |b| {
        b.iter(|| {
            black_box(engine.should_reason(
                black_box(query.as_slice()),
                0.5f32,
                black_box(&group_envs),
            ))
        });
    });
}

criterion_group!(
    benches,
    bench_single_pair,
    bench_query_vs_centroids,
    bench_band_coherence_pair,
    bench_reasoning_should_reason
);
criterion_main!(benches);
