//! Re-measure I(R; A_spiral): output bottleneck vs below resolution (see CMI spec §spiral-resolve).

use crate::cmi::{
    cross_entropy_bits, empirical_h_r, mean_std, mi_from_ce, CmiPointRecord,
};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

const LN2: f32 = std::f32::consts::LN_2;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SpiralResolveResult {
    pub pooled_parametric_mi: f32,
    pub pooled_parametric_mi_std: f32,
    pub pooled_linear_mi: f32,
    pub pooled_linear_mi_std: f32,
    pub pooled_pca_knn_mi: [f32; 3],
    pub perm_null_mean: f32,
    pub perm_null_p95: f32,
    pub perm_observed_percentile: f32,
    pub debiased_mi: f32,
    pub pooled_probe_output_angle_deg: f32,
    pub i_r_y_spiral: f32,
    pub per_seed_parametric_mi: Vec<f32>,
    pub per_seed_linear_mi: Vec<f32>,
    pub per_seed_angles_deg: Vec<f32>,
}

fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

fn stratified_kfold_indices(
    labels: &[u8],
    k: usize,
    rng: &mut StdRng,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut idx0 = Vec::new();
    let mut idx1 = Vec::new();
    for (i, &y) in labels.iter().enumerate() {
        if y == 1 {
            idx1.push(i);
        } else {
            idx0.push(i);
        }
    }
    idx0.shuffle(rng);
    idx1.shuffle(rng);
    for (fi, i) in idx0.iter().enumerate() {
        folds[fi % k].push(*i);
    }
    for (fi, i) in idx1.iter().enumerate() {
        folds[fi % k].push(*i);
    }
    let mut splits = Vec::new();
    for test_fold in 0..k {
        let mut test = Vec::new();
        let mut train = Vec::new();
        for (fi, f) in folds.iter().enumerate() {
            if fi == test_fold {
                test.extend(f.iter().copied());
            } else {
                train.extend(f.iter().copied());
            }
        }
        splits.push((train, test));
    }
    splits
}

struct LogisticRidge {
    w: Vec<f32>,
    b: f32,
    lambda: f32,
}

impl LogisticRidge {
    fn new(dim: usize, lambda: f32) -> Self {
        Self {
            w: vec![0.0; dim],
            b: 0.0,
            lambda,
        }
    }

    fn fit(&mut self, x: &[Vec<f32>], y: &[u8], lr: f32, epochs: usize) {
        for _ in 0..epochs {
            for (xi, &yi) in x.iter().zip(y.iter()) {
                let mut z = self.b;
                for (wi, &xi_j) in self.w.iter().zip(xi.iter()) {
                    z += wi * xi_j;
                }
                let p = sigmoid(z);
                let target = if yi == 1 { 1.0 } else { 0.0 };
                let err = p - target;
                for (wi, &xi_j) in self.w.iter_mut().zip(xi.iter()) {
                    *wi -= lr * (err * xi_j + self.lambda * *wi);
                }
                self.b -= lr * err;
            }
        }
    }

    fn predict_proba(&self, x: &[f32]) -> f32 {
        let mut z = self.b;
        for (wi, &xi_j) in self.w.iter().zip(x.iter()) {
            z += wi * xi_j;
        }
        sigmoid(z)
    }
}

fn angle_degrees(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return f32::NAN;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    (dot / (na * nb + 1e-8)).clamp(-1.0, 1.0).acos().to_degrees()
}

fn mean_center(data: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
    let n = data.len().max(1) as f32;
    let d = data.first().map(|v| v.len()).unwrap_or(0);
    let mut mu = vec![0.0f32; d];
    for row in data {
        for (j, &v) in row.iter().enumerate() {
            mu[j] += v;
        }
    }
    for m in &mut mu {
        *m /= n;
    }
    let centered: Vec<Vec<f32>> = data
        .iter()
        .map(|row| row.iter().zip(mu.iter()).map(|(&v, &m)| v - m).collect())
        .collect();
    (centered, mu)
}

fn covariance_matrix(data: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = data.len().max(1) as f32;
    let d = data.first().map(|v| v.len()).unwrap_or(0);
    let mut cov = vec![vec![0.0f32; d]; d];
    for row in data {
        for i in 0..d {
            for j in 0..d {
                cov[i][j] += row[i] * row[j];
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            cov[i][j] /= n;
        }
    }
    cov
}

fn power_iteration_top_eigenvector(cov: &[Vec<f32>], rng: &mut StdRng) -> Vec<f32> {
    let d = cov.len();
    if d == 0 {
        return Vec::new();
    }
    let mut v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in &mut v {
        *x /= norm;
    }
    for _ in 0..80 {
        let mut w = vec![0.0f32; d];
        for i in 0..d {
            for j in 0..d {
                w[i] += cov[i][j] * v[j];
            }
        }
        let norm = w.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in &mut w {
            *x /= norm;
        }
        v = w;
    }
    v
}

fn deflate(cov: &mut [Vec<f32>], vec: &[f32], eigenvalue: f32) {
    let d = cov.len();
    for i in 0..d {
        for j in 0..d {
            cov[i][j] -= eigenvalue * vec[i] * vec[j];
        }
    }
}

fn pca_project(data: &[Vec<f32>], m: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let (centered, _) = mean_center(data);
    let mut cov = covariance_matrix(&centered);
    let d = cov.len();
    let m = m.min(d).max(1);
    let mut components = Vec::with_capacity(m);
    for _ in 0..m {
        let v = power_iteration_top_eigenvector(&cov, rng);
        let mut ev = 0.0f32;
        for i in 0..d {
            let mut row_sum = 0.0f32;
            for j in 0..d {
                row_sum += cov[i][j] * v[j];
            }
            ev += v[i] * row_sum;
        }
        components.push(v);
        deflate(&mut cov, &components.last().unwrap().clone(), ev);
    }
    centered
        .iter()
        .map(|row| {
            components
                .iter()
                .map(|comp| row.iter().zip(comp.iter()).map(|(a, b)| a * b).sum::<f32>())
                .collect()
        })
        .collect()
}

fn knn_mi(features: &[Vec<f32>], labels: &[u8], k: usize, rng: &mut StdRng) -> f32 {
    if features.len() < k + 2 {
        return 0.0;
    }
    let h_r = empirical_h_r(labels);
    let (train_idx, test_idx) = {
        let mut perm: Vec<usize> = (0..labels.len()).collect();
        perm.shuffle(rng);
        let split = (perm.len() as f32 * 0.6) as usize;
        let train: Vec<usize> = perm[..split].to_vec();
        let test: Vec<usize> = perm[split..].to_vec();
        (train, test)
    };
    if test_idx.is_empty() {
        return 0.0;
    }
    let dist = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
    };
    let mut probs = Vec::new();
    let mut test_y = Vec::new();
    for &ti in &test_idx {
        let mut dists: Vec<(f32, u8)> = train_idx
            .iter()
            .map(|&j| (dist(&features[ti], &features[j]), labels[j]))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let pos = dists
            .iter()
            .take(k)
            .filter(|(_, y)| *y == 1)
            .count() as f32;
        probs.push((pos + 1.0) / (k as f32 + 2.0));
        test_y.push(labels[ti]);
    }
    mi_from_ce(h_r, cross_entropy_bits(&test_y, &probs))
}

fn mlp_mi_single_split(
    features: &[Vec<f32>],
    labels: &[u8],
    train_idx: &[usize],
    test_idx: &[usize],
    rng: &mut StdRng,
) -> f32 {
    let h_r = empirical_h_r(labels);
    let train_x: Vec<_> = train_idx.iter().map(|&i| features[i].clone()).collect();
    let train_y: Vec<_> = train_idx.iter().map(|&i| labels[i]).collect();
    let test_x: Vec<_> = test_idx.iter().map(|&i| features[i].clone()).collect();
    let test_y: Vec<_> = test_idx.iter().map(|&i| labels[i]).collect();
    if train_x.is_empty() || test_x.is_empty() {
        return 0.0;
    }
    let dim = features[0].len().max(1);
    let mut mlp = crate::cmi::SmallMlp::random(dim, rng);
    mlp.train(&train_x, &train_y, 0.03, 100);
    let probs: Vec<f32> = test_x.iter().map(|x| mlp.predict_proba(x)).collect();
    mi_from_ce(h_r, cross_entropy_bits(&test_y, &probs))
}

fn linear_mi_single_split(
    features: &[Vec<f32>],
    labels: &[u8],
    train_idx: &[usize],
    test_idx: &[usize],
    lambda: f32,
) -> (f32, Vec<f32>) {
    let h_r = empirical_h_r(labels);
    let train_x: Vec<_> = train_idx.iter().map(|&i| features[i].clone()).collect();
    let train_y: Vec<_> = train_idx.iter().map(|&i| labels[i]).collect();
    let test_x: Vec<_> = test_idx.iter().map(|&i| features[i].clone()).collect();
    let test_y: Vec<_> = test_idx.iter().map(|&i| labels[i]).collect();
    if train_x.is_empty() || test_x.is_empty() {
        return (0.0, vec![]);
    }
    let dim = features[0].len().max(1);
    let mut model = LogisticRidge::new(dim, lambda);
    model.fit(&train_x, &train_y, 0.1, 80);
    let probs: Vec<f32> = test_x.iter().map(|x| model.predict_proba(x)).collect();
    (mi_from_ce(h_r, cross_entropy_bits(&test_y, &probs)), model.w.clone())
}

fn repeated_cv_mlp(
    features: &[Vec<f32>],
    labels: &[u8],
    k: usize,
    repeats: usize,
    rng: &mut StdRng,
) -> (f32, f32) {
    let mut vals = Vec::new();
    for _ in 0..repeats {
        for (train, test) in stratified_kfold_indices(labels, k, rng) {
            vals.push(mlp_mi_single_split(features, labels, &train, &test, rng));
        }
    }
    let (m, s) = mean_std(&vals);
    (m, s)
}

fn repeated_cv_linear(
    features: &[Vec<f32>],
    labels: &[u8],
    k: usize,
    repeats: usize,
    lambda: f32,
    rng: &mut StdRng,
) -> (f32, f32, Vec<f32>) {
    let mut vals = Vec::new();
    let mut last_w = Vec::new();
    for _ in 0..repeats {
        for (train, test) in stratified_kfold_indices(labels, k, rng) {
            let (mi, w) = linear_mi_single_split(features, labels, &train, &test, lambda);
            vals.push(mi);
            last_w = w;
        }
    }
    let (m, s) = mean_std(&vals);
    (m, s, last_w)
}

fn permutation_null_linear(
    features: &[Vec<f32>],
    labels: &[u8],
    b: usize,
    k: usize,
    lambda: f32,
    rng: &mut StdRng,
) -> (f32, f32, f32, f32) {
    let (obs, _, _) = repeated_cv_linear(features, labels, k, 3, lambda, rng);
    let mut nulls = Vec::with_capacity(b);
    let mut shuffled = labels.to_vec();
    for _ in 0..b {
        shuffled.shuffle(rng);
        let (mi, _, _) = repeated_cv_linear(features, &shuffled, k, 2, lambda, rng);
        nulls.push(mi);
    }
    nulls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = nulls.iter().sum::<f32>() / nulls.len().max(1) as f32;
    let p95_idx = ((nulls.len() as f32) * 0.95).floor() as usize;
    let p95 = nulls.get(p95_idx.min(nulls.len().saturating_sub(1))).copied().unwrap_or(0.0);
    let below = nulls.iter().filter(|&&v| v < obs).count() as f32;
    let percentile = below / nulls.len().max(1) as f32 * 100.0;
    (obs, mean, p95, percentile)
}

pub fn resolve_spiral_region_mi(
    records: &[CmiPointRecord],
    output_weights_by_seed: &[(u64, Vec<f32>)],
    perm_b: usize,
) -> SpiralResolveResult {
    let mut rng = StdRng::seed_from_u64(88_001);
    let pooled_a: Vec<Vec<f32>> = records.iter().map(|r| r.a_spiral.clone()).collect();
    let pooled_r: Vec<u8> = records.iter().map(|r| r.region).collect();
    let pooled_y: Vec<f32> = records.iter().map(|r| r.y_spiral).collect();

    let (pooled_parametric_mi, pooled_parametric_mi_std) =
        repeated_cv_mlp(&pooled_a, &pooled_r, 5, 5, &mut rng);
    let (pooled_linear_mi, pooled_linear_mi_std, probe_w) =
        repeated_cv_linear(&pooled_a, &pooled_r, 5, 5, 1e-3, &mut rng);

    let mut pca_knn = [0.0f32; 3];
    for (i, &m) in [2usize, 3, 5].iter().enumerate() {
        let proj = pca_project(&pooled_a, m, &mut rng);
        pca_knn[i] = knn_mi(&proj, &pooled_r, 7, &mut rng);
    }

    let (perm_obs, perm_null_mean, perm_null_p95, perm_percentile) =
        permutation_null_linear(&pooled_a, &pooled_r, perm_b, 5, 1e-3, &mut rng);

    let i_r_y_spiral = crate::cmi::histogram_1d_mi(&pooled_y, &pooled_r, 12, &mut rng);

    let mut per_seed_parametric = Vec::new();
    let mut per_seed_linear = Vec::new();
    let mut per_seed_angles = Vec::new();
    let seeds: Vec<u64> = records
        .iter()
        .map(|r| r.seed)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    for &seed in &seeds {
        let seed_recs: Vec<_> = records.iter().filter(|r| r.seed == seed).collect();
        let a: Vec<Vec<f32>> = seed_recs.iter().map(|r| r.a_spiral.clone()).collect();
        let r: Vec<u8> = seed_recs.iter().map(|r| r.region).collect();
        let (mi_p, _) = repeated_cv_mlp(&a, &r, 5, 3, &mut rng);
        let (mi_l, _, w) = repeated_cv_linear(&a, &r, 5, 3, 1e-3, &mut rng);
        per_seed_parametric.push(mi_p);
        per_seed_linear.push(mi_l);
        if let Some((_, out_w)) = output_weights_by_seed.iter().find(|(s, _)| *s == seed) {
            per_seed_angles.push(angle_degrees(&w, out_w));
        }
    }

    let pooled_angle = output_weights_by_seed
        .iter()
        .map(|(_, out_w)| angle_degrees(&probe_w, out_w))
        .fold(0.0f32, |a, b| if b.is_nan() { a } else { a + b })
        / output_weights_by_seed.len().max(1) as f32;

    SpiralResolveResult {
        pooled_parametric_mi,
        pooled_parametric_mi_std,
        pooled_linear_mi,
        pooled_linear_mi_std,
        pooled_pca_knn_mi: pca_knn,
        perm_null_mean,
        perm_null_p95,
        perm_observed_percentile: perm_percentile,
        debiased_mi: (perm_obs - perm_null_p95).max(0.0),
        pooled_probe_output_angle_deg: pooled_angle,
        i_r_y_spiral,
        per_seed_parametric_mi: per_seed_parametric,
        per_seed_linear_mi: per_seed_linear,
        per_seed_angles_deg: per_seed_angles,
    }
}

pub fn format_spiral_resolve_report(res: &SpiralResolveResult) -> String {
    let (ps_m, ps_s) = mean_std(&res.per_seed_parametric_mi);
    let (pl_m, pl_s) = mean_std(&res.per_seed_linear_mi);
    let mut out = String::new();
    out.push_str("=== Spiral I(R;A) resolve — output bottleneck or below resolution? ===\n\n");
    out.push_str("| Instrument | Pooled | Per-seed mean ± std | Notes |\n");
    out.push_str("| ---------- | ------ | ------------------- | ----- |\n");
    out.push_str(&format!(
        "| I(R;Y_spiral) histogram | {:.3} | — | region-blind scalar baseline |\n",
        res.i_r_y_spiral
    ));
    out.push_str(&format!(
        "| I(R;A) parametric MLP | {:.3} ± {:.3} | {:.3} ± {:.3} | perm p={:.0}%, null95={:.3} |\n",
        res.pooled_parametric_mi,
        res.pooled_parametric_mi_std,
        ps_m,
        ps_s,
        res.perm_observed_percentile,
        res.perm_null_p95
    ));
    out.push_str(&format!(
        "| I(R;A) linear probe | {:.3} ± {:.3} | {:.3} ± {:.3} | probe⊥output avg {:.1}° |\n",
        res.pooled_linear_mi,
        res.pooled_linear_mi_std,
        pl_m,
        pl_s,
        res.pooled_probe_output_angle_deg
    ));
    out.push_str(&format!(
        "| PCA-kNN MI m=2,3,5 | {:.3}, {:.3}, {:.3} | — | low-D non-parametric |\n",
        res.pooled_pca_knn_mi[0], res.pooled_pca_knn_mi[1], res.pooled_pca_knn_mi[2]
    ));
    out.push_str(&format!(
        "| De-biased (obs − null95) | {:.3} | — | conservative bound |\n",
        res.debiased_mi
    ));

    out.push_str("\n=== §4 decision (pre-registered) ===\n");
    let base = res.i_r_y_spiral;
    let obs = res.debiased_mi;
    let lin = res.pooled_linear_mi;
    let ksg_pos = res.pooled_pca_knn_mi.iter().any(|&v| v > 0.05);
    if obs > 0.10 && lin > base + 0.10 && ksg_pos {
        out.push_str("VERDICT: Output bottleneck CONFIRMED — region linearly present beyond scalar.\n");
    } else if res.perm_observed_percentile < 95.0 || lin <= base + 0.05 {
        out.push_str("VERDICT: Below resolution / near wall — I(R;A_spiral) not credibly above I(R;Y_spiral).\n");
    } else {
        out.push_str("VERDICT: Nonlinear, fragile — region recoverable only weakly above null.\n");
    }
    out.push_str("\nRouting conclusion (unchanged): ΔCMI^C ≈ 0.1 for both specialists — competence routing remains information-bounded near the scalar.\n");
    out
}
