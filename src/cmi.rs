//! Conditional mutual information estimators for Task E dissociation measurement.
//! See docs/CMI_MEASUREMENT_SPEC.md (companion to COMPETENCE_ROUTING_SPEC).

use rand::Rng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

const LN2: f32 = std::f32::consts::LN_2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CmiPointRecord {
    pub seed: u64,
    pub x: f32,
    pub y_coord: f32,
    pub r: f32,
    pub region: u8,
    pub c_spiral: u8,
    pub c_circles: u8,
    pub y_spiral: f32,
    pub y_circles: f32,
    pub a_spiral: Vec<f32>,
    pub a_circles: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct CmiSeedEstimates {
    pub seed: u64,
    pub h_r: f32,
    pub i_r_y_spiral_mlp: f32,
    pub i_r_y_circles_mlp: f32,
    pub i_r_a_spiral_mlp: f32,
    pub i_r_a_circles_mlp: f32,
    pub delta_cmi_spiral_mlp: f32,
    pub delta_cmi_circles_mlp: f32,
    pub i_r_joint_y_mlp: f32,
    pub i_c_a_spiral_mlp: f32,
    pub i_c_y_spiral_mlp: f32,
    pub delta_cmi_c_spiral_mlp: f32,
    pub i_c_a_circles_mlp: f32,
    pub i_c_y_circles_mlp: f32,
    pub delta_cmi_c_circles_mlp: f32,
    pub i_r_a_spiral_knn: f32,
    pub i_r_a_circles_knn: f32,
    pub i_r_y_spiral_knn: f32,
    pub i_r_y_circles_knn: f32,
    pub backend_disagreement_spiral: f32,
    pub backend_disagreement_circles: f32,
}

pub fn binary_entropy(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
}

pub fn empirical_h_r(labels: &[u8]) -> f32 {
    if labels.is_empty() {
        return 0.0;
    }
    let p = labels.iter().filter(|&&l| l == 1).count() as f32 / labels.len() as f32;
    binary_entropy(p)
}

pub fn cross_entropy_bits(labels: &[u8], probs: &[f32]) -> f32 {
    if labels.is_empty() {
        return 0.0;
    }
    let mut nll = 0.0f32;
    for (&y, &p) in labels.iter().zip(probs.iter()) {
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        let py = if y == 1 { p } else { 1.0 - p };
        nll -= py.ln();
    }
    nll / labels.len() as f32 / LN2
}

pub fn mi_from_ce(h_r: f32, h_r_given_s: f32) -> f32 {
    (h_r - h_r_given_s).max(0.0)
}

fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

pub struct SmallMlp {
    input_dim: usize,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
}

impl SmallMlp {
    pub fn random(input_dim: usize, rng: &mut StdRng) -> Self {
        let hidden = 16usize;
        let scale1 = (2.0 / (input_dim + hidden) as f32).sqrt();
        let scale2 = (2.0 / (hidden + 1) as f32).sqrt();
        let w1 = (0..input_dim)
            .map(|_| {
                (0..hidden)
                    .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale1)
                    .collect()
            })
            .collect();
        Self {
            input_dim,
            w1,
            b1: vec![0.0; hidden],
            w2: (0..hidden)
                .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale2)
                .collect(),
            b2: 0.0,
        }
    }

    pub fn predict_proba(&self, x: &[f32]) -> f32 {
        let hidden = self.b1.len();
        let mut h = vec![0.0f32; hidden];
        for j in 0..hidden {
            let mut z = self.b1[j];
            for i in 0..self.input_dim.min(x.len()) {
                z += self.w1[i][j] * x[i];
            }
            h[j] = z.max(0.0);
        }
        let mut logit = self.b2;
        for (j, hj) in h.iter().enumerate() {
            logit += self.w2[j] * hj;
        }
        sigmoid(logit)
    }

    pub fn train(&mut self, features: &[Vec<f32>], labels: &[u8], lr: f32, epochs: usize) {
        for _ in 0..epochs {
            for (x, &y) in features.iter().zip(labels.iter()) {
                let hidden = self.b1.len();
                let mut h = vec![0.0f32; hidden];
                for j in 0..hidden {
                    let mut z = self.b1[j];
                    for i in 0..self.input_dim.min(x.len()) {
                        z += self.w1[i][j] * x[i];
                    }
                    h[j] = z.max(0.0);
                }
                let mut logit = self.b2;
                for (j, hj) in h.iter().enumerate() {
                    logit += self.w2[j] * hj;
                }
                let p = sigmoid(logit);
                let target = if y == 1 { 1.0 } else { 0.0 };
                let err = p - target;
                for j in 0..hidden {
                    let dh = if h[j] > 0.0 { err * self.w2[j] } else { 0.0 };
                    self.w2[j] -= lr * err * h[j];
                    for i in 0..self.input_dim.min(x.len()) {
                        self.w1[i][j] -= lr * dh * x[i];
                    }
                    self.b1[j] -= lr * dh;
                }
                self.b2 -= lr * err;
            }
        }
    }
}

fn stratified_split(
    n: usize,
    labels: &[u8],
    train_frac: f32,
    rng: &mut StdRng,
) -> (Vec<usize>, Vec<usize>) {
    let mut idx0 = Vec::new();
    let mut idx1 = Vec::new();
    for i in 0..n {
        if labels[i] == 1 {
            idx1.push(i);
        } else {
            idx0.push(i);
        }
    }
    idx0.shuffle(rng);
    idx1.shuffle(rng);
    let n_train0 = ((idx0.len() as f32) * train_frac).round() as usize;
    let n_train1 = ((idx1.len() as f32) * train_frac).round() as usize;
    let mut train = Vec::new();
    let mut test = Vec::new();
    train.extend(idx0.iter().take(n_train0));
    test.extend(idx0.iter().skip(n_train0));
    train.extend(idx1.iter().take(n_train1));
    test.extend(idx1.iter().skip(n_train1));
    train.shuffle(rng);
    test.shuffle(rng);
    (train, test)
}

fn mlp_region_mi(
    features: &[Vec<f32>],
    labels: &[u8],
    rng: &mut StdRng,
) -> (f32, f32) {
    if features.is_empty() || labels.is_empty() {
        return (0.0, 0.0);
    }
    let h_r = empirical_h_r(labels);
    let (train_idx, test_idx) = stratified_split(features.len(), labels, 0.6, rng);
    let train_x: Vec<_> = train_idx.iter().map(|&i| features[i].clone()).collect();
    let train_y: Vec<_> = train_idx.iter().map(|&i| labels[i]).collect();
    let test_x: Vec<_> = test_idx.iter().map(|&i| features[i].clone()).collect();
    let test_y: Vec<_> = test_idx.iter().map(|&i| labels[i]).collect();
    if train_x.is_empty() || test_x.is_empty() {
        return (0.0, h_r);
    }
    let dim = features[0].len().max(1);
    let mut mlp = SmallMlp::random(dim, rng);
    mlp.train(&train_x, &train_y, 0.05, 120);
    let probs: Vec<f32> = test_x.iter().map(|x| mlp.predict_proba(x)).collect();
    let h_r_given = cross_entropy_bits(&test_y, &probs);
    (mi_from_ce(h_r, h_r_given), h_r_given)
}

fn knn_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn knn_region_mi(features: &[Vec<f32>], labels: &[u8], k: usize, rng: &mut StdRng) -> f32 {
    if features.len() < k + 2 {
        return 0.0;
    }
    let h_r = empirical_h_r(labels);
    let (train_idx, test_idx) = stratified_split(features.len(), labels, 0.6, rng);
    if test_idx.is_empty() {
        return 0.0;
    }
    let mut probs = Vec::with_capacity(test_idx.len());
    let mut test_y = Vec::with_capacity(test_idx.len());
    for &ti in &test_idx {
        let mut dists: Vec<(f32, u8)> = train_idx
            .iter()
            .map(|&j| (knn_distance(&features[ti], &features[j]), labels[j]))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let neighbors = dists.iter().take(k).collect::<Vec<_>>();
        let pos = neighbors.iter().filter(|(_, y)| *y == 1).count() as f32;
        let p = (pos + 1.0) / (neighbors.len() as f32 + 2.0);
        probs.push(p);
        test_y.push(labels[ti]);
    }
    let h_r_given = cross_entropy_bits(&test_y, &probs);
    mi_from_ce(h_r, h_r_given)
}

pub fn histogram_1d_mi(scalars: &[f32], labels: &[u8], n_bins: usize, rng: &mut StdRng) -> f32 {
    if scalars.is_empty() {
        return 0.0;
    }
    let h_r = empirical_h_r(labels);
    let (train_idx, test_idx) = stratified_split(scalars.len(), labels, 0.6, rng);
    if test_idx.is_empty() || train_idx.is_empty() {
        return 0.0;
    }
    let train_s: Vec<f32> = train_idx.iter().map(|&i| scalars[i]).collect();
    let train_y: Vec<u8> = train_idx.iter().map(|&i| labels[i]).collect();
    let min = train_s.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = train_s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let span = (max - min).max(1e-4);
    let mut counts = vec![0usize; n_bins];
    let mut pos_counts = vec![0usize; n_bins];
    for (&s, &y) in train_s.iter().zip(train_y.iter()) {
        let b = (((s - min) / span) * n_bins as f32).floor() as usize;
        let b = b.min(n_bins - 1);
        counts[b] += 1;
        if y == 1 {
            pos_counts[b] += 1;
        }
    }
    let mut probs = Vec::with_capacity(test_idx.len());
    let mut test_y = Vec::with_capacity(test_idx.len());
    for &ti in &test_idx {
        let s = scalars[ti];
        let b = (((s - min) / span) * n_bins as f32).floor() as usize;
        let b = b.min(n_bins - 1);
        let p = (pos_counts[b] as f32 + 1.0) / (counts[b] as f32 + 2.0);
        probs.push(p);
        test_y.push(labels[ti]);
    }
    let h_r_given = cross_entropy_bits(&test_y, &probs);
    mi_from_ce(h_r, h_r_given)
}

pub fn estimate_cmi_seed(records: &[CmiPointRecord], seed: u64) -> CmiSeedEstimates {
    let seed_records: Vec<_> = records.iter().filter(|r| r.seed == seed).collect();
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(77_001));

    let region: Vec<u8> = seed_records.iter().map(|r| r.region).collect();
    let c_spiral: Vec<u8> = seed_records.iter().map(|r| r.c_spiral).collect();
    let c_circles: Vec<u8> = seed_records.iter().map(|r| r.c_circles).collect();
    let y_spiral: Vec<f32> = seed_records.iter().map(|r| r.y_spiral).collect();
    let y_circles: Vec<f32> = seed_records.iter().map(|r| r.y_circles).collect();
    let a_spiral: Vec<Vec<f32>> = seed_records.iter().map(|r| r.a_spiral.clone()).collect();
    let a_circles: Vec<Vec<f32>> = seed_records.iter().map(|r| r.a_circles.clone()).collect();
    let joint_y: Vec<Vec<f32>> = seed_records
        .iter()
        .map(|r| vec![r.y_spiral, r.y_circles])
        .collect();

    let h_r = empirical_h_r(&region);

    let (i_r_y_spiral_mlp, _) = mlp_region_mi(
        &y_spiral.iter().map(|&v| vec![v]).collect::<Vec<_>>(),
        &region,
        &mut rng,
    );
    let (i_r_y_circles_mlp, _) =
        mlp_region_mi(&y_circles.iter().map(|&v| vec![v]).collect::<Vec<_>>(), &region, &mut rng);
    let (i_r_a_spiral_mlp, _) = mlp_region_mi(&a_spiral, &region, &mut rng);
    let (i_r_a_circles_mlp, _) = mlp_region_mi(&a_circles, &region, &mut rng);
    let (i_r_joint_y_mlp, _) = mlp_region_mi(&joint_y, &region, &mut rng);

    let (i_c_a_spiral_mlp, _) = mlp_region_mi(&a_spiral, &c_spiral, &mut rng);
    let (i_c_y_spiral_mlp, _) = mlp_region_mi(
        &y_spiral.iter().map(|&v| vec![v]).collect::<Vec<_>>(),
        &c_spiral,
        &mut rng,
    );
    let (i_c_a_circles_mlp, _) = mlp_region_mi(&a_circles, &c_circles, &mut rng);
    let (i_c_y_circles_mlp, _) = mlp_region_mi(
        &y_circles.iter().map(|&v| vec![v]).collect::<Vec<_>>(),
        &c_circles,
        &mut rng,
    );

    let i_r_a_spiral_knn = knn_region_mi(&a_spiral, &region, 7, &mut rng);
    let i_r_a_circles_knn = knn_region_mi(&a_circles, &region, 7, &mut rng);
    let i_r_y_spiral_knn = histogram_1d_mi(&y_spiral, &region, 12, &mut rng);
    let i_r_y_circles_knn = histogram_1d_mi(&y_circles, &region, 12, &mut rng);

    CmiSeedEstimates {
        seed,
        h_r,
        i_r_y_spiral_mlp,
        i_r_y_circles_mlp,
        i_r_a_spiral_mlp,
        i_r_a_circles_mlp,
        delta_cmi_spiral_mlp: i_r_a_spiral_mlp - i_r_y_spiral_mlp,
        delta_cmi_circles_mlp: i_r_a_circles_mlp - i_r_y_circles_mlp,
        i_r_joint_y_mlp,
        i_c_a_spiral_mlp,
        i_c_y_spiral_mlp,
        delta_cmi_c_spiral_mlp: i_c_a_spiral_mlp - i_c_y_spiral_mlp,
        i_c_a_circles_mlp,
        i_c_y_circles_mlp,
        delta_cmi_c_circles_mlp: i_c_a_circles_mlp - i_c_y_circles_mlp,
        i_r_a_spiral_knn,
        i_r_a_circles_knn,
        i_r_y_spiral_knn,
        i_r_y_circles_knn,
        backend_disagreement_spiral: (i_r_a_spiral_mlp - i_r_a_spiral_knn).abs(),
        backend_disagreement_circles: (i_r_a_circles_mlp - i_r_a_circles_knn).abs(),
    }
}

pub fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    if values.len() == 1 {
        return (mean, 0.0);
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;
    (mean, var.sqrt())
}

pub fn format_cmi_report(estimates: &[CmiSeedEstimates]) -> String {
    let mut out = String::new();
    if estimates.is_empty() {
        return out;
    }

    let agg = |f: fn(&CmiSeedEstimates) -> f32| -> String {
        let vals: Vec<f32> = estimates.iter().map(f).collect();
        let (m, s) = mean_std(&vals);
        format!("{:.3} ± {:.3}", m, s)
    };

    let agg_ms = |f: fn(&CmiSeedEstimates) -> f32| -> (f32, f32) {
        let vals: Vec<f32> = estimates.iter().map(f).collect();
        mean_std(&vals)
    };

    out.push_str("=== CMI measurement (per-seed held-out classifiers, bits) ===\n\n");
    out.push_str("| Quantity | Mean ± std (backend-range) | §6 prediction |\n");
    out.push_str("| -------- | -------------------------- | ------------- |\n");
    out.push_str(&format!(
        "| H(R) | {} | ~1.0 (balanced) |\n",
        agg(|e| e.h_r)
    ));
    out.push_str(&format!(
        "| I(R; Y_spiral) MLP | {} | ≈ 0 |\n",
        agg(|e| e.i_r_y_spiral_mlp)
    ));
    out.push_str(&format!(
        "| I(R; Y_circles) MLP | {} | > 0 |\n",
        agg(|e| e.i_r_y_circles_mlp)
    ));
    let (ias_m, ias_s) = agg_ms(|e| e.i_r_a_spiral_mlp);
    let (bad_s_m, _) = agg_ms(|e| e.backend_disagreement_spiral);
    out.push_str(&format!(
        "| I(R; A_spiral) MLP | {:.3} ± {:.3} (±{:.3}) | ≈ I(R;Y_spiral) |\n",
        ias_m, ias_s, bad_s_m
    ));
    let (iac_m, iac_s) = agg_ms(|e| e.i_r_a_circles_mlp);
    let (bad_c_m, _) = agg_ms(|e| e.backend_disagreement_circles);
    out.push_str(&format!(
        "| I(R; A_circles) MLP | {:.3} ± {:.3} (±{:.3}) | ≈ I(R;Y_circles) |\n",
        iac_m, iac_s, bad_c_m
    ));
    out.push_str(&format!(
        "| **ΔCMI_spiral = I(R;A)−I(R;Y)** | **{}** | **≈ 0** |\n",
        agg(|e| e.delta_cmi_spiral_mlp)
    ));
    out.push_str(&format!(
        "| **ΔCMI_circles = I(R;A)−I(R;Y)** | **{}** | **≈ 0** |\n",
        agg(|e| e.delta_cmi_circles_mlp)
    ));
    out.push_str(&format!(
        "| I(R; (Y₁,Y₂)) MLP | {} | appreciable |\n",
        agg(|e| e.i_r_joint_y_mlp)
    ));
    out.push_str(&format!(
        "| ΔCMI^C_spiral (correctness) | {} | ≈ 0 — **load-bearing** |\n",
        agg(|e| e.delta_cmi_c_spiral_mlp)
    ));
    out.push_str(&format!(
        "| ΔCMI^C_circles (correctness) | {} | ≈ 0 |\n",
        agg(|e| e.delta_cmi_c_circles_mlp)
    ));
    out.push_str(&format!(
        "| Backend disagree I(R;A_spiral) | {} | < 0.1 bit |\n",
        agg(|e| e.backend_disagreement_spiral)
    ));
    out.push_str(&format!(
        "| Backend disagree I(R;A_circles) | {} | < 0.1 bit |\n",
        agg(|e| e.backend_disagreement_circles)
    ));

    out.push_str("\n| seed | ΔCMI_spiral | ΔCMI_circles | I(R;joint Y) |\n");
    out.push_str("| ---- | ----------- | ------------ | ------------ |\n");
    for e in estimates {
        out.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} |\n",
            e.seed, e.delta_cmi_spiral_mlp, e.delta_cmi_circles_mlp, e.i_r_joint_y_mlp
        ));
    }

    let delta_spiral: Vec<f32> = estimates.iter().map(|e| e.delta_cmi_spiral_mlp).collect();
    let delta_circles: Vec<f32> = estimates.iter().map(|e| e.delta_cmi_circles_mlp).collect();
    let (ds_m, _) = mean_std(&delta_spiral);
    let (dc_m, _) = mean_std(&delta_circles);
    let joint: Vec<f32> = estimates.iter().map(|e| e.i_r_joint_y_mlp).collect();
    let (j_m, _) = mean_std(&joint);

    out.push_str("\n=== Falsification check (§7) ===\n");
    if ds_m > 0.15 || dc_m > 0.15 {
        out.push_str("⚠ ΔCMI clearly > 0.15 — entanglement thesis may be WRONG; reopen competence routing.\n");
    } else {
        out.push_str("ΔCMI ≈ 0 — entanglement thesis supported (conservative held-out classifiers).\n");
    }
    if j_m < 0.1 {
        out.push_str("⚠ I(R;(Y₁,Y₂)) near zero — cross-specialist joint does not carry region; bivector direction dead.\n");
    } else {
        out.push_str("I(R;(Y₁,Y₂)) appreciable — cross-specialist direction remains motivated.\n");
    }

    out
}
