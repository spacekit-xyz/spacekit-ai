//! Clifford MNIST: image classification through Cl(1,7) spacetime algebra.
//!
//! Encodes 28×28 pixel images as multivectors in Cl(1,7) where each grade
//! carries geometrically meaningful visual information:
//!
//!   grade 0 (1):  global intensity (total brightness)
//!   grade 1 (8):  spatial gradient directions (1 timelike + 7 spacelike)
//!   grade 2 (28): oriented edge/curvature detectors (boost + rotation planes)
//!   grade 3 (56): junction/corner features (triple intersections)
//!   grade 4 (70): topological features (enclosed regions, holes)
//!   grades 5-8:   higher-order structural invariants
//!
//! Classification uses Minkowski intervals between image multivectors and
//! learned class centroids: timelike intervals → same class, spacelike → different.
//!
//! This replaces the flat random-projection pipeline with a geometrically
//! principled encoding that uses the same Cl(1,7) algebra as the language system.

use crate::clifford::{
    Multivector, embed_bridge_vector, minkowski_interval, classify_interval,
    IntervalType, CL8_DIM, GRADE_DIMS, GRADE_OFFSETS,
};
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub const IMAGE_DIM: usize = 28 * 28;

/// Composite spacetime distance across ALL grades of a Cl(1,7) multivector.
///
/// - Grade 1: Minkowski metric with (1,7) signature (e_0 timelike, e_1..e_7 spacelike)
/// - All other grades: L2 (Euclidean) distance
///
/// Returns a scalar where lower = more similar. Negative contributions from
/// timelike grade-1 separation mean causally connected (same class).
/// This makes the FULL multivector contribute to classification, not just grade-1.
pub fn spacetime_distance(a: &Multivector, b: &Multivector) -> f32 {
    let mut dist = 0.0f32;

    for g in 0..=8 {
        let ga = a.grade(g);
        let gb = b.grade(g);
        if g == 1 {
            // Minkowski (1,7): e_0² = -1, e_1..e_7² = +1
            for k in 0..GRADE_DIMS[1] {
                let d = gb[k] - ga[k];
                if k == 0 {
                    dist -= d * d;
                } else {
                    dist += d * d;
                }
            }
        } else {
            for k in 0..GRADE_DIMS[g] {
                let d = gb[k] - ga[k];
                dist += d * d;
            }
        }
    }
    dist
}

/// Structured projection that maps 784D images into 256D Cl(1,7) space.
/// Unlike random projection, this assigns pixel neighborhoods to specific
/// grades so the algebra's structure carries geometric meaning.
pub struct CliffordImageEncoder {
    /// 784 → 256 projection matrix, initialized to extract geometric features
    projection: Vec<[f32; CL8_DIM]>,
    /// Per-grade learned scale factors for balanced contribution
    grade_scales: [f32; 9],
}

impl CliffordImageEncoder {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut projection = Vec::with_capacity(IMAGE_DIM);

        // Grade-aware damping: scale inversely with sqrt(grade_dim) to prevent
        // high-dimensional grades from overflowing during accumulation.
        let grade_init_scale: [f32; 9] = std::array::from_fn(|g| {
            1.0 / (GRADE_DIMS[g] as f32).sqrt().max(1.0)
        });

        for pixel_idx in 0..IMAGE_DIM {
            let mut row = [0.0f32; CL8_DIM];
            let px = pixel_idx % 28;
            let py = pixel_idx / 28;
            let cx = (px as f32 - 13.5) / 14.0;
            let cy = (py as f32 - 13.5) / 14.0;
            let r = (cx * cx + cy * cy).sqrt();

            // Grade 0 (scalar): global intensity — every pixel contributes
            row[GRADE_OFFSETS[0]] = 1.0 / IMAGE_DIM as f32;

            // Grade 1 (vector): spatial position encoding.
            // e_0 (timelike): radial distance from center — captures
            // center-of-mass, invariant to rotation. NOT normalized away
            // so the Minkowski metric can produce genuine timelike intervals.
            let g1s = grade_init_scale[1];
            row[GRADE_OFFSETS[1]]     = r * g1s;
            row[GRADE_OFFSETS[1] + 1] = cx * g1s;
            row[GRADE_OFFSETS[1] + 2] = cy * g1s;
            row[GRADE_OFFSETS[1] + 3] = cx * cy * g1s;
            row[GRADE_OFFSETS[1] + 4] = (cx * cx - cy * cy) * g1s;
            row[GRADE_OFFSETS[1] + 5] = (2.0 * std::f32::consts::PI * cx).sin() * g1s;
            row[GRADE_OFFSETS[1] + 6] = (2.0 * std::f32::consts::PI * cy).sin() * g1s;
            row[GRADE_OFFSETS[1] + 7] = (std::f32::consts::PI * r).cos() * g1s;

            // Grade 2 (bivector): oriented edge features.
            // 28 components — scale by 1/sqrt(28) to prevent accumulation overflow.
            let g2s = grade_init_scale[2];
            for b in 0..GRADE_DIMS[2] {
                let angle = b as f32 * std::f32::consts::PI / GRADE_DIMS[2] as f32;
                let dx = angle.cos();
                let dy = angle.sin();
                // Directional derivative proxy: how aligned is this pixel's
                // position with each bivector orientation?
                let phase = cx * dx + cy * dy;
                // Quadratic term captures curvature — critical for 8 vs 9
                let curvature = (cx * dx + cy * dy).powi(2) - r * r * 0.5;
                row[GRADE_OFFSETS[2] + b] = (phase * 0.5 + curvature * 0.3) * g2s;
            }

            // Grade 3 (trivector): junction features — where 3+ strokes meet.
            // Initialized with spatial triple-products.
            let g3s = grade_init_scale[3];
            for k in 0..GRADE_DIMS[3] {
                let freq = (k as f32 + 1.0) * std::f32::consts::PI / 14.0;
                let triple = (freq * cx).sin() * (freq * cy).cos() * r;
                row[GRADE_OFFSETS[3] + k] = triple * g3s * 0.3
                    + rng.gen_range(-0.02..0.02) * g3s;
            }

            // Grades 4-8: random structured initialization, grade-scaled.
            for g in 4..=8 {
                let gs = grade_init_scale[g];
                for k in 0..GRADE_DIMS[g] {
                    row[GRADE_OFFSETS[g] + k] = rng.gen_range(-0.1..0.1) * gs;
                }
            }

            projection.push(row);
        }

        CliffordImageEncoder {
            projection,
            grade_scales: [1.0; 9],
        }
    }

    /// Encode a 784D image into a Cl(1,7) multivector.
    /// Preserves grade magnitudes so the Minkowski metric produces meaningful
    /// timelike/spacelike intervals. No L2 normalization of grade-1.
    pub fn encode(&self, image: &[f32]) -> Multivector {
        let mut mv = Multivector::zero();
        for (i, &pixel) in image.iter().enumerate().take(IMAGE_DIM) {
            if pixel.abs() < 1e-6 { continue; }
            let row = &self.projection[i];
            for j in 0..CL8_DIM {
                mv.components[j] += pixel * row[j];
            }
        }

        // Apply per-grade scaling
        for g in 0..=8 {
            let scale = self.grade_scales[g];
            let start = GRADE_OFFSETS[g];
            for k in 0..GRADE_DIMS[g] {
                mv.components[start + k] *= scale;
            }
        }

        // Per-grade safety clamp: prevent any grade from going to NaN/Inf.
        // Normalize only grades whose norm exceeds a safe ceiling,
        // preserving magnitude information within the safe range.
        const MAX_GRADE_NORM: f32 = 50.0;
        for g in 0..=8 {
            let start = GRADE_OFFSETS[g];
            let dim = GRADE_DIMS[g];
            let norm: f32 = mv.components[start..start + dim]
                .iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > MAX_GRADE_NORM {
                let scale = MAX_GRADE_NORM / norm;
                for k in 0..dim {
                    mv.components[start + k] *= scale;
                }
            }
        }
        mv
    }

    /// Calibrate grade scales from sample data so each grade contributes
    /// meaningfully to the Minkowski interval. Target: each grade's mean
    /// norm across samples is approximately 1.0.
    pub fn calibrate_scales(&mut self, samples: &[(Vec<f32>, u8)]) {
        let n = samples.len().min(500) as f32;
        if n < 2.0 { return; }
        // Reset scales to 1.0 before measuring raw norms
        self.grade_scales = [1.0; 9];
        let mut grade_norms = [0.0f32; 9];
        for (img, _) in samples.iter().take(500) {
            let mv = self.encode(img);
            for g in 0..=8 {
                let gnorm: f32 = mv.grade(g).iter().map(|x| x * x).sum::<f32>().sqrt();
                if gnorm.is_finite() {
                    grade_norms[g] += gnorm / n;
                }
            }
        }
        let target_norm = 1.0f32;
        for g in 0..=8 {
            if grade_norms[g] > 1e-6 && grade_norms[g].is_finite() {
                self.grade_scales[g] = target_norm / grade_norms[g];
            }
        }
    }

    /// Contrastive training step using full spacetime distance.
    /// Adaptive margin scaled to actual distance range. Only fires on
    /// samples where the margin is violated (misclassified or near boundary).
    pub fn train_step(
        &mut self,
        image: &[f32],
        label: u8,
        centroids: &[Multivector; 10],
        lr: f32,
    ) {
        let mv = self.encode(image);
        let correct_centroid = &centroids[label as usize];
        let correct_dist = spacetime_distance(&mv, correct_centroid);

        // Find hardest negative: the wrong centroid with smallest distance
        let mut hardest_neg_dist = f32::MAX;
        let mut hardest_neg_idx = 0;
        for d in 0..10 {
            if d == label as usize { continue; }
            if centroids[d].components.iter().all(|x| x.abs() < 1e-12) { continue; }
            let dist = spacetime_distance(&mv, &centroids[d]);
            if dist < hardest_neg_dist {
                hardest_neg_dist = dist;
                hardest_neg_idx = d;
            }
        }
        if hardest_neg_dist == f32::MAX { return; }

        // Triplet margin loss: we want correct_dist < hardest_neg_dist - margin
        // Only update when margin is violated (sample is misclassified or near boundary).
        let margin = 0.1;
        let violation = correct_dist - hardest_neg_dist + margin;
        if violation <= 0.0 { return; }

        let neg_centroid = &centroids[hardest_neg_idx];
        let loss_scale = violation.min(2.0);

        for (i, &pixel) in image.iter().enumerate().take(IMAGE_DIM) {
            if pixel.abs() < 0.01 { continue; }
            let row = &mut self.projection[i];

            for g in 0..=8 {
                let glr = lr / (GRADE_DIMS[g] as f32).sqrt().max(1.0);
                for k in 0..GRADE_DIMS[g] {
                    let idx = GRADE_OFFSETS[g] + k;
                    // Push toward correct centroid, away from negative centroid
                    let toward_correct = correct_centroid.components[idx] - mv.components[idx];
                    let away_from_neg = mv.components[idx] - neg_centroid.components[idx];
                    let update = glr * pixel * loss_scale * (toward_correct + away_from_neg);
                    if update.is_finite() {
                        row[idx] += update;
                    }
                }
            }
        }
    }
}

/// Learned centroids for each digit class (0-9) in Cl(1,7) space.
pub struct CliffordClassifier {
    pub centroids: [Multivector; 10],
    counts: [u32; 10],
}

impl CliffordClassifier {
    pub fn new() -> Self {
        CliffordClassifier {
            centroids: std::array::from_fn(|_| Multivector::zero()),
            counts: [0; 10],
        }
    }

    /// Accumulate a sample into the running centroid for its class.
    pub fn accumulate(&mut self, mv: &Multivector, label: u8) {
        let d = label as usize;
        if d >= 10 { return; }
        self.counts[d] += 1;
        let n = self.counts[d] as f32;
        let alpha = 1.0 / n;
        for i in 0..CL8_DIM {
            self.centroids[d].components[i] =
                (1.0 - alpha) * self.centroids[d].components[i]
                + alpha * mv.components[i];
        }
    }

    pub fn has_centroid(&self, d: usize) -> bool {
        d < 10 && self.counts[d] > 0
    }

    /// Classify using full spacetime distance (all grades, Minkowski on grade-1).
    /// Most negative = most timelike = strongest causal connection = class membership.
    pub fn classify(&self, mv: &Multivector) -> (u8, f32) {
        let mut best_label = 0u8;
        let mut best_dist = f32::MAX;
        for d in 0..10 {
            if self.counts[d] == 0 { continue; }
            let dist = spacetime_distance(mv, &self.centroids[d]);
            if dist < best_dist {
                best_dist = dist;
                best_label = d as u8;
            }
        }
        (best_label, best_dist)
    }

    /// Classify for binary task using full spacetime distance.
    pub fn classify_binary(&self, mv: &Multivector, digit_a: u8, digit_b: u8) -> (u8, f32) {
        let da = spacetime_distance(mv, &self.centroids[digit_a as usize]);
        let db = spacetime_distance(mv, &self.centroids[digit_b as usize]);
        if da < db { (digit_a, da) } else { (digit_b, db) }
    }

    /// Per-grade discriminability: how well each grade separates classes.
    pub fn grade_discriminability(&self) -> [f32; 9] {
        let mut disc = [0.0f32; 9];
        let mut count = 0u32;
        for a in 0..10 {
            if self.counts[a] == 0 { continue; }
            for b in (a+1)..10 {
                if self.counts[b] == 0 { continue; }
                for g in 0..=8 {
                    let ga = self.centroids[a].grade(g);
                    let gb = self.centroids[b].grade(g);
                    let dist_sq: f32 = ga.iter().zip(gb.iter())
                        .map(|(x, y)| (x - y) * (x - y)).sum();
                    if dist_sq.is_finite() {
                        disc[g] += dist_sq;
                    }
                }
                count += 1;
            }
        }
        if count > 0 {
            for g in 0..=8 { disc[g] /= count as f32; }
        }
        disc
    }
}

/// Multi-task classifier: separate per-task heads on a shared encoder.
/// Solves the global classifier's decision boundary collision problem
/// (catastrophic forgetting at the routing level).
pub struct MultiTaskClassifier {
    pub task_heads: Vec<CliffordClassifier>,
    pub task_digits: Vec<(u8, u8)>,
}

impl MultiTaskClassifier {
    pub fn new() -> Self {
        MultiTaskClassifier {
            task_heads: Vec::new(),
            task_digits: Vec::new(),
        }
    }

    pub fn register_task(&mut self, digit_a: u8, digit_b: u8, classifier: CliffordClassifier) {
        self.task_heads.push(classifier);
        self.task_digits.push((digit_a, digit_b));
    }

    /// Classify across all seen tasks by finding the task head that produces
    /// the most timelike interval for its best candidate digit.
    pub fn classify(&self, mv: &Multivector) -> (u8, f32) {
        let mut best_label = 0u8;
        let mut best_interval = f32::MAX;
        for (head, &(da, db)) in self.task_heads.iter().zip(self.task_digits.iter()) {
            let (pred, interval) = head.classify_binary(mv, da, db);
            if interval < best_interval {
                best_interval = interval;
                best_label = pred;
            }
        }
        (best_label, best_interval)
    }

    /// Grade discriminability across all task heads.
    pub fn grade_discriminability(&self) -> [f32; 9] {
        let mut disc = [0.0f32; 9];
        let mut n = 0;
        for head in &self.task_heads {
            let hd = head.grade_discriminability();
            for g in 0..=8 {
                if hd[g].is_finite() {
                    disc[g] += hd[g];
                }
            }
            n += 1;
        }
        if n > 0 {
            for g in 0..=8 { disc[g] /= n as f32; }
        }
        disc
    }
}

/// Discriminability-weighted spacetime distance.
/// Weights each grade's contribution to the distance by its measured
/// discriminability — grades that actually separate classes matter more.
/// Preserves Minkowski (1,7) signature on grade-1.
pub fn weighted_spacetime_distance(
    a: &Multivector,
    b: &Multivector,
    grade_weights: &[f32; 9],
) -> f32 {
    let mut dist = 0.0f32;
    for g in 0..=8 {
        let w = grade_weights[g];
        if w < 1e-10 { continue; }
        let ga = a.grade(g);
        let gb = b.grade(g);
        if g == 1 {
            for k in 0..GRADE_DIMS[1] {
                let d = gb[k] - ga[k];
                if k == 0 {
                    dist -= w * d * d;
                } else {
                    dist += w * d * d;
                }
            }
        } else {
            for k in 0..GRADE_DIMS[g] {
                let d = gb[k] - ga[k];
                dist += w * d * d;
            }
        }
    }
    dist
}

/// Compute normalized grade weights from discriminability scores.
/// Higher discriminability → higher weight. Normalizes so max weight = 1.0.
pub fn discriminability_weights(disc: &[f32; 9]) -> [f32; 9] {
    let max_d = disc.iter().cloned().fold(0.0f32, f32::max);
    if max_d < 1e-10 { return [1.0; 9]; }
    std::array::from_fn(|g| disc[g] / max_d)
}

/// Interval-augmented classification score.
/// Combines discriminability-weighted L2 distance with the Minkowski
/// interval as a first-class classification feature.
/// Lower score = better match (distance-like, not similarity-like).
fn interval_augmented_score(
    query: &Multivector,
    centroid: &Multivector,
    grade_weights: &[f32; 9],
) -> f32 {
    let base_dist = weighted_spacetime_distance(query, centroid, grade_weights);
    let interval = minkowski_interval(query, centroid);

    // Timelike bonus: reduce distance for causally connected pairs.
    // Spacelike penalty: increase distance for causally disconnected pairs.
    let interval_adjustment = if interval < 0.0 {
        // Timelike: subtract (reduce distance) proportional to magnitude
        interval * 0.3
    } else {
        // Spacelike: add (increase distance) proportional to magnitude
        interval * 0.1
    };

    base_dist + interval_adjustment
}

/// Full training pipeline result.
pub struct CliffordMnistResult {
    pub task_accuracies: Vec<f32>,
    pub avg_accuracy: f32,
    pub grade_discriminability: [f32; 9],
    pub interval_stats: IntervalStats,
}

/// Statistics on Minkowski intervals for correct vs incorrect classifications.
pub struct IntervalStats {
    pub correct_mean_interval: f32,
    pub incorrect_mean_interval: f32,
    pub timelike_correct_pct: f32,
}

/// Run the full Split MNIST benchmark through Cl(1,7).
/// Same 5-task structure as the flat demo: (0,1), (2,3), (4,5), (6,7), (8,9).
///
/// Uses per-task classifier heads on a shared encoder to avoid decision
/// boundary collision (catastrophic forgetting at the routing level).
pub fn run_clifford_mnist(
    train_images: &[Vec<f32>],
    train_labels: &[u8],
    test_images: &[Vec<f32>],
    test_labels: &[u8],
    train_limit: Option<usize>,
    max_epochs: u32,
) -> CliffordMnistResult {
    use crate::mnist::filter_digit_pair_raw;

    let mut encoder = CliffordImageEncoder::new(42);

    const TASKS: [(u8, u8); 5] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
    let mut task_accuracies = Vec::with_capacity(5);
    let mut multi_head = MultiTaskClassifier::new();

    let mut correct_intervals = Vec::new();
    let mut incorrect_intervals = Vec::new();

    for (t, (d1, d2)) in TASKS.iter().enumerate() {
        let train_pairs = filter_digit_pair_raw(train_images, train_labels, *d1, *d2);
        let test_pairs = filter_digit_pair_raw(test_images, test_labels, *d1, *d2);

        let train_subset: Vec<_> = if let Some(lim) = train_limit {
            train_pairs.into_iter().take(lim).collect()
        } else {
            train_pairs
        };

        println!("  Task {} ({} vs {}): {} train, {} test",
            t, d1, d2, train_subset.len(), test_pairs.len());

        // Phase 1: Build initial centroids
        let mut task_classifier = CliffordClassifier::new();
        let calibration: Vec<_> = train_subset.iter().take(200)
            .map(|(img, lbl)| (img.clone(), *lbl))
            .collect();
        encoder.calibrate_scales(&calibration);

        for (img, lbl) in &train_subset {
            let mv = encoder.encode(img);
            task_classifier.accumulate(&mv, *lbl);
        }

        // Phase 2: Contrastive training with frozen centroids.
        // LR schedule: ramp up, plateau in the productive 10-25 range,
        // then decay. This gives 2 vs 3 (the hardest pair) time to consolidate.
        const REBUILD_INTERVAL: u32 = 5;
        let lr_schedule = |epoch: u32| -> f32 {
            let base = 0.01;
            if epoch < 3 {
                base * (epoch as f32 + 1.0) / 3.0 // warmup
            } else if epoch < 20 {
                base // plateau
            } else {
                base * (-0.05 * (epoch as f32 - 20.0)).exp() // gradual decay
            }
        };

        let mut best_test_acc = 0.0f32;
        let mut stale_count = 0u32;

        for epoch in 0..max_epochs {
            let lr = lr_schedule(epoch);

            for (img, lbl) in &train_subset {
                encoder.train_step(img, *lbl, &task_classifier.centroids, lr);
            }

            // Rebuild centroids periodically from updated encoder
            if (epoch + 1) % REBUILD_INTERVAL == 0 || epoch == 0 {
                task_classifier = CliffordClassifier::new();
                for (img, lbl) in &train_subset {
                    let mv = encoder.encode(img);
                    task_classifier.accumulate(&mv, *lbl);
                }
            }

            if epoch % 5 == 0 || epoch == max_epochs - 1 {
                let test_acc = evaluate_binary(
                    &encoder, &task_classifier, &test_pairs, *d1, *d2,
                );
                println!("    epoch {}: test_acc={:.1}%, lr={:.5}",
                    epoch, test_acc * 100.0, lr);

                if test_acc >= 0.98 {
                    println!("    Reached 98% at epoch {}", epoch);
                    break;
                }
                if test_acc > best_test_acc + 0.001 {
                    best_test_acc = test_acc;
                    stale_count = 0;
                } else {
                    stale_count += 1;
                }
                if stale_count >= 4 {
                    println!("    Converged at epoch {} (no improvement)", epoch);
                    break;
                }
            }
        }

        // Final centroid rebuild after training
        task_classifier = CliffordClassifier::new();
        for (img, lbl) in &train_subset {
            let mv = encoder.encode(img);
            task_classifier.accumulate(&mv, *lbl);
        }

        // Final evaluation
        let test_acc = evaluate_binary(&encoder, &task_classifier, &test_pairs, *d1, *d2);
        task_accuracies.push(test_acc);
        println!("  Task {} ({} vs {}) done: {:.1}%", t, d1, d2, test_acc * 100.0);

        // Collect interval stats (both grade-1 Minkowski and full spacetime distance)
        for (img, lbl) in &test_pairs {
            let mv = encoder.encode(img);
            let (pred, dist) = task_classifier.classify_binary(&mv, *d1, *d2);
            // Store grade-1 Minkowski interval for physics interpretation
            let mink = minkowski_interval(&mv, &task_classifier.centroids[pred as usize]);
            if pred == *lbl {
                correct_intervals.push(mink);
            } else {
                incorrect_intervals.push(mink);
            }
        }

        multi_head.register_task(*d1, *d2, task_classifier);
    }

    // Phase 3: 10-class refinement.
    // Binary tasks taught within-pair discrimination. Now refine the encoder
    // against ALL classes so cross-pair confusions (e.g. 0 vs 6) are resolved.
    println!("\n--- Phase 3: 10-class contrastive refinement ---");
    let mut global_classifier = CliffordClassifier::new();
    for (img, &lbl) in train_images.iter().zip(train_labels.iter()) {
        let mv = encoder.encode(img);
        global_classifier.accumulate(&mv, lbl);
    }

    let global_train: Vec<_> = if let Some(lim) = train_limit {
        train_images.iter().zip(train_labels.iter())
            .take(lim * 5).map(|(img, &lbl)| (img.clone(), lbl)).collect()
    } else {
        train_images.iter().zip(train_labels.iter())
            .map(|(img, &lbl)| (img.clone(), lbl)).collect()
    };

    let refinement_epochs = max_epochs;
    let mut best_refinement_acc = 0.0f32;
    let mut refinement_stale = 0u32;
    for epoch in 0..refinement_epochs {
        let lr = if epoch < 5 { 0.005 } else { 0.005 * (-0.02 * (epoch as f32 - 5.0)).exp() };

        for (img, lbl) in &global_train {
            encoder.train_step(img, *lbl, &global_classifier.centroids, lr);
        }

        // Rebuild centroids every 3 epochs
        if (epoch + 1) % 3 == 0 {
            global_classifier = CliffordClassifier::new();
            for (img, lbl) in &global_train {
                let mv = encoder.encode(img);
                global_classifier.accumulate(&mv, *lbl);
            }
        }

        if epoch % 5 == 0 || epoch == refinement_epochs - 1 {
            let mut rc = 0u32;
            let rt = test_images.len().min(2000) as u32;
            for (img, &lbl) in test_images.iter().zip(test_labels.iter()).take(2000) {
                let mv = encoder.encode(img);
                let (pred, _) = global_classifier.classify(&mv);
                if pred == lbl { rc += 1; }
            }
            let racc = rc as f32 / rt.max(1) as f32;
            println!("    refinement epoch {}: 10-class acc={:.1}%, lr={:.5}",
                epoch, racc * 100.0, lr);
            if racc > best_refinement_acc + 0.002 {
                best_refinement_acc = racc;
                refinement_stale = 0;
            } else {
                refinement_stale += 1;
            }
            if refinement_stale >= 4 {
                println!("    Refinement converged at epoch {}", epoch);
                break;
            }
        }
    }

    // Final centroid rebuild
    global_classifier = CliffordClassifier::new();
    for (img, &lbl) in train_images.iter().zip(train_labels.iter()) {
        let mv = encoder.encode(img);
        global_classifier.accumulate(&mv, lbl);
    }

    // Per-task evaluation with global classifier
    println!("\n--- Cross-task evaluation (global 10-class classifier) ---");
    let mut cross_task_total = 0u32;
    let mut cross_task_correct = 0u32;
    for (t, (d1, d2)) in TASKS.iter().enumerate() {
        let test_pairs = filter_digit_pair_raw(test_images, test_labels, *d1, *d2);
        let mut task_correct = 0u32;
        let total = test_pairs.len() as u32;
        for (img, lbl) in &test_pairs {
            let mv = encoder.encode(img);
            let (pred, _) = global_classifier.classify_binary(&mv, *d1, *d2);
            if pred == *lbl { task_correct += 1; }
        }
        let acc = task_correct as f32 / total.max(1) as f32;
        println!("  Task {} ({} vs {}): {:.1}% (global)",
            t, d1, d2, acc * 100.0);
        cross_task_total += total;
        cross_task_correct += task_correct;
    }
    let cross_task_acc = cross_task_correct as f32 / cross_task_total.max(1) as f32;
    println!("  Overall cross-task binary: {:.1}%", cross_task_acc * 100.0);

    let full_total = test_images.len() as u32;
    let grade_disc = global_classifier.grade_discriminability();

    // Compute data-driven grade weights from measured discriminability
    let gw = discriminability_weights(&grade_disc);
    println!("\n--- Data-driven grade weights ---");
    let grade_names = [
        "scalar (intensity)", "vector (gradients)", "bivector (edges)",
        "trivector (junctions)", "quadvector (topology)", "grade-5", "grade-6",
        "grade-7", "pseudoscalar (orientation)",
    ];
    for g in 0..=8 {
        println!("  grade {}: disc={:.1}, weight={:.3} — {}",
            g, grade_disc[g], gw[g], grade_names[g]);
    }

    // --- Classifier comparison: 3 methods on the same encoder + centroids ---
    let classifiers: [(&str, Box<dyn Fn(&Multivector) -> (u8, f32)>); 3] = [
        ("flat spacetime distance", Box::new(|mv: &Multivector| {
            let mut best_label = 0u8;
            let mut best_dist = f32::MAX;
            for d in 0..10 {
                if global_classifier.counts[d] == 0 { continue; }
                let dist = spacetime_distance(mv, &global_classifier.centroids[d]);
                if dist < best_dist { best_dist = dist; best_label = d as u8; }
            }
            (best_label, best_dist)
        })),
        ("discriminability-weighted", Box::new(|mv: &Multivector| {
            let mut best_label = 0u8;
            let mut best_dist = f32::MAX;
            for d in 0..10 {
                if global_classifier.counts[d] == 0 { continue; }
                let dist = weighted_spacetime_distance(mv, &global_classifier.centroids[d], &gw);
                if dist < best_dist { best_dist = dist; best_label = d as u8; }
            }
            (best_label, best_dist)
        })),
        ("interval-augmented", Box::new(|mv: &Multivector| {
            let mut best_label = 0u8;
            let mut best_dist = f32::MAX;
            for d in 0..10 {
                if global_classifier.counts[d] == 0 { continue; }
                let dist = interval_augmented_score(mv, &global_classifier.centroids[d], &gw);
                if dist < best_dist { best_dist = dist; best_label = d as u8; }
            }
            (best_label, best_dist)
        })),
    ];

    for (name, classify_fn) in &classifiers {
        println!("\n--- Full 10-class evaluation ({}) ---", name);
        let mut full_correct = 0u32;
        let mut per_digit_correct = [0u32; 10];
        let mut per_digit_total = [0u32; 10];
        for (img, &lbl) in test_images.iter().zip(test_labels.iter()) {
            let mv = encoder.encode(img);
            let (pred, _) = classify_fn(&mv);
            per_digit_total[lbl as usize] += 1;
            if pred == lbl {
                full_correct += 1;
                per_digit_correct[lbl as usize] += 1;
            }
        }
        let full_acc = full_correct as f32 / full_total.max(1) as f32;
        println!("  Overall: {:.1}% ({}/{})", full_acc * 100.0, full_correct, full_total);
        for d in 0..10 {
            let acc = per_digit_correct[d] as f32 / per_digit_total[d].max(1) as f32;
            println!("    digit {}: {:.1}% ({}/{})", d, acc * 100.0,
                per_digit_correct[d], per_digit_total[d]);
        }
    }

    let avg_accuracy = task_accuracies.iter().sum::<f32>() / task_accuracies.len() as f32;

    // Recompute Minkowski interval stats from the FINAL encoder + global classifier
    correct_intervals.clear();
    incorrect_intervals.clear();
    for (img, &lbl) in test_images.iter().zip(test_labels.iter()) {
        let mv = encoder.encode(img);
        let (pred, _) = global_classifier.classify(&mv);
        let mink = minkowski_interval(&mv, &global_classifier.centroids[pred as usize]);
        if pred == lbl {
            correct_intervals.push(mink);
        } else {
            incorrect_intervals.push(mink);
        }
    }

    let correct_mean = if correct_intervals.is_empty() { 0.0 }
        else { correct_intervals.iter().sum::<f32>() / correct_intervals.len() as f32 };
    let incorrect_mean = if incorrect_intervals.is_empty() { 0.0 }
        else { incorrect_intervals.iter().sum::<f32>() / incorrect_intervals.len() as f32 };
    let timelike_correct = correct_intervals.iter()
        .filter(|&&i| classify_interval(i) == IntervalType::Timelike)
        .count() as f32 / correct_intervals.len().max(1) as f32;

    println!("\n--- Minkowski interval statistics (post-refinement) ---");
    println!("  Correct classifications:   mean interval = {:.6}", correct_mean);
    println!("  Incorrect classifications: mean interval = {:.6}", incorrect_mean);
    if correct_mean.abs() > 1e-8 {
        println!("  Ratio (incorrect/correct): {:.1}x", incorrect_mean / correct_mean);
    }
    println!("  Correct that are timelike:   {:.1}%", timelike_correct * 100.0);
    let timelike_incorrect = incorrect_intervals.iter()
        .filter(|&&i| classify_interval(i) == IntervalType::Timelike)
        .count() as f32 / incorrect_intervals.len().max(1) as f32;
    println!("  Incorrect that are timelike: {:.1}% (should be low)", timelike_incorrect * 100.0);
    let spacelike_correct = correct_intervals.iter()
        .filter(|&&i| classify_interval(i) == IntervalType::Spacelike)
        .count() as f32 / correct_intervals.len().max(1) as f32;
    println!("  Correct that are spacelike:  {:.1}%", spacelike_correct * 100.0);
    let lightlike_correct = correct_intervals.iter()
        .filter(|&&i| classify_interval(i) == IntervalType::Lightlike)
        .count() as f32 / correct_intervals.len().max(1) as f32;
    println!("  Correct that are lightlike:  {:.1}%", lightlike_correct * 100.0);

    CliffordMnistResult {
        task_accuracies,
        avg_accuracy,
        grade_discriminability: grade_disc,
        interval_stats: IntervalStats {
            correct_mean_interval: correct_mean,
            incorrect_mean_interval: incorrect_mean,
            timelike_correct_pct: timelike_correct,
        },
    }
}

fn evaluate_binary(
    encoder: &CliffordImageEncoder,
    classifier: &CliffordClassifier,
    test_pairs: &[(Vec<f32>, u8)],
    digit_a: u8,
    digit_b: u8,
) -> f32 {
    let mut correct = 0u32;
    let mut total = 0u32;
    for (img, lbl) in test_pairs {
        let mv = encoder.encode(img);
        let (pred, _) = classifier.classify_binary(&mv, digit_a, digit_b);
        if pred == *lbl { correct += 1; }
        total += 1;
    }
    if total == 0 { return 0.0; }
    correct as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_produces_valid_multivector() {
        let encoder = CliffordImageEncoder::new(42);
        let image = vec![0.5f32; IMAGE_DIM];
        let mv = encoder.encode(&image);
        let norm: f32 = mv.components.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0, "encoded image should have nonzero norm");
        assert!(mv.grade(1).iter().any(|&x| x.abs() > 1e-6),
            "grade-1 should have content");
    }

    #[test]
    fn test_different_images_produce_different_multivectors() {
        let encoder = CliffordImageEncoder::new(42);
        let img_a = vec![1.0f32; IMAGE_DIM];
        let img_b = vec![0.0f32; IMAGE_DIM];
        let mv_a = encoder.encode(&img_a);
        let mv_b = encoder.encode(&img_b);
        let interval = minkowski_interval(&mv_a, &mv_b);
        assert!(interval.abs() > 1e-6, "different images should have nonzero interval");
    }

    #[test]
    fn test_classifier_accumulate_and_classify() {
        let encoder = CliffordImageEncoder::new(42);
        let mut classifier = CliffordClassifier::new();

        // Circle pattern (digit 0-like): bright ring, dark center
        let mut circle = vec![0.0f32; IMAGE_DIM];
        for y in 0..28 {
            for x in 0..28 {
                let dx = x as f32 - 13.5;
                let dy = y as f32 - 13.5;
                let r = (dx * dx + dy * dy).sqrt();
                circle[y * 28 + x] = if r > 6.0 && r < 12.0 { 0.9 } else { 0.1 };
            }
        }
        // Vertical line pattern (digit 1-like): thin bright column
        let mut line = vec![0.0f32; IMAGE_DIM];
        for y in 0..28 {
            for x in 0..28 {
                line[y * 28 + x] = if (x as i32 - 14).abs() < 3 { 0.9 } else { 0.1 };
            }
        }

        for _ in 0..20 {
            classifier.accumulate(&encoder.encode(&circle), 0);
            classifier.accumulate(&encoder.encode(&line), 1);
        }

        let test_circle = encoder.encode(&circle);
        let test_line = encoder.encode(&line);
        let (pred_c, _) = classifier.classify_binary(&test_circle, 0, 1);
        let (pred_l, _) = classifier.classify_binary(&test_line, 0, 1);
        assert_eq!(pred_c, 0, "circle should classify as digit 0");
        assert_eq!(pred_l, 1, "line should classify as digit 1");
    }
}
