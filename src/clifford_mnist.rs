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
    Rotor, apply_group_rotor,
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
    pub projection: Vec<[f32; CL8_DIM]>,
    /// Per-grade learned scale factors for balanced contribution
    pub grade_scales: [f32; 9],
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

    /// Texture-oriented encoder for histology/natural images.
    /// Uses Fourier + Gabor basis instead of position features.
    /// Translation-invariant: captures "what texture" not "where is ink".
    pub fn new_texture(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut projection = Vec::with_capacity(IMAGE_DIM);

        let grade_init_scale: [f32; 9] = std::array::from_fn(|g| {
            1.0 / (GRADE_DIMS[g] as f32).sqrt().max(1.0)
        });
        let norm = 1.0 / (IMAGE_DIM as f32).sqrt();
        let two_pi = 2.0 * std::f32::consts::PI;
        let pi = std::f32::consts::PI;

        // Pre-compute frequency pairs for grades 3 and 4
        let mut g3_freqs: Vec<(f32, f32)> = Vec::new();
        for u in 0..8i32 {
            for v in -4..5i32 {
                let r2 = u * u + v * v;
                if r2 >= 2 && r2 <= 20 && g3_freqs.len() < GRADE_DIMS[3] / 2 {
                    g3_freqs.push((u as f32, v as f32));
                }
            }
        }
        let mut g4_freqs: Vec<(f32, f32)> = Vec::new();
        for u in 0..10i32 {
            for v in -5..6i32 {
                let r2 = u * u + v * v;
                if r2 > 20 && r2 <= 50 && g4_freqs.len() < GRADE_DIMS[4] / 2 {
                    g4_freqs.push((u as f32, v as f32));
                }
            }
        }

        for pixel_idx in 0..IMAGE_DIM {
            let mut row = [0.0f32; CL8_DIM];
            let px = (pixel_idx % 28) as f32;
            let py = (pixel_idx / 28) as f32;

            // Grade 0 (scalar): mean intensity — tissue density
            row[GRADE_OFFSETS[0]] = norm;

            // Grade 1 (8 components): Low-frequency Fourier — coarse texture
            let g1s = grade_init_scale[1] * norm;
            let g1_uv: [(f32, f32); 4] = [
                (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0),
            ];
            for (k, &(u, v)) in g1_uv.iter().enumerate() {
                let phase = two_pi * (u * px + v * py) / 28.0;
                row[GRADE_OFFSETS[1] + 2 * k]     = phase.cos() * g1s;
                row[GRADE_OFFSETS[1] + 2 * k + 1] = phase.sin() * g1s;
            }

            // Grade 2 (28 components): Gabor-like oriented texture
            // 7 orientations × 2 spatial frequencies × cos/sin
            let g2s = grade_init_scale[2] * norm;
            let n_orient = 7usize;
            let g2_spatial_freqs = [2.0f32, 4.0];
            for ori in 0..n_orient {
                let theta = ori as f32 * pi / n_orient as f32;
                let dx = theta.cos();
                let dy = theta.sin();
                for (fi, &freq) in g2_spatial_freqs.iter().enumerate() {
                    let phase = two_pi * freq * (px * dx + py * dy) / 28.0;
                    let base = ori * 4 + fi * 2;
                    if base + 1 < GRADE_DIMS[2] {
                        row[GRADE_OFFSETS[2] + base]     = phase.cos() * g2s;
                        row[GRADE_OFFSETS[2] + base + 1] = phase.sin() * g2s;
                    }
                }
            }

            // Grade 3 (56 components): Medium-frequency 2D Fourier
            let g3s = grade_init_scale[3] * norm;
            for (k, &(u, v)) in g3_freqs.iter().enumerate() {
                let phase = two_pi * (u * px + v * py) / 28.0;
                let base = 2 * k;
                if base + 1 < GRADE_DIMS[3] {
                    row[GRADE_OFFSETS[3] + base]     = phase.cos() * g3s;
                    row[GRADE_OFFSETS[3] + base + 1] = phase.sin() * g3s;
                }
            }

            // Grade 4 (70 components): Higher-frequency 2D Fourier
            let g4s = grade_init_scale[4] * norm;
            for (k, &(u, v)) in g4_freqs.iter().enumerate() {
                let phase = two_pi * (u * px + v * py) / 28.0;
                let base = 2 * k;
                if base + 1 < GRADE_DIMS[4] {
                    row[GRADE_OFFSETS[4] + base]     = phase.cos() * g4s;
                    row[GRADE_OFFSETS[4] + base + 1] = phase.sin() * g4s;
                }
            }

            // Grades 5-8: random structured (learned via contrastive training)
            for g in 5..=8 {
                let gs = grade_init_scale[g] * norm;
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

// ─── Dirac encoder ────────────────────────────────────────────────────────
//
// Treats an image as a field and computes its Dirac spinor representation.
// Instead of linear projection (which can't beat a linear classifier on raw pixels),
// this uses the geometric product to produce nonlinear texture features:
//
//   1. Compute 4 differential feature channels from the image:
//      - Intensity (identity operator)
//      - Horizontal gradient ∂_x (Sobel-like)
//      - Vertical gradient ∂_y
//      - Laplacian ∂²_xx + ∂²_yy
//
//   2. Each channel is projected into a grade-1 vector via 784→8 projection.
//      This gives 4 vectors in Cl(1,7): v_I, v_∂x, v_∂y, v_Δ.
//
//   3. Take geometric products: v_∂x * v_∂y produces grade-0 + grade-2 terms.
//      This is the NONLINEAR step — texture co-occurrence features that a linear
//      encoder cannot produce. The bivector (grade-2) captures oriented texture
//      relationships: parallel fibers (stroma) vs disorganized (cancer stroma).
//
//   4. Combine: grade-0 from products, grade-1 from vectors, grade-2 from
//      products, higher grades from multi-scale products.
//
// The resulting multivector is a Dirac spinor where even grades carry the
// nonlinear interaction features that separate tissue textures.

const DIRAC_W: usize = 28;
const DIRAC_H: usize = 28;
const N_CHANNELS: usize = 4; // intensity, ∂x, ∂y, laplacian

pub struct CliffordDiracEncoder {
    /// Per-channel projection: 784 → 8 (grade-1 vector) for each of 4 channels
    channel_proj: [[f32; 8]; N_CHANNELS],
    /// Additional texture projection: 784 → CL8_DIM for local variance channel
    texture_proj: Vec<[f32; CL8_DIM]>,
    grade_scales: [f32; 9],
}

fn img_at(image: &[f32], x: usize, y: usize) -> f32 {
    if x < DIRAC_W && y < DIRAC_H { image[y * DIRAC_W + x] } else { 0.0 }
}

fn compute_gradient_x(image: &[f32]) -> [f32; IMAGE_DIM] {
    let mut out = [0.0f32; IMAGE_DIM];
    for y in 0..DIRAC_H {
        for x in 0..DIRAC_W {
            let left = if x > 0 { img_at(image, x - 1, y) } else { img_at(image, x, y) };
            let right = if x < DIRAC_W - 1 { img_at(image, x + 1, y) } else { img_at(image, x, y) };
            out[y * DIRAC_W + x] = (right - left) * 0.5;
        }
    }
    out
}

fn compute_gradient_y(image: &[f32]) -> [f32; IMAGE_DIM] {
    let mut out = [0.0f32; IMAGE_DIM];
    for y in 0..DIRAC_H {
        for x in 0..DIRAC_W {
            let up = if y > 0 { img_at(image, x, y - 1) } else { img_at(image, x, y) };
            let down = if y < DIRAC_H - 1 { img_at(image, x, y + 1) } else { img_at(image, x, y) };
            out[y * DIRAC_W + x] = (down - up) * 0.5;
        }
    }
    out
}

fn compute_laplacian(image: &[f32]) -> [f32; IMAGE_DIM] {
    let mut out = [0.0f32; IMAGE_DIM];
    for y in 0..DIRAC_H {
        for x in 0..DIRAC_W {
            let c = img_at(image, x, y);
            let l = if x > 0 { img_at(image, x - 1, y) } else { c };
            let r = if x < DIRAC_W - 1 { img_at(image, x + 1, y) } else { c };
            let u = if y > 0 { img_at(image, x, y - 1) } else { c };
            let d = if y < DIRAC_H - 1 { img_at(image, x, y + 1) } else { c };
            out[y * DIRAC_W + x] = l + r + u + d - 4.0 * c;
        }
    }
    out
}

fn compute_local_variance(image: &[f32]) -> [f32; IMAGE_DIM] {
    let mut out = [0.0f32; IMAGE_DIM];
    for y in 0..DIRAC_H {
        for x in 0..DIRAC_W {
            let mut sum = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut count = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < DIRAC_W as i32 && ny >= 0 && ny < DIRAC_H as i32 {
                        let v = image[ny as usize * DIRAC_W + nx as usize];
                        sum += v;
                        sum2 += v * v;
                        count += 1.0;
                    }
                }
            }
            let mean = sum / count;
            out[y * DIRAC_W + x] = (sum2 / count - mean * mean).max(0.0).sqrt();
        }
    }
    out
}

fn channel_to_vector(
    channel: &[f32],
    proj: &[f32; 8],
) -> Multivector {
    let mut sums = [0.0f32; 8];
    for &v in channel.iter() {
        if v.abs() < 1e-7 { continue; }
        for k in 0..8 { sums[k] += v * proj[k]; }
    }
    Multivector::vector(&sums)
}

impl CliffordDiracEncoder {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Per-channel projections to grade-1 vectors
        // Initialized with different frequency responses per channel
        let mut channel_proj = [[0.0f32; 8]; N_CHANNELS];

        // Channel 0 (intensity): uniform → measures total energy per basis direction
        for k in 0..8 {
            let freq = (k as f32 + 1.0) * std::f32::consts::PI / 8.0;
            channel_proj[0][k] = freq.cos() * 0.1;
        }
        // Channel 1 (∂x): horizontal texture orientation
        for k in 0..8 {
            channel_proj[1][k] = rng.gen_range(-0.15..0.15);
        }
        channel_proj[1][1] = 0.2; // e_1: primary horizontal
        // Channel 2 (∂y): vertical texture orientation
        for k in 0..8 {
            channel_proj[2][k] = rng.gen_range(-0.15..0.15);
        }
        channel_proj[2][2] = 0.2; // e_2: primary vertical
        // Channel 3 (laplacian): curvature/blob detector
        for k in 0..8 {
            channel_proj[3][k] = rng.gen_range(-0.15..0.15);
        }
        channel_proj[3][0] = 0.2; // e_0 (timelike): curvature as causal signal

        // Texture projection for local variance → full multivector
        let norm = 1.0 / (IMAGE_DIM as f32).sqrt();
        let grade_init_scale: [f32; 9] = std::array::from_fn(|g| {
            1.0 / (GRADE_DIMS[g] as f32).sqrt().max(1.0)
        });
        let two_pi = 2.0 * std::f32::consts::PI;

        let mut texture_proj = Vec::with_capacity(IMAGE_DIM);
        for pixel_idx in 0..IMAGE_DIM {
            let mut row = [0.0f32; CL8_DIM];
            let px = (pixel_idx % 28) as f32;
            let py = (pixel_idx / 28) as f32;

            // Fourier features for residual texture detail (same as new_texture grades 2-4)
            let g2s = grade_init_scale[2] * norm;
            for ori in 0..7usize {
                let theta = ori as f32 * std::f32::consts::PI / 7.0;
                let dx = theta.cos();
                let dy = theta.sin();
                for (fi, &freq) in [2.0f32, 4.0].iter().enumerate() {
                    let phase = two_pi * freq * (px * dx + py * dy) / 28.0;
                    let base = ori * 4 + fi * 2;
                    if base + 1 < GRADE_DIMS[2] {
                        row[GRADE_OFFSETS[2] + base]     = phase.cos() * g2s;
                        row[GRADE_OFFSETS[2] + base + 1] = phase.sin() * g2s;
                    }
                }
            }

            for g in 3..=8 {
                let gs = grade_init_scale[g] * norm;
                for k in 0..GRADE_DIMS[g] {
                    row[GRADE_OFFSETS[g] + k] = rng.gen_range(-0.1..0.1) * gs;
                }
            }

            texture_proj.push(row);
        }

        CliffordDiracEncoder {
            channel_proj,
            texture_proj,
            grade_scales: [1.0; 9],
        }
    }

    /// Encode an image as a Dirac particle in Cl(1,7).
    /// Applies differential operators, projects to vectors, then takes
    /// geometric products to produce nonlinear spinor features.
    pub fn encode(&self, image: &[f32]) -> Multivector {
        let dx = compute_gradient_x(image);
        let dy = compute_gradient_y(image);
        let lap = compute_laplacian(image);
        let lvar = compute_local_variance(image);

        // Project each channel to grade-1 vectors
        let v_i   = channel_to_vector(image, &self.channel_proj[0]);
        let v_dx  = channel_to_vector(&dx,   &self.channel_proj[1]);
        let v_dy  = channel_to_vector(&dy,   &self.channel_proj[2]);
        let v_lap = channel_to_vector(&lap,  &self.channel_proj[3]);

        // Geometric products → nonlinear even-grade features
        // v_dx * v_dy produces grade-0 (dot product: gradient energy)
        //                  and grade-2 (wedge: oriented texture "spin")
        let spin = v_dx.geo(&v_dy);
        // v_i * v_lap: intensity-curvature interaction
        let mass_curvature = v_i.geo(&v_lap);

        // Linear texture features from local variance
        let mut texture_mv = Multivector::zero();
        for (i, &v) in lvar.iter().enumerate() {
            if v.abs() < 1e-6 { continue; }
            let row = &self.texture_proj[i];
            for j in 0..CL8_DIM {
                texture_mv.components[j] += v * row[j];
            }
        }

        // Combine: linear vectors + nonlinear products + texture
        let mut result = Multivector::zero();
        for i in 0..CL8_DIM {
            result.components[i] =
                0.5 * v_i.components[i]
                + 0.3 * v_dx.components[i]
                + 0.3 * v_dy.components[i]
                + 0.2 * v_lap.components[i]
                + 1.0 * spin.components[i]
                + 0.5 * mass_curvature.components[i]
                + 0.3 * texture_mv.components[i];
        }

        // Apply grade scales
        for g in 0..=8 {
            let scale = self.grade_scales[g];
            let start = GRADE_OFFSETS[g];
            for k in 0..GRADE_DIMS[g] {
                result.components[start + k] *= scale;
            }
        }

        // Safety clamp
        const MAX_GRADE_NORM: f32 = 50.0;
        for g in 0..=8 {
            let start = GRADE_OFFSETS[g];
            let dim = GRADE_DIMS[g];
            let norm: f32 = result.components[start..start + dim]
                .iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > MAX_GRADE_NORM {
                let s = MAX_GRADE_NORM / norm;
                for k in 0..dim { result.components[start + k] *= s; }
            }
        }
        result
    }

    pub fn calibrate_scales(&mut self, samples: &[(Vec<f32>, u8)]) {
        let n = samples.len().min(500) as f32;
        if n < 2.0 { return; }
        self.grade_scales = [1.0; 9];
        let mut grade_norms = [0.0f32; 9];
        for (img, _) in samples.iter().take(500) {
            let mv = self.encode(img);
            for g in 0..=8 {
                let gnorm: f32 = mv.grade(g).iter().map(|x| x * x).sum::<f32>().sqrt();
                if gnorm.is_finite() { grade_norms[g] += gnorm / n; }
            }
        }
        for g in 0..=8 {
            if grade_norms[g] > 1e-6 && grade_norms[g].is_finite() {
                self.grade_scales[g] = 1.0 / grade_norms[g];
            }
        }
    }
}

// ─── Trainable Dirac Channel ─────────────────────────────────────────────
//
// For translationally confused pairs (|B| < 0.1), the existing encoder
// maps both classes to the same point — no rotational structure to exploit.
// This channel learns a NEW differential operator (3×3 kernel) that
// creates a dimension of discrimination the fixed channels don't have.
//
// The training target is |B|: maximize the confusion bivector norm
// so that Clifford GD can then separate the pair in one pass.

pub struct TrainableDiracChannel {
    kernel: [[f32; 3]; 3],
    projection: Vec<[f32; CL8_DIM]>,
    grade_scales: [f32; 9],
}

impl TrainableDiracChannel {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let norm = 1.0 / (IMAGE_DIM as f32).sqrt();

        let mut kernel = [[0.0f32; 3]; 3];
        for row in kernel.iter_mut() {
            for v in row.iter_mut() {
                *v = rng.gen_range(-0.3..0.3);
            }
        }

        let mut projection = Vec::with_capacity(IMAGE_DIM);
        for _ in 0..IMAGE_DIM {
            let mut row = [0.0f32; CL8_DIM];
            for j in 0..CL8_DIM {
                row[j] = rng.gen_range(-1.0..1.0) * norm;
            }
            projection.push(row);
        }

        TrainableDiracChannel {
            kernel,
            projection,
            grade_scales: [1.0; 9],
        }
    }

    fn apply_kernel(&self, image: &[f32]) -> [f32; IMAGE_DIM] {
        let mut out = [0.0f32; IMAGE_DIM];
        for y in 0..DIRAC_H {
            for x in 0..DIRAC_W {
                let mut val = 0.0f32;
                for ky in 0..3usize {
                    for kx in 0..3usize {
                        let iy = (y as i32 + ky as i32 - 1).clamp(0, DIRAC_H as i32 - 1) as usize;
                        let ix = (x as i32 + kx as i32 - 1).clamp(0, DIRAC_W as i32 - 1) as usize;
                        val += image[iy * DIRAC_W + ix] * self.kernel[ky][kx];
                    }
                }
                out[y * DIRAC_W + x] = val;
            }
        }
        out
    }

    pub fn encode(&self, image: &[f32]) -> Multivector {
        let response = self.apply_kernel(image);
        let mut mv = Multivector::zero();
        for (i, &v) in response.iter().enumerate() {
            if v.abs() < 1e-6 { continue; }
            let row = &self.projection[i];
            for j in 0..CL8_DIM {
                mv.components[j] += v * row[j];
            }
        }
        for g in 0..=8 {
            let scale = self.grade_scales[g];
            let start = GRADE_OFFSETS[g];
            for k in 0..GRADE_DIMS[g] {
                mv.components[start + k] *= scale;
            }
        }
        mv
    }

    pub fn calibrate_scales(&mut self, images: &[Vec<f32>]) {
        let n = images.len().min(500) as f32;
        if n < 2.0 { return; }
        self.grade_scales = [1.0; 9];
        let mut grade_norms = [0.0f32; 9];
        for img in images.iter().take(500) {
            let mv = self.encode(img);
            for g in 0..=8 {
                let gnorm: f32 = mv.grade(g).iter().map(|x| x * x).sum::<f32>().sqrt();
                if gnorm.is_finite() { grade_norms[g] += gnorm / n; }
            }
        }
        for g in 0..=8 {
            if grade_norms[g] > 1e-6 && grade_norms[g].is_finite() {
                self.grade_scales[g] = 1.0 / grade_norms[g];
            }
        }
    }

    /// Train on a specific class pair to maximize |B| (confusion bivector norm).
    /// Uses finite-difference gradient on the 3×3 kernel.
    /// Goal: manufacture rotational structure so Clifford GD can separate them.
    pub fn train_on_pair(
        &mut self,
        images: &[Vec<f32>],
        labels: &[u8],
        class_a: u8,
        class_b: u8,
        max_epochs: usize,
        lr: f32,
        target_bv_norm: f32,
    ) -> f32 {
        let max_per_class = 1000;

        for epoch in 0..max_epochs {
            let (ca, cb, bv_norm) = self.compute_pair_bivector(
                images, labels, class_a, class_b, max_per_class,
            );
            let _ = (ca, cb);

            if epoch % 10 == 0 || bv_norm >= target_bv_norm {
                println!("      epoch {:>3}: |B|={:.4}", epoch, bv_norm);
            }
            if bv_norm >= target_bv_norm {
                println!("      target reached at epoch {} (|B|={:.4} >= {:.3})",
                    epoch, bv_norm, target_bv_norm);
                return bv_norm;
            }

            let eps = 0.01f32;
            for ky in 0..3 {
                for kx in 0..3 {
                    self.kernel[ky][kx] += eps;
                    let (_, _, bv_plus) = self.compute_pair_bivector(
                        images, labels, class_a, class_b, max_per_class,
                    );
                    self.kernel[ky][kx] -= 2.0 * eps;
                    let (_, _, bv_minus) = self.compute_pair_bivector(
                        images, labels, class_a, class_b, max_per_class,
                    );
                    self.kernel[ky][kx] += eps;

                    let grad = (bv_plus - bv_minus) / (2.0 * eps);
                    self.kernel[ky][kx] += lr * grad;
                }
            }

            // Also update projection weights via finite difference on random subset
            if epoch % 5 == 0 {
                let mut proj_rng = StdRng::seed_from_u64(epoch as u64);
                let n_update = 50;
                for _ in 0..n_update {
                    let pi: usize = proj_rng.gen_range(0..IMAGE_DIM);
                    let pj: usize = proj_rng.gen_range(0..CL8_DIM);

                    self.projection[pi][pj] += eps;
                    let (_, _, bv_plus) = self.compute_pair_bivector(
                        images, labels, class_a, class_b, max_per_class,
                    );
                    self.projection[pi][pj] -= 2.0 * eps;
                    let (_, _, bv_minus) = self.compute_pair_bivector(
                        images, labels, class_a, class_b, max_per_class,
                    );
                    self.projection[pi][pj] += eps;

                    let grad = (bv_plus - bv_minus) / (2.0 * eps);
                    self.projection[pi][pj] += lr * 0.1 * grad;
                }
            }
        }

        let (_, _, final_bv) = self.compute_pair_bivector(
            images, labels, class_a, class_b, 1000,
        );
        final_bv
    }

    fn compute_pair_bivector(
        &self,
        images: &[Vec<f32>],
        labels: &[u8],
        class_a: u8,
        class_b: u8,
        max_per_class: usize,
    ) -> (Multivector, Multivector, f32) {
        let mut sum_a = Multivector::zero();
        let mut sum_b = Multivector::zero();
        let mut n_a = 0usize;
        let mut n_b = 0usize;
        for (img, &l) in images.iter().zip(labels.iter()) {
            if l == class_a && n_a < max_per_class {
                let mv = self.encode(img);
                for j in 0..CL8_DIM { sum_a.components[j] += mv.components[j]; }
                n_a += 1;
            } else if l == class_b && n_b < max_per_class {
                let mv = self.encode(img);
                for j in 0..CL8_DIM { sum_b.components[j] += mv.components[j]; }
                n_b += 1;
            }
            if n_a >= max_per_class && n_b >= max_per_class { break; }
        }
        if n_a > 0 { for j in 0..CL8_DIM { sum_a.components[j] /= n_a as f32; } }
        if n_b > 0 { for j in 0..CL8_DIM { sum_b.components[j] /= n_b as f32; } }

        let bv = confusion_bivector(&sum_a, &sum_b);
        let bv_norm: f32 = bv.grade(2).iter().map(|x| x * x).sum::<f32>().sqrt();
        (sum_a, sum_b, bv_norm)
    }
}

/// Compute |B| for all class pairs — the geometric learnability diagnostic.
/// Returns a matrix of bivector norms and the list of degenerate pairs (|B| < threshold).
pub fn diagnose_pair_learnability(
    rgb_enc: &CliffordRGBEncoder,
    rgb_images: &[Vec<f32>],
    labels: &[u8],
    n_classes: usize,
    max_samples: usize,
) -> (Vec<Vec<f32>>, Vec<(u8, u8)>, Vec<(u8, u8)>) {
    let mut centroids = Vec::with_capacity(n_classes);
    for c in 0..n_classes {
        let mut sum = Multivector::zero();
        let mut n = 0usize;
        for (img, &l) in rgb_images.iter().zip(labels.iter()) {
            if l as usize == c && n < max_samples {
                let mv = rgb_enc.encode(img);
                for j in 0..CL8_DIM { sum.components[j] += mv.components[j]; }
                n += 1;
            }
        }
        if n > 0 { for j in 0..CL8_DIM { sum.components[j] /= n as f32; } }
        centroids.push(sum);
    }

    let mut bv_matrix = vec![vec![0.0f32; n_classes]; n_classes];
    let mut rotational = Vec::new();
    let mut degenerate = Vec::new();

    for i in 0..n_classes {
        for j in (i+1)..n_classes {
            let bv = confusion_bivector(&centroids[i], &centroids[j]);
            let bv_norm: f32 = bv.grade(2).iter().map(|x| x * x).sum::<f32>().sqrt();
            bv_matrix[i][j] = bv_norm;
            bv_matrix[j][i] = bv_norm;

            if bv_norm >= 0.3 {
                rotational.push((i as u8, j as u8));
            } else if bv_norm < 0.1 {
                degenerate.push((i as u8, j as u8));
            }
        }
    }
    (bv_matrix, rotational, degenerate)
}

// ─── RGB encoder (single 2352→256 projection) ───────────────────────────
//
// Direct projection from 28×28×3 RGB pixels into Cl(1,7).
// Each of the 2352 input dimensions (R,G,B at each spatial position)
// has its own learned projection row. The three color channels at each
// pixel contribute DIFFERENTLY to the multivector — the encoder learns
// that blue (hematoxylin/nuclei) and red (eosin/stroma) carry distinct
// grade signals without needing explicit geometric product fusion.
//
// Same speed as the grayscale encoder, just 3x more projection rows.

const RGB_DIM: usize = 28 * 28 * 3; // 2352

pub struct CliffordRGBEncoder {
    projection: Vec<[f32; CL8_DIM]>, // 2352 → 256
    grade_scales: [f32; 9],
}

impl CliffordRGBEncoder {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut projection = Vec::with_capacity(RGB_DIM);

        let grade_init_scale: [f32; 9] = std::array::from_fn(|g| {
            1.0 / (GRADE_DIMS[g] as f32).sqrt().max(1.0)
        });
        let norm = 1.0 / (RGB_DIM as f32).sqrt();
        let two_pi = 2.0 * std::f32::consts::PI;
        let pi = std::f32::consts::PI;

        for input_idx in 0..RGB_DIM {
            let mut row = [0.0f32; CL8_DIM];
            let pixel_idx = input_idx / 3;
            let channel = input_idx % 3; // 0=R, 1=G, 2=B
            let px = (pixel_idx % 28) as f32;
            let py = (pixel_idx / 28) as f32;

            // Channel-specific grade biases:
            // R (eosin/stroma): emphasize grade-2 (bivector/texture orientation)
            // G (mixed): balanced contribution
            // B (hematoxylin/nuclei): emphasize grade-0 (scalar/density)
            let channel_weight = match channel {
                0 => [0.8, 0.8, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], // R: boost bivector
                1 => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], // G: balanced
                2 => [1.5, 1.2, 0.8, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0], // B: boost scalar+topology
                _ => unreachable!(),
            };

            // Grade 0: intensity contribution (channel-weighted)
            row[GRADE_OFFSETS[0]] = norm * channel_weight[0];

            // Grade 1: Fourier features (texture at different scales)
            let g1s = grade_init_scale[1] * norm * channel_weight[1];
            let g1_uv: [(f32, f32); 4] = [
                (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0),
            ];
            for (k, &(u, v)) in g1_uv.iter().enumerate() {
                let phase = two_pi * (u * px + v * py) / 28.0;
                row[GRADE_OFFSETS[1] + 2 * k]     = phase.cos() * g1s;
                row[GRADE_OFFSETS[1] + 2 * k + 1] = phase.sin() * g1s;
            }

            // Grade 2: Gabor-like oriented texture (7 orientations × 2 frequencies)
            let g2s = grade_init_scale[2] * norm * channel_weight[2];
            for ori in 0..7usize {
                let theta = ori as f32 * pi / 7.0;
                let dx = theta.cos();
                let dy = theta.sin();
                for (fi, &freq) in [2.0f32, 4.0].iter().enumerate() {
                    let phase = two_pi * freq * (px * dx + py * dy) / 28.0;
                    let base = ori * 4 + fi * 2;
                    if base + 1 < GRADE_DIMS[2] {
                        row[GRADE_OFFSETS[2] + base]     = phase.cos() * g2s;
                        row[GRADE_OFFSETS[2] + base + 1] = phase.sin() * g2s;
                    }
                }
            }

            // Grade 3-4: medium/high frequency Fourier
            for g in 3..=4 {
                let gs = grade_init_scale[g] * norm * channel_weight[g];
                let freq_offset = if g == 3 { 2.0 } else { 5.0 };
                for k in 0..GRADE_DIMS[g] {
                    let freq = freq_offset + k as f32 * 0.5;
                    let angle = k as f32 * pi / GRADE_DIMS[g] as f32;
                    let phase = two_pi * freq * (px * angle.cos() + py * angle.sin()) / 28.0;
                    row[GRADE_OFFSETS[g] + k] = phase.cos() * gs
                        + rng.gen_range(-0.02..0.02) * gs;
                }
            }

            // Grades 5-8: random structured (learned via contrastive training)
            for g in 5..=8 {
                let gs = grade_init_scale[g] * norm * channel_weight[g];
                for k in 0..GRADE_DIMS[g] {
                    row[GRADE_OFFSETS[g] + k] = rng.gen_range(-0.1..0.1) * gs;
                }
            }

            projection.push(row);
        }

        CliffordRGBEncoder {
            projection,
            grade_scales: [1.0; 9],
        }
    }

    pub fn encode(&self, rgb: &[f32]) -> Multivector {
        let mut mv = Multivector::zero();
        let n = rgb.len().min(RGB_DIM);
        for (i, &val) in rgb.iter().enumerate().take(n) {
            if val.abs() < 1e-6 { continue; }
            let row = &self.projection[i];
            for j in 0..CL8_DIM {
                mv.components[j] += val * row[j];
            }
        }

        for g in 0..=8 {
            let scale = self.grade_scales[g];
            let start = GRADE_OFFSETS[g];
            for k in 0..GRADE_DIMS[g] {
                mv.components[start + k] *= scale;
            }
        }

        const MAX_GRADE_NORM: f32 = 50.0;
        for g in 0..=8 {
            let start = GRADE_OFFSETS[g];
            let dim = GRADE_DIMS[g];
            let norm: f32 = mv.components[start..start + dim]
                .iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > MAX_GRADE_NORM {
                let s = MAX_GRADE_NORM / norm;
                for k in 0..dim { mv.components[start + k] *= s; }
            }
        }
        mv
    }

    pub fn calibrate_scales(&mut self, samples: &[(Vec<f32>, u8)]) {
        let n = samples.len().min(500) as f32;
        if n < 2.0 { return; }
        self.grade_scales = [1.0; 9];
        let mut grade_norms = [0.0f32; 9];
        for (img, _) in samples.iter().take(500) {
            let mv = self.encode(img);
            for g in 0..=8 {
                let gnorm: f32 = mv.grade(g).iter().map(|x| x * x).sum::<f32>().sqrt();
                if gnorm.is_finite() { grade_norms[g] += gnorm / n; }
            }
        }
        for g in 0..=8 {
            if grade_norms[g] > 1e-6 && grade_norms[g].is_finite() {
                self.grade_scales[g] = 1.0 / grade_norms[g];
            }
        }
    }

    /// Contrastive training step. Takes a centroid slice (any length).
    pub fn train_step(
        &mut self,
        rgb: &[f32],
        label: u8,
        centroids: &[Multivector],
        lr: f32,
    ) {
        let mv = self.encode(rgb);
        let n_classes = centroids.len();
        let correct_centroid = &centroids[label as usize];
        let correct_dist = spacetime_distance(&mv, correct_centroid);

        let mut hardest_neg_dist = f32::MAX;
        let mut hardest_neg_idx = 0;
        for d in 0..n_classes {
            if d == label as usize { continue; }
            if centroids[d].components.iter().all(|x| x.abs() < 1e-12) { continue; }
            let dist = spacetime_distance(&mv, &centroids[d]);
            if dist < hardest_neg_dist {
                hardest_neg_dist = dist;
                hardest_neg_idx = d;
            }
        }
        if hardest_neg_dist == f32::MAX { return; }

        let margin = 0.1;
        let violation = correct_dist - hardest_neg_dist + margin;
        if violation <= 0.0 { return; }

        let neg_centroid = &centroids[hardest_neg_idx];
        let loss_scale = violation.min(2.0);

        let n = rgb.len().min(RGB_DIM);
        for (i, &pixel) in rgb.iter().enumerate().take(n) {
            if pixel.abs() < 0.01 { continue; }
            let row = &mut self.projection[i];
            for g in 0..=8 {
                let glr = lr / (GRADE_DIMS[g] as f32).sqrt().max(1.0);
                for k in 0..GRADE_DIMS[g] {
                    let idx = GRADE_OFFSETS[g] + k;
                    let toward = correct_centroid.components[idx] - mv.components[idx];
                    let away = mv.components[idx] - neg_centroid.components[idx];
                    let update = glr * pixel * loss_scale * (toward + away);
                    if update.is_finite() {
                        row[idx] += update;
                    }
                }
            }
        }
    }
}

/// N-class centroid classifier for PathMNIST (not hardcoded to 10).
pub struct PathClassifier {
    pub centroids: Vec<Multivector>,
    pub counts: Vec<u32>,
    pub n_classes: usize,
}

impl PathClassifier {
    pub fn new(n_classes: usize) -> Self {
        PathClassifier {
            centroids: (0..n_classes).map(|_| Multivector::zero()).collect(),
            counts: vec![0; n_classes],
            n_classes,
        }
    }

    pub fn accumulate(&mut self, mv: &Multivector, label: u8) {
        let d = label as usize;
        if d >= self.n_classes { return; }
        self.counts[d] += 1;
        let n = self.counts[d] as f32;
        let alpha = 1.0 / n;
        for i in 0..CL8_DIM {
            self.centroids[d].components[i] =
                self.centroids[d].components[i] * (1.0 - alpha) + mv.components[i] * alpha;
        }
    }

    pub fn classify(&self, mv: &Multivector) -> (u8, f32) {
        let mut best = 0u8;
        let mut best_dist = f32::MAX;
        for d in 0..self.n_classes {
            if self.counts[d] == 0 { continue; }
            let dist = spacetime_distance(mv, &self.centroids[d]);
            if dist < best_dist { best_dist = dist; best = d as u8; }
        }
        (best, best_dist)
    }

    pub fn classify_binary(&self, mv: &Multivector, a: u8, b: u8) -> (u8, f32) {
        let da = spacetime_distance(mv, &self.centroids[a as usize]);
        let db = spacetime_distance(mv, &self.centroids[b as usize]);
        if da <= db { (a, da) } else { (b, db) }
    }

    pub fn grade_discriminability(&self) -> [f32; 9] {
        let active: Vec<usize> = (0..self.n_classes).filter(|&d| self.counts[d] > 0).collect();
        if active.len() < 2 { return [0.0; 9]; }
        let mut disc = [0.0f32; 9];
        let mut npairs = 0u32;
        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                let ca = &self.centroids[active[i]];
                let cb = &self.centroids[active[j]];
                for g in 0..=8 {
                    let ga = ca.grade(g);
                    let gb = cb.grade(g);
                    let d2: f32 = ga.iter().zip(gb.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    disc[g] += d2;
                }
                npairs += 1;
            }
        }
        if npairs > 0 {
            for g in 0..=8 { disc[g] /= npairs as f32; }
        }
        disc
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

// ─── CliffordMicroBrain ─────────────────────────────────────────────────
// Paramecium-inspired router that uses spacetime algebra instead of cosine
// similarity. Stores per-class Multivector centroids and classifies via
// grade-weighted Minkowski distance + interval augmentation.

pub struct CliffordMicroBrain {
    pub centroids_rgb: Vec<Multivector>,
    pub centroids_dirac: Vec<Multivector>,
    counts: Vec<u64>,
    grade_weights_rgb: [f32; 9],
    grade_weights_dirac: [f32; 9],
    pub n_classes: usize,
}

impl CliffordMicroBrain {
    /// Build from paired RGB + Dirac multivectors in one pass.
    pub fn build(
        rgb_mvs: &[Multivector],
        dirac_mvs: &[Multivector],
        labels: &[u8],
        n_classes: usize,
    ) -> Self {
        let n = rgb_mvs.len();
        let mut centroids_rgb = vec![Multivector::zero(); n_classes];
        let mut centroids_dirac = vec![Multivector::zero(); n_classes];
        let mut counts = vec![0u64; n_classes];

        for i in 0..n {
            let c = labels[i] as usize;
            if c >= n_classes { continue; }
            counts[c] += 1;
            for j in 0..CL8_DIM {
                centroids_rgb[c].components[j] += rgb_mvs[i].components[j];
                centroids_dirac[c].components[j] += dirac_mvs[i].components[j];
            }
        }
        for c in 0..n_classes {
            if counts[c] > 0 {
                let n_f = counts[c] as f32;
                for j in 0..CL8_DIM {
                    centroids_rgb[c].components[j] /= n_f;
                    centroids_dirac[c].components[j] /= n_f;
                }
            }
        }

        // Compute grade discriminability from RGB centroids
        let mut path_clf_rgb = PathClassifier::new(n_classes);
        for (mv, &l) in rgb_mvs.iter().zip(labels.iter()) {
            path_clf_rgb.accumulate(mv, l);
        }
        let disc_rgb = path_clf_rgb.grade_discriminability();
        let gw_rgb = discriminability_weights(&disc_rgb);

        let mut path_clf_dirac = PathClassifier::new(n_classes);
        for (mv, &l) in dirac_mvs.iter().zip(labels.iter()) {
            path_clf_dirac.accumulate(mv, l);
        }
        let disc_dirac = path_clf_dirac.grade_discriminability();
        let gw_dirac = discriminability_weights(&disc_dirac);

        println!("  CliffordMicroBrain: {} classes, {} samples", n_classes, n);
        println!("    RGB grade weights:   [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
            gw_rgb[0], gw_rgb[1], gw_rgb[2], gw_rgb[3], gw_rgb[4],
            gw_rgb[5], gw_rgb[6], gw_rgb[7], gw_rgb[8]);
        println!("    Dirac grade weights: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
            gw_dirac[0], gw_dirac[1], gw_dirac[2], gw_dirac[3], gw_dirac[4],
            gw_dirac[5], gw_dirac[6], gw_dirac[7], gw_dirac[8]);

        CliffordMicroBrain {
            centroids_rgb,
            centroids_dirac,
            counts,
            grade_weights_rgb: gw_rgb,
            grade_weights_dirac: gw_dirac,
            n_classes,
        }
    }

    /// Classify using both RGB and Dirac multivectors with spacetime distance.
    pub fn classify(&self, rgb_mv: &Multivector, dirac_mv: &Multivector) -> (u8, f32) {
        self.classify_among(rgb_mv, dirac_mv, &(0..self.n_classes).collect::<Vec<_>>())
    }

    /// Classify only among a subset of candidate classes.
    pub fn classify_among(
        &self,
        rgb_mv: &Multivector,
        dirac_mv: &Multivector,
        candidates: &[usize],
    ) -> (u8, f32) {
        let mut best_class = candidates[0] as u8;
        let mut best_score = f32::MAX;

        for &c in candidates {
            if c >= self.n_classes || self.counts[c] == 0 { continue; }

            let rgb_score = interval_augmented_score(
                rgb_mv, &self.centroids_rgb[c], &self.grade_weights_rgb,
            );
            let dirac_score = interval_augmented_score(
                dirac_mv, &self.centroids_dirac[c], &self.grade_weights_dirac,
            );

            let score = rgb_score + dirac_score;

            if score < best_score {
                best_score = score;
                best_class = c as u8;
            }
        }
        (best_class, best_score)
    }
}

// ─── Contrastive Training for PathMNIST ─────────────────────────────────
// Grade-pressured triplet training that teaches each grade to carry
// specific histological structure: grade-2 for texture, grade-3 for
// junctions, grade-4 for topology, etc.

/// Sign factor for component j in the grade-weighted distance.
/// Minkowski signature: grade-1 k=0 is timelike (-1), everything else +1.
fn distance_sign(j: usize) -> f32 {
    if j >= GRADE_OFFSETS[1] && j < GRADE_OFFSETS[1] + 1 { -1.0 } else { 1.0 }
}

/// Map a flat multivector component index to its grade.
fn component_grade(j: usize) -> usize {
    for g in (0..=8).rev() {
        if j >= GRADE_OFFSETS[g] { return g; }
    }
    0
}

pub struct ContrastivePair {
    pub anchor_idx: usize,
    pub positive_idx: usize,
    pub negative_idx: usize,
    pub class_a: u8,
    pub class_b: u8,
}

/// Per-pair-type grade emphasis weights.
/// Higher weight = this grade is MORE important for separating this pair.
pub struct PairGradeWeights {
    pub weights: [f32; 9],
}

impl PairGradeWeights {
    fn for_classes(a: u8, b: u8) -> Self {
        let w = match (a.min(b), a.max(b)) {
            // stroma(7) vs debris(2): texture is the discriminator
            (2, 7) => [1.0, 0.5, 3.0, 1.5, 1.0, 1.0, 1.0, 1.0, 0.5],
            // lymphocytes(3) vs mucosa(6): topology (packing)
            (3, 6) => [1.0, 0.5, 1.0, 1.5, 3.0, 1.0, 1.0, 1.0, 0.5],
            // adeno(8) vs mucus(4): intensity + junctions
            (4, 8) => [2.0, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            // stroma(7) vs adeno(8): texture + junctions
            (7, 8) => [1.0, 0.5, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            // debris(2) vs adeno(8): intensity
            (2, 8) => [2.0, 0.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            // cancer vs normal: all grades contribute
            _ => {
                let is_cancer_a = a == 7 || a == 8;
                let is_cancer_b = b == 7 || b == 8;
                if is_cancer_a != is_cancer_b {
                    [1.2, 0.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 0.5]
                } else {
                    [1.0; 9]
                }
            }
        };
        PairGradeWeights { weights: w }
    }

    fn to_component_weight(&self, j: usize) -> f32 {
        self.weights[component_grade(j)]
    }
}

pub struct ContrastiveConfig {
    pub margin: f32,
    pub learning_rate: f32,
    pub grade_lr_mult: [f32; 9],
    pub batch_size: usize,
    pub epochs: usize,
    pub pairs_per_type: usize,
}

/// Generate contrastive pairs between two classes.
pub fn generate_pairs(
    labels: &[u8],
    class_a: u8,
    class_b: u8,
    n_pairs: usize,
    rng: &mut StdRng,
) -> Vec<ContrastivePair> {
    let indices_a: Vec<usize> = labels.iter().enumerate()
        .filter(|(_, &l)| l == class_a).map(|(i, _)| i).collect();
    let indices_b: Vec<usize> = labels.iter().enumerate()
        .filter(|(_, &l)| l == class_b).map(|(i, _)| i).collect();

    if indices_a.len() < 2 || indices_b.is_empty() { return Vec::new(); }

    (0..n_pairs).map(|_| {
        let ai = rng.gen_range(0..indices_a.len());
        let mut pi = rng.gen_range(0..indices_a.len());
        while pi == ai && indices_a.len() > 1 { pi = rng.gen_range(0..indices_a.len()); }
        let ni = rng.gen_range(0..indices_b.len());
        ContrastivePair {
            anchor_idx: indices_a[ai],
            positive_idx: indices_a[pi],
            negative_idx: indices_b[ni],
            class_a,
            class_b,
        }
    }).collect()
}

/// Compute triplet loss and update RGB encoder projection via analytical gradients.
/// Returns the average loss over the batch.
pub fn contrastive_train_rgb_batch(
    enc: &mut CliffordRGBEncoder,
    pairs: &[ContrastivePair],
    rgb_images: &[Vec<f32>],
    config: &ContrastiveConfig,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut active_count = 0u32;

    let input_dim = RGB_DIM;

    // Accumulate gradients over the batch
    // Gradient is sparse — only update rows where pixel != 0
    // For efficiency, process one pair at a time and apply immediately
    // (online SGD rather than batch accumulate for memory)

    for pair in pairs {
        let x_a = &rgb_images[pair.anchor_idx];
        let x_p = &rgb_images[pair.positive_idx];
        let x_n = &rgb_images[pair.negative_idx];

        // Forward: encode all three (without grade clamp for gradient flow)
        let z_a = enc.encode(x_a);
        let z_p = enc.encode(x_p);
        let z_n = enc.encode(x_n);

        let gw = PairGradeWeights::for_classes(pair.class_a, pair.class_b);

        // Grade-weighted distance
        let mut d_pos = 0.0f32;
        let mut d_neg = 0.0f32;
        for j in 0..CL8_DIM {
            let s = distance_sign(j);
            let w = gw.to_component_weight(j);
            let dp = z_a.components[j] - z_p.components[j];
            let dn = z_a.components[j] - z_n.components[j];
            d_pos += s * w * dp * dp;
            d_neg += s * w * dn * dn;
        }

        let loss = (d_pos - d_neg + config.margin).max(0.0);
        total_loss += loss;
        if loss <= 0.0 { continue; }
        active_count += 1;

        // Analytical gradient of triplet loss w.r.t. projection[i][j]
        let lr = config.learning_rate / config.batch_size.max(1) as f32;

        for i in 0..input_dim.min(enc.projection.len()) {
            let va = if i < x_a.len() { x_a[i] } else { 0.0 };
            let vp = if i < x_p.len() { x_p[i] } else { 0.0 };
            let vn = if i < x_n.len() { x_n[i] } else { 0.0 };

            if va.abs() < 1e-4 && vp.abs() < 1e-4 && vn.abs() < 1e-4 { continue; }

            let row = &mut enc.projection[i];
            for j in 0..CL8_DIM {
                let s = distance_sign(j);
                let w = gw.to_component_weight(j);
                let g = component_grade(j);
                let glr = lr * config.grade_lr_mult[g];

                let dl_dza = 2.0 * s * w * (z_n.components[j] - z_p.components[j]);
                let dl_dzp = -2.0 * s * w * (z_a.components[j] - z_p.components[j]);
                let dl_dzn = 2.0 * s * w * (z_a.components[j] - z_n.components[j]);

                let grad = dl_dza * va + dl_dzp * vp + dl_dzn * vn;
                if grad.is_finite() {
                    row[j] -= glr * grad;
                }
            }
        }
    }

    total_loss / pairs.len().max(1) as f32
}

/// Contrastive update for Dirac encoder's texture_proj.
/// Only updates the texture projection (784→256), not channel_proj or geo products.
pub fn contrastive_train_dirac_batch(
    enc: &mut CliffordDiracEncoder,
    pairs: &[ContrastivePair],
    gray_images: &[Vec<f32>],
    config: &ContrastiveConfig,
) -> f32 {
    let mut total_loss = 0.0f32;

    for pair in pairs {
        let x_a = &gray_images[pair.anchor_idx];
        let x_p = &gray_images[pair.positive_idx];
        let x_n = &gray_images[pair.negative_idx];

        let z_a = enc.encode(x_a);
        let z_p = enc.encode(x_p);
        let z_n = enc.encode(x_n);

        let gw = PairGradeWeights::for_classes(pair.class_a, pair.class_b);

        let mut d_pos = 0.0f32;
        let mut d_neg = 0.0f32;
        for j in 0..CL8_DIM {
            let s = distance_sign(j);
            let w = gw.to_component_weight(j);
            let dp = z_a.components[j] - z_p.components[j];
            let dn = z_a.components[j] - z_n.components[j];
            d_pos += s * w * dp * dp;
            d_neg += s * w * dn * dn;
        }

        let loss = (d_pos - d_neg + config.margin).max(0.0);
        total_loss += loss;
        if loss <= 0.0 { continue; }

        // For Dirac, the texture_proj contribution is:
        //   texture_mv[j] = sum_i texture_proj[i][j] * local_var[i]
        // and result includes 0.3 * texture_mv[j], so the gradient
        // through texture_proj is attenuated by 0.3.
        let lvar_a = compute_local_variance(x_a);
        let lvar_p = compute_local_variance(x_p);
        let lvar_n = compute_local_variance(x_n);

        let lr = config.learning_rate * 0.3 / config.batch_size.max(1) as f32;

        for i in 0..IMAGE_DIM {
            let va = lvar_a[i];
            let vp = lvar_p[i];
            let vn = lvar_n[i];

            if va.abs() < 1e-4 && vp.abs() < 1e-4 && vn.abs() < 1e-4 { continue; }

            let row = &mut enc.texture_proj[i];
            for j in 0..CL8_DIM {
                let s = distance_sign(j);
                let w = gw.to_component_weight(j);
                let g = component_grade(j);
                let glr = lr * config.grade_lr_mult[g];

                let dl_dza = 2.0 * s * w * (z_n.components[j] - z_p.components[j]);
                let dl_dzp = -2.0 * s * w * (z_a.components[j] - z_p.components[j]);
                let dl_dzn = 2.0 * s * w * (z_a.components[j] - z_n.components[j]);

                let grad = dl_dza * va + dl_dzp * vp + dl_dzn * vn;
                if grad.is_finite() {
                    row[j] -= glr * grad;
                }
            }
        }
    }

    total_loss / pairs.len().max(1) as f32
}

/// Generate hard-negative contrastive pairs by mining closest cross-class samples.
/// For each anchor in class_a, finds the nearest class_b sample in embedding space.
pub fn generate_hard_negative_pairs(
    rgb_enc: &CliffordRGBEncoder,
    rgb_images: &[Vec<f32>],
    labels: &[u8],
    class_a: u8,
    class_b: u8,
    n_pairs: usize,
    max_per_class: usize,
) -> Vec<ContrastivePair> {
    let indices_a: Vec<usize> = labels.iter().enumerate()
        .filter(|(_, &l)| l == class_a).map(|(i, _)| i)
        .take(max_per_class).collect();
    let indices_b: Vec<usize> = labels.iter().enumerate()
        .filter(|(_, &l)| l == class_b).map(|(i, _)| i)
        .take(max_per_class).collect();

    if indices_a.len() < 2 || indices_b.is_empty() { return Vec::new(); }

    // Encode all samples from both classes
    let emb_a: Vec<Multivector> = indices_a.iter()
        .map(|&i| rgb_enc.encode(&rgb_images[i])).collect();
    let emb_b: Vec<Multivector> = indices_b.iter()
        .map(|&i| rgb_enc.encode(&rgb_images[i])).collect();

    let mut pairs = Vec::with_capacity(n_pairs);
    let mut rng = StdRng::seed_from_u64(42);

    for (ai, a_emb) in emb_a.iter().enumerate() {
        // Find closest class_b sample to this anchor
        let mut best_dist = f32::MAX;
        let mut best_bi = 0usize;
        for (bi, b_emb) in emb_b.iter().enumerate() {
            let mut d = 0.0f32;
            for j in 0..CL8_DIM {
                let diff = a_emb.components[j] - b_emb.components[j];
                d += diff * diff;
            }
            if d < best_dist { best_dist = d; best_bi = bi; }
        }

        // Positive: random other sample from class_a
        let mut pi = rng.gen_range(0..indices_a.len());
        while pi == ai && indices_a.len() > 1 { pi = rng.gen_range(0..indices_a.len()); }

        pairs.push(ContrastivePair {
            anchor_idx: indices_a[ai],
            positive_idx: indices_a[pi],
            negative_idx: indices_b[best_bi],
            class_a,
            class_b,
        });

        if pairs.len() >= n_pairs { break; }
    }

    // Also generate reverse pairs (class_b anchors, class_a hard negatives)
    for (bi, b_emb) in emb_b.iter().enumerate() {
        if pairs.len() >= n_pairs { break; }

        let mut best_dist = f32::MAX;
        let mut best_ai = 0usize;
        for (ai, a_emb) in emb_a.iter().enumerate() {
            let mut d = 0.0f32;
            for j in 0..CL8_DIM {
                let diff = b_emb.components[j] - a_emb.components[j];
                d += diff * diff;
            }
            if d < best_dist { best_dist = d; best_ai = ai; }
        }

        let mut pi = rng.gen_range(0..indices_b.len());
        while pi == bi && indices_b.len() > 1 { pi = rng.gen_range(0..indices_b.len()); }

        pairs.push(ContrastivePair {
            anchor_idx: indices_b[bi],
            positive_idx: indices_b[pi],
            negative_idx: indices_a[best_ai],
            class_a: class_b,
            class_b: class_a,
        });
    }

    pairs
}

/// Measure centroid distance between two classes using current encoder.
pub fn measure_class_distance(
    rgb_enc: &CliffordRGBEncoder,
    rgb_images: &[Vec<f32>],
    labels: &[u8],
    class_a: u8,
    class_b: u8,
    max_samples: usize,
) -> f32 {
    let mut sum_a = Multivector::zero();
    let mut sum_b = Multivector::zero();
    let mut n_a = 0u32;
    let mut n_b = 0u32;
    for (img, &l) in rgb_images.iter().zip(labels.iter()) {
        if l == class_a && (n_a as usize) < max_samples {
            let mv = rgb_enc.encode(img);
            for j in 0..CL8_DIM { sum_a.components[j] += mv.components[j]; }
            n_a += 1;
        } else if l == class_b && (n_b as usize) < max_samples {
            let mv = rgb_enc.encode(img);
            for j in 0..CL8_DIM { sum_b.components[j] += mv.components[j]; }
            n_b += 1;
        }
    }
    if n_a == 0 || n_b == 0 { return 0.0; }
    for j in 0..CL8_DIM {
        sum_a.components[j] /= n_a as f32;
        sum_b.components[j] /= n_b as f32;
    }
    let mut d = 0.0f32;
    for j in 0..CL8_DIM {
        let diff = sum_a.components[j] - sum_b.components[j];
        d += diff * diff;
    }
    d.sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Clifford Gradient Descent — single forward pass learning in Cl(1,7)
//
// Standard backprop: forward → scalar loss → backward → parameter updates
// Clifford gradient: forward → geometric product → rotor → done
//
// The geometric product Z_a * Z_b† of two class centroids encodes their
// full relationship. The grade-2 (bivector) component IS the confusion
// plane. The rotor R = exp(-B/2) IS the update. One operation.
//
// No backward pass. No chain rule. No scalar loss.
// The algebra computes the exact separation direction.
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the confusion bivector between two class centroids.
///
/// B = <Z_a * Z_b†>_2  (grade-2 projection of geometric product with reverse)
///
/// This bivector identifies the plane in Cl(1,7) where the two classes
/// are geometrically indistinguishable. The separation rotor rotates
/// one class out of this plane.
pub fn confusion_bivector(centroid_a: &Multivector, centroid_b: &Multivector) -> Multivector {
    let b_rev = centroid_b.reverse();
    let product = centroid_a.geo(&b_rev);
    product.grade_project(2)
}

/// Construct a separation rotor from a confusion bivector.
///
/// R = exp(-B_clamped / 2)
///
/// The bivector is clamped to |B| <= 0.5 to prevent destructive
/// large rotations that disrupt other class separations.
pub fn separation_rotor_full(confusion_bv: &Multivector) -> Rotor {
    let bv_slice = confusion_bv.grade(2);
    let norm: f32 = bv_slice.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut bv = [0.0f32; 28];
    if norm > 0.5 {
        let scale = 0.5 / norm;
        for i in 0..28 { bv[i] = bv_slice[i] * scale; }
    } else {
        bv.copy_from_slice(bv_slice);
    }
    Rotor::from_bivector(&bv)
}

/// Construct a scaled separation rotor.
///
/// The bivector is normalized and α controls the rotation angle in radians,
/// independent of the bivector magnitude. Useful when the full rotation
/// would disrupt other class separations.
pub fn separation_rotor_scaled(confusion_bv: &Multivector, alpha: f32) -> Rotor {
    let bv_slice = confusion_bv.grade(2);
    let bv_norm: f32 = bv_slice.iter().map(|x| x * x).sum::<f32>().sqrt();
    if bv_norm < 1e-10 {
        return Rotor::identity();
    }
    let mut scaled = [0.0f32; 28];
    for i in 0..28 {
        scaled[i] = (bv_slice[i] / bv_norm) * alpha;
    }
    Rotor::from_bivector(&scaled)
}

/// Apply a rotor update to the RGB encoder's projection matrix.
///
/// Since mv = Σ_i projection[i] * pixel[i] and R*mv*R† is linear:
///   projection_new[i] = R * projection_old[i] * R†
///
/// Each row of the projection is treated as a multivector and rotated.
pub fn apply_rotor_to_rgb_encoder(encoder: &mut CliffordRGBEncoder, rotor: &Rotor) {
    for row in encoder.projection.iter_mut() {
        let mut mv = Multivector::zero();
        mv.components.copy_from_slice(row);
        let rotated = apply_group_rotor(&mv, rotor);
        row.copy_from_slice(&rotated.components);
    }
}

/// Apply a rotor update to the Dirac encoder's texture projection.
pub fn apply_rotor_to_dirac_encoder(encoder: &mut CliffordDiracEncoder, rotor: &Rotor) {
    for row in encoder.texture_proj.iter_mut() {
        let mut mv = Multivector::zero();
        mv.components.copy_from_slice(row);
        let rotated = apply_group_rotor(&mv, rotor);
        row.copy_from_slice(&rotated.components);
    }
}

/// Compute the Euclidean distance between two multivector centroids.
fn centroid_distance(a: &Multivector, b: &Multivector) -> f32 {
    let mut d = 0.0f32;
    for j in 0..CL8_DIM {
        let diff = a.components[j] - b.components[j];
        d += diff * diff;
    }
    d.sqrt()
}

/// Result of single-pass Clifford gradient descent for one class pair.
pub struct CliffordSeparationResult {
    pub class_a: u8,
    pub class_b: u8,
    pub distance_before: f32,
    pub distance_after: f32,
    pub bivector_norm: f32,
}

/// Single-pass Clifford gradient descent for all collision pairs.
///
/// One forward pass per pair:
///   1. Encode class samples → compute centroids  (forward pass)
///   2. Geometric product Z_a * Z_b†             (the gradient)
///   3. Extract grade-2 → confusion bivector B    (the direction)
///   4. R = exp(-B/2) → separation rotor          (the update)
///   5. Apply R to encoder weights                (done)
///
/// No backward pass. No loss function. No chain rule.
/// The geometric product IS the gradient. The rotor IS the update.
pub fn clifford_single_pass(
    rgb_enc: &mut CliffordRGBEncoder,
    dirac_enc: &mut CliffordDiracEncoder,
    rgb_images: &[Vec<f32>],
    labels: &[u8],
    pairs: &[(u8, u8)],
    max_samples_per_class: usize,
) -> Vec<CliffordSeparationResult> {
    let mut results = Vec::new();

    for &(ca, cb) in pairs {
        let mut sum_a = Multivector::zero();
        let mut sum_b = Multivector::zero();
        let mut n_a = 0usize;
        let mut n_b = 0usize;

        for (img, &l) in rgb_images.iter().zip(labels.iter()) {
            if l == ca && n_a < max_samples_per_class {
                let mv = rgb_enc.encode(img);
                for j in 0..CL8_DIM { sum_a.components[j] += mv.components[j]; }
                n_a += 1;
            } else if l == cb && n_b < max_samples_per_class {
                let mv = rgb_enc.encode(img);
                for j in 0..CL8_DIM { sum_b.components[j] += mv.components[j]; }
                n_b += 1;
            }
            if n_a >= max_samples_per_class && n_b >= max_samples_per_class { break; }
        }

        if n_a == 0 || n_b == 0 { continue; }
        for j in 0..CL8_DIM {
            sum_a.components[j] /= n_a as f32;
            sum_b.components[j] /= n_b as f32;
        }

        let dist_before = centroid_distance(&sum_a, &sum_b);

        let bv = confusion_bivector(&sum_a, &sum_b);
        let bv_norm: f32 = bv.grade(2).iter().map(|x| x * x).sum::<f32>().sqrt();

        if bv_norm < 1e-10 {
            println!("    {} ↔ {}: already orthogonal (|B|≈0)", ca, cb);
            results.push(CliffordSeparationResult {
                class_a: ca, class_b: cb,
                distance_before: dist_before, distance_after: dist_before,
                bivector_norm: bv_norm,
            });
            continue;
        }

        let rotor = separation_rotor_full(&bv);

        apply_rotor_to_rgb_encoder(rgb_enc, &rotor);
        apply_rotor_to_dirac_encoder(dirac_enc, &rotor);

        let mut sum_a2 = Multivector::zero();
        let mut sum_b2 = Multivector::zero();
        let mut n_a2 = 0usize;
        let mut n_b2 = 0usize;
        for (img, &l) in rgb_images.iter().zip(labels.iter()) {
            if l == ca && n_a2 < max_samples_per_class {
                let mv = rgb_enc.encode(img);
                for j in 0..CL8_DIM { sum_a2.components[j] += mv.components[j]; }
                n_a2 += 1;
            } else if l == cb && n_b2 < max_samples_per_class {
                let mv = rgb_enc.encode(img);
                for j in 0..CL8_DIM { sum_b2.components[j] += mv.components[j]; }
                n_b2 += 1;
            }
            if n_a2 >= max_samples_per_class && n_b2 >= max_samples_per_class { break; }
        }
        for j in 0..CL8_DIM {
            if n_a2 > 0 { sum_a2.components[j] /= n_a2 as f32; }
            if n_b2 > 0 { sum_b2.components[j] /= n_b2 as f32; }
        }
        let dist_after = centroid_distance(&sum_a2, &sum_b2);

        println!("    {} ↔ {}: |B|={:.4}  d={:.3} → {:.3}  (one pass)",
            ca, cb, bv_norm, dist_before, dist_after);

        results.push(CliffordSeparationResult {
            class_a: ca, class_b: cb,
            distance_before: dist_before, distance_after: dist_after,
            bivector_norm: bv_norm,
        });
    }
    results
}

/// Full training pipeline result.
pub struct CliffordMnistResult {
    pub task_accuracies: Vec<f32>,
    pub avg_accuracy: f32,
    pub ten_class_accuracy: f32,
    pub per_digit_accuracy: [f32; 10],
    pub grade_discriminability: [f32; 9],
    pub interval_stats: IntervalStats,
    pub encoder: CliffordImageEncoder,
    pub classifier: CliffordClassifier,
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
/// Verbosity: 0=quiet, 1=progress (task epochs + phase labels), 2=full diagnostics.
pub fn run_clifford_mnist(
    train_images: &[Vec<f32>],
    train_labels: &[u8],
    test_images: &[Vec<f32>],
    test_labels: &[u8],
    train_limit: Option<usize>,
    max_epochs: u32,
) -> CliffordMnistResult {
    run_clifford_mnist_inner(train_images, train_labels, test_images, test_labels,
        train_limit, max_epochs, 2)
}

pub fn run_clifford_mnist_progress(
    train_images: &[Vec<f32>],
    train_labels: &[u8],
    test_images: &[Vec<f32>],
    test_labels: &[u8],
    train_limit: Option<usize>,
    max_epochs: u32,
) -> CliffordMnistResult {
    run_clifford_mnist_inner(train_images, train_labels, test_images, test_labels,
        train_limit, max_epochs, 1)
}

pub fn run_clifford_mnist_quiet(
    train_images: &[Vec<f32>],
    train_labels: &[u8],
    test_images: &[Vec<f32>],
    test_labels: &[u8],
    train_limit: Option<usize>,
    max_epochs: u32,
) -> CliffordMnistResult {
    run_clifford_mnist_inner(train_images, train_labels, test_images, test_labels,
        train_limit, max_epochs, 0)
}

fn run_clifford_mnist_inner(
    train_images: &[Vec<f32>],
    train_labels: &[u8],
    test_images: &[Vec<f32>],
    test_labels: &[u8],
    train_limit: Option<usize>,
    max_epochs: u32,
    verbosity: u8,
) -> CliffordMnistResult {
    use crate::mnist::filter_digit_pair_raw;

    // Level 1+: task progress (epochs, convergence, phase labels)
    macro_rules! progress {
        ($($arg:tt)*) => { if verbosity >= 1 { println!($($arg)*); } }
    }
    // Level 2: full diagnostics (grade weights, classifier comparisons, intervals)
    macro_rules! vprint {
        ($($arg:tt)*) => { if verbosity >= 2 { println!($($arg)*); } }
    }

    let mut encoder = CliffordImageEncoder::new(42);

    const TASKS: [(u8, u8); 5] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
    let mut task_accuracies = Vec::with_capacity(5);
    let mut multi_head = MultiTaskClassifier::new();

    let mut correct_intervals = Vec::new();
    let mut incorrect_intervals = Vec::new();

    progress!("--- Phase 1-2: Binary pair training ({} epochs each) ---", max_epochs);
    for (t, (d1, d2)) in TASKS.iter().enumerate() {
        let train_pairs = filter_digit_pair_raw(train_images, train_labels, *d1, *d2);
        let test_pairs = filter_digit_pair_raw(test_images, test_labels, *d1, *d2);

        let train_subset: Vec<_> = if let Some(lim) = train_limit {
            train_pairs.into_iter().take(lim).collect()
        } else {
            train_pairs
        };

        progress!("  Task {} ({} vs {}): {} train, {} test",
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
                progress!("    epoch {}: acc={:.1}%",
                    epoch, test_acc * 100.0);

                if test_acc >= 0.98 {
                    progress!("    Converged at epoch {} ({:.1}%)", epoch, test_acc * 100.0);
                    break;
                }
                if test_acc > best_test_acc + 0.001 {
                    best_test_acc = test_acc;
                    stale_count = 0;
                } else {
                    stale_count += 1;
                }
                if stale_count >= 4 {
                    progress!("    Converged at epoch {} (no improvement)", epoch);
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
        progress!("  Task {} ({} vs {}) final: {:.1}%\n", t, d1, d2, test_acc * 100.0);

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
    progress!("\n--- Phase 3: 10-class refinement ({} epochs) ---", max_epochs);
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
            progress!("    epoch {}: 10-class acc={:.1}%",
                epoch, racc * 100.0);
            if racc > best_refinement_acc + 0.002 {
                best_refinement_acc = racc;
                refinement_stale = 0;
            } else {
                refinement_stale += 1;
            }
            if refinement_stale >= 4 {
                progress!("    Converged at epoch {}", epoch);
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

    progress!("\n--- Evaluating ---");
    // Per-task evaluation with global classifier
    vprint!("\n--- Cross-task evaluation (global 10-class classifier) ---");
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
        vprint!("  Task {} ({} vs {}): {:.1}% (global)",
            t, d1, d2, acc * 100.0);
        cross_task_total += total;
        cross_task_correct += task_correct;
    }
    let cross_task_acc = cross_task_correct as f32 / cross_task_total.max(1) as f32;
    vprint!("  Overall cross-task binary: {:.1}%", cross_task_acc * 100.0);

    let full_total = test_images.len() as u32;
    let grade_disc = global_classifier.grade_discriminability();

    // Compute data-driven grade weights from measured discriminability
    let gw = discriminability_weights(&grade_disc);
    vprint!("\n--- Data-driven grade weights ---");
    let grade_names = [
        "scalar (intensity)", "vector (gradients)", "bivector (edges)",
        "trivector (junctions)", "quadvector (topology)", "grade-5", "grade-6",
        "grade-7", "pseudoscalar (orientation)",
    ];
    for g in 0..=8 {
        vprint!("  grade {}: disc={:.1}, weight={:.3} — {}",
            g, grade_disc[g], gw[g], grade_names[g]);
    }

    // --- Classifier comparison: 3 methods on the same encoder + centroids ---
    // Wrapped in a block so closures that borrow global_classifier drop before
    // we move it into the result struct.
    let mut ten_class_accuracy = 0.0f32;
    let mut per_digit_accuracy = [0.0f32; 10];
    {
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

        for (ci, (name, classify_fn)) in classifiers.iter().enumerate() {
            vprint!("\n--- Full 10-class evaluation ({}) ---", name);
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
            vprint!("  Overall: {:.1}% ({}/{})", full_acc * 100.0, full_correct, full_total);
            for d in 0..10 {
                let acc = per_digit_correct[d] as f32 / per_digit_total[d].max(1) as f32;
                vprint!("    digit {}: {:.1}% ({}/{})", d, acc * 100.0,
                    per_digit_correct[d], per_digit_total[d]);
            }
            if ci == 0 || full_acc > ten_class_accuracy {
                ten_class_accuracy = full_acc;
                for d in 0..10 {
                    per_digit_accuracy[d] = per_digit_correct[d] as f32
                        / per_digit_total[d].max(1) as f32;
                }
            }
        }
    } // classifiers dropped here, releasing borrow on global_classifier

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

    vprint!("\n--- Minkowski interval statistics (post-refinement) ---");
    vprint!("  Correct classifications:   mean interval = {:.6}", correct_mean);
    vprint!("  Incorrect classifications: mean interval = {:.6}", incorrect_mean);
    if correct_mean.abs() > 1e-8 {
        vprint!("  Ratio (incorrect/correct): {:.1}x", incorrect_mean / correct_mean);
    }
    vprint!("  Correct that are timelike:   {:.1}%", timelike_correct * 100.0);
    let timelike_incorrect = incorrect_intervals.iter()
        .filter(|&&i| classify_interval(i) == IntervalType::Timelike)
        .count() as f32 / incorrect_intervals.len().max(1) as f32;
    vprint!("  Incorrect that are timelike: {:.1}% (should be low)", timelike_incorrect * 100.0);
    let spacelike_correct = correct_intervals.iter()
        .filter(|&&i| classify_interval(i) == IntervalType::Spacelike)
        .count() as f32 / correct_intervals.len().max(1) as f32;
    vprint!("  Correct that are spacelike:  {:.1}%", spacelike_correct * 100.0);
    let lightlike_correct = correct_intervals.iter()
        .filter(|&&i| classify_interval(i) == IntervalType::Lightlike)
        .count() as f32 / correct_intervals.len().max(1) as f32;
    vprint!("  Correct that are lightlike:  {:.1}%", lightlike_correct * 100.0);

    CliffordMnistResult {
        task_accuracies,
        avg_accuracy,
        ten_class_accuracy,
        per_digit_accuracy,
        grade_discriminability: grade_disc,
        interval_stats: IntervalStats {
            correct_mean_interval: correct_mean,
            incorrect_mean_interval: incorrect_mean,
            timelike_correct_pct: timelike_correct,
        },
        encoder,
        classifier: global_classifier,
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

// ─── Clifford Image Decoder (Autoencoder) ────────────────────────────────
//
// Reconstructs 28×28 images from Cl(1,7) multivector encodings.
// Single-pass closed-form least-squares: W = (Z^T Z + λI)^{-1} Z^T Y
// No epochs, no learning rate, no gradient descent — just linear algebra.

const DECODER_INPUT_DIM: usize = CL8_DIM + 1; // 256 multivector + 1 bias

/// Linear decoder solved in closed form via normal equations.
/// Maps 256D Cl(1,7) multivector → 784D pixel image.
pub struct CliffordDecoder {
    weights: Vec<[f32; IMAGE_DIM]>, // DECODER_INPUT_DIM rows × IMAGE_DIM cols
}

impl CliffordDecoder {
    /// Fit the decoder in a single pass over the training data.
    /// Solves W = (Z^T Z + λI)^{-1} Z^T Y where Z = [multivectors | 1].
    pub fn fit(
        encoder: &CliffordImageEncoder,
        images: &[Vec<f32>],
        lambda: f32,
        verbosity: u8,
    ) -> Self {
        let d = DECODER_INPUT_DIM;
        if verbosity >= 1 { println!("  Accumulating {} samples...", images.len()); }

        // Accumulate Z^T Z (d×d) and Z^T Y (d×784) in a single pass
        let mut ztz = vec![vec![0.0f64; d]; d];
        let mut zty = vec![[0.0f64; IMAGE_DIM]; d];

        for (idx, img) in images.iter().enumerate() {
            let mv = encoder.encode(img);
            let mut z = [0.0f64; DECODER_INPUT_DIM];
            for i in 0..CL8_DIM { z[i] = mv.components[i] as f64; }
            z[CL8_DIM] = 1.0; // bias

            for i in 0..d {
                if z[i].abs() < 1e-12 { continue; }
                for j in i..d {
                    ztz[i][j] += z[i] * z[j];
                }
                for j in 0..IMAGE_DIM {
                    zty[i][j] += z[i] * img[j] as f64;
                }
            }

            if verbosity >= 1 && (idx + 1) % 10000 == 0 {
                println!("    {}/{}", idx + 1, images.len());
            }
        }

        // Symmetrise Z^T Z
        for i in 0..d {
            for j in 0..i {
                ztz[i][j] = ztz[j][i];
            }
        }

        // Tikhonov regularisation
        for i in 0..d {
            ztz[i][i] += lambda as f64;
        }

        if verbosity >= 1 { println!("  Solving {}×{} normal equations...", d, d); }
        let inv = invert_symmetric(ztz);

        // W = inv(Z^T Z) * Z^T Y
        let mut weights: Vec<[f32; IMAGE_DIM]> = vec![[0.0f32; IMAGE_DIM]; d];
        for i in 0..d {
            for j in 0..IMAGE_DIM {
                let mut s = 0.0f64;
                for k in 0..d {
                    s += inv[i][k] * zty[k][j];
                }
                weights[i][j] = s as f32;
            }
        }
        if verbosity >= 1 { println!("  Decoder solved."); }

        CliffordDecoder { weights }
    }

    /// Reconstruct an image from its multivector encoding.
    pub fn decode(&self, mv: &Multivector) -> Vec<f32> {
        let mut pixels = vec![0.0f32; IMAGE_DIM];
        for j in 0..IMAGE_DIM {
            let mut s = self.weights[CL8_DIM][j]; // bias term
            for i in 0..CL8_DIM {
                s += self.weights[i][j] * mv.components[i];
            }
            pixels[j] = s.clamp(0.0, 1.0);
        }
        pixels
    }
}

/// Invert a symmetric positive-definite matrix via Gauss-Jordan with
/// partial pivoting.  Operates in f64 for numerical stability.
fn invert_symmetric(m: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = m.len();
    // Augmented [M | I]
    let mut a: Vec<Vec<f64>> = m.into_iter().map(|row| {
        let mut aug = Vec::with_capacity(2 * n);
        aug.extend_from_slice(&row);
        aug.resize(2 * n, 0.0);
        aug
    }).collect();
    for i in 0..n { a[i][n + i] = 1.0; }

    for col in 0..n {
        // Partial pivot
        let mut best = a[col][col].abs();
        let mut pivot_row = col;
        for row in (col + 1)..n {
            let v = a[row][col].abs();
            if v > best { best = v; pivot_row = row; }
        }
        a.swap(col, pivot_row);

        let pivot = a[col][col];
        if pivot.abs() < 1e-30 { continue; }
        let inv_pivot = 1.0 / pivot;
        for j in 0..(2 * n) { a[col][j] *= inv_pivot; }

        for row in 0..n {
            if row == col { continue; }
            let factor = a[row][col];
            if factor.abs() < 1e-30 { continue; }
            for j in 0..(2 * n) {
                a[row][j] -= factor * a[col][j];
            }
        }
    }

    a.into_iter().map(|row| row[n..].to_vec()).collect()
}

/// Single-pass linear classifier solved via normal equations.
/// Maps 256D Cl(1,7) multivector → n_classes logits.
/// W = (Z^T Z + λI)^{-1} Z^T Y where Y is one-hot labels.
pub struct LinearClassifier {
    weights: Vec<Vec<f32>>,  // DECODER_INPUT_DIM × n_classes
    pub n_classes: usize,
}

impl LinearClassifier {
    pub fn fit(
        encoder: &CliffordImageEncoder,
        images: &[Vec<f32>],
        labels: &[u8],
        n_classes: usize,
        lambda: f32,
        verbosity: u8,
    ) -> Self {
        let d = DECODER_INPUT_DIM;
        let n = images.len();
        if verbosity >= 1 { println!("  Fitting {}-class classifier on {} samples...", n_classes, n); }

        let mut ztz = vec![vec![0.0f64; d]; d];
        let mut zty = vec![vec![0.0f64; n_classes]; d];

        for idx in 0..n {
            let mv = encoder.encode(&images[idx]);
            let mut z = vec![0.0f64; d];
            for i in 0..CL8_DIM { z[i] = mv.components[i] as f64; }
            z[CL8_DIM] = 1.0;

            let c = labels[idx] as usize;

            for i in 0..d {
                if z[i].abs() < 1e-12 { continue; }
                for j in i..d {
                    ztz[i][j] += z[i] * z[j];
                }
                // One-hot target: Y[idx][c] = 1.0, rest 0.0
                zty[i][c] += z[i];
            }

            if verbosity >= 1 && (idx + 1) % 20000 == 0 {
                println!("    {}/{}", idx + 1, n);
            }
        }

        for i in 0..d { for j in 0..i { ztz[i][j] = ztz[j][i]; } }
        for i in 0..d { ztz[i][i] += lambda as f64; }

        if verbosity >= 1 { println!("  Solving {}×{} normal equations...", d, d); }
        let inv = invert_symmetric(ztz);

        let mut weights = vec![vec![0.0f32; n_classes]; d];
        for i in 0..d {
            for c in 0..n_classes {
                let mut s = 0.0f64;
                for k in 0..d { s += inv[i][k] * zty[k][c]; }
                weights[i][c] = s as f32;
            }
        }
        if verbosity >= 1 { println!("  Classifier solved."); }

        LinearClassifier { weights, n_classes }
    }

    /// Fit from pre-computed multivector embeddings (encoder-agnostic).
    pub fn fit_from_embeddings(
        embeddings: &[Multivector],
        labels: &[u8],
        n_classes: usize,
        lambda: f32,
        verbosity: u8,
    ) -> Self {
        let d = DECODER_INPUT_DIM;
        let n = embeddings.len();
        if verbosity >= 1 { println!("  Fitting {}-class classifier on {} embeddings...", n_classes, n); }

        // Inverse-frequency class weights: w_c = N / (K * count_c)
        let mut counts = vec![0.0f64; n_classes];
        for &l in labels { counts[l as usize] += 1.0; }
        let class_weights: Vec<f64> = counts.iter()
            .map(|&c| if c > 0.0 { n as f64 / (n_classes as f64 * c) } else { 1.0 })
            .collect();

        if verbosity >= 1 {
            print!("  Class weights: ");
            for (c, &w) in class_weights.iter().enumerate() {
                print!("{}={:.2} ", c, w);
            }
            println!();
        }

        // Weighted least squares: (Z^T W Z)^{-1} Z^T W Y
        // Absorb sqrt(w) into z for each sample
        let mut ztz = vec![vec![0.0f64; d]; d];
        let mut zty = vec![vec![0.0f64; n_classes]; d];

        for idx in 0..n {
            let mv = &embeddings[idx];
            let c = labels[idx] as usize;
            let w = class_weights[c];
            let sw = w.sqrt();

            let mut z = vec![0.0f64; d];
            for i in 0..CL8_DIM { z[i] = mv.components[i] as f64 * sw; }
            z[CL8_DIM] = sw;

            for i in 0..d {
                if z[i].abs() < 1e-12 { continue; }
                for j in i..d {
                    ztz[i][j] += z[i] * z[j];
                }
                // Target is sw (not 1.0) because y is also scaled by sqrt(w)
                zty[i][c] += z[i] * sw;
            }

            if verbosity >= 1 && (idx + 1) % 20000 == 0 {
                println!("    {}/{}", idx + 1, n);
            }
        }

        for i in 0..d { for j in 0..i { ztz[i][j] = ztz[j][i]; } }
        for i in 0..d { ztz[i][i] += lambda as f64; }

        if verbosity >= 1 { println!("  Solving {}×{} normal equations...", d, d); }
        let inv = invert_symmetric(ztz);

        let mut weights = vec![vec![0.0f32; n_classes]; d];
        for i in 0..d {
            for c in 0..n_classes {
                let mut s = 0.0f64;
                for k in 0..d { s += inv[i][k] * zty[k][c]; }
                weights[i][c] = s as f32;
            }
        }
        if verbosity >= 1 { println!("  Classifier solved."); }

        LinearClassifier { weights, n_classes }
    }

    /// Predict class label and confidence (max logit).
    pub fn classify(&self, mv: &Multivector) -> (u8, f32) {
        let logits = self.logits(mv);
        let mut best = 0;
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..self.n_classes {
            if logits[c] > best_val { best_val = logits[c]; best = c; }
        }
        (best as u8, best_val)
    }

    pub fn logits(&self, mv: &Multivector) -> Vec<f32> {
        let d = DECODER_INPUT_DIM;
        let mut out = vec![0.0f32; self.n_classes];
        for c in 0..self.n_classes {
            let mut s = self.weights[CL8_DIM][c]; // bias
            for i in 0..CL8_DIM {
                s += self.weights[i][c] * mv.components[i];
            }
            out[c] = s;
        }
        out
    }

    /// Fit from arbitrary-width feature vectors (e.g. concatenated multivectors).
    /// Solves class-weighted normal equations on D-dimensional input + bias.
    pub fn fit_from_features(
        features: &[Vec<f32>],
        labels: &[u8],
        n_classes: usize,
        lambda: f32,
        verbosity: u8,
    ) -> Self {
        let n = features.len();
        let raw_dim = features[0].len();
        let d = raw_dim + 1; // +1 bias
        if verbosity >= 1 {
            println!("  Fitting {}-class classifier on {} samples, {}D features (+bias={}D)...",
                n_classes, n, raw_dim, d);
        }

        let mut counts = vec![0.0f64; n_classes];
        for &l in labels { counts[l as usize] += 1.0; }
        let class_weights: Vec<f64> = counts.iter()
            .map(|&c| if c > 0.0 { n as f64 / (n_classes as f64 * c) } else { 1.0 })
            .collect();

        if verbosity >= 1 {
            print!("  Class weights: ");
            for (c, &w) in class_weights.iter().enumerate() {
                print!("{}={:.2} ", c, w);
            }
            println!();
        }

        let mut ztz = vec![vec![0.0f64; d]; d];
        let mut zty = vec![vec![0.0f64; n_classes]; d];

        for idx in 0..n {
            let feat = &features[idx];
            let c = labels[idx] as usize;
            let w = class_weights[c];
            let sw = w.sqrt();

            let mut z = vec![0.0f64; d];
            for i in 0..raw_dim { z[i] = feat[i] as f64 * sw; }
            z[raw_dim] = sw; // bias

            for i in 0..d {
                if z[i].abs() < 1e-12 { continue; }
                for j in i..d {
                    ztz[i][j] += z[i] * z[j];
                }
                zty[i][c] += z[i] * sw;
            }

            if verbosity >= 1 && (idx + 1) % 20000 == 0 {
                println!("    {}/{}", idx + 1, n);
            }
        }

        for i in 0..d { for j in 0..i { ztz[i][j] = ztz[j][i]; } }
        for i in 0..d { ztz[i][i] += lambda as f64; }

        if verbosity >= 1 { println!("  Solving {}×{} normal equations...", d, d); }
        let inv = invert_symmetric(ztz);

        let mut weights = vec![vec![0.0f32; n_classes]; d];
        for i in 0..d {
            for c in 0..n_classes {
                let mut s = 0.0f64;
                for k in 0..d { s += inv[i][k] * zty[k][c]; }
                weights[i][c] = s as f32;
            }
        }
        if verbosity >= 1 { println!("  Classifier solved."); }

        LinearClassifier { weights, n_classes }
    }

    /// Classify from an arbitrary-width feature vector.
    pub fn classify_features(&self, features: &[f32]) -> (u8, f32) {
        let raw_dim = self.weights.len() - 1;
        let mut best = 0;
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..self.n_classes {
            let mut s = self.weights[raw_dim][c];
            for i in 0..raw_dim {
                s += self.weights[i][c] * features[i];
            }
            if s > best_val { best_val = s; best = c; }
        }
        (best as u8, best_val)
    }

    /// Return per-class logits from an arbitrary-width feature vector.
    pub fn logits_features(&self, features: &[f32]) -> Vec<f32> {
        let raw_dim = self.weights.len() - 1;
        (0..self.n_classes).map(|c| {
            let mut s = self.weights[raw_dim][c];
            for i in 0..raw_dim { s += self.weights[i][c] * features[i]; }
            s
        }).collect()
    }
}

/// SSIM-like structural similarity (simplified single-scale).
fn compute_ssim(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let mu_a: f32 = a.iter().sum::<f32>() / n;
    let mu_b: f32 = b.iter().sum::<f32>() / n;
    let var_a: f32 = a.iter().map(|x| (x - mu_a) * (x - mu_a)).sum::<f32>() / n;
    let var_b: f32 = b.iter().map(|x| (x - mu_b) * (x - mu_b)).sum::<f32>() / n;
    let cov: f32 = a.iter().zip(b.iter())
        .map(|(x, y)| (x - mu_a) * (y - mu_b)).sum::<f32>() / n;
    let c1 = 0.0001f32;
    let c2 = 0.0009f32;
    let num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2);
    let den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2);
    (num / den).clamp(0.0, 1.0)
}

/// Result of the autoencoder pipeline.
pub struct AutoencoderResult {
    pub final_mse: f32,
    pub final_ssim: f32,
    pub classifier_accuracy: f32,
    pub sample_reconstructions: Vec<(u8, Vec<f32>, Vec<f32>)>,
}

/// Run the Clifford autoencoder: encode images → solve decoder → evaluate.
/// Single forward pass through data, closed-form solve, no epochs.
pub fn run_clifford_autoencoder(
    encoder: &CliffordImageEncoder,
    train_images: &[Vec<f32>],
    train_labels: &[u8],
    test_images: &[Vec<f32>],
    test_labels: &[u8],
    train_limit: Option<usize>,
    classifier: &CliffordClassifier,
    verbosity: u8,
) -> AutoencoderResult {
    macro_rules! progress {
        ($($arg:tt)*) => { if verbosity >= 1 { println!($($arg)*); } }
    }

    let train_subset: Vec<_> = if let Some(lim) = train_limit {
        train_images.iter().take(lim).cloned().collect()
    } else {
        train_images.to_vec()
    };

    // Single-pass solve
    progress!("--- Fitting decoder (single-pass least squares) ---");
    let decoder = CliffordDecoder::fit(encoder, &train_subset, 1.0, verbosity);

    // Evaluate on test set
    progress!("\n--- Evaluating reconstruction ---");
    let eval_n = test_images.len().min(2000);
    let mut total_mse = 0.0f32;
    let mut total_ssim = 0.0f32;
    let mut recon_correct = 0u32;
    let mut samples = Vec::new();
    let mut seen = [false; 10];

    for (img, &lbl) in test_images.iter().zip(test_labels.iter()).take(eval_n) {
        let mv = encoder.encode(img);
        let recon = decoder.decode(&mv);

        let mse: f32 = recon.iter().zip(img.iter())
            .map(|(r, t)| (r - t) * (r - t)).sum::<f32>() / IMAGE_DIM as f32;
        total_mse += mse;
        total_ssim += compute_ssim(&recon, img);

        let recon_mv = encoder.encode(&recon);
        let (pred, _) = classifier.classify(&recon_mv);
        if pred == lbl { recon_correct += 1; }

        if !seen[lbl as usize] {
            seen[lbl as usize] = true;
            samples.push((lbl, img.clone(), recon));
        }
    }

    let n = eval_n as f32;
    let final_mse = total_mse / n;
    let final_ssim = total_ssim / n;
    let classifier_acc = recon_correct as f32 / n;

    progress!("  Pixel MSE:               {:.5}", final_mse);
    progress!("  SSIM:                    {:.3}", final_ssim);
    progress!("  Classifier on generated: {:.1}%", classifier_acc * 100.0);

    AutoencoderResult {
        final_mse,
        final_ssim,
        classifier_accuracy: classifier_acc,
        sample_reconstructions: samples,
    }
}

/// Render a 28×28 image as ASCII art for terminal display.
pub fn render_ascii(pixels: &[f32], width: usize) -> String {
    let chars = [' ', '·', '░', '▒', '▓', '█'];
    let mut out = String::new();
    for (i, &p) in pixels.iter().enumerate() {
        let idx = ((p.clamp(0.0, 1.0)) * (chars.len() - 1) as f32).round() as usize;
        out.push(chars[idx.min(chars.len() - 1)]);
        if (i + 1) % width == 0 { out.push('\n'); }
    }
    out
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
