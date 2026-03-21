//! Cl(1,7) Clifford Algebra — SpaceTime-inspired geometric product engine.
//!
//! Provides the 256-dimensional real Clifford algebra over R^{1,7},
//! where e_0 is **timelike** (e_0² = −1) and e_1..e_7 are spacelike (e_i² = +1).
//! Multivectors are decomposed by grade:
//!
//!   grade 0: 1 scalar           (routing similarity)
//!   grade 1: 8 vectors          (1 timelike + 7 spacelike)
//!   grade 2: 28 bivectors       (7 boost planes + 21 rotation planes)
//!   grade 3: 56 trivectors
//!   grade 4: 70 quadvectors
//!   grade 5: 56
//!   grade 6: 28
//!   grade 7: 8
//!   grade 8: 1 pseudoscalar     (orientation)
//!   total:   256 basis blades
//!
//! The mixed metric signature gives the algebra causal structure:
//!   - Boost bivectors (e_0∧e_i) encode temporal/causal/sequential relations
//!   - Rotation bivectors (e_i∧e_j, i,j≥1) encode spatial/structural relations
//!   - Rotors include both spatial rotations and Lorentz boosts
//!
//! The geometric product `uv = u·v + u∧v` replaces:
//!   - bridge projection (grade extraction)
//!   - per-group adapters (rotor sandwich `R x R†`)
//!   - E8 quantization (grade-1 in Cl(1,7) = Minkowski-like vector space)
//!
//! Compact representation: only store and compute grades needed for each operation.

use serde::{Deserialize, Serialize};

/// Number of basis blades in Cl(1,7) = 2^8
pub const CL8_DIM: usize = 256;
pub const CL8_VECTOR_DIM: usize = 8;

/// Metric signature mask: bit i is set if e_i squares to −1 (timelike).
/// Cl(1,7): e_0 is timelike, e_1..e_7 are spacelike.
pub const TIMELIKE_MASK: u8 = 0b0000_0001;

/// Number of boost bivectors (timelike ∧ spacelike): 7
pub const BOOST_BIVECTOR_COUNT: usize = 7;
/// Number of rotation bivectors (spacelike ∧ spacelike): 21
pub const ROTATION_BIVECTOR_COUNT: usize = 21;

/// Binomial coefficients C(8,k) — dimensions of each grade
pub const GRADE_DIMS: [usize; 9] = [1, 8, 28, 56, 70, 56, 28, 8, 1];

/// Cumulative offsets into the flat 256-element representation
pub const GRADE_OFFSETS: [usize; 9] = [0, 1, 9, 37, 93, 163, 219, 247, 255];

/// A multivector in Cl(1,7), stored as 256 real components.
#[derive(Clone, Debug)]
pub struct Multivector {
    pub components: [f32; CL8_DIM],
}

impl Default for Multivector {
    fn default() -> Self {
        Self { components: [0.0; CL8_DIM] }
    }
}

impl Serialize for Multivector {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.components.as_slice().serialize(s)
    }
}

impl<'de> Deserialize<'de> for Multivector {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v: Vec<f32> = Vec::deserialize(d)?;
        let mut components = [0.0f32; CL8_DIM];
        for (i, &val) in v.iter().enumerate().take(CL8_DIM) {
            components[i] = val;
        }
        Ok(Self { components })
    }
}

/// A rotor (even-grade multivector) for representing rotations in Cl(8).
/// Stores only even grades (0, 2, 4, 6, 8) = 1 + 28 + 70 + 28 + 1 = 128 components.
#[derive(Clone, Debug)]
pub struct Rotor {
    pub components: [f32; 128],
}

impl Default for Rotor {
    fn default() -> Self {
        let mut r = Self { components: [0.0; 128] };
        r.components[0] = 1.0; // identity rotor
        r
    }
}

impl Serialize for Rotor {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.components.as_slice().serialize(s)
    }
}

impl<'de> Deserialize<'de> for Rotor {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v: Vec<f32> = Vec::deserialize(d)?;
        let mut components = [0.0f32; 128];
        for (i, &val) in v.iter().enumerate().take(128) {
            components[i] = val;
        }
        Ok(Self { components })
    }
}

/// Geometric product sign and result blade for Cl(1,7).
///
/// For basis blades e_I and e_J, returns (sign, K) where e_I * e_J = sign * e_K.
/// The sign accounts for both transposition parity AND the metric signature:
/// each shared timelike basis vector (e_0) contributes an extra −1.
///
/// Blade indices use the canonical bitmap encoding:
///   e_0 = 0b00000001, e_1 = 0b00000010, ..., e_7 = 0b10000000
///   e_{01} = 0b00000011, etc.
fn geo_sign_and_index(a: u8, b: u8) -> (f32, u8) {
    let result_blade = a ^ b;
    let shared = a & b;

    // Count transpositions: for each bit in b, count how many higher bits in a
    // must be passed through (each swap contributes a sign flip).
    let mut swaps = 0u32;
    let mut b_remaining = b;
    while b_remaining != 0 {
        let lowest_b = b_remaining & b_remaining.wrapping_neg();
        let a_above = a & !((lowest_b << 1).wrapping_sub(1));
        swaps += a_above.count_ones();
        b_remaining &= b_remaining - 1;
    }

    // Metric contractions: each shared basis vector that is timelike (squares to −1)
    // contributes an additional sign flip.  In Cl(1,7), only e_0 is timelike.
    let timelike_contractions = (shared & TIMELIKE_MASK).count_ones();

    let total_negatives = swaps + timelike_contractions;
    let sign = if total_negatives % 2 == 0 { 1.0 } else { -1.0 };
    (sign, result_blade)
}

/// Map a blade bitmap (0..255) to its grade (popcount).
#[inline]
fn blade_grade(blade: u8) -> usize {
    blade.count_ones() as usize
}

/// Map a blade bitmap to its index within its grade.
/// Blades of grade k are enumerated in ascending bitmap order.
fn blade_to_grade_index(blade: u8) -> usize {
    let grade = blade_grade(blade);
    let mut idx = 0;
    for b in 0..blade {
        if (b as u8).count_ones() as usize == grade {
            idx += 1;
        }
    }
    idx
}

/// Map a blade bitmap to its flat index in the 256-element multivector.
#[inline]
fn blade_flat_index(blade: u8) -> usize {
    let grade = blade_grade(blade);
    GRADE_OFFSETS[grade] + blade_to_grade_index(blade)
}

impl Multivector {
    pub fn zero() -> Self {
        Self::default()
    }

    pub fn scalar(s: f32) -> Self {
        let mut mv = Self::zero();
        mv.components[0] = s;
        mv
    }

    /// Construct a grade-1 vector from 8 components.
    pub fn vector(v: &[f32; CL8_VECTOR_DIM]) -> Self {
        let mut mv = Self::zero();
        mv.components[GRADE_OFFSETS[1]..GRADE_OFFSETS[1] + 8]
            .copy_from_slice(v);
        mv
    }

    /// Construct a grade-1 vector from a slice (takes first 8 elements).
    pub fn vector_from_slice(v: &[f32]) -> Self {
        let mut arr = [0.0f32; 8];
        for (i, a) in arr.iter_mut().enumerate() {
            *a = v.get(i).copied().unwrap_or(0.0);
        }
        Self::vector(&arr)
    }

    /// Extract grade-k components as a slice.
    pub fn grade(&self, k: usize) -> &[f32] {
        if k > 8 { return &[]; }
        let start = GRADE_OFFSETS[k];
        let end = start + GRADE_DIMS[k];
        &self.components[start..end]
    }

    /// Extract grade-k components as a mutable slice.
    pub fn grade_mut(&mut self, k: usize) -> &mut [f32] {
        if k > 8 { return &mut []; }
        let start = GRADE_OFFSETS[k];
        let end = start + GRADE_DIMS[k];
        &mut self.components[start..end]
    }

    /// Scalar part (grade 0).
    pub fn scalar_part(&self) -> f32 {
        self.components[0]
    }

    /// Vector part (grade 1) — the 8d E8-space component.
    pub fn vector_part(&self) -> [f32; 8] {
        let mut v = [0.0f32; 8];
        v.copy_from_slice(self.grade(1));
        v
    }

    /// Bivector part (grade 2) — 28 components encoding rotational structure.
    pub fn bivector_part(&self) -> &[f32] {
        self.grade(2)
    }

    /// Pseudoscalar part (grade 8).
    pub fn pseudoscalar_part(&self) -> f32 {
        self.components[CL8_DIM - 1]
    }

    /// Full geometric product: self * other.
    /// O(256²) = O(65536) multiply-adds, but most are zero in practice
    /// when inputs are sparse (low-grade).
    pub fn geo(&self, other: &Multivector) -> Multivector {
        let mut result = Multivector::zero();
        for a in 0u16..256 {
            let sa = self.components[blade_flat_index(a as u8)];
            if sa.abs() < 1e-12 { continue; }
            for b in 0u16..256 {
                let sb = other.components[blade_flat_index(b as u8)];
                if sb.abs() < 1e-12 { continue; }
                let (sign, blade) = geo_sign_and_index(a as u8, b as u8);
                result.components[blade_flat_index(blade)] += sign * sa * sb;
            }
        }
        result
    }

    /// Inner product (grade-lowering): ⟨self, other⟩
    /// Extracts grade |r-s| from the geometric product of grade-r and grade-s parts.
    pub fn inner(&self, other: &Multivector) -> f32 {
        let product = self.geo(other);
        product.scalar_part()
    }

    /// Outer (wedge) product: self ∧ other
    /// Extracts grade r+s from the geometric product of grade-r and grade-s parts.
    pub fn wedge(&self, other: &Multivector) -> Multivector {
        let mut result = Multivector::zero();
        for a in 0u16..256 {
            let sa = self.components[blade_flat_index(a as u8)];
            if sa.abs() < 1e-12 { continue; }
            let grade_a = blade_grade(a as u8);
            for b in 0u16..256 {
                let sb = other.components[blade_flat_index(b as u8)];
                if sb.abs() < 1e-12 { continue; }
                let grade_b = blade_grade(b as u8);
                let (sign, blade) = geo_sign_and_index(a as u8, b as u8);
                if blade_grade(blade) == grade_a + grade_b {
                    result.components[blade_flat_index(blade)] += sign * sa * sb;
                }
            }
        }
        result
    }

    /// Reverse: reverse the order of basis vectors in each blade.
    /// For a grade-k blade: rev(e_{i1..ik}) = (-1)^{k(k-1)/2} e_{i1..ik}
    pub fn reverse(&self) -> Multivector {
        let mut result = self.clone();
        for blade in 0u16..256 {
            let k = blade_grade(blade as u8);
            // Reversal sign: (-1)^{k(k-1)/2} — grades 2,3,6,7 flip sign
            let sign = match k % 4 { 2 | 3 => -1.0, _ => 1.0 };
            result.components[blade_flat_index(blade as u8)] *= sign;
        }
        result
    }

    /// Squared magnitude: self * reverse(self) scalar part.
    pub fn norm_sq(&self) -> f32 {
        let rev = self.reverse();
        self.inner(&rev)
    }

    /// L2 norm of all 256 components (not the Clifford norm, but useful for normalization).
    pub fn component_norm(&self) -> f32 {
        self.components.iter().map(|c| c * c).sum::<f32>().sqrt()
    }

    /// Normalize all components to unit L2 norm.
    pub fn normalize(&mut self) {
        let n = self.component_norm();
        if n > 1e-12 {
            for c in self.components.iter_mut() {
                *c /= n;
            }
        }
    }

    /// Add two multivectors.
    pub fn add(&self, other: &Multivector) -> Multivector {
        let mut result = Multivector::zero();
        for i in 0..CL8_DIM {
            result.components[i] = self.components[i] + other.components[i];
        }
        result
    }

    /// Scale by a scalar.
    pub fn scale(&self, s: f32) -> Multivector {
        let mut result = self.clone();
        for c in result.components.iter_mut() {
            *c *= s;
        }
        result
    }

    /// Project to a specific grade, zeroing all others.
    pub fn grade_project(&self, k: usize) -> Multivector {
        let mut result = Multivector::zero();
        if k <= 8 {
            let start = GRADE_OFFSETS[k];
            let dim = GRADE_DIMS[k];
            result.components[start..start + dim]
                .copy_from_slice(&self.components[start..start + dim]);
        }
        result
    }
}

impl Rotor {
    /// Identity rotor (scalar = 1, everything else 0).
    pub fn identity() -> Self {
        Self::default()
    }

    /// Build a rotor from a bivector: R = exp(B/2) ≈ 1 + B/2 for small B.
    /// For exact computation, uses the Taylor series.
    pub fn from_bivector(bivector: &[f32]) -> Self {
        debug_assert!(bivector.len() == 28);
        let mut r = Self::identity();
        // Copy bivector into grade-2 slot (offset 1 in even-grade representation)
        // Even grades: 0(1), 2(28), 4(70), 6(28), 8(1) = 128
        // Offsets:      0,    1,     29,    99,    127
        let b_norm_sq: f32 = bivector.iter().map(|b| b * b).sum();
        let b_norm = b_norm_sq.sqrt();
        if b_norm < 1e-10 {
            return r;
        }
        let half_angle = b_norm / 2.0;
        let cos_ha = half_angle.cos();
        let sinc_ha = if half_angle.abs() < 1e-6 {
            0.5 // lim sin(x/2)/(x) as x->0 = 1/2
        } else {
            half_angle.sin() / b_norm
        };
        r.components[0] = cos_ha;
        for (i, &b) in bivector.iter().enumerate() {
            r.components[1 + i] = -sinc_ha * b; // R = cos(θ/2) - sin(θ/2) * B̂
        }
        r
    }

    /// Apply rotor to a grade-1 vector: v' = R v R†
    /// This is the sandwich product that performs rotation.
    pub fn rotate_vector(&self, v: &[f32; 8]) -> [f32; 8] {
        // For efficiency, implement the sandwich product directly for vectors.
        // Full version: convert rotor to Multivector, compute R * v * R†
        let mv_v = Multivector::vector(v);
        let mv_r = self.to_multivector();
        let mv_r_rev = mv_r.reverse();
        let rotated = mv_r.geo(&mv_v).geo(&mv_r_rev);
        rotated.vector_part()
    }

    /// Convert rotor to full Multivector (only even grades populated).
    pub fn to_multivector(&self) -> Multivector {
        let mut mv = Multivector::zero();
        // grade 0: 1 component
        mv.components[GRADE_OFFSETS[0]] = self.components[0];
        // grade 2: 28 components
        mv.components[GRADE_OFFSETS[2]..GRADE_OFFSETS[2] + 28]
            .copy_from_slice(&self.components[1..29]);
        // grade 4: 70 components
        mv.components[GRADE_OFFSETS[4]..GRADE_OFFSETS[4] + 70]
            .copy_from_slice(&self.components[29..99]);
        // grade 6: 28 components
        mv.components[GRADE_OFFSETS[6]..GRADE_OFFSETS[6] + 28]
            .copy_from_slice(&self.components[99..127]);
        // grade 8: 1 component
        mv.components[GRADE_OFFSETS[8]] = self.components[127];
        mv
    }

    /// Number of trainable parameters (just the bivector part for simple rotors).
    pub fn param_count() -> usize {
        28
    }
}

/// Embed an n-dimensional bridge vector into Cl(8) by chunking into 8d blocks.
/// Each block becomes a grade-1 vector; the full embedding is their sum
/// (capturing the complete signal across E8 subspaces).
pub fn embed_bridge_vector(v: &[f32]) -> Multivector {
    embed_bridge_vector_with_goal(v, 0.0)
}

/// Embed an N-dimensional vector into Cl(1,7), with an explicit goal magnitude
/// injected into the timelike (e_0) component.  When `goal_mag` is 0.0 this
/// is identical to the original embedding; when non-zero it gives the
/// multivector a causal "direction" that distinguishes passive queries from
/// directed actions.
pub fn embed_bridge_vector_with_goal(v: &[f32], goal_mag: f32) -> Multivector {
    let num_blocks = (v.len() + 7) / 8;
    let mut result = Multivector::zero();
    for block in 0..num_blocks {
        let offset = block * 8;
        let mut chunk = [0.0f32; 8];
        for i in 0..8 {
            chunk[i] = v.get(offset + i).copied().unwrap_or(0.0);
        }
        let block_mv = Multivector::vector(&chunk);
        if block == 0 {
            for i in 0..CL8_DIM {
                result.components[i] = block_mv.components[i];
            }
        } else {
            let wedge = result.wedge(&block_mv);
            result = result.add(&block_mv);
            let alpha = 1.0 / (block as f32 + 1.0);
            for i in GRADE_OFFSETS[2]..GRADE_OFFSETS[2] + GRADE_DIMS[2] {
                result.components[i] += alpha * wedge.components[i];
            }
        }
    }

    // Inject goal magnitude into the timelike dimension (e_0, grade-1 index 0).
    // This replaces the accumulated e_0 content with the intentional goal signal,
    // blended with whatever content was already there.
    if goal_mag.abs() > 1e-8 {
        let e0_idx = GRADE_OFFSETS[1]; // first grade-1 component = e_0
        let existing = result.components[e0_idx];
        result.components[e0_idx] = existing * 0.5 + goal_mag * 0.5;
    }

    result
}

/// Trainable per-group rotor with SPSA-based learning.
/// Wraps a Rotor (28 bivector parameters) with training state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroupRotor {
    pub bivector: Vec<f32>,  // 28 trainable parameters
    pub frozen: bool,
    pub l2_weight: f32,
}

impl GroupRotor {
    pub fn new() -> Self {
        Self {
            bivector: vec![0.0f32; 28],
            frozen: false,
            l2_weight: 1e-4,
        }
    }

    pub fn rotor(&self) -> Rotor {
        Rotor::from_bivector(&self.bivector)
    }

    /// Full Clifford conditioning pipeline:
    /// raw 768d → embed into Cl(1,7) → rotate/boost by group rotor → extract flat vector.
    pub fn condition(&self, h_raw: &[f32], target_dim: usize) -> Vec<f32> {
        let mv = embed_bridge_vector(h_raw);
        let rotor = self.rotor();
        let rotated = apply_group_rotor(&mv, &rotor);
        extract_conditioning(&rotated, target_dim)
    }

    /// SPSA training step: perturb all 28 bivector params simultaneously,
    /// evaluate loss in both directions, estimate gradient, update.
    /// `loss_fn` takes a conditioning vector and returns a scalar loss.
    pub fn train_step_spsa<F>(&mut self, h_raw: &[f32], target_dim: usize, mut loss_fn: F, lr: f32)
    where
        F: FnMut(&[f32]) -> f32,
    {
        if self.frozen { return; }
        let eps = 0.02f32;
        let mut perturb = vec![0.0f32; 28];
        let mut seed = self.bivector.iter().map(|b| (b * 1000.0) as u64).sum::<u64>().wrapping_add(7);
        for p in perturb.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *p = if seed % 2 == 0 { 1.0 } else { -1.0 };
        }

        // Evaluate loss with +perturbation
        let mut bv_plus = self.bivector.clone();
        let mut bv_minus = self.bivector.clone();
        for i in 0..28 {
            bv_plus[i] += eps * perturb[i];
            bv_minus[i] -= eps * perturb[i];
        }

        let r_plus = Rotor::from_bivector(&bv_plus);
        let mv = embed_bridge_vector(h_raw);
        let cond_plus = extract_conditioning(&apply_group_rotor(&mv, &r_plus), target_dim);

        let r_minus = Rotor::from_bivector(&bv_minus);
        let cond_minus = extract_conditioning(&apply_group_rotor(&mv, &r_minus), target_dim);

        let l_plus = loss_fn(&cond_plus);
        let l_minus = loss_fn(&cond_minus);
        let scale = (l_plus - l_minus) / (2.0 * eps);

        for i in 0..28 {
            let grad = scale / perturb[i] + self.l2_weight * self.bivector[i];
            self.bivector[i] -= lr * grad;
        }
    }

    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    pub fn param_count() -> usize {
        28
    }
}

/// Extract routing score: the scalar part of the geometric product of two multivectors.
/// Higher values = more aligned in Cl(1,7) space. Note: the timelike inner product
/// contributes −u₀v₀, so agreement on the timelike axis *lowers* similarity while
/// agreement on spacelike axes raises it — encoding causal distinctness vs content overlap.
pub fn geometric_similarity(a: &Multivector, b: &Multivector) -> f32 {
    a.inner(b)
}

/// Apply a per-group rotor to adapt a multivector for generation conditioning.
/// Returns a new multivector rotated by the group's learned rotor.
pub fn apply_group_rotor(mv: &Multivector, rotor: &Rotor) -> Multivector {
    let r = rotor.to_multivector();
    let r_rev = r.reverse();
    r.geo(mv).geo(&r_rev)
}

/// Extract a flat conditioning vector from a multivector.
/// Uses grade-1 (8d) + grade-2 (28d) = 36d, or padded to target_dim.
pub fn extract_conditioning(mv: &Multivector, target_dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(target_dim);
    // Grade 1: direct signal (8 components: 1 timelike + 7 spacelike)
    out.extend_from_slice(mv.grade(1));
    // Grade 2 boost bivectors: causal/temporal structure (7 components)
    let bv = mv.grade(2);
    out.extend_from_slice(&bv[..BOOST_BIVECTOR_COUNT]);
    // Grade 2 rotation bivectors: spatial/relational structure (21 components)
    out.extend_from_slice(&bv[BOOST_BIVECTOR_COUNT..]);
    // Grade 0: scalar similarity (1 component)
    out.push(mv.scalar_part());
    // Pad or truncate to target_dim
    out.resize(target_dim, 0.0);
    out
}

// ---------------------------------------------------------------------------
// Understanding primitives — grade-separated reasoning over Cl(8)
// ---------------------------------------------------------------------------

/// Extract the *structural fingerprint* of a multivector: grade-2 bivector
/// components that encode pairwise feature relationships, independent of
/// the specific content (grade-1).  Two inputs about different topics but
/// with the same relational structure will have similar fingerprints.
pub fn structural_fingerprint(mv: &Multivector) -> [f32; 28] {
    let mut fp = [0.0f32; 28];
    fp.copy_from_slice(mv.grade(2));
    fp
}

/// Extract the 7d causal fingerprint: boost bivector components (e_0∧e_i).
/// These encode temporal/sequential/causal relationships — "what leads to what."
/// In the canonical blade ordering, the first 7 bivectors are e_{01}..e_{07},
/// i.e. exactly the boost planes involving the timelike dimension.
pub fn causal_fingerprint(mv: &Multivector) -> [f32; BOOST_BIVECTOR_COUNT] {
    let mut fp = [0.0f32; BOOST_BIVECTOR_COUNT];
    let bv = mv.grade(2);
    fp.copy_from_slice(&bv[..BOOST_BIVECTOR_COUNT]);
    fp
}

/// Extract the 21d spatial fingerprint: rotation bivector components (e_i∧e_j, i,j≥1).
/// These encode structural/relational similarity — "what kind of thing is this."
pub fn spatial_fingerprint(mv: &Multivector) -> [f32; ROTATION_BIVECTOR_COUNT] {
    let mut fp = [0.0f32; ROTATION_BIVECTOR_COUNT];
    let bv = mv.grade(2);
    fp.copy_from_slice(&bv[BOOST_BIVECTOR_COUNT..]);
    fp
}

/// Cosine similarity between two causal fingerprints.
/// High value means two inputs share the same goal/action/temporal structure.
pub fn causal_similarity(a: &Multivector, b: &Multivector) -> f32 {
    let fa = causal_fingerprint(a);
    let fb = causal_fingerprint(b);
    let dot: f32 = fa.iter().zip(fb.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = fa.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = fb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 { return 0.0; }
    dot / (na * nb)
}

/// Cosine similarity in bivector (grade-2) space only.
/// Measures whether two inputs share the same *structure* regardless of
/// surface content.  Used for understanding-based routing: a novel topic
/// routes to the group whose training data shares its relational pattern.
pub fn structural_similarity(a: &Multivector, b: &Multivector) -> f32 {
    let fa = structural_fingerprint(a);
    let fb = structural_fingerprint(b);
    let dot: f32 = fa.iter().zip(fb.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = fa.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = fb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 { return 0.0; }
    dot / (na * nb)
}

/// Abstraction: project a multivector to grade-2+, zeroing grade-0 and
/// grade-1.  The result retains relational/structural information while
/// discarding topic-specific features.  Two inputs that are "about the
/// same kind of thing" collapse to nearby points.
pub fn abstract_mv(mv: &Multivector) -> Multivector {
    let mut result = mv.clone();
    result.components[GRADE_OFFSETS[0]] = 0.0;
    for i in GRADE_OFFSETS[1]..GRADE_OFFSETS[1] + GRADE_DIMS[1] {
        result.components[i] = 0.0;
    }
    result
}

/// Cross-domain transfer: given a source rotor (learned on domain A) and a
/// target rotor (learned on domain B), compute the *transfer rotor* that
/// maps A-structure to B-structure.  Applying this to a novel input from
/// domain A produces a conditioning vector appropriate for domain B's
/// generation head — analogical reasoning via rotor composition.
///
/// transfer = R_target · R_source^{-1}
/// For unit rotors, R^{-1} = R̃ (reverse).
pub fn transfer_rotor(source: &Rotor, target: &Rotor) -> Rotor {
    let s_mv = source.to_multivector();
    let t_mv = target.to_multivector();
    let s_rev = s_mv.reverse();
    let transfer_mv = t_mv.geo(&s_rev);
    // Extract even-grade components back into a Rotor
    let mut r = Rotor::identity();
    r.components[0] = transfer_mv.components[GRADE_OFFSETS[0]];
    for i in 0..28 {
        r.components[1 + i] = transfer_mv.components[GRADE_OFFSETS[2] + i];
    }
    for i in 0..70 {
        r.components[29 + i] = transfer_mv.components[GRADE_OFFSETS[4] + i];
    }
    for i in 0..28 {
        r.components[99 + i] = transfer_mv.components[GRADE_OFFSETS[6] + i];
    }
    r.components[127] = transfer_mv.components[GRADE_OFFSETS[8]];
    r
}

/// Understanding-aware conditioning: embed the raw input, separate content
/// (grade-1: timelike goal + spacelike topic) from structure (grade-2:
/// boost causality + rotation similarity), apply the group rotor to structure
/// only, then recombine.  The rotor now includes both spatial rotations
/// (adapting relational pattern) and boosts (adapting causal/temporal ordering).
pub fn condition_with_understanding(
    h_raw: &[f32],
    rotor: &Rotor,
    target_dim: usize,
) -> (Vec<f32>, [f32; 28]) {
    condition_with_understanding_goal(h_raw, rotor, target_dim, 0.0)
}

/// Goal-aware variant: injects the goal magnitude from the UnderstandingLayer
/// into the timelike dimension before separating content and structure.
pub fn condition_with_understanding_goal(
    h_raw: &[f32],
    rotor: &Rotor,
    target_dim: usize,
    goal_mag: f32,
) -> (Vec<f32>, [f32; 28]) {
    let mv = embed_bridge_vector_with_goal(h_raw, goal_mag);
    let content = mv.grade_project(1);
    let structure = abstract_mv(&mv);
    let rotated_structure = apply_group_rotor(&structure, rotor);
    let combined = content.add(&rotated_structure);
    let fingerprint = structural_fingerprint(&combined);
    (extract_conditioning(&combined, target_dim), fingerprint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grade_dimensions_sum_to_256() {
        assert_eq!(GRADE_DIMS.iter().sum::<usize>(), CL8_DIM);
    }

    #[test]
    fn test_grade_offsets_consistent() {
        for k in 0..8 {
            assert_eq!(GRADE_OFFSETS[k + 1], GRADE_OFFSETS[k] + GRADE_DIMS[k]);
        }
        assert_eq!(GRADE_OFFSETS[8] + GRADE_DIMS[8], CL8_DIM);
    }

    #[test]
    fn test_geo_sign_basis_vectors() {
        // e_0 * e_0 = -1 in Cl(1,7) (timelike: squares to -1)
        let (sign, blade) = geo_sign_and_index(0b00000001, 0b00000001);
        assert_eq!(blade, 0); // scalar
        assert_eq!(sign, -1.0);

        // e_1 * e_1 = +1 in Cl(1,7) (spacelike: squares to +1)
        let (sign, blade) = geo_sign_and_index(0b00000010, 0b00000010);
        assert_eq!(blade, 0);
        assert_eq!(sign, 1.0);

        // e_0 * e_1 = e_{01} (no shared basis, no metric effect)
        let (sign, blade) = geo_sign_and_index(0b00000001, 0b00000010);
        assert_eq!(blade, 0b00000011);
        assert_eq!(sign, 1.0);

        // e_1 * e_0 = -e_{01} (anticommutative for orthogonal vectors)
        let (sign, blade) = geo_sign_and_index(0b00000010, 0b00000001);
        assert_eq!(blade, 0b00000011);
        assert_eq!(sign, -1.0);
    }

    #[test]
    fn test_vector_construction_and_extraction() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mv = Multivector::vector(&v);
        assert_eq!(mv.vector_part(), v);
        assert_eq!(mv.scalar_part(), 0.0);
        assert!(mv.bivector_part().iter().all(|&b| b == 0.0));
    }

    #[test]
    fn test_geometric_product_vectors() {
        // For two grade-1 vectors u, v: uv = u·v + u∧v
        let u = Multivector::vector(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = Multivector::vector(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let uv = u.geo(&v);
        // Scalar part (u·v) should be 0 (orthogonal)
        assert!(uv.scalar_part().abs() < 1e-6);
        // Bivector part should be non-zero (e12 component)
        assert!(uv.bivector_part().iter().any(|&b| b.abs() > 0.5));
    }

    #[test]
    fn test_geometric_product_parallel_vectors_timelike() {
        // e_0 is timelike: u = 3·e_0, u·u = -9 (Minkowski inner product)
        let u = Multivector::vector(&[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let uu = u.geo(&u);
        assert!((uu.scalar_part() - (-9.0)).abs() < 1e-5,
            "timelike u·u should be -9 in Cl(1,7), got {}", uu.scalar_part());
        assert!(uu.bivector_part().iter().all(|&b| b.abs() < 1e-6));
    }

    #[test]
    fn test_geometric_product_parallel_vectors_spacelike() {
        // e_1 is spacelike: v = 3·e_1, v·v = +9 (Euclidean)
        let v = Multivector::vector(&[0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let vv = v.geo(&v);
        assert!((vv.scalar_part() - 9.0).abs() < 1e-5,
            "spacelike v·v should be +9 in Cl(1,7), got {}", vv.scalar_part());
        assert!(vv.bivector_part().iter().all(|&b| b.abs() < 1e-6));
    }

    #[test]
    fn test_reverse_grades() {
        let v = Multivector::vector(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let rev = v.reverse();
        // Grade 1 is unchanged under reversal: (-1)^{1*0/2} = 1
        assert_eq!(v.vector_part(), rev.vector_part());
    }

    #[test]
    fn test_rotor_identity_preserves_vector() {
        let r = Rotor::identity();
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rotated = r.rotate_vector(&v);
        for i in 0..8 {
            assert!((rotated[i] - v[i]).abs() < 1e-4,
                "identity rotor should preserve vector: {} vs {}", rotated[i], v[i]);
        }
    }

    #[test]
    fn test_rotor_rotation_preserves_norm() {
        let bivector = [0.3f32, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let r = Rotor::from_bivector(&bivector);
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rotated = r.rotate_vector(&v);

        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_after: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 0.1 * norm_before,
            "rotor should approximately preserve norm: {} vs {}", norm_before, norm_after);
    }

    #[test]
    fn test_embed_bridge_vector() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01).sin()).collect();
        let mv = embed_bridge_vector(&v);
        // Should have non-zero grade-1 and grade-2 components
        assert!(mv.grade(1).iter().any(|&x| x.abs() > 0.001));
        assert!(mv.grade(2).iter().any(|&x| x.abs() > 0.001));
    }

    #[test]
    fn test_geometric_similarity_self_spacelike() {
        // Spacelike vector: self-similarity is positive
        let v = Multivector::vector(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let sim = geometric_similarity(&v, &v);
        assert!(sim > 0.0, "spacelike self-similarity should be positive: {}", sim);
    }

    #[test]
    fn test_geometric_similarity_self_timelike() {
        // Timelike vector: self-inner-product is negative in Cl(1,7)
        let v = Multivector::vector(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let sim = geometric_similarity(&v, &v);
        assert!(sim < 0.0, "timelike self-similarity should be negative in Cl(1,7): {}", sim);
    }

    #[test]
    fn test_geometric_similarity_orthogonal() {
        let u = Multivector::vector(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = Multivector::vector(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let sim = geometric_similarity(&u, &v);
        assert!(sim.abs() < 1e-6, "orthogonal similarity should be ~0: {}", sim);
    }

    #[test]
    fn test_extract_conditioning_size() {
        let mv = Multivector::vector(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let cond = extract_conditioning(&mv, 128);
        assert_eq!(cond.len(), 128);
        assert!(cond[0..8].iter().any(|&x| x.abs() > 0.5));
    }

    #[test]
    fn test_rotor_different_from_identity() {
        let bivector = [0.5f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let r = Rotor::from_bivector(&bivector);
        let v = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let rotated = r.rotate_vector(&v);
        assert!(rotated != v, "non-identity rotor should change vector");
    }

    #[test]
    fn test_wedge_anticommutative() {
        let u = Multivector::vector(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let v = Multivector::vector(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let uv = u.wedge(&v);
        let vu = v.wedge(&u);
        // u∧v = -v∧u for grade-1 vectors
        for i in 0..CL8_DIM {
            assert!((uv.components[i] + vu.components[i]).abs() < 1e-6,
                "wedge should be anticommutative at component {}: {} vs {}",
                i, uv.components[i], vu.components[i]);
        }
    }

    #[test]
    fn test_grade_projection() {
        let u = Multivector::vector(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let v = Multivector::vector(&[0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let product = u.geo(&v);

        let grade0 = product.grade_project(0);
        let grade2 = product.grade_project(2);

        // Grade-0 projection should only have scalar
        assert!(grade0.grade(1).iter().all(|&x| x.abs() < 1e-6));
        assert!(grade0.grade(2).iter().all(|&x| x.abs() < 1e-6));

        // Grade-2 projection should only have bivector
        assert!(grade2.scalar_part().abs() < 1e-6);
        assert!(grade2.grade(1).iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_structural_fingerprint_is_grade2() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.03).sin()).collect();
        let mv = embed_bridge_vector(&v);
        let fp = structural_fingerprint(&mv);
        assert_eq!(fp.len(), 28);
        assert!(fp.iter().any(|&x| x.abs() > 0.001), "fingerprint should be non-trivial");
    }

    #[test]
    fn test_structural_similarity_same_structure() {
        // Two inputs with same structure but different magnitude
        let v1: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let v2: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin() * 2.0).collect();
        let mv1 = embed_bridge_vector(&v1);
        let mv2 = embed_bridge_vector(&v2);
        let sim = structural_similarity(&mv1, &mv2);
        assert!(sim > 0.9, "same-structure inputs should have high structural similarity: {}", sim);
    }

    #[test]
    fn test_structural_similarity_different_structure() {
        // Sparse: only first block has signal
        let mut v1 = vec![0.0f32; 128];
        for i in 0..8 { v1[i] = (i as f32 + 1.0) * 0.5; }
        // Dense: all blocks have signal with varying magnitudes
        let v2: Vec<f32> = (0..128).map(|i| ((i as f32 * 0.37).cos()) * (1.0 + (i / 8) as f32)).collect();
        let mv1 = embed_bridge_vector(&v1);
        let mv2 = embed_bridge_vector(&v2);
        let sim = structural_similarity(&mv1, &mv2);
        // Single-block vs multi-block should produce different wedge structure
        assert!(sim < 0.95, "sparse vs dense inputs should have distinct structural similarity: {}", sim);
    }

    #[test]
    fn test_abstract_mv_zeroes_content() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.02).sin()).collect();
        let mv = embed_bridge_vector(&v);
        let abs = abstract_mv(&mv);
        assert!(abs.scalar_part().abs() < 1e-10, "abstraction should zero grade-0");
        assert!(abs.grade(1).iter().all(|&x| x.abs() < 1e-10), "abstraction should zero grade-1");
        assert!(abs.grade(2).iter().any(|&x| x.abs() > 0.001), "abstraction should preserve grade-2");
    }

    #[test]
    fn test_transfer_rotor_identity_roundtrip() {
        let bv = [0.1f32, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let r = Rotor::from_bivector(&bv);
        let id = Rotor::identity();
        let t = transfer_rotor(&id, &r);
        // Transfer from identity to R should ≈ R
        let v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let direct = r.rotate_vector(&v);
        let via_transfer = t.rotate_vector(&v);
        for i in 0..8 {
            assert!((direct[i] - via_transfer[i]).abs() < 0.15,
                "transfer from identity should approximate source rotor: dim {} {} vs {}",
                i, direct[i], via_transfer[i]);
        }
    }

    #[test]
    fn test_condition_with_understanding_produces_output() {
        let h_raw: Vec<f32> = (0..768).map(|i| (i as f32 * 0.007).sin()).collect();
        let bv = [0.05f32; 28];
        let r = Rotor::from_bivector(&bv);
        let (cond, fp) = condition_with_understanding(&h_raw, &r, 128);
        assert_eq!(cond.len(), 128);
        assert!(cond.iter().any(|&x| x.abs() > 0.001), "conditioning should be non-trivial");
        assert!(fp.iter().any(|&x| x.abs() > 0.001), "fingerprint should be non-trivial");
    }

    #[test]
    fn test_understanding_preserves_content_adapts_structure() {
        let h_raw: Vec<f32> = (0..768).map(|i| (i as f32 * 0.007).sin()).collect();
        let id = Rotor::identity();
        let bv = [0.2f32, -0.1, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let r = Rotor::from_bivector(&bv);
        let (cond_id, fp_id) = condition_with_understanding(&h_raw, &id, 128);
        let (cond_r, fp_r) = condition_with_understanding(&h_raw, &r, 128);
        // Same input: grade-1 content should be identical (first 8 components)
        for i in 0..8 {
            assert!((cond_id[i] - cond_r[i]).abs() < 1e-4,
                "content (grade-1) should be preserved: dim {} {} vs {}", i, cond_id[i], cond_r[i]);
        }
        // Structure (fingerprint) should differ due to rotor
        let fp_diff: f32 = fp_id.iter().zip(fp_r.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(fp_diff > 0.01, "structural fingerprint should change under non-identity rotor: diff={}", fp_diff);
    }

    #[test]
    fn test_causal_fingerprint_size() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.03).sin()).collect();
        let mv = embed_bridge_vector(&v);
        let cfp = causal_fingerprint(&mv);
        assert_eq!(cfp.len(), BOOST_BIVECTOR_COUNT);
        assert_eq!(cfp.len(), 7);
    }

    #[test]
    fn test_spatial_fingerprint_size() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.03).sin()).collect();
        let mv = embed_bridge_vector(&v);
        let sfp = spatial_fingerprint(&mv);
        assert_eq!(sfp.len(), ROTATION_BIVECTOR_COUNT);
        assert_eq!(sfp.len(), 21);
    }

    #[test]
    fn test_causal_plus_spatial_equals_structural() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let mv = embed_bridge_vector(&v);
        let full = structural_fingerprint(&mv);
        let causal = causal_fingerprint(&mv);
        let spatial = spatial_fingerprint(&mv);
        for i in 0..7 {
            assert!((full[i] - causal[i]).abs() < 1e-10,
                "causal fingerprint should be first 7 of structural");
        }
        for i in 0..21 {
            assert!((full[7 + i] - spatial[i]).abs() < 1e-10,
                "spatial fingerprint should be last 21 of structural");
        }
    }

    #[test]
    fn test_goal_magnitude_affects_timelike() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let mv0 = embed_bridge_vector_with_goal(&v, 0.0);
        let mv1 = embed_bridge_vector_with_goal(&v, 1.0);
        let e0_idx = GRADE_OFFSETS[1]; // timelike component
        assert!((mv0.components[e0_idx] - mv1.components[e0_idx]).abs() > 0.01,
            "goal magnitude should change the timelike component");
        // Spacelike components (indices 1..7 of grade-1) should be identical
        for i in 1..8 {
            assert!((mv0.grade(1)[i] - mv1.grade(1)[i]).abs() < 1e-10,
                "goal magnitude should not affect spacelike components");
        }
    }

    #[test]
    fn test_metric_mixed_vector_inner_product() {
        // v = (1, 1, 0, ...) → v·v = -1 + 1 = 0 (null/lightlike vector)
        let v = Multivector::vector(&[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let sim = geometric_similarity(&v, &v);
        assert!(sim.abs() < 1e-5, "lightlike vector self-product should be ~0: {}", sim);
    }
}
