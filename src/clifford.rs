//! Cl(8,0) Clifford Algebra — geometric product engine for Growformer.
//!
//! Provides the 256-dimensional real Clifford algebra over R^8,
//! where multivectors are decomposed by grade:
//!
//!   grade 0: 1 scalar           (routing similarity)
//!   grade 1: 8 vectors          (raw signal)
//!   grade 2: 28 bivectors       (per-group conditioning / rotations)
//!   grade 3: 56 trivectors
//!   grade 4: 70 quadvectors
//!   grade 5: 56
//!   grade 6: 28
//!   grade 7: 8
//!   grade 8: 1 pseudoscalar     (orientation)
//!   total:   256 basis blades
//!
//! The geometric product `uv = u·v + u∧v` replaces:
//!   - bridge projection (grade extraction)
//!   - per-group adapters (rotor sandwich `R x R†`)
//!   - E8 quantization (grade-1 in Cl(8) = E8 vector space)
//!
//! Compact representation: only store and compute grades needed for each operation.

use serde::{Deserialize, Serialize};

/// Number of basis blades in Cl(8,0) = 2^8
pub const CL8_DIM: usize = 256;
pub const CL8_VECTOR_DIM: usize = 8;

/// Binomial coefficients C(8,k) — dimensions of each grade
pub const GRADE_DIMS: [usize; 9] = [1, 8, 28, 56, 70, 56, 28, 8, 1];

/// Cumulative offsets into the flat 256-element representation
pub const GRADE_OFFSETS: [usize; 9] = [0, 1, 9, 37, 93, 163, 219, 247, 255];

/// A multivector in Cl(8,0), stored as 256 real components.
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

/// Precomputed geometric product sign table for Cl(8,0).
/// For basis blades e_I and e_J, `geo_product_sign(I, J)` returns the sign
/// and the resulting blade index K such that e_I * e_J = sign * e_K.
///
/// Blade indices use the canonical bitmap encoding:
///   e_0 = 0b00000001, e_1 = 0b00000010, ..., e_7 = 0b10000000
///   e_{01} = 0b00000011, etc.
fn geo_sign_and_index(a: u8, b: u8) -> (f32, u8) {
    let result_blade = a ^ b;
    // Count transpositions: for each bit in b, count how many higher bits in a
    // must be passed through (each swap contributes a sign flip).
    let mut swaps = 0u32;
    let mut b_remaining = b;
    while b_remaining != 0 {
        let lowest_b = b_remaining & b_remaining.wrapping_neg(); // isolate lowest bit
        let a_above = a & !((lowest_b << 1).wrapping_sub(1)); // bits in a above this position
        swaps += a_above.count_ones();
        b_remaining &= b_remaining - 1; // clear lowest bit
    }
    let sign = if swaps % 2 == 0 { 1.0 } else { -1.0 };
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
    let num_blocks = (v.len() + 7) / 8;
    let mut result = Multivector::zero();
    for block in 0..num_blocks {
        let offset = block * 8;
        let mut chunk = [0.0f32; 8];
        for i in 0..8 {
            chunk[i] = v.get(offset + i).copied().unwrap_or(0.0);
        }
        // Weighted sum: each block contributes to the same grade-1 space
        // but we also capture inter-block structure via grade-2 wedge products
        let block_mv = Multivector::vector(&chunk);
        if block == 0 {
            for i in 0..CL8_DIM {
                result.components[i] = block_mv.components[i];
            }
        } else {
            // Accumulate: add vector part, wedge for bivector structure
            let wedge = result.wedge(&block_mv);
            result = result.add(&block_mv);
            // Blend in a fraction of the wedge product to preserve inter-block correlations
            let alpha = 1.0 / (block as f32 + 1.0);
            for i in GRADE_OFFSETS[2]..GRADE_OFFSETS[2] + GRADE_DIMS[2] {
                result.components[i] += alpha * wedge.components[i];
            }
        }
    }
    result
}

/// Extract routing score: the scalar part of the geometric product of two multivectors.
/// Higher values = more aligned in Cl(8) space.
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
    // Grade 1: direct signal (8 components)
    out.extend_from_slice(mv.grade(1));
    // Grade 2: rotational/relational structure (28 components)
    out.extend_from_slice(mv.grade(2));
    // Grade 0: scalar similarity (1 component)
    out.push(mv.scalar_part());
    // Pad or truncate to target_dim
    out.resize(target_dim, 0.0);
    out
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
        // e1 * e1 = +1 (in Cl(p,0), all basis vectors square to +1)
        let (sign, blade) = geo_sign_and_index(0b00000001, 0b00000001);
        assert_eq!(blade, 0); // scalar
        assert_eq!(sign, 1.0);

        // e1 * e2 = e12
        let (sign, blade) = geo_sign_and_index(0b00000001, 0b00000010);
        assert_eq!(blade, 0b00000011);
        assert_eq!(sign, 1.0);

        // e2 * e1 = -e12 (anticommutative for orthogonal vectors)
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
    fn test_geometric_product_parallel_vectors() {
        // u·u = |u|² (scalar), u∧u = 0
        let u = Multivector::vector(&[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let uu = u.geo(&u);
        assert!((uu.scalar_part() - 9.0).abs() < 1e-5, "u·u should be 9, got {}", uu.scalar_part());
        assert!(uu.bivector_part().iter().all(|&b| b.abs() < 1e-6));
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
    fn test_geometric_similarity_self() {
        let v = Multivector::vector(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let sim = geometric_similarity(&v, &v);
        assert!(sim > 0.0, "self-similarity should be positive: {}", sim);
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
}
