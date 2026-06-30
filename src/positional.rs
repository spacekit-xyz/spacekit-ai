// positional.rs — Rotor-based positional encoding for Clifford LLM
//
// Standard transformers inject position with additive sinusoids.  In a Clifford
// model a more principled approach is available: encode position t as a *rotor*
// R(t) and apply the sandwich product  R(t) ⊛ x ⊛ R̃(t).
//
// A rotor in Cl(1,3) lives in the even sub-algebra (grades 0 and 2).  The
// simplest family uses a single unit bivector B̂:
//
//   R(t) = exp(−θ(t) B̂ / 2) = cos(θ(t)/2)  −  sin(θ(t)/2) B̂
//
// Different bivector planes (e01, e12, e23, …) give independent "rotation axes",
// so you can tile multiple planes across the d_model dimension — exactly
// analogous to RoPE's sinusoidal basis pairs, but geometrically meaningful.
//
// For STA the bivectors split into two families:
//   - Spatial rotations: e12, e13, e23
//   - Boosts (Lorentz):  e01, e02, e03
// Spatial planes produce ordinary rotations; boost planes produce hyperbolic
// "rotations" (cosh/sinh).  Both are supported below.

use crate::Multivector;
use crate::cayley_const::CliffordAlgebraConst;
use crate::blade::{E01, E02, E03, E12, E13, E23, SCALAR};

// ─── Bivector plane descriptor ────────────────────────────────────────────────

/// The family of a bivector plane determines whether it generates a trigonometric
/// rotation or a hyperbolic (Lorentz boost) transformation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlaneKind {
    /// Spatial plane (e12, e13, e23): B̂² = −1 → uses cos/sin.
    Rotation,
    /// Boost plane (e01, e02, e03): B̂² = +1 → uses cosh/sinh.
    Boost,
}

/// A single bivector plane used as a rotation axis.
#[derive(Clone, Copy, Debug)]
pub struct BivectorPlane {
    pub blade_idx: usize,   // which component of c[…] to use (e.g. E12 = 6)
    pub kind:      PlaneKind,
}

impl BivectorPlane {
    pub const E01: Self = Self { blade_idx: E01, kind: PlaneKind::Boost    };
    pub const E02: Self = Self { blade_idx: E02, kind: PlaneKind::Boost    };
    pub const E03: Self = Self { blade_idx: E03, kind: PlaneKind::Boost    };
    pub const E12: Self = Self { blade_idx: E12, kind: PlaneKind::Rotation };
    pub const E13: Self = Self { blade_idx: E13, kind: PlaneKind::Rotation };
    pub const E23: Self = Self { blade_idx: E23, kind: PlaneKind::Rotation };
}

/// Default plane schedule: alternate rotation and boost planes across the
/// model dimension.  Matches the 6 independent bivectors in Cl(1,3).
pub const ALL_PLANES: [BivectorPlane; 6] = [
    BivectorPlane::E12,
    BivectorPlane::E01,
    BivectorPlane::E13,
    BivectorPlane::E02,
    BivectorPlane::E23,
    BivectorPlane::E03,
];

// ─── Rotor construction ───────────────────────────────────────────────────────

/// Build a rotor in the given bivector plane for angle θ.
///
/// - Rotation plane (B̂² = −1):  R = cos(θ/2) − sin(θ/2)·B̂
/// - Boost plane    (B̂² = +1):  R = cosh(θ/2) − sinh(θ/2)·B̂
///
/// The returned Multivector lives in the even sub-algebra (grades 0 and 2).
pub fn make_rotor(theta: f32, plane: BivectorPlane) -> Multivector {
    let half = theta * 0.5;
    let mut r = Multivector::zero();
    match plane.kind {
        PlaneKind::Rotation => {
            r.c[SCALAR]          =  half.cos();
            r.c[plane.blade_idx] = -half.sin();
        }
        PlaneKind::Boost => {
            r.c[SCALAR]          =  half.cosh();
            r.c[plane.blade_idx] = -half.sinh();
        }
    }
    r
}

/// Apply the sandwich product: R ⊛ x ⊛ R̃ to a single multivector.
#[inline]
pub fn apply_rotor(alg: &CliffordAlgebraConst, r: &Multivector, x: &Multivector) -> Multivector {
    alg.sandwich(r, x)
}

// ─── Rotor positional encoding ────────────────────────────────────────────────
//
// Each position t gets a unique set of rotors, one per model-dimension slot.
// The angles are spaced logarithmically (same intuition as sinusoidal PE):
//   θ(t, p) = t / 10000^(2p / d_model)
//
// With d_model multivectors and 6 available planes we cycle through ALL_PLANES.

pub struct RotorPositionalEncoding {
    pub d_model: usize,
    pub planes:  Vec<BivectorPlane>,  // length d_model
    pub base:    f32,                 // frequency base (default 10000.0)
}

impl RotorPositionalEncoding {
    /// Build a positional encoding for a model of `d_model` multivectors per token.
    /// Cycles through `ALL_PLANES` across the dimension axis.
    pub fn new(d_model: usize) -> Self {
        let planes: Vec<BivectorPlane> = (0..d_model)
            .map(|i| ALL_PLANES[i % ALL_PLANES.len()])
            .collect();
        Self { d_model, planes, base: 10_000.0 }
    }

    /// Compute the angle for dimension slot `d` at position `t`.
    fn angle(&self, t: usize, d: usize) -> f32 {
        let freq = 1.0 / self.base.powf(2.0 * d as f32 / self.d_model as f32);
        t as f32 * freq
    }

    /// Apply positional encoding to a single token position.
    ///
    /// `x`         — the d_model multivectors for this position
    /// `position`  — the integer sequence position (0-indexed)
    pub fn encode_position(
        &self,
        alg:      &CliffordAlgebraConst,
        x:        &[Multivector],
        position: usize,
    ) -> Vec<Multivector> {
        x.iter().enumerate().map(|(d, mv)| {
            let theta = self.angle(position, d);
            let r     = make_rotor(theta, self.planes[d]);
            apply_rotor(alg, &r, mv)
        }).collect()
    }

    /// Apply positional encoding to an entire sequence.
    ///
    /// `x` — [seq_len][d_model] multivectors (the output of the embedding lookup)
    ///
    /// Returns [seq_len][d_model] with each position rotated by its own rotor.
    pub fn encode(
        &self,
        alg: &CliffordAlgebraConst,
        x:   &[Vec<Multivector>],
    ) -> Vec<Vec<Multivector>> {
        x.iter().enumerate()
            .map(|(t, xi)| self.encode_position(alg, xi, t))
            .collect()
    }

    /// Pre-compute and cache all rotors up to `max_seq_len`.
    /// Useful during inference to avoid repeated trig evaluations.
    pub fn precompute_rotors(&self, max_seq_len: usize) -> Vec<Vec<Multivector>> {
        (0..max_seq_len).map(|t| {
            (0..self.d_model).map(|d| {
                make_rotor(self.angle(t, d), self.planes[d])
            }).collect()
        }).collect()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blade::vector;

    const ALG: CliffordAlgebraConst = CliffordAlgebraConst::new();

    #[test]
    fn rotation_rotor_is_unit() {
        // A rotor should satisfy R R̃ = 1
        let r = make_rotor(0.7, BivectorPlane::E12);
        let r_rev = ALG.reverse(&r);
        let rr_rev = ALG.geo_product(&r, &r_rev);
        // Should be the scalar 1
        assert!((rr_rev.c[0] - 1.0).abs() < 1e-6, "R R̃ scalar part should be 1");
        for i in 1..16 {
            assert!(rr_rev.c[i].abs() < 1e-6, "R R̃ should have no grade > 0");
        }
    }

    #[test]
    fn boost_rotor_is_unit() {
        let r = make_rotor(0.5, BivectorPlane::E01);
        let r_rev = ALG.reverse(&r);
        let rr_rev = ALG.geo_product(&r, &r_rev);
        assert!((rr_rev.c[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_position_is_identity() {
        // At t=0 all angles are 0, so every rotor is the scalar 1,
        // and the sandwich product should leave x unchanged.
        let alg = CliffordAlgebraConst::new();
        let pe  = RotorPositionalEncoding::new(4);
        let x: Vec<Multivector> = vec![vector(1.0, 0.5, -0.3, 0.1); 4];
        let encoded = pe.encode_position(&alg, &x, 0);

        for (orig, enc) in x.iter().zip(encoded.iter()) {
            for k in 0..16 {
                assert!((orig.c[k] - enc.c[k]).abs() < 1e-5,
                    "position 0 should leave multivector unchanged");
            }
        }
    }

    #[test]
    fn different_positions_produce_different_encodings() {
        let alg = CliffordAlgebraConst::new();
        let pe  = RotorPositionalEncoding::new(4);
        // Use a spatial vector so the slot-0 E12 rotor changes the embedding (e0 alone is fixed).
        let x: Vec<Multivector> = vec![vector(0.0, 1.0, 0.0, 0.0); 4];
        let enc0 = pe.encode_position(&alg, &x, 0);
        let enc1 = pe.encode_position(&alg, &x, 1);
        let enc5 = pe.encode_position(&alg, &x, 5);

        // enc0 == x (identity at t=0), enc1 != enc5
        let same_1_5: bool = (0..16).all(|k| (enc1[0].c[k] - enc5[0].c[k]).abs() < 1e-5);
        assert!(!same_1_5, "positions 1 and 5 should give different encodings");
    }
}
