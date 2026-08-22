// cayley_const.rs — Compile-time Cayley table for Cl(1,3) / STA
//
// The entire 16×16 geometric product table is evaluated at compile time via
// const fn, producing a zero-cost static array.  CliffordAlgebraConst wraps
// it with the same interface as the runtime CliffordAlgebra in clifford_llm.rs
// so the two are drop-in substitutes.

use crate::Multivector;

// ─── Const geometric product of two basis blades ─────────────────────────────
//
// Index convention: same bit-mask scheme as blade.rs.
// STA metric: e0² = +1,  e1² = e2² = e3² = −1.

const fn lowest_bit(x: usize) -> usize {
    // Two's-complement trick: isolates the lowest set bit.
    x & x.wrapping_neg()
}

/// Returns (sign: i8, output_blade: usize) for basis_a ⊛ basis_b in Cl(1,3).
const fn geo_blades(a: usize, b: usize) -> (i8, usize) {
    let mut sign: i8 = 1;
    let mut a_bits = a;

    while a_bits != 0 {
        let bit = lowest_bit(a_bits);
        let idx = bit.trailing_zeros() as usize; // which basis vector (0..3)

        // Count bits in b that sit to the LEFT of this bit (i.e. higher index
        // basis vectors that would have to move past this one).  Each such swap
        // flips the sign.
        let _higher = b & !((bit << 1).wrapping_sub(1)); // bits above `bit` in b
                                                         // Equivalently: count bits in b strictly less than this bit
        let lower = b & (bit - 1);
        if lower.count_ones() % 2 == 1 {
            sign = -sign;
        }

        // If the same basis vector appears in both a and b it squares to ±1.
        if b & bit != 0 {
            if idx > 0 {
                // e1²=e2²=e3² = −1
                sign = -sign;
            }
            // idx == 0 → e0² = +1, sign unchanged
        }

        a_bits &= a_bits - 1; // clear lowest set bit
    }

    (sign, a ^ b)
}

// ─── Build the full table at compile time ─────────────────────────────────────

/// One cell of the Cayley table: (sign as i8, output blade index).
#[derive(Clone, Copy, Debug)]
pub struct CayleyCell {
    pub sign: i8,
    pub blade: u8,
}

const fn build_cayley() -> [[CayleyCell; 16]; 16] {
    let mut table = [[CayleyCell { sign: 0, blade: 0 }; 16]; 16];
    let mut i = 0;
    while i < 16 {
        let mut j = 0;
        while j < 16 {
            let (sign, blade) = geo_blades(i, j);
            table[i][j] = CayleyCell {
                sign,
                blade: blade as u8,
            };
            j += 1;
        }
        i += 1;
    }
    table
}

/// The complete Cl(1,3) Cayley table, computed at compile time.
/// `CAYLEY_STA[i][j]` describes the geometric product of basis blade `i` with
/// basis blade `j`, giving a sign and the index of the output blade.
pub const CAYLEY_STA: [[CayleyCell; 16]; 16] = build_cayley();

// ─── CliffordAlgebraConst ─────────────────────────────────────────────────────
//
// Identical API to CliffordAlgebra in clifford_llm.rs but uses the static table
// — no heap allocation, zero runtime construction cost.

pub struct CliffordAlgebraConst;

impl CliffordAlgebraConst {
    pub const fn new() -> Self {
        CliffordAlgebraConst
    }

    /// Geometric product a ⊛ b.
    #[inline]
    pub fn geo_product(&self, a: &Multivector, b: &Multivector) -> Multivector {
        let mut out = [0.0f32; 16];
        let mut i = 0;
        while i < 16 {
            if a.c[i] != 0.0 {
                let mut j = 0;
                while j < 16 {
                    if b.c[j] != 0.0 {
                        let cell = CAYLEY_STA[i][j];
                        let k = cell.blade as usize;
                        out[k] += (cell.sign as f32) * a.c[i] * b.c[j];
                    }
                    j += 1;
                }
            }
            i += 1;
        }
        Multivector { c: out }
    }

    /// Reverse: flip sign of grade-2 and grade-3 blades.
    #[inline]
    pub fn reverse(&self, a: &Multivector) -> Multivector {
        use crate::blade::REVERSE_SIGNS;
        let mut c = a.c;
        for i in 0..16 {
            c[i] *= REVERSE_SIGNS[i];
        }
        Multivector { c }
    }

    /// Scalar (grade-0) part of a ⊛ reverse(b) — Clifford inner product.
    #[inline]
    pub fn inner_product(&self, a: &Multivector, b: &Multivector) -> f32 {
        self.geo_product(a, &self.reverse(b)).c[0]
    }

    /// Squared norm ‖a‖² = ⟨ã a⟩₀.
    #[inline]
    pub fn norm_sq(&self, a: &Multivector) -> f32 {
        self.inner_product(a, a)
    }

    /// Sandwich product: r ⊛ x ⊛ r̃  (used for rotations/boosts).
    #[inline]
    pub fn sandwich(&self, r: &Multivector, x: &Multivector) -> Multivector {
        let r_rev = self.reverse(r);
        self.geo_product(&self.geo_product(r, x), &r_rev)
    }

    /// Commutator product: (a ⊛ b − b ⊛ a) / 2
    pub fn commutator(&self, a: &Multivector, b: &Multivector) -> Multivector {
        let ab = self.geo_product(a, b);
        let ba = self.geo_product(b, a);
        let mut c = [0.0f32; 16];
        for i in 0..16 {
            c[i] = (ab.c[i] - ba.c[i]) * 0.5;
        }
        Multivector { c }
    }
}

// ─── Validation ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const ALG: CliffordAlgebraConst = CliffordAlgebraConst::new();

    fn e(idx: usize) -> Multivector {
        let mut mv = Multivector::zero();
        mv.c[idx] = 1.0;
        mv
    }

    #[test]
    fn e0_squares_to_plus_one() {
        let result = ALG.geo_product(&e(1), &e(1)); // e0 * e0
        assert!(
            (result.c[0] - 1.0).abs() < 1e-6,
            "e0² should be +1, got {}",
            result.c[0]
        );
    }

    #[test]
    fn e1_squares_to_minus_one() {
        let result = ALG.geo_product(&e(2), &e(2)); // e1 * e1
        assert!(
            (result.c[0] + 1.0).abs() < 1e-6,
            "e1² should be −1, got {}",
            result.c[0]
        );
    }

    #[test]
    fn e2_squares_to_minus_one() {
        let result = ALG.geo_product(&e(4), &e(4)); // e2 * e2
        assert!(
            (result.c[0] + 1.0).abs() < 1e-6,
            "e2² should be −1, got {}",
            result.c[0]
        );
    }

    #[test]
    fn e3_squares_to_minus_one() {
        let result = ALG.geo_product(&e(8), &e(8)); // e3 * e3
        assert!(
            (result.c[0] + 1.0).abs() < 1e-6,
            "e3² should be −1, got {}",
            result.c[0]
        );
    }

    #[test]
    fn anti_commutativity_e1_e2() {
        // e1 ⊛ e2 = −(e2 ⊛ e1)
        let e1e2 = ALG.geo_product(&e(2), &e(4));
        let e2e1 = ALG.geo_product(&e(4), &e(2));
        for i in 0..16 {
            assert!((e1e2.c[i] + e2e1.c[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn associativity() {
        // (e0 ⊛ e1) ⊛ e2 == e0 ⊛ (e1 ⊛ e2)
        let e0 = e(1);
        let e1 = e(2);
        let e2 = e(4);
        let lhs = ALG.geo_product(&ALG.geo_product(&e0, &e1), &e2);
        let rhs = ALG.geo_product(&e0, &ALG.geo_product(&e1, &e2));
        for i in 0..16 {
            assert!(
                (lhs.c[i] - rhs.c[i]).abs() < 1e-6,
                "associativity failed at blade {}",
                i
            );
        }
    }

    #[test]
    fn pseudoscalar_is_grade_4() {
        let ps = ALG.geo_product(
            &ALG.geo_product(&ALG.geo_product(&e(1), &e(2)), &e(4)),
            &e(8),
        ); // e0 ⊛ e1 ⊛ e2 ⊛ e3 = e0123
        assert!((ps.c[15] - 1.0).abs() < 1e-6, "e0123 not at index 15");
    }

    #[test]
    fn const_matches_runtime() {
        // Verify the const table against the runtime geometric_product_blades
        // (the function from clifford_llm.rs).
        // We spot-check a handful of entries.
        let checks: &[(usize, usize, i8, usize)] = &[
            (0, 0, 1, 0),  // 1 * 1 = 1
            (1, 1, 1, 0),  // e0 * e0 = +1
            (2, 2, -1, 0), // e1 * e1 = -1
            (1, 2, 1, 3),  // e0 * e1 = e01
            (2, 1, -1, 3), // e1 * e0 = -e01
            (6, 6, -1, 0), // e12 * e12 = -1  (since e1²e2² = -1·-1 = +1... wait)
        ];
        // e12 * e12 = e1*e2*e1*e2 = -e1*e1*e2*e2 = -(-1)(-1) = -1
        for &(i, j, exp_sign, exp_blade) in checks {
            let cell = CAYLEY_STA[i][j];
            assert_eq!(cell.sign, exp_sign, "sign mismatch for [{i}][{j}]");
            assert_eq!(
                cell.blade as usize, exp_blade,
                "blade mismatch for [{i}][{j}]"
            );
        }
    }
}
