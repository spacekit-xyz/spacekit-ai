// blade.rs — Blade index constants and utilities for Cl(1,3) / STA
//
// Index convention: the integer used as an array index IS the bit-mask of the
// blade.  Bit k set means basis vector eₖ is present.
//
//   index (= bitmask)  blade   grade
//   0  = 0b0000        1       0      scalar
//   1  = 0b0001        e0      1
//   2  = 0b0010        e1      1
//   3  = 0b0011        e01     2
//   4  = 0b0100        e2      1
//   5  = 0b0101        e02     2
//   6  = 0b0110        e12     2
//   7  = 0b0111        e012    3
//   8  = 0b1000        e3      1
//   9  = 0b1001        e03     2
//  10  = 0b1010        e13     2
//  11  = 0b1011        e013    3
//  12  = 0b1100        e23     2
//  13  = 0b1101        e023    3
//  14  = 0b1110        e123    3
//  15  = 0b1111        e0123   4      pseudoscalar

use crate::Multivector;

// ─── Named blade indices ──────────────────────────────────────────────────────

pub const SCALAR: usize = 0b0000; //  0
pub const E0:     usize = 0b0001; //  1
pub const E1:     usize = 0b0010; //  2
pub const E01:    usize = 0b0011; //  3
pub const E2:     usize = 0b0100; //  4
pub const E02:    usize = 0b0101; //  5
pub const E12:    usize = 0b0110; //  6
pub const E012:   usize = 0b0111; //  7
pub const E3:     usize = 0b1000; //  8
pub const E03:    usize = 0b1001; //  9
pub const E13:    usize = 0b1010; // 10
pub const E013:   usize = 0b1011; // 11
pub const E23:    usize = 0b1100; // 12
pub const E023:   usize = 0b1101; // 13
pub const E123:   usize = 0b1110; // 14
pub const E0123:  usize = 0b1111; // 15

// ─── Lookup tables ────────────────────────────────────────────────────────────

/// Human-readable name for each blade index.
pub const BLADE_NAMES: [&str; 16] = [
    "1",
    "e0", "e1", "e01",
    "e2", "e02", "e12", "e012",
    "e3", "e03", "e13", "e013",
    "e23", "e023", "e123", "e0123",
];

/// Grade of each blade (= popcount of the index).
pub const BLADE_GRADES: [u8; 16] = [
    0,           //  0  scalar
    1, 1, 2,     //  1  e0 | e1 | e01
    1, 2, 2, 3,  //  4  e2 | e02 | e12 | e012
    1, 2, 2, 3,  //  8  e3 | e03 | e13 | e013
    2, 3, 3, 4,  // 12  e23 | e023 | e123 | e0123
];

/// Per-blade weight `|⟨B_k B̃_k⟩₀|` used in metric-aware layer norm (STA: all 1).
pub const BLADE_METRIC_WEIGHT: [f32; 16] = [
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
];
/// Blades of grade 0,1 keep sign; grade 2,3 flip; grade 4 keeps.
pub const REVERSE_SIGNS: [f32; 16] = [
    1.0,               //  0  grade 0
    1.0, 1.0, -1.0,   //  1..3  grade 1,1,2
    1.0, -1.0, -1.0, -1.0,  //  4..7  grade 1,2,2,3
    1.0, -1.0, -1.0, -1.0,  //  8..11
    -1.0, -1.0, -1.0, 1.0,  // 12..15  grade 2,3,3,4
];

// ─── Grade utilities ──────────────────────────────────────────────────────────

/// Return the grade of a blade (0–4).
#[inline]
pub fn grade_of(blade: usize) -> u8 {
    BLADE_GRADES[blade]
}

/// Return the indices of all blades of grade `k`.
pub fn blades_of_grade(k: u8) -> Vec<usize> {
    (0..16).filter(|&i| BLADE_GRADES[i] == k).collect()
}

/// Project a multivector onto a single grade, zeroing all other components.
pub fn project_grade(mv: &Multivector, k: u8) -> Multivector {
    let mut out = Multivector::zero();
    for i in 0..16 {
        if BLADE_GRADES[i] == k {
            out.c[i] = mv.c[i];
        }
    }
    out
}

/// Return the scalar (grade-0) part — alias for mv.c[SCALAR].
#[inline]
pub fn scalar_part(mv: &Multivector) -> f32 {
    mv.c[SCALAR]
}

/// Return the vector (grade-1) part as [e0, e1, e2, e3].
pub fn vector_part(mv: &Multivector) -> [f32; 4] {
    [mv.c[E0], mv.c[E1], mv.c[E2], mv.c[E3]]
}

/// Return the bivector (grade-2) part as [e01, e02, e12, e03, e13, e23].
pub fn bivector_part(mv: &Multivector) -> [f32; 6] {
    [mv.c[E01], mv.c[E02], mv.c[E12], mv.c[E03], mv.c[E13], mv.c[E23]]
}

/// Return the pseudoscalar (grade-4) component.
#[inline]
pub fn pseudoscalar_part(mv: &Multivector) -> f32 {
    mv.c[E0123]
}

// ─── Multivector construction helpers ─────────────────────────────────────────

/// Build a pure grade-1 vector: a0 e0 + a1 e1 + a2 e2 + a3 e3.
pub fn vector(a0: f32, a1: f32, a2: f32, a3: f32) -> Multivector {
    let mut mv = Multivector::zero();
    mv.c[E0] = a0;
    mv.c[E1] = a1;
    mv.c[E2] = a2;
    mv.c[E3] = a3;
    mv
}

/// Build a pure bivector from the six independent components.
/// Order: (e01, e02, e12, e03, e13, e23)
pub fn bivector(e01: f32, e02: f32, e12: f32, e03: f32, e13: f32, e23: f32) -> Multivector {
    let mut mv = Multivector::zero();
    mv.c[E01] = e01;
    mv.c[E02] = e02;
    mv.c[E12] = e12;
    mv.c[E03] = e03;
    mv.c[E13] = e13;
    mv.c[E23] = e23;
    mv
}

// ─── Display ──────────────────────────────────────────────────────────────────

/// Return a human-readable string for a multivector, omitting near-zero terms.
pub fn display(mv: &Multivector) -> String {
    let terms: Vec<String> = (0..16)
        .filter(|&i| mv.c[i].abs() > 1e-6)
        .map(|i| {
            let v = mv.c[i];
            if i == SCALAR {
                format!("{:.4}", v)
            } else if (v - 1.0).abs() < 1e-6 {
                BLADE_NAMES[i].to_string()
            } else if (v + 1.0).abs() < 1e-6 {
                format!("-{}", BLADE_NAMES[i])
            } else {
                format!("{:.4}{}", v, BLADE_NAMES[i])
            }
        })
        .collect();
    if terms.is_empty() { "0".to_string() } else { terms.join(" + ") }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blade_grade_counts() {
        assert_eq!(blades_of_grade(0).len(), 1);   // just scalar
        assert_eq!(blades_of_grade(1).len(), 4);   // e0..e3
        assert_eq!(blades_of_grade(2).len(), 6);   // all bivectors
        assert_eq!(blades_of_grade(3).len(), 4);
        assert_eq!(blades_of_grade(4).len(), 1);   // pseudoscalar
    }

    #[test]
    fn grade_of_blades() {
        assert_eq!(grade_of(SCALAR), 0);
        assert_eq!(grade_of(E0), 1);
        assert_eq!(grade_of(E12), 2);
        assert_eq!(grade_of(E012), 3);
        assert_eq!(grade_of(E0123), 4);
    }

    #[test]
    fn display_unit_vector() {
        let mv = vector(1.0, 0.0, 0.0, 0.0);
        assert_eq!(display(&mv), "e0");
    }
}
