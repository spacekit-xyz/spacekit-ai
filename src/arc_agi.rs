//! ARC-AGI benchmark through Cl(1,7) spacetime algebra.
//!
//! The Problem of Universals and the Problem of Induction are the same problem.
//! Clifford algebra dissolves both for the geometric subset of ARC:
//!   Universals = rotors (exact, composable, transferable)
//!   Induction  = |B| measurement (rotor consistency across examples)
//!   Generalization = rotor application (exact, not probabilistic)
//!
//! Solving strategies (priority order):
//!
//!   Exact structural rules (100% train accuracy required):
//!     D. Ring depth reversal — reverse concentric ring colors
//!     F. Ring color cycle — cyclic permutation of ring colors
//!     G. Diagonal X from special cell — draw diagonals through anomaly
//!     H. Spiral fill — clockwise spiral with alternating colors
//!     E. Depth fill / border frame — map cell depth to colors
//!     J. Object positional — detect connected components, extract position deltas
//!     K. Object recolor — detect connected components, learn color mapping
//!
//!   Competing heuristics (highest train cell accuracy wins):
//!     A. Color map — per-input-color majority in `color_vector_full` space
//!     L. Palette-constrained neighborhood — cell centroid restricted to output palette
//!     B. Neighborhood centroid — 3×3 `cell_feature`, nearest centroid by dot
//!     C. Adjacency Clifford score — precomputed pairwise scoring
//!
//!   Fallback:
//!     Grid-level rotor — whole-grid encoding for diff-dim tasks
//!
//! |B| diagnostic measures inductive consistency: low |B| means the same
//! geometric rule is being induced from each training example.

use crate::clifford::{
    Multivector, CL8_DIM, GRADE_OFFSETS,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use std::path::Path;

pub const MAX_GRID: usize = 30;
pub const NUM_COLORS: usize = 10;

// ─── Data structures ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Grid {
    pub cells: Vec<Vec<u8>>,
    pub height: usize,
    pub width: usize,
}

#[derive(Clone, Debug)]
pub struct ArcExample {
    pub input: Grid,
    pub output: Grid,
}

#[derive(Clone, Debug)]
pub struct ArcTask {
    pub id: String,
    pub train: Vec<ArcExample>,
    pub test: Vec<ArcExample>,
}

#[derive(Debug)]
pub struct TaskDiagnostic {
    pub id: String,
    pub n_train: usize,
    pub n_test: usize,
    pub same_dims: bool,
    pub rotor_consistency: f32,
    pub mean_bv_norm: f32,
    pub solved: bool,
    pub n_correct_cells: usize,
    pub n_total_cells: usize,
    pub strategy: &'static str,
}

// ─── JSON loading ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RawPair {
    input: Vec<Vec<u8>>,
    output: Vec<Vec<u8>>,
}

#[derive(Deserialize)]
struct RawTask {
    train: Vec<RawPair>,
    test: Vec<RawPair>,
}

fn grid_from_raw(raw: &[Vec<u8>]) -> Grid {
    let height = raw.len();
    let width = if height > 0 { raw[0].len() } else { 0 };
    Grid { cells: raw.to_vec(), height, width }
}

pub fn load_arc_tasks(dir: &Path) -> Vec<ArcTask> {
    let mut tasks = Vec::new();
    let entries: Vec<_> = match std::fs::read_dir(dir) {
        Ok(e) => e.filter_map(|e| e.ok()).collect(),
        Err(e) => {
            eprintln!("Cannot read ARC directory {:?}: {}", dir, e);
            return tasks;
        }
    };

    for entry in &entries {
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "json") { continue; }
        let id = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let data = match std::fs::read_to_string(&path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let raw: RawTask = match serde_json::from_str(&data) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let train = raw.train.iter().map(|p| ArcExample {
            input: grid_from_raw(&p.input),
            output: grid_from_raw(&p.output),
        }).collect();
        let test = raw.test.iter().map(|p| ArcExample {
            input: grid_from_raw(&p.input),
            output: grid_from_raw(&p.output),
        }).collect();

        tasks.push(ArcTask { id, train, test });
    }
    tasks.sort_by(|a, b| a.id.cmp(&b.id));
    tasks
}

// ─── Color encoding ────────────────────────────────────────────────────────
//
// For grid-level encoding, background (0) = zero (contributes nothing to sum).
// For cell-level strategies, all 10 colors need distinct non-zero vectors.

fn color_vector(color: u8) -> Multivector {
    match color {
        0 => Multivector::zero(),
        c @ 1..=8 => {
            let mut v = [0.0f32; 8];
            v[(c - 1) as usize] = 1.0;
            Multivector::vector(&v)
        }
        9 => {
            let s = 1.0 / 2.0f32.sqrt();
            Multivector::vector(&[s, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        }
        _ => Multivector::zero(),
    }
}

fn color_vector_full(color: u8) -> Multivector {
    match color {
        0 => {
            let s = 1.0 / 8.0f32.sqrt();
            Multivector::vector(&[-s, -s, -s, -s, -s, -s, -s, -s])
        }
        c @ 1..=8 => {
            let mut v = [0.0f32; 8];
            v[(c - 1) as usize] = 1.0;
            Multivector::vector(&v)
        }
        9 => {
            let s = 1.0 / 2.0f32.sqrt();
            Multivector::vector(&[s, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        }
        _ => Multivector::zero(),
    }
}

fn component_dot(a: &Multivector, b: &Multivector) -> f32 {
    a.components.iter().zip(b.components.iter()).map(|(x, y)| x * y).sum()
}

fn decode_color(mv: &Multivector) -> u8 {
    let mut best = 0u8;
    let mut best_score = f32::NEG_INFINITY;
    for c in 0..NUM_COLORS as u8 {
        let cv = color_vector_full(c);
        let score = component_dot(mv, &cv);
        if score > best_score {
            best_score = score;
            best = c;
        }
    }
    best
}

fn cell_get(grid: &Grid, r: isize, c: isize) -> u8 {
    if r >= 0 && r < grid.height as isize && c >= 0 && c < grid.width as isize {
        grid.cells[r as usize][c as usize]
    } else { 0 }
}

// ─── Strategy A: Color rotor ───────────────────────────────────────────────
//
// Global color substitution via Clifford algebra.
// For each input color, accumulate the output color vectors across all cells
// and all training examples. The dominant direction IS the mapped color.
// This is the Clifford analogue of a color lookup table: the algebra's
// orthogonality naturally decouples independent color channels.

fn solve_color_map(task: &ArcTask) -> ([u8; NUM_COLORS], f32) {
    let mut output_sums: Vec<Multivector> = (0..NUM_COLORS).map(|_| Multivector::zero()).collect();
    let mut counts = [0u32; NUM_COLORS];

    for ex in &task.train {
        if ex.input.height != ex.output.height || ex.input.width != ex.output.width { continue; }
        for r in 0..ex.input.height {
            for c in 0..ex.input.width {
                let in_c = ex.input.cells[r][c] as usize;
                let out_c = ex.output.cells[r][c];
                output_sums[in_c] = output_sums[in_c].add(&color_vector_full(out_c));
                counts[in_c] += 1;
            }
        }
    }

    let mut color_map = [0u8; NUM_COLORS];
    for c in 0..NUM_COLORS {
        color_map[c] = if counts[c] > 0 { decode_color(&output_sums[c]) } else { c as u8 };
    }

    let (mut correct, mut total) = (0usize, 0usize);
    for ex in &task.train {
        if ex.input.height != ex.output.height || ex.input.width != ex.output.width { continue; }
        for r in 0..ex.input.height {
            for c in 0..ex.input.width {
                if color_map[ex.input.cells[r][c] as usize] == ex.output.cells[r][c] { correct += 1; }
                total += 1;
            }
        }
    }

    (color_map, if total > 0 { correct as f32 / total as f32 } else { 0.0 })
}

fn apply_color_map(grid: &Grid, color_map: &[u8; NUM_COLORS]) -> Grid {
    Grid {
        cells: grid.cells.iter().map(|row|
            row.iter().map(|&c| color_map[c as usize]).collect()
        ).collect(),
        height: grid.height,
        width: grid.width,
    }
}

// ─── Strategy B: Neighborhood centroid classifier ──────────────────────────
//
// Each cell's feature encodes the 3×3 local neighborhood as a multivector:
//   - Center color → grade-1 (8 components)
//   - 8 neighbors → grades 2-3 (8×8 = 64 components, each neighbor
//     gets its own slice so directional information is preserved)
//
// For each output color, compute the centroid (mean feature) across all
// training cells that map to that color. Classify test cells by nearest
// centroid using component dot product — the algebra's inner product.
//
// This is a single forward pass: accumulate centroids, classify.
// No geometric products needed — O(N × 256) total.

fn cell_feature(grid: &Grid, r: usize, c: usize) -> Multivector {
    let mut mv = Multivector::zero();

    let center = color_vector_full(grid.cells[r][c]);
    for i in 0..8 {
        mv.components[GRADE_OFFSETS[1] + i] = center.components[GRADE_OFFSETS[1] + i];
    }

    const OFFSETS: [(isize, isize); 8] = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ];

    for (i, &(dr, dc)) in OFFSETS.iter().enumerate() {
        let nc = cell_get(grid, r as isize + dr, c as isize + dc);
        let ncv = color_vector_full(nc);
        let base = GRADE_OFFSETS[2] + i * 8;
        for j in 0..8 {
            if base + j < CL8_DIM {
                mv.components[base + j] = ncv.components[GRADE_OFFSETS[1] + j] * 0.5;
            }
        }
    }

    mv
}

fn classify_centroid(centroids: &[Multivector], feat: &Multivector) -> u8 {
    let mut best = 0u8;
    let mut best_score = f32::NEG_INFINITY;
    for (c, centroid) in centroids.iter().enumerate() {
        let score = component_dot(feat, centroid);
        if score > best_score {
            best_score = score;
            best = c as u8;
        }
    }
    best
}

fn solve_cell_centroid(task: &ArcTask) -> Option<(Vec<Multivector>, f32)> {
    if !task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width) {
        return None;
    }

    let mut sums: Vec<Multivector> = (0..NUM_COLORS).map(|_| Multivector::zero()).collect();
    let mut counts = [0u32; NUM_COLORS];

    for ex in &task.train {
        for r in 0..ex.input.height {
            for c in 0..ex.input.width {
                let feat = cell_feature(&ex.input, r, c);
                let out_c = ex.output.cells[r][c] as usize;
                sums[out_c] = sums[out_c].add(&feat);
                counts[out_c] += 1;
            }
        }
    }

    let centroids: Vec<Multivector> = sums.iter().zip(counts.iter())
        .map(|(sum, &count)| {
            if count > 0 { sum.scale(1.0 / count as f32) } else { Multivector::zero() }
        })
        .collect();

    let (mut correct, mut total) = (0usize, 0usize);
    for ex in &task.train {
        for r in 0..ex.input.height {
            for c in 0..ex.input.width {
                let feat = cell_feature(&ex.input, r, c);
                if classify_centroid(&centroids, &feat) == ex.output.cells[r][c] { correct += 1; }
                total += 1;
            }
        }
    }

    Some((centroids, if total > 0 { correct as f32 / total as f32 } else { 0.0 }))
}

fn apply_cell_centroid(centroids: &[Multivector], grid: &Grid) -> Grid {
    let mut cells = vec![vec![0u8; grid.width]; grid.height];
    for r in 0..grid.height {
        for c in 0..grid.width {
            let feat = cell_feature(grid, r, c);
            cells[r][c] = classify_centroid(centroids, &feat);
        }
    }
    Grid { cells, height: grid.height, width: grid.width }
}

// ─── Strategy C: Adjacency Clifford score (same-size only) ─────────────────
//
// For each undirected grid edge (right / down), encode the 4-tuple
// (in_a, out_a, in_b, out_b) into fixed multivector slots (grade-1 + grade-2),
// average over training examples → centroid per edge. Scoring a candidate
// output uses only component_dot against precomputed per-edge tables.

const MAX_EXHAUSTIVE_ASSIGNMENTS: u128 = 50_000_000;
const ADJ_SEARCH_SAMPLES_TRAIN: u64 = 24_000;
const ADJ_SEARCH_SAMPLES_TEST: u64 = 400_000;

fn grid_edges(height: usize, width: usize) -> Vec<(usize, usize, usize, usize)> {
    let mut e = Vec::new();
    for r in 0..height {
        for c in 0..width {
            if c + 1 < width {
                e.push((r, c, r, c + 1));
            }
            if r + 1 < height {
                e.push((r, c, r + 1, c));
            }
        }
    }
    e
}

fn adj_edge_feature(in_a: u8, out_a: u8, in_b: u8, out_b: u8) -> Multivector {
    let mut mv = Multivector::zero();
    let va = color_vector_full(in_a);
    let vb = color_vector_full(out_a);
    let vc = color_vector_full(in_b);
    let vd = color_vector_full(out_b);
    let g1 = GRADE_OFFSETS[1];
    for i in 0..8 {
        mv.components[g1 + i] = va.components[g1 + i];
    }
    let g2 = GRADE_OFFSETS[2];
    for i in 0..8 {
        mv.components[g2 + i] = vb.components[g1 + i];
        mv.components[g2 + 8 + i] = vc.components[g1 + i];
        mv.components[g2 + 16 + i] = vd.components[g1 + i];
    }
    mv
}

fn task_color_palette(task: &ArcTask) -> Vec<u8> {
    let mut seen = [false; NUM_COLORS];
    for ex in &task.train {
        for row in &ex.input.cells {
            for &c in row {
                seen[c as usize] = true;
            }
        }
        for row in &ex.output.cells {
            for &c in row {
                seen[c as usize] = true;
            }
        }
    }
    for ex in &task.test {
        for row in &ex.input.cells {
            for &c in row {
                seen[c as usize] = true;
            }
        }
    }
    let mut p: Vec<u8> = (0..NUM_COLORS as u8).filter(|&c| seen[c as usize]).collect();
    if p.is_empty() {
        p.extend(0..NUM_COLORS as u8);
    }
    p.sort_unstable();
    p.dedup();
    p
}

fn palette_state_count(k: usize, n_cells: usize) -> Option<u128> {
    let mut p = 1u128;
    for _ in 0..n_cells {
        p = p.checked_mul(k as u128)?;
    }
    Some(p)
}

fn build_adj_centroids(task: &ArcTask, h: usize, w: usize) -> Option<Vec<Multivector>> {
    if !task.train.iter().all(|ex|
        ex.input.height == h && ex.input.width == w
            && ex.output.height == h && ex.output.width == w) {
        return None;
    }
    let edges = grid_edges(h, w);
    if edges.is_empty() {
        return None;
    }
    let mut sums: Vec<Multivector> = (0..edges.len()).map(|_| Multivector::zero()).collect();
    let mut counts = vec![0u32; edges.len()];

    for ex in &task.train {
        for (ei, &(r1, c1, r2, c2)) in edges.iter().enumerate() {
            let ia = ex.input.cells[r1][c1];
            let ib = ex.input.cells[r2][c2];
            let oa = ex.output.cells[r1][c1];
            let ob = ex.output.cells[r2][c2];
            sums[ei] = sums[ei].add(&adj_edge_feature(ia, oa, ib, ob));
            counts[ei] += 1;
        }
    }

    for (sum, &cnt) in sums.iter_mut().zip(counts.iter()) {
        if cnt > 0 {
            *sum = sum.scale(1.0 / cnt as f32);
        }
    }
    Some(sums)
}

/// Per-edge table: flat index `oa * NUM_COLORS + ob`.
fn adj_precompute_tables(
    input: &Grid,
    centroids: &[Multivector],
    edges: &[(usize, usize, usize, usize)],
) -> Vec<Vec<f32>> {
    let mut tables = Vec::with_capacity(edges.len());
    for (ei, &(r1, c1, r2, c2)) in edges.iter().enumerate() {
        let ia = input.cells[r1][c1];
        let ib = input.cells[r2][c2];
        let rule = &centroids[ei];
        let mut t = vec![0.0f32; NUM_COLORS * NUM_COLORS];
        for oa in 0..NUM_COLORS {
            for ob in 0..NUM_COLORS {
                let mv = adj_edge_feature(ia, oa as u8, ib, ob as u8);
                t[oa * NUM_COLORS + ob] = component_dot(&mv, rule);
            }
        }
        tables.push(t);
    }
    tables
}

fn adj_score_flat(
    flat: &[u8],
    width: usize,
    edges: &[(usize, usize, usize, usize)],
    tables: &[Vec<f32>],
) -> f32 {
    let idx = |r: usize, c: usize| r * width + c;
    let mut s = 0.0f32;
    for (ei, e) in edges.iter().enumerate() {
        let i1 = idx(e.0, e.1);
        let i2 = idx(e.2, e.3);
        let oa = flat[i1] as usize;
        let ob = flat[i2] as usize;
        s += tables[ei][oa * NUM_COLORS + ob];
    }
    s
}

fn flat_to_grid(flat: &[u8], h: usize, w: usize) -> Grid {
    let mut cells = vec![vec![0u8; w]; h];
    for r in 0..h {
        for c in 0..w {
            cells[r][c] = flat[r * w + c];
        }
    }
    Grid { cells, height: h, width: w }
}

fn adjacency_search_best(
    input: &Grid,
    centroids: &[Multivector],
    edges: &[(usize, usize, usize, usize)],
    palette: &[u8],
    max_samples: u64,
    rng: &mut StdRng,
) -> Grid {
    let h = input.height;
    let w = input.width;
    let n = h * w;
    let k = palette.len().max(1);
    let tables = adj_precompute_tables(input, centroids, edges);

    let mut best_flat = vec![palette[0]; n];
    let mut best_score = f32::NEG_INFINITY;

    let states = palette_state_count(k, n);
    let use_exhaustive = states.map_or(false, |p| p > 0 && p <= MAX_EXHAUSTIVE_ASSIGNMENTS);

    if use_exhaustive {
        let total = states.unwrap() as u64;
        for code in 0u64..total {
            let mut flat = vec![0u8; n];
            let mut t = code;
            for cell in flat.iter_mut() {
                let pi = (t % k as u64) as usize;
                t /= k as u64;
                *cell = palette[pi];
            }
            let s = adj_score_flat(&flat, w, edges, &tables);
            if s > best_score {
                best_score = s;
                best_flat.copy_from_slice(&flat);
            }
        }
    } else {
        for _ in 0..max_samples {
            let mut flat = vec![0u8; n];
            for cell in flat.iter_mut() {
                *cell = palette[rng.gen_range(0..k)];
            }
            let s = adj_score_flat(&flat, w, edges, &tables);
            if s > best_score {
                best_score = s;
                best_flat.copy_from_slice(&flat);
            }
        }
    }

    flat_to_grid(&best_flat, h, w)
}

fn task_rng_seed(task: &ArcTask) -> u64 {
    task.id.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
}

fn solve_adjacency_train_acc(task: &ArcTask, h: usize, w: usize, centroids: &[Multivector]) -> f32 {
    let edges = grid_edges(h, w);
    if edges.is_empty() {
        return 0.0;
    }
    let palette = task_color_palette(task);
    let mut rng = StdRng::seed_from_u64(task_rng_seed(task).wrapping_add(0xA11ACE));
    let n = h * w;
    let k = palette.len().max(1);
    let states = palette_state_count(k, n);
    let use_exhaustive = states.map_or(false, |p| p > 0 && p <= MAX_EXHAUSTIVE_ASSIGNMENTS);
    let samples = if use_exhaustive { 0 } else { ADJ_SEARCH_SAMPLES_TRAIN };

    let (mut correct, mut total) = (0usize, 0usize);
    for ex in &task.train {
        let pred = adjacency_search_best(
            &ex.input,
            centroids,
            &edges,
            &palette,
            samples,
            &mut rng,
        );
        let (c, t) = grid_matches(&pred, &ex.output);
        correct += c;
        total += t;
    }
    if total > 0 {
        correct as f32 / total as f32
    } else {
        0.0
    }
}

// ─── Strategy D: Ring depth reversal ────────────────────────────────────────
//
// Detects concentric-ring structure (each "ring" at Chebyshev distance d from
// border shares a single color). If detected, reverses the depth ordering:
// ring at depth d gets the color of ring at depth (max_depth - d).

fn chebyshev_border_dist(r: usize, c: usize, h: usize, w: usize) -> usize {
    let top = r;
    let bot = h.saturating_sub(1).saturating_sub(r);
    let left = c;
    let right = w.saturating_sub(1).saturating_sub(c);
    top.min(bot).min(left).min(right)
}

fn detect_concentric_rings(grid: &Grid) -> Option<Vec<u8>> {
    let h = grid.height;
    let w = grid.width;
    if h == 0 || w == 0 { return None; }
    let max_depth = chebyshev_border_dist(h / 2, w / 2, h, w);

    let mut ring_colors: Vec<Option<u8>> = vec![None; max_depth + 1];

    for r in 0..h {
        for c in 0..w {
            let d = chebyshev_border_dist(r, c, h, w);
            let color = grid.cells[r][c];
            match ring_colors[d] {
                None => ring_colors[d] = Some(color),
                Some(existing) if existing != color => return None,
                _ => {}
            }
        }
    }

    Some(ring_colors.iter().map(|c| c.unwrap_or(0)).collect())
}

fn apply_ring_reversal(grid: &Grid, reversed_colors: &[u8]) -> Grid {
    let h = grid.height;
    let w = grid.width;
    let mut cells = vec![vec![0u8; w]; h];
    for r in 0..h {
        for c in 0..w {
            let d = chebyshev_border_dist(r, c, h, w);
            cells[r][c] = if d < reversed_colors.len() { reversed_colors[d] } else { 0 };
        }
    }
    Grid { cells, height: h, width: w }
}

fn solve_ring_reversal(task: &ArcTask) -> Option<f32> {
    for ex in &task.train {
        if ex.input.height != ex.output.height || ex.input.width != ex.output.width {
            return None;
        }
        let in_rings = detect_concentric_rings(&ex.input)?;
        let out_rings = detect_concentric_rings(&ex.output)?;
        if in_rings.len() != out_rings.len() { return None; }
        let n = in_rings.len();
        for d in 0..n {
            if in_rings[d] != out_rings[n - 1 - d] { return None; }
        }
    }

    let (mut correct, mut total) = (0usize, 0usize);
    for ex in &task.train {
        let in_rings = detect_concentric_rings(&ex.input).unwrap();
        let n = in_rings.len();
        let mut reversed = vec![0u8; n];
        for d in 0..n { reversed[d] = in_rings[n - 1 - d]; }
        let pred = apply_ring_reversal(&ex.input, &reversed);
        let (c, t) = grid_matches(&pred, &ex.output);
        correct += c;
        total += t;
    }
    Some(if total > 0 { correct as f32 / total as f32 } else { 0.0 })
}

fn apply_ring_reversal_to_test(grid: &Grid) -> Grid {
    let rings = match detect_concentric_rings(grid) {
        Some(r) => r,
        None => return grid.clone(),
    };
    let n = rings.len();
    let mut reversed = vec![0u8; n];
    for d in 0..n { reversed[d] = rings[n - 1 - d]; }
    apply_ring_reversal(grid, &reversed)
}

// ─── Strategy E: Border frame ──────────────────────────────────────────────
//
// All-uniform input → output has border cells set to a specific color, interior
// unchanged. Also handles concentric spiral / multi-ring fill patterns
// by learning a depth→color map from training examples.

fn is_uniform_grid(grid: &Grid) -> Option<u8> {
    if grid.height == 0 || grid.width == 0 { return None; }
    let c0 = grid.cells[0][0];
    for row in &grid.cells {
        for &c in row {
            if c != c0 { return None; }
        }
    }
    Some(c0)
}

fn learn_depth_color_map(task: &ArcTask) -> Option<(u8, Vec<(usize, u8)>)> {
    let uniform_color = is_uniform_grid(&task.train[0].input)?;
    for ex in &task.train {
        if is_uniform_grid(&ex.input) != Some(uniform_color) { return None; }
        if ex.input.height != ex.output.height || ex.input.width != ex.output.width {
            return None;
        }
    }

    let mut depth_map: Vec<(usize, u8)> = Vec::new();

    for ex in &task.train {
        let h = ex.output.height;
        let w = ex.output.width;
        let max_d = chebyshev_border_dist(h / 2, w / 2, h, w);

        for r in 0..h {
            for c in 0..w {
                let d = chebyshev_border_dist(r, c, h, w);
                let out_c = ex.output.cells[r][c];

                let relative_d = if max_d > 0 { d } else { 0 };

                if let Some(entry) = depth_map.iter().find(|&&(dd, _)| dd == relative_d) {
                    if entry.1 != out_c { return None; }
                } else {
                    depth_map.push((relative_d, out_c));
                }
            }
        }
    }

    depth_map.sort_by_key(|&(d, _)| d);
    depth_map.dedup();

    if depth_map.len() < 2 { return None; }

    let period = depth_map.len();
    let (mut correct, mut total) = (0usize, 0usize);
    for ex in &task.train {
        let h = ex.output.height;
        let w = ex.output.width;
        for r in 0..h {
            for c in 0..w {
                let d = chebyshev_border_dist(r, c, h, w);
                let idx = d % period;
                let pred_c = depth_map.get(idx).map(|&(_, c)| c).unwrap_or(uniform_color);
                if pred_c == ex.output.cells[r][c] { correct += 1; }
                total += 1;
            }
        }
    }

    if total > 0 && correct == total {
        Some((uniform_color, depth_map))
    } else {
        None
    }
}

fn apply_depth_color_map(grid: &Grid, depth_map: &[(usize, u8)], default_color: u8) -> Grid {
    let h = grid.height;
    let w = grid.width;
    let period = depth_map.len();
    let mut cells = vec![vec![default_color; w]; h];
    for r in 0..h {
        for c in 0..w {
            let d = chebyshev_border_dist(r, c, h, w);
            let idx = d % period;
            cells[r][c] = depth_map.get(idx).map(|&(_, col)| col).unwrap_or(default_color);
        }
    }
    Grid { cells, height: h, width: w }
}

// ─── Strategy F: Ring color cycle ──────────────────────────────────────────
//
// Concentric-ring tasks where the output is a cyclic color permutation:
// extract unique colors by depth order, shift each to its predecessor.
// (bda2d7a6-style: outer→innermost color, mid→outer color, etc.)

fn solve_ring_color_cycle(task: &ArcTask) -> Option<f32> {
    for ex in &task.train {
        if ex.input.height != ex.output.height || ex.input.width != ex.output.width {
            return None;
        }
        let in_rings = detect_concentric_rings(&ex.input)?;
        let out_rings = detect_concentric_rings(&ex.output)?;
        if in_rings.len() != out_rings.len() { return None; }

        let mut seen = [false; NUM_COLORS];
        let mut unique: Vec<u8> = Vec::new();
        for &c in &in_rings {
            if !seen[c as usize] { seen[c as usize] = true; unique.push(c); }
        }
        let k = unique.len();
        if k < 2 { return None; }

        let mut perm = [0u8; NUM_COLORS];
        for (i, &c) in unique.iter().enumerate() {
            perm[c as usize] = unique[(i + k - 1) % k];
        }

        for (&ic, &oc) in in_rings.iter().zip(out_rings.iter()) {
            if perm[ic as usize] != oc { return None; }
        }
    }

    let (mut correct, mut total) = (0usize, 0usize);
    for ex in &task.train {
        if let Some(pred) = apply_ring_color_cycle(&ex.input) {
            let (c, t) = grid_matches(&pred, &ex.output);
            correct += c;
            total += t;
        } else {
            return None;
        }
    }
    Some(if total > 0 { correct as f32 / total as f32 } else { 0.0 })
}

fn apply_ring_color_cycle(grid: &Grid) -> Option<Grid> {
    let rings = detect_concentric_rings(grid)?;
    let mut seen = [false; NUM_COLORS];
    let mut unique: Vec<u8> = Vec::new();
    for &c in &rings {
        if !seen[c as usize] { seen[c as usize] = true; unique.push(c); }
    }
    let k = unique.len();
    if k < 2 { return None; }

    let mut perm = [0u8; NUM_COLORS];
    for (i, &c) in unique.iter().enumerate() {
        perm[c as usize] = unique[(i + k - 1) % k];
    }

    let h = grid.height;
    let w = grid.width;
    let mut cells = vec![vec![0u8; w]; h];
    for r in 0..h {
        for c in 0..w {
            cells[r][c] = perm[grid.cells[r][c] as usize];
        }
    }
    Some(Grid { cells, height: h, width: w })
}

// ─── Strategy G: Diagonal X from special cell ──────────────────────────────
//
// Input: uniform grid with exactly one cell of a different color.
// Output: diagonal X (both diagonals) drawn through that cell.

fn find_special_cell(grid: &Grid) -> Option<(u8, usize, usize, u8)> {
    let h = grid.height;
    let w = grid.width;
    if h == 0 || w == 0 { return None; }

    let mut counts = [0u32; NUM_COLORS];
    for row in &grid.cells {
        for &c in row { counts[c as usize] += 1; }
    }

    let total = (h * w) as u32;
    let bg = counts.iter().enumerate().max_by_key(|&(_, &cnt)| cnt)?.0 as u8;
    if counts[bg as usize] != total - 1 { return None; }

    for r in 0..h {
        for c in 0..w {
            if grid.cells[r][c] != bg {
                return Some((bg, r, c, grid.cells[r][c]));
            }
        }
    }
    None
}

fn solve_diagonal_x(task: &ArcTask) -> Option<f32> {
    for ex in &task.train {
        if ex.input.height != ex.output.height || ex.input.width != ex.output.width {
            return None;
        }
        let (_bg, sr, sc, special) = find_special_cell(&ex.input)?;

        for r in 0..ex.output.height {
            for c in 0..ex.output.width {
                let on_diag = (r as isize - sr as isize).unsigned_abs()
                    == (c as isize - sc as isize).unsigned_abs();
                let expected = if on_diag { special } else { ex.input.cells[r][c] };
                if ex.output.cells[r][c] != expected { return None; }
            }
        }
    }

    let (mut correct, mut total) = (0usize, 0usize);
    for ex in &task.train {
        let pred = apply_diagonal_x(&ex.input)?;
        let (c, t) = grid_matches(&pred, &ex.output);
        correct += c;
        total += t;
    }
    Some(if total > 0 { correct as f32 / total as f32 } else { 0.0 })
}

fn apply_diagonal_x(grid: &Grid) -> Option<Grid> {
    let (_bg, sr, sc, special) = find_special_cell(grid)?;
    let h = grid.height;
    let w = grid.width;
    let mut cells = grid.cells.clone();
    for r in 0..h {
        for c in 0..w {
            if (r as isize - sr as isize).unsigned_abs()
                == (c as isize - sc as isize).unsigned_abs()
            {
                cells[r][c] = special;
            }
        }
    }
    Some(Grid { cells, height: h, width: w })
}

// ─── Strategy H: Clockwise spiral fill ─────────────────────────────────────
//
// All-uniform input → output is a clockwise inward spiral alternating two
// colors. Each ring's perimeter: (p-1) cells of current color, last cell
// of alternate. Single-cell rings get current color only.

fn spiral_fill(h: usize, w: usize, color_a: u8, color_b: u8) -> Grid {
    let mut cells = vec![vec![0u8; w]; h];
    let max_rings = (h.min(w) + 1) / 2;
    let mut cur = color_a;
    let mut alt = color_b;
    let mut ring = 0usize;

    loop {
        let top = ring;
        let left = ring;
        let bottom = if h > ring { h - 1 - ring } else { break };
        let right = if w > ring { w - 1 - ring } else { break };
        if top > bottom || left > right { break; }

        let mut path: Vec<(usize, usize)> = Vec::new();

        if top == bottom {
            for c in left..=right { path.push((top, c)); }
        } else if left == right {
            for r in top..=bottom { path.push((r, left)); }
        } else {
            for c in left..=right { path.push((top, c)); }
            for r in (top + 1)..=bottom { path.push((r, right)); }
            for c in (left..right).rev() { path.push((bottom, c)); }
            for r in ((top + 1)..bottom).rev() { path.push((r, left)); }
        }

        let n = path.len();
        let is_last = ring + 1 >= max_rings;
        let suppress = is_last && n <= 4 && max_rings % 2 == 1;

        for (i, &(r, c)) in path.iter().enumerate() {
            cells[r][c] = if n == 1 || suppress || i < n - 1 { cur } else { alt };
        }

        std::mem::swap(&mut cur, &mut alt);
        ring += 1;
    }

    Grid { cells, height: h, width: w }
}

fn detect_spiral_fill(task: &ArcTask) -> Option<(u8, u8)> {
    let input_color = is_uniform_grid(&task.train[0].input)?;
    for ex in &task.train {
        if is_uniform_grid(&ex.input) != Some(input_color) { return None; }
        if ex.input.height != ex.output.height || ex.input.width != ex.output.width {
            return None;
        }
    }

    let first_out = &task.train[0].output;
    let color_a = first_out.cells[0][0];
    let color_b = {
        let mut other = None;
        for row in &first_out.cells {
            for &c in row {
                if c != color_a { other = Some(c); break; }
            }
            if other.is_some() { break; }
        }
        other?
    };

    for ex in &task.train {
        let pred = spiral_fill(ex.output.height, ex.output.width, color_a, color_b);
        if !grid_exact_match(&pred, &ex.output) { return None; }
    }

    Some((color_a, color_b))
}

// ─── Connected component detection ─────────────────────────────────────────
//
// Flood fill (4-connected) over non-background cells. Each connected region
// of the same color is an "object" with position, size, and color.

#[derive(Clone, Debug)]
pub struct GridObject {
    pub color: u8,
    pub pixels: Vec<(usize, usize)>,
    pub min_r: usize,
    pub max_r: usize,
    pub min_c: usize,
    pub max_c: usize,
}

impl GridObject {
    fn centroid(&self) -> (f32, f32) {
        let n = self.pixels.len() as f32;
        let r: f32 = self.pixels.iter().map(|&(r, _)| r as f32).sum::<f32>() / n;
        let c: f32 = self.pixels.iter().map(|&(_, c)| c as f32).sum::<f32>() / n;
        (r, c)
    }
    fn size(&self) -> usize { self.pixels.len() }
    fn bbox_h(&self) -> usize { self.max_r - self.min_r + 1 }
    fn bbox_w(&self) -> usize { self.max_c - self.min_c + 1 }
}

pub fn find_objects(grid: &Grid, background: u8) -> Vec<GridObject> {
    let h = grid.height;
    let w = grid.width;
    let mut visited = vec![vec![false; w]; h];
    let mut objects = Vec::new();

    for r in 0..h {
        for c in 0..w {
            if visited[r][c] || grid.cells[r][c] == background { continue; }
            let color = grid.cells[r][c];
            let mut pixels = Vec::new();
            let mut stack = vec![(r, c)];
            visited[r][c] = true;

            while let Some((cr, cc)) = stack.pop() {
                pixels.push((cr, cc));
                for &(dr, dc) in &[(0isize, 1isize), (0, -1), (1, 0), (-1, 0)] {
                    let nr = cr as isize + dr;
                    let nc = cc as isize + dc;
                    if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                        let (nr, nc) = (nr as usize, nc as usize);
                        if !visited[nr][nc] && grid.cells[nr][nc] == color {
                            visited[nr][nc] = true;
                            stack.push((nr, nc));
                        }
                    }
                }
            }

            let min_r = pixels.iter().map(|&(r, _)| r).min().unwrap();
            let max_r = pixels.iter().map(|&(r, _)| r).max().unwrap();
            let min_c = pixels.iter().map(|&(_, c)| c).min().unwrap();
            let max_c = pixels.iter().map(|&(_, c)| c).max().unwrap();

            objects.push(GridObject { color, pixels, min_r, max_r, min_c, max_c });
        }
    }
    objects
}

fn most_common_color(grid: &Grid) -> u8 {
    let mut counts = [0u32; NUM_COLORS];
    for row in &grid.cells {
        for &c in row { counts[c as usize] += 1; }
    }
    counts.iter().enumerate().max_by_key(|&(_, &cnt)| cnt).map(|(i, _)| i as u8).unwrap_or(0)
}

// ─── Strategy J: Object-level encoding and rule extraction ─────────────────
//
// Encode each detected object as a multivector (color → grade-1,
// centroid position → grade-2, size/shape → scalar+grade-3).
// Match objects across input→output by color, extract positional delta,
// validate consistency across training examples, apply to test.

fn encode_object_mv(obj: &GridObject, h: usize, w: usize) -> Multivector {
    let mut mv = Multivector::zero();
    let cv = color_vector_full(obj.color);
    let g1 = GRADE_OFFSETS[1];
    for i in 0..8 { mv.components[g1 + i] = cv.components[g1 + i]; }

    let (cr, cc) = obj.centroid();
    let nr = if h > 1 { cr / (h - 1) as f32 } else { 0.5 };
    let nc = if w > 1 { cc / (w - 1) as f32 } else { 0.5 };
    let g2 = GRADE_OFFSETS[2];
    mv.components[g2] = nr;
    mv.components[g2 + 1] = nc;
    mv.components[g2 + 2] = obj.size() as f32 / (h * w) as f32;
    mv.components[g2 + 3] = obj.bbox_h() as f32 / h as f32;
    mv.components[g2 + 4] = obj.bbox_w() as f32 / w as f32;

    mv
}

fn match_objects_by_color<'a>(
    in_objs: &'a [GridObject],
    out_objs: &'a [GridObject],
) -> Vec<(&'a GridObject, &'a GridObject)> {
    let mut matched = Vec::new();
    let mut used_out = vec![false; out_objs.len()];

    for in_obj in in_objs {
        let mut best_j = None;
        let mut best_dist = f32::MAX;
        for (j, out_obj) in out_objs.iter().enumerate() {
            if used_out[j] || out_obj.color != in_obj.color { continue; }
            let (ir, ic) = in_obj.centroid();
            let (or, oc) = out_obj.centroid();
            let dist = (ir - or).powi(2) + (ic - oc).powi(2);
            if dist < best_dist {
                best_dist = dist;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            used_out[j] = true;
            matched.push((in_obj, &out_objs[j]));
        }
    }
    matched
}

fn solve_object_positional(task: &ArcTask) -> Option<f32> {
    if !task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width) {
        return None;
    }
    if task.train.is_empty() { return None; }

    let bg = most_common_color(&task.train[0].input);

    let mut consistent_deltas: Option<Vec<(u8, isize, isize)>> = None;

    for ex in &task.train {
        let in_objs = find_objects(&ex.input, bg);
        let out_objs = find_objects(&ex.output, bg);
        if in_objs.is_empty() || out_objs.is_empty() { return None; }

        let matched = match_objects_by_color(&in_objs, &out_objs);
        if matched.is_empty() { return None; }

        let mut deltas: Vec<(u8, isize, isize)> = matched.iter().map(|(inp, out)| {
            let (ir, ic) = inp.centroid();
            let (or, oc) = out.centroid();
            (inp.color, (or - ir).round() as isize, (oc - ic).round() as isize)
        }).collect();
        deltas.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

        match &consistent_deltas {
            None => consistent_deltas = Some(deltas),
            Some(prev) => {
                if prev.len() != deltas.len() { return None; }
                for (p, d) in prev.iter().zip(deltas.iter()) {
                    if p != d { return None; }
                }
            }
        }
    }

    let deltas = consistent_deltas?;
    if deltas.is_empty() { return None; }
    if deltas.iter().all(|&(_, dr, dc)| dr == 0 && dc == 0) { return None; }

    let (mut correct, mut total) = (0usize, 0usize);
    for ex in &task.train {
        let pred = apply_object_positional(&ex.input, bg, &deltas);
        let (c, t) = grid_matches(&pred, &ex.output);
        correct += c;
        total += t;
    }

    if total > 0 && correct == total {
        Some(1.0)
    } else {
        None
    }
}

fn apply_object_positional(grid: &Grid, bg: u8, deltas: &[(u8, isize, isize)]) -> Grid {
    let h = grid.height;
    let w = grid.width;
    let mut cells = vec![vec![bg; w]; h];

    for row in &grid.cells {
        for (c_idx, &cell) in row.iter().enumerate() {
            if cell == bg { cells[c_idx / w][c_idx % w] = bg; }
        }
    }
    for r in 0..h {
        for c in 0..w {
            if grid.cells[r][c] == bg { cells[r][c] = bg; }
        }
    }

    let objects = find_objects(grid, bg);

    for obj in &objects {
        let delta = deltas.iter().find(|&&(color, _, _)| color == obj.color);
        let (dr, dc) = match delta {
            Some(&(_, dr, dc)) => (dr, dc),
            None => (0, 0),
        };

        for &(pr, pc) in &obj.pixels {
            let nr = pr as isize + dr;
            let nc = pc as isize + dc;
            if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                cells[nr as usize][nc as usize] = obj.color;
            }
        }
    }

    Grid { cells, height: h, width: w }
}

// ─── Strategy K: Object color change ───────────────────────────────────────
//
// Objects stay in place but change color. Match objects by position overlap,
// learn the color mapping, apply to test.

fn solve_object_color_change(task: &ArcTask) -> Option<f32> {
    if !task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width) {
        return None;
    }
    if task.train.is_empty() { return None; }

    let bg = most_common_color(&task.train[0].input);
    let mut color_map = [None::<u8>; NUM_COLORS];

    for ex in &task.train {
        let in_objs = find_objects(&ex.input, bg);
        let out_objs = find_objects(&ex.output, bg);

        for in_obj in &in_objs {
            let (ir, ic) = in_obj.centroid();
            let mut best = None;
            let mut best_dist = f32::MAX;
            for out_obj in &out_objs {
                let (or, oc) = out_obj.centroid();
                let dist = (ir - or).powi(2) + (ic - oc).powi(2);
                if dist < best_dist {
                    best_dist = dist;
                    best = Some(out_obj.color);
                }
            }
            if let Some(out_c) = best {
                match color_map[in_obj.color as usize] {
                    None => color_map[in_obj.color as usize] = Some(out_c),
                    Some(prev) if prev != out_c => return None,
                    _ => {}
                }
            }
        }
    }

    let mut cmap = [0u8; NUM_COLORS];
    for c in 0..NUM_COLORS {
        cmap[c] = color_map[c].unwrap_or(c as u8);
    }
    cmap[bg as usize] = bg;

    let (mut correct, mut total) = (0usize, 0usize);
    for ex in &task.train {
        let pred = apply_object_color_map(&ex.input, &cmap);
        let (c, t) = grid_matches(&pred, &ex.output);
        correct += c;
        total += t;
    }

    if total > 0 && correct == total {
        Some(1.0)
    } else {
        None
    }
}

fn apply_object_color_map(grid: &Grid, cmap: &[u8; NUM_COLORS]) -> Grid {
    let mut cells = grid.cells.clone();
    for row in &mut cells {
        for cell in row {
            *cell = cmap[*cell as usize];
        }
    }
    Grid { cells, height: grid.height, width: grid.width }
}

// ─── Strategy M: L-shape diagonal extension ────────────────────────────────
//
// Find 3-cell L-shaped objects. For each L, identify the missing corner of its
// 2×2 bounding box. Draw a diagonal line from that corner in the direction
// away from the elbow, extending to the grid edge.

fn is_l_shape(obj: &GridObject) -> Option<((usize, usize), (isize, isize))> {
    if obj.pixels.len() != 3 { return None; }
    if obj.bbox_h() != 2 || obj.bbox_w() != 2 { return None; }

    let corners = [
        (obj.min_r, obj.min_c),
        (obj.min_r, obj.max_c),
        (obj.max_r, obj.min_c),
        (obj.max_r, obj.max_c),
    ];

    let mut missing = None;
    for &(r, c) in &corners {
        if !obj.pixels.contains(&(r, c)) {
            missing = Some((r, c));
            break;
        }
    }
    let (mr, mc) = missing?;

    let elbow_r = if mr == obj.min_r { obj.max_r } else { obj.min_r };
    let elbow_c = if mc == obj.min_c { obj.max_c } else { obj.min_c };
    let dr = mr as isize - elbow_r as isize;
    let dc = mc as isize - elbow_c as isize;

    Some(((mr, mc), (dr, dc)))
}

fn solve_l_diagonal(task: &ArcTask) -> Option<f32> {
    if !task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width) {
        return None;
    }
    if task.train.is_empty() { return None; }

    let (mut correct, mut total) = (0usize, 0usize);
    for ex in &task.train {
        let pred = apply_l_diagonal(&ex.input)?;
        let (c, t) = grid_matches(&pred, &ex.output);
        correct += c;
        total += t;
    }
    if total > 0 && correct == total { Some(1.0) } else { None }
}

fn apply_l_diagonal(grid: &Grid) -> Option<Grid> {
    let bg = most_common_color(grid);
    let objects = find_objects(grid, bg);
    if objects.is_empty() { return None; }
    if !objects.iter().all(|o| is_l_shape(o).is_some()) { return None; }

    let h = grid.height;
    let w = grid.width;
    let mut cells = grid.cells.clone();

    for obj in &objects {
        let ((mr, mc), (dr, dc)) = is_l_shape(obj)?;
        let color = obj.color;
        let mut r = mr as isize + dr;
        let mut c = mc as isize + dc;
        while r >= 0 && r < h as isize && c >= 0 && c < w as isize {
            cells[r as usize][c as usize] = color;
            r += dr;
            c += dc;
        }
    }

    Some(Grid { cells, height: h, width: w })
}

// ─── Strategy N: Subgrid extraction (diff-dim) ─────────────────────────────
//
// For tasks where output is smaller than input: search all subgrids of the
// input that match the output dimensions. If a consistent extraction rule
// exists across training examples, apply it to the test input.

fn extract_subgrid(grid: &Grid, r0: usize, c0: usize, h: usize, w: usize) -> Grid {
    let cells: Vec<Vec<u8>> = (0..h)
        .map(|r| (0..w).map(|c| grid.cells[r0 + r][c0 + c]).collect())
        .collect();
    Grid { cells, height: h, width: w }
}

fn find_matching_subgrids(
    input: &Grid, output: &Grid,
) -> Vec<(usize, usize)> {
    let oh = output.height;
    let ow = output.width;
    if oh > input.height || ow > input.width { return vec![]; }

    let mut matches = Vec::new();
    for r0 in 0..=(input.height - oh) {
        for c0 in 0..=(input.width - ow) {
            let sub = extract_subgrid(input, r0, c0, oh, ow);
            if grid_exact_match(&sub, output) {
                matches.push((r0, c0));
            }
        }
    }
    matches
}

/// Try fixed-offset extraction: same (r,c) offset across all training examples.
fn solve_subgrid_fixed_offset(task: &ArcTask) -> Option<(usize, usize, usize, usize)> {
    if task.train.is_empty() { return None; }

    let oh = task.train[0].output.height;
    let ow = task.train[0].output.width;
    if !task.train.iter().all(|ex|
        ex.output.height == oh && ex.output.width == ow
        && ex.output.height <= ex.input.height && ex.output.width <= ex.input.width
    ) { return None; }

    let first_matches = find_matching_subgrids(&task.train[0].input, &task.train[0].output);
    if first_matches.is_empty() { return None; }

    for &(r0, c0) in &first_matches {
        let all_match = task.train[1..].iter().all(|ex| {
            if r0 + oh > ex.input.height || c0 + ow > ex.input.width { return false; }
            let sub = extract_subgrid(&ex.input, r0, c0, oh, ow);
            grid_exact_match(&sub, &ex.output)
        });
        if all_match {
            return Some((r0, c0, oh, ow));
        }
    }
    None
}

/// Try object-bounding-box extraction: output = bbox of a specific colored region.
fn solve_subgrid_object_bbox(task: &ArcTask) -> Option<u8> {
    if task.train.is_empty() { return None; }
    let bg = most_common_color(&task.train[0].input);

    let mut consistent_color: Option<u8> = None;

    for ex in &task.train {
        let objects = find_objects(&ex.input, bg);
        let oh = ex.output.height;
        let ow = ex.output.width;

        let mut found_color = None;
        for obj in &objects {
            if obj.bbox_h() != oh || obj.bbox_w() != ow { continue; }
            let sub = extract_subgrid(&ex.input, obj.min_r, obj.min_c, oh, ow);
            if grid_exact_match(&sub, &ex.output) {
                found_color = Some(obj.color);
                break;
            }
        }
        match (found_color, &consistent_color) {
            (None, _) => return None,
            (Some(c), None) => consistent_color = Some(c),
            (Some(c), Some(prev)) if c != *prev => return None,
            _ => {}
        }
    }
    consistent_color
}

/// Extract the bounding box of all non-background content.
fn solve_subgrid_content_bbox(task: &ArcTask) -> Option<()> {
    if task.train.is_empty() { return None; }

    for ex in &task.train {
        let bg = most_common_color(&ex.input);
        let bbox = content_bbox(&ex.input, bg)?;
        let (r0, c0, bh, bw) = bbox;
        if bh != ex.output.height || bw != ex.output.width { return None; }
        let sub = extract_subgrid(&ex.input, r0, c0, bh, bw);
        if !grid_exact_match(&sub, &ex.output) { return None; }
    }
    Some(())
}

fn content_bbox(grid: &Grid, bg: u8) -> Option<(usize, usize, usize, usize)> {
    let mut min_r = grid.height;
    let mut max_r = 0usize;
    let mut min_c = grid.width;
    let mut max_c = 0usize;
    let mut found = false;
    for r in 0..grid.height {
        for c in 0..grid.width {
            if grid.cells[r][c] != bg {
                found = true;
                min_r = min_r.min(r);
                max_r = max_r.max(r);
                min_c = min_c.min(c);
                max_c = max_c.max(c);
            }
        }
    }
    if !found { return None; }
    Some((min_r, min_c, max_r - min_r + 1, max_c - min_c + 1))
}

/// Unique-subgrid extraction: output is the only subgrid of its dimensions
/// that exists exactly once in the input. Validated on training, applied to test.
fn solve_subgrid_unique(task: &ArcTask) -> Option<()> {
    if task.train.is_empty() { return None; }
    for ex in &task.train {
        if ex.output.height > ex.input.height || ex.output.width > ex.input.width {
            return None;
        }
        let matches = find_matching_subgrids(&ex.input, &ex.output);
        if matches.len() != 1 { return None; }
    }
    Some(())
}

/// Extract the bounding box of the MINORITY color (non-background, non-dominant).
fn solve_subgrid_minority_bbox(task: &ArcTask) -> Option<u8> {
    if task.train.is_empty() { return None; }

    let mut consistent_color: Option<u8> = None;

    for ex in &task.train {
        let bg = most_common_color(&ex.input);
        let mut color_counts = [0u32; NUM_COLORS];
        for row in &ex.input.cells {
            for &c in row {
                if c != bg { color_counts[c as usize] += 1; }
            }
        }
        let minority = color_counts.iter().enumerate()
            .filter(|&(i, &cnt)| cnt > 0 && i != bg as usize)
            .min_by_key(|&(_, &cnt)| cnt)
            .map(|(i, _)| i as u8)?;

        let objects = find_objects(&ex.input, bg);
        let target_objs: Vec<&GridObject> = objects.iter()
            .filter(|o| o.color == minority)
            .collect();
        if target_objs.is_empty() { return None; }

        let min_r = target_objs.iter().map(|o| o.min_r).min().unwrap();
        let max_r = target_objs.iter().map(|o| o.max_r).max().unwrap();
        let min_c = target_objs.iter().map(|o| o.min_c).min().unwrap();
        let max_c = target_objs.iter().map(|o| o.max_c).max().unwrap();
        let bh = max_r - min_r + 1;
        let bw = max_c - min_c + 1;

        if bh != ex.output.height || bw != ex.output.width { return None; }
        let sub = extract_subgrid(&ex.input, min_r, min_c, bh, bw);
        if !grid_exact_match(&sub, &ex.output) { return None; }

        match consistent_color {
            None => consistent_color = Some(minority),
            Some(prev) if prev != minority => return None,
            _ => {}
        }
    }
    consistent_color
}

// ─── Strategy O: Tiling (diff-dim, output >= input) ────────────────────────
//
// When output dimensions are integer multiples of input dimensions, check
// if the output is the input tiled repeatedly.

fn solve_tiling(task: &ArcTask) -> Option<(usize, usize)> {
    if task.train.is_empty() { return None; }

    let mut tile_dims: Option<(usize, usize)> = None;

    for ex in &task.train {
        let ih = ex.input.height;
        let iw = ex.input.width;
        let oh = ex.output.height;
        let ow = ex.output.width;

        if oh < ih || ow < iw { return None; }
        if oh % ih != 0 || ow % iw != 0 { return None; }

        let tr = oh / ih;
        let tc = ow / iw;
        if tr < 1 || tc < 1 { return None; }

        let pred = tile_grid(&ex.input, tr, tc);
        if !grid_exact_match(&pred, &ex.output) { return None; }

        match tile_dims {
            None => tile_dims = Some((tr, tc)),
            Some((pr, pc)) if pr != tr || pc != tc => return None,
            _ => {}
        }
    }
    tile_dims
}

fn tile_grid(grid: &Grid, tile_r: usize, tile_c: usize) -> Grid {
    let ih = grid.height;
    let iw = grid.width;
    let oh = ih * tile_r;
    let ow = iw * tile_c;
    let cells: Vec<Vec<u8>> = (0..oh)
        .map(|r| (0..ow).map(|c| grid.cells[r % ih][c % iw]).collect())
        .collect();
    Grid { cells, height: oh, width: ow }
}

// ─── Strategy S: Canvas embedding (diff-dim, output contains input) ────────
//
// Input is placed at a consistent offset inside a larger output, with the
// remaining cells filled by a fixed background color.

fn solve_canvas(task: &ArcTask) -> Option<(isize, isize, u8)> {
    if task.train.is_empty() { return None; }

    let mut consistent: Option<(isize, isize, u8)> = None;

    for ex in &task.train {
        let ih = ex.input.height;
        let iw = ex.input.width;
        let oh = ex.output.height;
        let ow = ex.output.width;
        if oh < ih || ow < iw { return None; }

        let mut found = None;
        'search: for r0 in 0..=(oh - ih) {
            for c0 in 0..=(ow - iw) {
                let mut ok = true;
                for r in 0..ih {
                    for c in 0..iw {
                        if ex.output.cells[r0 + r][c0 + c] != ex.input.cells[r][c] {
                            ok = false;
                            break;
                        }
                    }
                    if !ok { break; }
                }
                if ok {
                    found = Some((r0 as isize, c0 as isize));
                    break 'search;
                }
            }
        }
        let (r0, c0) = found?;

        let mut bg_color = None;
        for r in 0..oh {
            for c in 0..ow {
                let in_region = r >= r0 as usize && r < r0 as usize + ih
                    && c >= c0 as usize && c < c0 as usize + iw;
                if !in_region {
                    let v = ex.output.cells[r][c];
                    match bg_color {
                        None => bg_color = Some(v),
                        Some(prev) if prev != v => return None,
                        _ => {}
                    }
                }
            }
        }
        let bg = bg_color.unwrap_or(0);

        match consistent {
            None => consistent = Some((r0, c0, bg)),
            Some((pr, pc, pb)) if pr != r0 || pc != c0 || pb != bg => return None,
            _ => {}
        }
    }
    consistent
}

fn apply_canvas(grid: &Grid, r0: isize, c0: isize, oh: usize, ow: usize, bg: u8) -> Grid {
    let mut cells = vec![vec![bg; ow]; oh];
    for r in 0..grid.height {
        for c in 0..grid.width {
            let tr = r as isize + r0;
            let tc = c as isize + c0;
            if tr >= 0 && tr < oh as isize && tc >= 0 && tc < ow as isize {
                cells[tr as usize][tc as usize] = grid.cells[r][c];
            }
        }
    }
    Grid { cells, height: oh, width: ow }
}

// ─── Strategy T: Downscale (diff-dim, input shrinks by integer factor) ─────
//
// Each NxN block of the input maps to one output cell. Try different
// aggregation methods: majority vote, non-background winner.

fn solve_downscale(task: &ArcTask) -> Option<(usize, usize, u8)> {
    if task.train.is_empty() { return None; }

    for method in 0u8..3 {
        let mut consistent_factor: Option<(usize, usize)> = None;
        let mut all_ok = true;

        for ex in &task.train {
            let ih = ex.input.height;
            let iw = ex.input.width;
            let oh = ex.output.height;
            let ow = ex.output.width;

            if oh > ih || ow > iw { all_ok = false; break; }
            if ih % oh != 0 || iw % ow != 0 { all_ok = false; break; }

            let sr = ih / oh;
            let sc = iw / ow;
            if sr < 2 || sc < 2 { all_ok = false; break; }

            match consistent_factor {
                None => consistent_factor = Some((sr, sc)),
                Some((pr, pc)) if pr != sr || pc != sc => { all_ok = false; break; }
                _ => {}
            }

            let pred = downscale_grid(&ex.input, sr, sc, method);
            if !grid_exact_match(&pred, &ex.output) { all_ok = false; break; }
        }

        if all_ok {
            let (sr, sc) = consistent_factor?;
            return Some((sr, sc, method));
        }
    }
    None
}

fn downscale_grid(grid: &Grid, sr: usize, sc: usize, method: u8) -> Grid {
    let oh = grid.height / sr;
    let ow = grid.width / sc;
    let bg = most_common_color(grid);
    let mut cells = vec![vec![0u8; ow]; oh];

    for r in 0..oh {
        for c in 0..ow {
            let mut counts = [0u32; NUM_COLORS];
            for dr in 0..sr {
                for dc in 0..sc {
                    counts[grid.cells[r * sr + dr][c * sc + dc] as usize] += 1;
                }
            }
            cells[r][c] = match method {
                0 => {
                    // Majority vote
                    counts.iter().enumerate()
                        .max_by_key(|&(_, &cnt)| cnt)
                        .map(|(i, _)| i as u8).unwrap_or(0)
                }
                1 => {
                    // Non-background winner (most common non-bg, or bg if all bg)
                    let non_bg: Vec<(usize, u32)> = counts.iter().enumerate()
                        .filter(|&(i, &cnt)| cnt > 0 && i != bg as usize)
                        .map(|(i, &cnt)| (i, cnt))
                        .collect();
                    if non_bg.is_empty() { bg }
                    else { non_bg.iter().max_by_key(|&&(_, cnt)| cnt).unwrap().0 as u8 }
                }
                _ => {
                    // Minority non-bg (least common non-bg, or bg if all bg)
                    let non_bg: Vec<(usize, u32)> = counts.iter().enumerate()
                        .filter(|&(i, &cnt)| cnt > 0 && i != bg as usize)
                        .map(|(i, &cnt)| (i, cnt))
                        .collect();
                    if non_bg.is_empty() { bg }
                    else { non_bg.iter().min_by_key(|&&(_, cnt)| cnt).unwrap().0 as u8 }
                }
            };
        }
    }
    Grid { cells, height: oh, width: ow }
}

// ─── Strategy U: Mirror tiling (diff-dim) ──────────────────────────────────
//
// Output is the input reflected along one or both axes:
//   mode 0: 2x2 quadrant mirror (TL=inp, TR=hflip, BL=vflip, BR=rot180)
//   mode 1: horizontal concat (L=inp, R=hflip)
//   mode 2: vertical concat (T=inp, B=vflip)

fn solve_mirror_tile(task: &ArcTask) -> Option<u8> {
    if task.train.is_empty() { return None; }

    for mode in 0u8..4 {
        let all_ok = task.train.iter().all(|ex| {
            let pred = apply_mirror_tile(&ex.input, mode);
            match pred {
                Some(ref p) => grid_exact_match(p, &ex.output),
                None => false,
            }
        });
        if all_ok { return Some(mode); }
    }
    None
}

fn apply_mirror_tile(grid: &Grid, mode: u8) -> Option<Grid> {
    let ih = grid.height;
    let iw = grid.width;

    match mode {
        0 => {
            let oh = ih * 2;
            let ow = iw * 2;
            let cells: Vec<Vec<u8>> = (0..oh).map(|r| {
                (0..ow).map(|c| {
                    let sr = if r < ih { r } else { oh - 1 - r };
                    let sc = if c < iw { c } else { ow - 1 - c };
                    grid.cells[sr][sc]
                }).collect()
            }).collect();
            Some(Grid { cells, height: oh, width: ow })
        }
        1 => {
            let ow = iw * 2;
            let cells: Vec<Vec<u8>> = (0..ih).map(|r| {
                (0..ow).map(|c| {
                    if c < iw { grid.cells[r][c] }
                    else { grid.cells[r][ow - 1 - c] }
                }).collect()
            }).collect();
            Some(Grid { cells, height: ih, width: ow })
        }
        2 => {
            let oh = ih * 2;
            let cells: Vec<Vec<u8>> = (0..oh).map(|r| {
                (0..iw).map(|c| {
                    if r < ih { grid.cells[r][c] }
                    else { grid.cells[oh - 1 - r][c] }
                }).collect()
            }).collect();
            Some(Grid { cells, height: oh, width: iw })
        }
        3 => {
            // Rot90 quadrant: TL=inp, TR=rot90_cw, BL=rot90_ccw, BR=rot180
            if ih != iw { return None; }
            let n = ih;
            let oh = n * 2;
            let ow = n * 2;
            let cells: Vec<Vec<u8>> = (0..oh).map(|r| {
                (0..ow).map(|c| {
                    let qr = r / n;
                    let qc = c / n;
                    let lr = r % n;
                    let lc = c % n;
                    match (qr, qc) {
                        (0, 0) => grid.cells[lr][lc],
                        (0, 1) => grid.cells[n - 1 - lc][lr],   // rot90_cw
                        (1, 0) => grid.cells[lc][n - 1 - lr],   // rot90_ccw
                        _      => grid.cells[n - 1 - lr][n - 1 - lc], // rot180
                    }
                }).collect()
            }).collect();
            Some(Grid { cells, height: oh, width: ow })
        }
        _ => None,
    }
}

// ─── Strategy P: Grid scaling (diff-dim, output = input × factor) ──────────
//
// Each input cell becomes a factor_r × factor_c block of the same color.
// Tiling repeats the whole grid; scaling magnifies each cell.

/// Detects pixel scaling: each input cell becomes an sr×sc block.
/// Scale factor can vary per example (derived from output/input ratio).
/// Returns true if all training examples are valid pixel-scale transforms.
/// Also returns whether the factor is fixed (same across examples) for apply logic.
fn solve_scale(task: &ArcTask) -> Option<(usize, usize)> {
    if task.train.is_empty() { return None; }

    let mut fixed_scale: Option<(usize, usize)> = None;
    let mut all_same = true;

    for ex in &task.train {
        let ih = ex.input.height;
        let iw = ex.input.width;
        let oh = ex.output.height;
        let ow = ex.output.width;

        if oh < ih || ow < iw { return None; }
        if oh % ih != 0 || ow % iw != 0 { return None; }

        let sr = oh / ih;
        let sc = ow / iw;
        if sr < 2 && sc < 2 { return None; }

        let pred = scale_grid(&ex.input, sr, sc);
        if !grid_exact_match(&pred, &ex.output) { return None; }

        match fixed_scale {
            None => fixed_scale = Some((sr, sc)),
            Some((pr, pc)) if pr != sr || pc != sc => all_same = false,
            _ => {}
        }
    }

    // Return (0, 0) as sentinel for "variable scale — derive from output dims"
    if all_same { fixed_scale } else { Some((0, 0)) }
}

fn scale_grid(grid: &Grid, sr: usize, sc: usize) -> Grid {
    let oh = grid.height * sr;
    let ow = grid.width * sc;
    let cells: Vec<Vec<u8>> = (0..oh)
        .map(|r| (0..ow).map(|c| grid.cells[r / sr][c / sc]).collect())
        .collect();
    Grid { cells, height: oh, width: ow }
}

// ─── Strategy Q2: Fractal tile (self-referencing) ───────────────────────────
//
// The input is used as both content and mask: for each cell (r,c),
// if input[r][c] != bg, place a copy of input at tile position (r,c);
// otherwise fill that tile with bg. Requires oh = ih*ih, ow = iw*iw.

fn solve_fractal_tile(task: &ArcTask) -> bool {
    if task.train.is_empty() { return false; }
    task.train.iter().all(|ex| {
        let ih = ex.input.height;
        let iw = ex.input.width;
        let oh = ex.output.height;
        let ow = ex.output.width;
        if oh != ih * ih || ow != iw * iw { return false; }
        let bg = 0u8;
        for tr in 0..ih {
            for tc in 0..iw {
                for r in 0..ih {
                    for c in 0..iw {
                        let expected = if ex.input.cells[tr][tc] != bg {
                            ex.input.cells[r][c]
                        } else {
                            bg
                        };
                        if ex.output.cells[tr * ih + r][tc * iw + c] != expected {
                            return false;
                        }
                    }
                }
            }
        }
        true
    })
}

fn apply_fractal_tile(grid: &Grid) -> Grid {
    let ih = grid.height;
    let iw = grid.width;
    let oh = ih * ih;
    let ow = iw * iw;
    let bg = 0u8;
    let mut cells = vec![vec![bg; ow]; oh];
    for tr in 0..ih {
        for tc in 0..iw {
            if grid.cells[tr][tc] != bg {
                for r in 0..ih {
                    for c in 0..iw {
                        cells[tr * ih + r][tc * iw + c] = grid.cells[r][c];
                    }
                }
            }
        }
    }
    Grid { cells, height: oh, width: ow }
}

// ─── Strategy Q: Gravity — objects fall toward a wall ──────────────────────
//
// All non-background cells slide in a fixed direction (down, up, left, right)
// until blocked by the grid edge or another non-background cell.

fn solve_gravity(task: &ArcTask) -> Option<u8> {
    if task.train.is_empty() { return None; }
    if !task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width) {
        return None;
    }

    for dir in 0u8..4 {
        let all_match = task.train.iter().all(|ex| {
            let pred = apply_gravity(&ex.input, dir);
            grid_exact_match(&pred, &ex.output)
        });
        if all_match { return Some(dir); }
    }
    None
}

fn apply_gravity(grid: &Grid, dir: u8) -> Grid {
    let h = grid.height;
    let w = grid.width;
    let bg = most_common_color(grid);
    let mut cells = vec![vec![bg; w]; h];

    match dir {
        0 => { // down
            for c in 0..w {
                let mut write = h;
                for r in (0..h).rev() {
                    if grid.cells[r][c] != bg {
                        write -= 1;
                        cells[write][c] = grid.cells[r][c];
                    }
                }
            }
        }
        1 => { // up
            for c in 0..w {
                let mut write = 0;
                for r in 0..h {
                    if grid.cells[r][c] != bg {
                        cells[write][c] = grid.cells[r][c];
                        write += 1;
                    }
                }
            }
        }
        2 => { // right
            for r in 0..h {
                let mut write = w;
                for c in (0..w).rev() {
                    if grid.cells[r][c] != bg {
                        write -= 1;
                        cells[r][write] = grid.cells[r][c];
                    }
                }
            }
        }
        3 => { // left
            for r in 0..h {
                let mut write = 0;
                for c in 0..w {
                    if grid.cells[r][c] != bg {
                        cells[r][write] = grid.cells[r][c];
                        write += 1;
                    }
                }
            }
        }
        _ => return grid.clone(),
    }

    Grid { cells, height: h, width: w }
}

// ─── Strategy R: Symmetry completion ───────────────────────────────────────
//
// Detect axis of symmetry in the output, check if the output is the input
// completed to be symmetric along that axis.

fn solve_symmetry(task: &ArcTask) -> Option<u8> {
    if task.train.is_empty() { return None; }
    if !task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width) {
        return None;
    }

    for axis in 0u8..4 {
        let all_match = task.train.iter().all(|ex| {
            let pred = apply_symmetry(&ex.input, axis);
            grid_exact_match(&pred, &ex.output)
        });
        if all_match { return Some(axis); }
    }
    None
}

fn apply_symmetry(grid: &Grid, axis: u8) -> Grid {
    let h = grid.height;
    let w = grid.width;
    let bg = most_common_color(grid);
    let mut cells = grid.cells.clone();

    match axis {
        0 => { // horizontal: mirror top→bottom
            for r in 0..h {
                for c in 0..w {
                    let mr = h - 1 - r;
                    if cells[r][c] == bg && cells[mr][c] != bg {
                        cells[r][c] = cells[mr][c];
                    }
                }
            }
        }
        1 => { // vertical: mirror left→right
            for r in 0..h {
                for c in 0..w {
                    let mc = w - 1 - c;
                    if cells[r][c] == bg && cells[r][mc] != bg {
                        cells[r][c] = cells[r][mc];
                    }
                }
            }
        }
        2 => { // both horizontal + vertical
            for r in 0..h {
                for c in 0..w {
                    let mr = h - 1 - r;
                    let mc = w - 1 - c;
                    if cells[r][c] == bg {
                        if cells[mr][c] != bg { cells[r][c] = cells[mr][c]; }
                        else if cells[r][mc] != bg { cells[r][c] = cells[r][mc]; }
                        else if cells[mr][mc] != bg { cells[r][c] = cells[mr][mc]; }
                    }
                }
            }
        }
        3 => { // diagonal (transpose): mirror across main diagonal
            if h != w { return grid.clone(); }
            for r in 0..h {
                for c in 0..w {
                    if cells[r][c] == bg && cells[c][r] != bg {
                        cells[r][c] = cells[c][r];
                    }
                }
            }
        }
        _ => {}
    }

    Grid { cells, height: h, width: w }
}

// ─── Strategy W: 3×3 neighborhood lookup table ─────────────────────────────
//
// For each cell, extract the 3×3 neighborhood (9 values, bg-padded at edges).
// Build a lookup table mapping each observed neighborhood → output color.
// If consistent across all training examples AND all test neighborhoods are
// covered, apply the table to produce predictions. Zero parameters learned —
// the table IS the rule.

fn solve_nbr_lookup(task: &ArcTask) -> Option<std::collections::HashMap<[u8; 9], u8>> {
    if task.train.is_empty() { return None; }
    if !task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width) {
        return None;
    }

    let bg = most_common_color(&task.train[0].input);
    let mut lut: std::collections::HashMap<[u8; 9], u8> = std::collections::HashMap::new();

    for ex in &task.train {
        let h = ex.input.height;
        let w = ex.input.width;
        for r in 0..h {
            for c in 0..w {
                let nbr = nbr_key(&ex.input, r, c, bg);
                let out_color = ex.output.cells[r][c];
                match lut.get(&nbr) {
                    Some(&existing) if existing != out_color => return None,
                    _ => { lut.insert(nbr, out_color); }
                }
            }
        }
    }

    // Verify all test neighborhoods are covered
    for test_ex in &task.test {
        let h = test_ex.input.height;
        let w = test_ex.input.width;
        for r in 0..h {
            for c in 0..w {
                let nbr = nbr_key(&test_ex.input, r, c, bg);
                if !lut.contains_key(&nbr) { return None; }
            }
        }
    }

    Some(lut)
}

fn nbr_key(grid: &Grid, r: usize, c: usize, pad: u8) -> [u8; 9] {
    let h = grid.height as isize;
    let w = grid.width as isize;
    let mut key = [pad; 9];
    let mut i = 0;
    for dr in -1i32..=1 {
        for dc in -1i32..=1 {
            let nr = r as isize + dr as isize;
            let nc = c as isize + dc as isize;
            if nr >= 0 && nr < h && nc >= 0 && nc < w {
                key[i] = grid.cells[nr as usize][nc as usize];
            }
            i += 1;
        }
    }
    key
}

fn apply_nbr_lookup(grid: &Grid, lut: &std::collections::HashMap<[u8; 9], u8>) -> Grid {
    let bg = most_common_color(grid);
    let h = grid.height;
    let w = grid.width;
    let mut cells = grid.cells.clone();
    for r in 0..h {
        for c in 0..w {
            let nbr = nbr_key(grid, r, c, bg);
            if let Some(&color) = lut.get(&nbr) {
                cells[r][c] = color;
            }
        }
    }
    Grid { cells, height: h, width: w }
}

// ─── Strategy O: Connect lines ─────────────────────────────────────────────
//
// Fill bg cells between same-colored cells in the same row or column.
// Only connects the nearest pair; stops at different-colored obstacles.

fn solve_connect_lines(task: &ArcTask) -> bool {
    if task.train.is_empty() { return false; }
    if !task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width) {
        return false;
    }
    task.train.iter().all(|ex| {
        let pred = apply_connect_lines(&ex.input);
        grid_exact_match(&pred, &ex.output)
    })
}

fn apply_connect_lines(grid: &Grid) -> Grid {
    let h = grid.height;
    let w = grid.width;
    let bg = most_common_color(grid);
    let mut cells = grid.cells.clone();

    // Horizontal connections
    for r in 0..h {
        let mut c = 0;
        while c < w {
            if grid.cells[r][c] == bg { c += 1; continue; }
            let color = grid.cells[r][c];
            let mut c2 = c + 1;
            while c2 < w && grid.cells[r][c2] == bg { c2 += 1; }
            if c2 < w && grid.cells[r][c2] == color {
                for fill_c in (c + 1)..c2 {
                    if cells[r][fill_c] == bg { cells[r][fill_c] = color; }
                }
            }
            c = c2;
        }
    }

    // Vertical connections
    for c in 0..w {
        let mut r = 0;
        while r < h {
            if grid.cells[r][c] == bg { r += 1; continue; }
            let color = grid.cells[r][c];
            let mut r2 = r + 1;
            while r2 < h && grid.cells[r2][c] == bg { r2 += 1; }
            if r2 < h && grid.cells[r2][c] == color {
                for fill_r in (r + 1)..r2 {
                    if cells[fill_r][c] == bg { cells[fill_r][c] = color; }
                }
            }
            r = r2;
        }
    }

    Grid { cells, height: h, width: w }
}

// ─── Strategy P2: Enclosed region extraction ────────────────────────────────
//
// Finds bg cells not reachable from grid edges (enclosed by non-bg cells).
// Extracts the bounding box of enclosed interior as the output.

fn find_enclosed_bbox(grid: &Grid, bg: u8) -> Option<(usize, usize, usize, usize)> {
    let h = grid.height;
    let w = grid.width;
    let mut outside = vec![vec![false; w]; h];
    let mut queue = std::collections::VecDeque::new();

    // Seed BFS from all edge cells that are bg
    for r in 0..h {
        for c in 0..w {
            if (r == 0 || r == h - 1 || c == 0 || c == w - 1) && grid.cells[r][c] == bg {
                if !outside[r][c] {
                    outside[r][c] = true;
                    queue.push_back((r, c));
                }
            }
        }
    }
    while let Some((r, c)) = queue.pop_front() {
        for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                let (nr, nc) = (nr as usize, nc as usize);
                if !outside[nr][nc] && grid.cells[nr][nc] == bg {
                    outside[nr][nc] = true;
                    queue.push_back((nr, nc));
                }
            }
        }
    }

    // Find bbox of enclosed bg cells (not outside, not on border path)
    let mut min_r = h;
    let mut max_r = 0;
    let mut min_c = w;
    let mut max_c = 0;
    for r in 0..h {
        for c in 0..w {
            if grid.cells[r][c] == bg && !outside[r][c] {
                min_r = min_r.min(r);
                max_r = max_r.max(r);
                min_c = min_c.min(c);
                max_c = max_c.max(c);
            }
        }
    }
    if min_r > max_r { return None; }
    Some((min_r, min_c, max_r - min_r + 1, max_c - min_c + 1))
}

fn solve_enclosed_region(task: &ArcTask) -> bool {
    if task.train.is_empty() { return false; }
    // Output must be smaller than input (extraction)
    if task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width) {
        return false;
    }

    task.train.iter().all(|ex| {
        let bg = most_common_color(&ex.input);
        if let Some((r0, c0, bh, bw)) = find_enclosed_bbox(&ex.input, bg) {
            if bh == ex.output.height && bw == ex.output.width {
                let sub = extract_subgrid(&ex.input, r0, c0, bh, bw);
                return grid_exact_match(&sub, &ex.output);
            }
        }
        false
    })
}

fn apply_enclosed_region(grid: &Grid) -> Option<Grid> {
    let bg = most_common_color(grid);
    if let Some((r0, c0, bh, bw)) = find_enclosed_bbox(grid, bg) {
        Some(extract_subgrid(grid, r0, c0, bh, bw))
    } else {
        None
    }
}

// ─── Strategy Z: Operator composition search ───────────────────────────────
//
// Instead of hand-coding each task's strategy, define a set of primitive
// operators and search for short compositions (depth ≤ 2) that transform
// input → output for all training examples. The operator set:
//   Size-changing: Tile, Scale, FractalTile, DiagTile, MirrorTile(4 modes)
//   Geometric:     HFlip, VFlip, Rot90CW, Rot180, Transpose
//   Color:         ColorSub(a, b)
//
// Search budget: ~60 ops × 60 ops = 3,600 depth-2 candidates.
// Validation: exact match on all training examples.

#[derive(Clone, Debug)]
enum ComposeOp {
    Tile(usize, usize),
    Scale(usize, usize),
    ColorSub(u8, u8),
    HFlip,
    VFlip,
    Rot90CW,
    Rot180,
    Transpose,
    FractalTile,
    DiagTile,
    MirrorTile(u8),
}

fn apply_compose_op(grid: &Grid, op: &ComposeOp) -> Option<Grid> {
    match op {
        ComposeOp::Tile(nh, nw) => Some(tile_grid(grid, *nh, *nw)),
        ComposeOp::Scale(sr, sc) => Some(scale_grid(grid, *sr, *sc)),
        ComposeOp::ColorSub(a, b) => {
            let mut g = grid.clone();
            for row in g.cells.iter_mut() {
                for cell in row.iter_mut() {
                    if *cell == *a { *cell = *b; }
                }
            }
            Some(g)
        }
        ComposeOp::HFlip => Some(apply_geometric(grid, 3)),
        ComposeOp::VFlip => Some(apply_geometric(grid, 4)),
        ComposeOp::Rot90CW => Some(apply_geometric(grid, 0)),
        ComposeOp::Rot180 => Some(apply_geometric(grid, 2)),
        ComposeOp::Transpose => Some(apply_geometric(grid, 5)),
        ComposeOp::FractalTile => {
            let ih = grid.height;
            let iw = grid.width;
            if ih * ih > 60 || iw * iw > 60 { return None; }
            Some(apply_fractal_tile(grid))
        }
        ComposeOp::DiagTile => {
            let n = grid.height;
            if n != grid.width || n * n > 60 { return None; }
            let on = n * n;
            let mut cells = vec![vec![0u8; on]; on];
            for t in 0..n {
                for r in 0..n {
                    for c in 0..n {
                        cells[t * n + r][t * n + c] = grid.cells[r][c];
                    }
                }
            }
            Some(Grid { cells, height: on, width: on })
        }
        ComposeOp::MirrorTile(mode) => apply_mirror_tile(grid, *mode),
    }
}

fn compose_output_dims(ih: usize, iw: usize, op: &ComposeOp) -> Option<(usize, usize)> {
    match op {
        ComposeOp::Tile(nh, nw) => Some((ih * nh, iw * nw)),
        ComposeOp::Scale(sr, sc) => Some((ih * sr, iw * sc)),
        ComposeOp::ColorSub(_, _) | ComposeOp::Rot180 => Some((ih, iw)),
        ComposeOp::HFlip | ComposeOp::VFlip => Some((ih, iw)),
        ComposeOp::Rot90CW | ComposeOp::Transpose => Some((iw, ih)),
        ComposeOp::FractalTile => Some((ih * ih, iw * iw)),
        ComposeOp::DiagTile => {
            if ih != iw { return None; }
            Some((ih * ih, iw * iw))
        }
        ComposeOp::MirrorTile(mode) => match mode {
            0 | 3 => Some((ih * 2, iw * 2)),
            1 => Some((ih, iw * 2)),
            2 => Some((ih * 2, iw)),
            _ => None,
        },
    }
}

fn solve_compose_ops(task: &ArcTask) -> Option<Vec<ComposeOp>> {
    if task.train.is_empty() || task.test.is_empty() { return None; }

    let oh = task.train[0].output.height;
    let ow = task.train[0].output.width;
    let ih = task.train[0].input.height;
    let iw = task.train[0].input.width;

    let mut ops: Vec<ComposeOp> = Vec::new();

    // Geometric
    ops.push(ComposeOp::HFlip);
    ops.push(ComposeOp::VFlip);
    ops.push(ComposeOp::Rot90CW);
    ops.push(ComposeOp::Rot180);
    ops.push(ComposeOp::Transpose);

    // Color substitutions from palette
    let mut all_colors = [false; 10];
    let mut out_colors = [false; 10];
    for ex in &task.train {
        for row in &ex.input.cells { for &v in row { all_colors[v as usize] = true; } }
        for row in &ex.output.cells { for &v in row { out_colors[v as usize] = true; all_colors[v as usize] = true; } }
    }
    for a in 0u8..10 {
        for b in 0u8..10 {
            if a != b && all_colors[a as usize] && out_colors[b as usize] {
                ops.push(ComposeOp::ColorSub(a, b));
            }
        }
    }

    // Size-changing ops from dimension ratios
    let mut added_tile = std::collections::HashSet::new();
    for ex in &task.train {
        let eih = ex.input.height;
        let eiw = ex.input.width;
        let eoh = ex.output.height;
        let eow = ex.output.width;
        if eoh >= eih && eow >= eiw && eih > 0 && eiw > 0 && eoh % eih == 0 && eow % eiw == 0 {
            let nh = eoh / eih;
            let nw = eow / eiw;
            if (nh > 1 || nw > 1) && !added_tile.contains(&(nh, nw)) {
                added_tile.insert((nh, nw));
                ops.push(ComposeOp::Tile(nh, nw));
                ops.push(ComposeOp::Scale(nh, nw));
            }
        }
    }
    for f in 2usize..=5 {
        if !added_tile.contains(&(f, f)) {
            ops.push(ComposeOp::Tile(f, f));
            ops.push(ComposeOp::Scale(f, f));
        }
        if !added_tile.contains(&(f, 1)) { ops.push(ComposeOp::Tile(f, 1)); }
        if !added_tile.contains(&(1, f)) { ops.push(ComposeOp::Tile(1, f)); }
    }

    ops.push(ComposeOp::FractalTile);
    ops.push(ComposeOp::DiagTile);
    for m in 0u8..4 { ops.push(ComposeOp::MirrorTile(m)); }

    // Dedup
    ops.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
    ops.dedup_by(|a, b| format!("{:?}", a) == format!("{:?}", b));

    // Helper: validate sequence on ALL training examples
    let validate = |seq: &[&ComposeOp]| -> bool {
        task.train.iter().all(|ex| {
            let mut g = ex.input.clone();
            for op in seq {
                match apply_compose_op(&g, op) {
                    Some(next) => g = next,
                    None => return false,
                }
            }
            g.height == ex.output.height && g.width == ex.output.width
                && grid_exact_match(&g, &ex.output)
        })
    };

    // Depth-1
    for op in &ops {
        if let Some((odh, odw)) = compose_output_dims(ih, iw, op) {
            if odh != oh || odw != ow { continue; }
        }
        if validate(&[op]) {
            return Some(vec![op.clone()]);
        }
    }

    // Depth-2: dim-filter to prune impossible pairs
    for op1 in &ops {
        let mid = compose_output_dims(ih, iw, op1);
        if mid.is_none() { continue; }
        let (mh, mw) = mid.unwrap();
        if mh > 60 || mw > 60 { continue; }

        for op2 in &ops {
            if let Some((fh, fw)) = compose_output_dims(mh, mw, op2) {
                if fh != oh || fw != ow { continue; }
            } else { continue; }

            if validate(&[op1, op2]) {
                return Some(vec![op1.clone(), op2.clone()]);
            }
        }
    }

    None
}

// ─── Strategy N: Geometric transforms (rot90, rot180, hflip, vflip, transpose)
//
// Detects if the output is a simple geometric transform of the input.
// Tries 6 transforms and returns the one consistent across all training examples.

fn solve_geometric(task: &ArcTask) -> Option<u8> {
    if task.train.is_empty() { return None; }
    if !task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width) {
        return None;
    }

    // 0=rot90_cw, 1=rot90_ccw, 2=rot180, 3=hflip, 4=vflip, 5=transpose
    for tf in 0u8..6 {
        let all_match = task.train.iter().all(|ex| {
            let h = ex.input.height;
            let w = ex.input.width;
            if (tf == 0 || tf == 1 || tf == 5) && h != w { return false; }
            (0..h).all(|r| (0..w).all(|c| {
                let (tr, tc) = match tf {
                    0 => (c, h - 1 - r),       // rot90 cw
                    1 => (w - 1 - c, r),        // rot90 ccw
                    2 => (h - 1 - r, w - 1 - c), // rot180
                    3 => (r, w - 1 - c),        // hflip
                    4 => (h - 1 - r, c),        // vflip
                    5 => (c, r),                // transpose
                    _ => (r, c),
                };
                ex.output.cells[tr][tc] == ex.input.cells[r][c]
            }))
        });
        if all_match { return Some(tf); }
    }
    None
}

fn apply_geometric(grid: &Grid, tf: u8) -> Grid {
    let h = grid.height;
    let w = grid.width;
    let (oh, ow) = match tf {
        0 | 1 | 5 => (w, h), // rot90 variants and transpose swap dims
        _ => (h, w),
    };
    let mut cells = vec![vec![0u8; ow]; oh];
    for r in 0..h {
        for c in 0..w {
            let (tr, tc) = match tf {
                0 => (c, h - 1 - r),
                1 => (w - 1 - c, r),
                2 => (h - 1 - r, w - 1 - c),
                3 => (r, w - 1 - c),
                4 => (h - 1 - r, c),
                5 => (c, r),
                _ => (r, c),
            };
            cells[tr][tc] = grid.cells[r][c];
        }
    }
    Grid { cells, height: oh, width: ow }
}

// ─── Strategy L: Palette-constrained neighborhood ──────────────────────────
//
// Same as neighborhood centroid, but classification is restricted to colors
// that appear in the task's training outputs. Reduces discretization error
// when tasks use only 2-4 of the 10 possible colors.

fn solve_cell_centroid_palette(task: &ArcTask) -> Option<(Vec<Multivector>, Vec<u8>, f32)> {
    if !task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width) {
        return None;
    }

    let palette = task_color_palette(task);

    let mut sums: Vec<Multivector> = (0..NUM_COLORS).map(|_| Multivector::zero()).collect();
    let mut counts = [0u32; NUM_COLORS];

    for ex in &task.train {
        for r in 0..ex.input.height {
            for c in 0..ex.input.width {
                let feat = cell_feature(&ex.input, r, c);
                let out_c = ex.output.cells[r][c] as usize;
                sums[out_c] = sums[out_c].add(&feat);
                counts[out_c] += 1;
            }
        }
    }

    let centroids: Vec<Multivector> = sums.iter().zip(counts.iter())
        .map(|(sum, &count)| {
            if count > 0 { sum.scale(1.0 / count as f32) } else { Multivector::zero() }
        })
        .collect();

    let (mut correct, mut total) = (0usize, 0usize);
    for ex in &task.train {
        for r in 0..ex.input.height {
            for c in 0..ex.input.width {
                let feat = cell_feature(&ex.input, r, c);
                let pred = classify_palette(&centroids, &feat, &palette);
                if pred == ex.output.cells[r][c] { correct += 1; }
                total += 1;
            }
        }
    }

    Some((centroids, palette, if total > 0 { correct as f32 / total as f32 } else { 0.0 }))
}

fn classify_palette(centroids: &[Multivector], feat: &Multivector, palette: &[u8]) -> u8 {
    let mut best = palette[0];
    let mut best_score = f32::NEG_INFINITY;
    for &c in palette {
        let score = component_dot(feat, &centroids[c as usize]);
        if score > best_score {
            best_score = score;
            best = c;
        }
    }
    best
}

fn apply_cell_centroid_palette(centroids: &[Multivector], grid: &Grid, palette: &[u8]) -> Grid {
    let mut cells = vec![vec![0u8; grid.width]; grid.height];
    for r in 0..grid.height {
        for c in 0..grid.width {
            let feat = cell_feature(grid, r, c);
            cells[r][c] = classify_palette(centroids, &feat, palette);
        }
    }
    Grid { cells, height: grid.height, width: grid.width }
}

// ─── Strategy M: Grid-separator summarization ──────────────────────────────
//
// Many ARC tasks present an input grid divided by separator lines (rows/cols
// of a single color) into NxN sub-regions. The output is a small grid (e.g.
// 3x3) summarizing each sub-region. We detect the separator structure,
// extract sub-regions, and try multiple summarization rules (majority color,
// minority non-bg color, unique color, presence of target color, etc.).

/// Detect grid-separator structure: a single non-bg color used for full
/// row/column separator lines dividing the grid into sub-regions.
/// Returns (sep_color, separator_row_indices, separator_col_indices).
fn detect_grid_separators(grid: &Grid, bg: u8) -> Option<(u8, Vec<usize>, Vec<usize>)> {
    // Try each non-bg color as potential separator
    for sep_c in 0..NUM_COLORS as u8 {
        if sep_c == bg { continue; }
        let sep_rows: Vec<usize> = (0..grid.height)
            .filter(|&r| grid.cells[r].iter().all(|&c| c == sep_c))
            .collect();
        let sep_cols: Vec<usize> = (0..grid.width)
            .filter(|&col| (0..grid.height).all(|r| grid.cells[r][col] == sep_c))
            .collect();
        if sep_rows.is_empty() && sep_cols.is_empty() { continue; }
        // Must have at least one row OR col separator
        return Some((sep_c, sep_rows, sep_cols));
    }
    None
}

/// Given separator rows and cols, extract the rectangular sub-regions between them.
/// Returns sub-regions as a 2D vec indexed by (region_row, region_col).
fn extract_meta_regions(grid: &Grid, sep_rows: &[usize], sep_cols: &[usize])
    -> Option<Vec<Vec<Vec<Vec<u8>>>>>
{
    // Build row bands: ranges between separators
    let mut row_bands: Vec<(usize, usize)> = Vec::new();
    let mut prev = 0usize;
    for &sr in sep_rows {
        if sr > prev { row_bands.push((prev, sr)); }
        prev = sr + 1;
    }
    if prev < grid.height { row_bands.push((prev, grid.height)); }

    let mut col_bands: Vec<(usize, usize)> = Vec::new();
    let mut prev = 0usize;
    for &sc in sep_cols {
        if sc > prev { col_bands.push((prev, sc)); }
        prev = sc + 1;
    }
    if prev < grid.width { col_bands.push((prev, grid.width)); }

    if row_bands.is_empty() || col_bands.is_empty() { return None; }

    let mut regions = Vec::new();
    for &(r0, r1) in &row_bands {
        let mut row_regions = Vec::new();
        for &(c0, c1) in &col_bands {
            let sub: Vec<Vec<u8>> = (r0..r1)
                .map(|r| grid.cells[r][c0..c1].to_vec())
                .collect();
            row_regions.push(sub);
        }
        regions.push(row_regions);
    }
    Some(regions)
}

/// Compute a feature vector for a sub-region: [count_per_color_0..9, total_non_bg, n_distinct].
/// bg and sep are both excluded from "interesting" counts.
fn region_features(region: &[Vec<u8>], bg: u8, sep: u8) -> [u32; 12] {
    let mut counts = [0u32; NUM_COLORS];
    for row in region {
        for &c in row { counts[c as usize] += 1; }
    }
    counts[bg as usize] = 0;
    if sep != bg { counts[sep as usize] = 0; }
    let total: u32 = counts.iter().sum();
    let distinct = counts.iter().filter(|&&c| c > 0).count() as u32;
    let mut feat = [0u32; 12];
    feat[..10].copy_from_slice(&counts);
    feat[10] = total;
    feat[11] = distinct;
    feat
}

/// Extract a single scalar feature from the feature vector, used as a lookup key.
fn region_scalar(feat: &[u32; 12], rule: usize) -> u32 {
    match rule {
        0 => feat[10],                    // total non-bg count
        1 => feat[11],                    // distinct non-bg colors
        2 => {                            // majority non-bg color index
            let mut best = 0u8;
            let mut best_v = 0u32;
            for i in 0..10 { if feat[i] > best_v { best_v = feat[i]; best = i as u8; } }
            best as u32
        }
        3 => {                            // minority non-bg color index
            let mut best = 0u8;
            let mut best_v = u32::MAX;
            for i in 0..10 { if feat[i] > 0 && feat[i] < best_v { best_v = feat[i]; best = i as u8; } }
            if best_v == u32::MAX { 0 } else { best as u32 }
        }
        4 => if feat[10] > 1 { 1 } else { 0 },  // has_multiple flag
        5 => {                            // color with count=1 (unique cell)
            for i in 0..10 { if feat[i] == 1 { return i as u32; } }
            0
        }
        _ => 0,
    }
}

const NUM_REGION_RULES: usize = 6;

/// Returns (best_rule, learned_mapping) if separator structure solves training.
/// learned_mapping[scalar_value] = output_color.
fn solve_grid_separator(task: &ArcTask) -> Option<(usize, [u8; 32])> {
    let oh = task.train[0].output.height;
    let ow = task.train[0].output.width;
    if !task.train.iter().all(|ex| ex.output.height == oh && ex.output.width == ow) {
        return None;
    }

    for rule in 0..NUM_REGION_RULES {
        // mapping[scalar_value] = output_color (0xFF = unset)
        let mut mapping = [0xFFu8; 32];
        let mut ok = true;

        for ex in &task.train {
            let bg = most_common_color(&ex.input);
            let (sep_c, sep_rows, sep_cols) = match detect_grid_separators(&ex.input, bg) {
                Some(v) => v,
                None => { ok = false; break; }
            };
            let regions = match extract_meta_regions(&ex.input, &sep_rows, &sep_cols) {
                Some(r) => r,
                None => { ok = false; break; }
            };
            let nr = regions.len();
            let nc = if nr > 0 { regions[0].len() } else { 0 };
            if nr != oh || nc != ow { ok = false; break; }

            for ri in 0..nr {
                for ci in 0..nc {
                    let feat = region_features(&regions[ri][ci], bg, sep_c);
                    let key = region_scalar(&feat, rule) as usize;
                    if key >= 32 { ok = false; break; }
                    let expected = ex.output.cells[ri][ci];
                    if mapping[key] == 0xFF {
                        mapping[key] = expected;
                    } else if mapping[key] != expected {
                        ok = false; break;
                    }
                }
                if !ok { break; }
            }
            if !ok { break; }
        }

        if ok {
            return Some((rule, mapping));
        }
    }
    None
}

fn apply_grid_separator(input: &Grid, rule: usize, mapping: &[u8; 32], oh: usize, ow: usize) -> Grid {
    let bg = most_common_color(input);
    let (sep_c, sep_rows, sep_cols) = detect_grid_separators(input, bg)
        .unwrap_or((bg, vec![], vec![]));

    if let Some(regions) = extract_meta_regions(input, &sep_rows, &sep_cols) {
        let nr = regions.len();
        let nc = if nr > 0 { regions[0].len() } else { 0 };
        if nr == oh && nc == ow {
            let mut cells = vec![vec![0u8; ow]; oh];
            for ri in 0..nr {
                for ci in 0..nc {
                    let feat = region_features(&regions[ri][ci], bg, sep_c);
                    let key = region_scalar(&feat, rule) as usize;
                    cells[ri][ci] = if key < 32 && mapping[key] != 0xFF {
                        mapping[key]
                    } else { 0 };
                }
            }
            return Grid { cells, height: oh, width: ow };
        }
    }
    Grid { cells: vec![vec![0u8; ow]; oh], height: oh, width: ow }
}

// ─── Strategy G: Grid-level rotor (fallback) ───────────────────────────────

fn position_vector(r: usize, c: usize, h: usize, w: usize) -> Multivector {
    let pi = std::f32::consts::PI;
    let u = if h > 1 { r as f32 / (h - 1) as f32 } else { 0.5 };
    let v = if w > 1 { c as f32 / (w - 1) as f32 } else { 0.5 };
    let pv = [
        (pi * u).sin(), (pi * u).cos(),
        (pi * v).sin(), (pi * v).cos(),
        (2.0 * pi * u).sin(), (2.0 * pi * v).sin(),
        (pi * (u + v)).sin(), (pi * (u - v + 1.0)).sin(),
    ];
    let norm: f32 = pv.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let mut normed = [0.0f32; 8];
    for i in 0..8 { normed[i] = pv[i] / norm; }
    Multivector::vector(&normed)
}

pub fn encode_grid(grid: &Grid) -> Multivector {
    let mut mv = Multivector::zero();
    for r in 0..grid.height {
        for c in 0..grid.width {
            let color = grid.cells[r][c];
            if color == 0 { continue; }
            let col_mv = color_vector(color);
            let pos_mv = position_vector(r, c, grid.height, grid.width);
            mv = mv.add(&pos_mv.geo(&col_mv));
        }
    }
    let n = mv.component_norm();
    if n > 1e-8 { mv = mv.scale(1.0 / n); }
    mv
}

pub fn decode_grid(pred_mv: &Multivector, h: usize, w: usize) -> Grid {
    let mut cells = vec![vec![0u8; w]; h];
    let color_bases: Vec<Multivector> = (1..NUM_COLORS as u8).map(|c| color_vector(c)).collect();
    for r in 0..h {
        for c in 0..w {
            let pos = position_vector(r, c, h, w);
            let mut best_color = 0u8;
            let mut best_score = f32::NEG_INFINITY;
            for (idx, col_mv) in color_bases.iter().enumerate() {
                let cell_mv = pos.geo(col_mv);
                let score = component_dot(pred_mv, &cell_mv);
                if score > best_score {
                    best_score = score;
                    best_color = (idx + 1) as u8;
                }
            }
            if best_score > 0.0 { cells[r][c] = best_color; }
        }
    }
    Grid { cells, height: h, width: w }
}

// ─── Clifford normal equations ─────────────────────────────────────────────

pub fn solve_normal_equations(
    inputs: &[Multivector],
    outputs: &[Multivector],
) -> Multivector {
    assert_eq!(inputs.len(), outputs.len());

    let mut a = Multivector::zero();
    for (inp, out) in inputs.iter().zip(outputs.iter()) {
        let inp_rev = inp.reverse();
        a = a.add(&out.geo(&inp_rev));
    }

    let mut b = Multivector::zero();
    for inp in inputs {
        let inp_rev = inp.reverse();
        b = b.add(&inp.geo(&inp_rev));
    }

    let b_rev = b.reverse();
    let bb_rev = b.geo(&b_rev);
    let denom = bb_rev.components[0];

    if denom.abs() < 1e-12 { return a; }
    let b_inv = b_rev.scale(1.0 / denom);
    a.geo(&b_inv)
}

// ─── Rule extraction & |B| diagnostic ──────────────────────────────────────

pub fn extract_rule(input_mv: &Multivector, output_mv: &Multivector) -> Multivector {
    let input_rev = input_mv.reverse();
    output_mv.geo(&input_rev)
}

pub fn rotor_consistency(rules: &[Multivector]) -> (f32, Vec<f32>) {
    let n = rules.len();
    if n < 2 { return (0.0, vec![]); }

    let mut bv_norms = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let rj_rev = rules[j].reverse();
            let product = rules[i].geo(&rj_rev);
            let g2 = product.grade(2);
            let bv_norm: f32 = g2.iter().map(|x| x * x).sum::<f32>().sqrt();
            bv_norms.push(bv_norm);
        }
    }
    let mean = bv_norms.iter().sum::<f32>() / bv_norms.len() as f32;
    (mean, bv_norms)
}

// ─── Evaluation ────────────────────────────────────────────────────────────

pub fn grid_matches(predicted: &Grid, expected: &Grid) -> (usize, usize) {
    if predicted.height != expected.height || predicted.width != expected.width {
        return (0, expected.height * expected.width);
    }
    let mut correct = 0;
    let total = expected.height * expected.width;
    for r in 0..expected.height {
        for c in 0..expected.width {
            if predicted.cells[r][c] == expected.cells[r][c] { correct += 1; }
        }
    }
    (correct, total)
}

pub fn grid_exact_match(predicted: &Grid, expected: &Grid) -> bool {
    let (correct, total) = grid_matches(predicted, expected);
    correct == total && total > 0
}

// ─── Multi-strategy solver ─────────────────────────────────────────────────
//
// Try applicable same-dimension strategies on training accuracy, apply the
// winner to test. Adjacency uses search (exhaustive or sampled); others are
// single-pass.

fn is_same_dims(task: &ArcTask) -> bool {
    task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width)
    && task.test.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width)
}

pub fn solve_task(task: &ArcTask) -> TaskDiagnostic {
    let same_dims = is_same_dims(task);

    // |B| diagnostic (grid-level, for reporting)
    let train_inputs: Vec<_> = task.train.iter().map(|ex| encode_grid(&ex.input)).collect();
    let train_outputs: Vec<_> = task.train.iter().map(|ex| encode_grid(&ex.output)).collect();
    let rules: Vec<_> = train_inputs.iter().zip(train_outputs.iter())
        .map(|(i, o)| extract_rule(i, o)).collect();
    let (mean_bv, _) = rotor_consistency(&rules);

    let mut total_correct = 0usize;
    let mut total_cells = 0usize;
    let mut all_exact = true;
    let best_strategy: &'static str;

    if same_dims {
        let (h, w) = task.train.first().map(|e| (e.input.height, e.input.width)).unwrap_or((0, 0));

        // Priority 1: Exact structural rules — each must achieve 100% on training.
        let ring_rev_acc = solve_ring_reversal(task);
        let ring_cyc_acc = solve_ring_color_cycle(task);
        let diag_x_acc = solve_diagonal_x(task);
        let spiral_result = detect_spiral_fill(task);
        let depth_map_result = learn_depth_color_map(task);

        if ring_rev_acc == Some(1.0) {
            best_strategy = "ring_reversal";
            for test_ex in &task.test {
                let pred = apply_ring_reversal_to_test(&test_ex.input);
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else if ring_cyc_acc == Some(1.0) {
            best_strategy = "ring_cycle";
            for test_ex in &task.test {
                let pred = apply_ring_color_cycle(&test_ex.input)
                    .unwrap_or_else(|| test_ex.input.clone());
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else if diag_x_acc == Some(1.0) {
            best_strategy = "diagonal_x";
            for test_ex in &task.test {
                let pred = apply_diagonal_x(&test_ex.input)
                    .unwrap_or_else(|| test_ex.input.clone());
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else if let Some((ca, cb)) = spiral_result {
            best_strategy = "spiral_fill";
            for test_ex in &task.test {
                let pred = spiral_fill(test_ex.output.height, test_ex.output.width, ca, cb);
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else if let Some((default_c, ref dmap)) = depth_map_result {
            best_strategy = "depth_fill";
            for test_ex in &task.test {
                let pred = apply_depth_color_map(&test_ex.input, dmap, default_c);
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else if solve_l_diagonal(task) == Some(1.0) {
            best_strategy = "l_diagonal";
            for test_ex in &task.test {
                let pred = apply_l_diagonal(&test_ex.input)
                    .unwrap_or_else(|| test_ex.input.clone());
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else if let Some(dir) = solve_gravity(task) {
            best_strategy = "gravity";
            for test_ex in &task.test {
                let pred = apply_gravity(&test_ex.input, dir);
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else if let Some(axis) = solve_symmetry(task) {
            best_strategy = "symmetry";
            for test_ex in &task.test {
                let pred = apply_symmetry(&test_ex.input, axis);
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else if let Some(tf) = solve_geometric(task) {
            best_strategy = "geometric";
            for test_ex in &task.test {
                let pred = apply_geometric(&test_ex.input, tf);
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else if let Some(lut) = solve_nbr_lookup(task) {
            best_strategy = "nbr_lookup";
            for test_ex in &task.test {
                let pred = apply_nbr_lookup(&test_ex.input, &lut);
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else if solve_connect_lines(task) {
            best_strategy = "connect_lines";
            for test_ex in &task.test {
                let pred = apply_connect_lines(&test_ex.input);
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else if solve_object_positional(task) == Some(1.0) {
            best_strategy = "obj_positional";
            let bg = most_common_color(&task.train[0].input);
            let in_objs = find_objects(&task.train[0].input, bg);
            let out_objs = find_objects(&task.train[0].output, bg);
            let matched = match_objects_by_color(&in_objs, &out_objs);
            let mut deltas: Vec<(u8, isize, isize)> = matched.iter().map(|(inp, out)| {
                let (ir, ic) = inp.centroid();
                let (or, oc) = out.centroid();
                (inp.color, (or - ir).round() as isize, (oc - ic).round() as isize)
            }).collect();
            deltas.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

            for test_ex in &task.test {
                let pred = apply_object_positional(&test_ex.input, bg, &deltas);
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else if solve_object_color_change(task) == Some(1.0) {
            best_strategy = "obj_recolor";
            let bg = most_common_color(&task.train[0].input);
            let mut cmap = [0u8; NUM_COLORS];
            for c in 0..NUM_COLORS { cmap[c] = c as u8; }
            cmap[bg as usize] = bg;

            for ex in &task.train {
                let in_objs = find_objects(&ex.input, bg);
                let out_objs = find_objects(&ex.output, bg);
                for in_obj in &in_objs {
                    let (ir, ic) = in_obj.centroid();
                    let mut best_out_c = in_obj.color;
                    let mut best_dist = f32::MAX;
                    for out_obj in &out_objs {
                        let (or, oc) = out_obj.centroid();
                        let dist = (ir - or).powi(2) + (ic - oc).powi(2);
                        if dist < best_dist { best_dist = dist; best_out_c = out_obj.color; }
                    }
                    cmap[in_obj.color as usize] = best_out_c;
                }
            }

            for test_ex in &task.test {
                let pred = apply_object_color_map(&test_ex.input, &cmap);
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            }
        } else {
            // Priority 2: Compete color vs neighborhood vs palette-neighborhood vs adjacency
            let (color_map, color_acc) = solve_color_map(task);

            let (centroids, centroid_acc) = match solve_cell_centroid(task) {
                Some((c, a)) => (Some(c), a),
                None => (None, 0.0),
            };

            let (pal_centroids, pal_palette, pal_acc) = match solve_cell_centroid_palette(task) {
                Some((c, p, a)) => (Some(c), Some(p), a),
                None => (None, None, 0.0),
            };

            let (adj_centroids, adj_acc) = match build_adj_centroids(task, h, w) {
                Some(c) => {
                    let a = solve_adjacency_train_acc(task, h, w, &c);
                    (Some(c), Some(a))
                }
                None => (None, None),
            };

            // Highest training accuracy; ties: color > pal_neighborhood > neighborhood > adjacency
            let mut best_acc = color_acc;
            let mut pick: u8 = 0;
            if pal_centroids.is_some() && pal_acc > best_acc {
                best_acc = pal_acc;
                pick = 3;
            }
            if centroids.is_some() && centroid_acc > best_acc {
                best_acc = centroid_acc;
                pick = 1;
            }
            if let Some(a) = adj_acc {
                if a > best_acc {
                    pick = 2;
                }
            }

            match pick {
                0 => {
                    best_strategy = "color";
                    for test_ex in &task.test {
                        let pred = apply_color_map(&test_ex.input, &color_map);
                        let (correct, total) = grid_matches(&pred, &test_ex.output);
                        total_correct += correct;
                        total_cells += total;
                        if correct != total { all_exact = false; }
                    }
                }
                1 => {
                    best_strategy = "neighborhood";
                    let cents = centroids.unwrap();
                    for test_ex in &task.test {
                        let pred = apply_cell_centroid(&cents, &test_ex.input);
                        let (correct, total) = grid_matches(&pred, &test_ex.output);
                        total_correct += correct;
                        total_cells += total;
                        if correct != total { all_exact = false; }
                    }
                }
                2 => {
                    best_strategy = "adjacency";
                    let cents = adj_centroids.unwrap();
                    let edges = grid_edges(h, w);
                    let palette = task_color_palette(task);
                    let mut rng = StdRng::seed_from_u64(task_rng_seed(task));
                    for test_ex in &task.test {
                        let pred = adjacency_search_best(
                            &test_ex.input,
                            &cents,
                            &edges,
                            &palette,
                            ADJ_SEARCH_SAMPLES_TEST,
                            &mut rng,
                        );
                        let (correct, total) = grid_matches(&pred, &test_ex.output);
                        total_correct += correct;
                        total_cells += total;
                        if correct != total { all_exact = false; }
                    }
                }
                _ => {
                    best_strategy = "pal_neighborhood";
                    let cents = pal_centroids.unwrap();
                    let pal = pal_palette.unwrap();
                    for test_ex in &task.test {
                        let pred = apply_cell_centroid_palette(&cents, &test_ex.input, &pal);
                        let (correct, total) = grid_matches(&pred, &test_ex.output);
                        total_correct += correct;
                        total_cells += total;
                        if correct != total { all_exact = false; }
                    }
                }
            }
        }
    } else if solve_enclosed_region(task) {
        best_strategy = "enclosed_region";
        for test_ex in &task.test {
            if let Some(pred) = apply_enclosed_region(&test_ex.input) {
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            } else {
                let total = test_ex.output.height * test_ex.output.width;
                total_cells += total;
                all_exact = false;
            }
        }
    } else if let Some((rule, mapping)) = solve_grid_separator(task) {
        best_strategy = "grid_separator";
        let oh = task.train[0].output.height;
        let ow = task.train[0].output.width;
        for test_ex in &task.test {
            let pred = apply_grid_separator(&test_ex.input, rule, &mapping, oh, ow);
            let (correct, total) = grid_matches(&pred, &test_ex.output);
            total_correct += correct;
            total_cells += total;
            if correct != total { all_exact = false; }
        }
    } else if let Some((r0, c0, oh, ow)) = solve_subgrid_fixed_offset(task) {
        best_strategy = "subgrid_fixed";
        for test_ex in &task.test {
            if r0 + oh <= test_ex.input.height && c0 + ow <= test_ex.input.width {
                let pred = extract_subgrid(&test_ex.input, r0, c0, oh, ow);
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            } else {
                total_cells += test_ex.output.height * test_ex.output.width;
                all_exact = false;
            }
        }
    } else if solve_subgrid_unique(task).is_some() {
        best_strategy = "subgrid_unique";
        for test_ex in &task.test {
            let matches = find_matching_subgrids(&test_ex.input, &test_ex.output);
            if matches.len() == 1 {
                let (r0, c0) = matches[0];
                let pred = extract_subgrid(
                    &test_ex.input, r0, c0,
                    test_ex.output.height, test_ex.output.width,
                );
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            } else {
                total_cells += test_ex.output.height * test_ex.output.width;
                all_exact = false;
            }
        }
    } else if let Some(obj_color) = solve_subgrid_object_bbox(task) {
        best_strategy = "subgrid_bbox";
        for test_ex in &task.test {
            let bg = most_common_color(&test_ex.input);
            let objects = find_objects(&test_ex.input, bg);
            let target = objects.iter().find(|o| o.color == obj_color);
            if let Some(obj) = target {
                let pred = extract_subgrid(
                    &test_ex.input, obj.min_r, obj.min_c, obj.bbox_h(), obj.bbox_w(),
                );
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            } else {
                total_cells += test_ex.output.height * test_ex.output.width;
                all_exact = false;
            }
        }
    } else if solve_subgrid_content_bbox(task).is_some() {
        best_strategy = "subgrid_content";
        for test_ex in &task.test {
            let bg = most_common_color(&test_ex.input);
            if let Some((r0, c0, bh, bw)) = content_bbox(&test_ex.input, bg) {
                let pred = extract_subgrid(&test_ex.input, r0, c0, bh, bw);
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            } else {
                total_cells += test_ex.output.height * test_ex.output.width;
                all_exact = false;
            }
        }
    } else if let Some(minority_color) = solve_subgrid_minority_bbox(task) {
        best_strategy = "subgrid_minority";
        for test_ex in &task.test {
            let bg = most_common_color(&test_ex.input);
            let objects = find_objects(&test_ex.input, bg);
            let target_objs: Vec<&GridObject> = objects.iter()
                .filter(|o| o.color == minority_color)
                .collect();
            if !target_objs.is_empty() {
                let min_r = target_objs.iter().map(|o| o.min_r).min().unwrap();
                let max_r = target_objs.iter().map(|o| o.max_r).max().unwrap();
                let min_c = target_objs.iter().map(|o| o.min_c).min().unwrap();
                let max_c = target_objs.iter().map(|o| o.max_c).max().unwrap();
                let pred = extract_subgrid(
                    &test_ex.input, min_r, min_c,
                    max_r - min_r + 1, max_c - min_c + 1,
                );
                let (correct, total) = grid_matches(&pred, &test_ex.output);
                total_correct += correct;
                total_cells += total;
                if correct != total { all_exact = false; }
            } else {
                total_cells += test_ex.output.height * test_ex.output.width;
                all_exact = false;
            }
        }
    } else if let Some(mode) = solve_mirror_tile(task) {
        best_strategy = "mirror_tile";
        for test_ex in &task.test {
            let pred = apply_mirror_tile(&test_ex.input, mode)
                .unwrap_or_else(|| test_ex.input.clone());
            let (correct, total) = grid_matches(&pred, &test_ex.output);
            total_correct += correct;
            total_cells += total;
            if correct != total { all_exact = false; }
        }
    } else if let Some((r0, c0, bg)) = solve_canvas(task) {
        best_strategy = "canvas";
        for test_ex in &task.test {
            let pred = apply_canvas(
                &test_ex.input, r0, c0,
                test_ex.output.height, test_ex.output.width, bg,
            );
            let (correct, total) = grid_matches(&pred, &test_ex.output);
            total_correct += correct;
            total_cells += total;
            if correct != total { all_exact = false; }
        }
    } else if let Some((sr, sc, method)) = solve_downscale(task) {
        best_strategy = "downscale";
        for test_ex in &task.test {
            let pred = downscale_grid(&test_ex.input, sr, sc, method);
            let (correct, total) = grid_matches(&pred, &test_ex.output);
            total_correct += correct;
            total_cells += total;
            if correct != total { all_exact = false; }
        }
    } else if let Some((sr, sc)) = solve_scale(task) {
        best_strategy = "scale";
        for test_ex in &task.test {
            // (0,0) = variable scale: derive from test output dims
            let (actual_sr, actual_sc) = if sr == 0 && sc == 0 {
                let ih = test_ex.input.height;
                let iw = test_ex.input.width;
                let oh = test_ex.output.height;
                let ow = test_ex.output.width;
                if ih > 0 && iw > 0 && oh % ih == 0 && ow % iw == 0 {
                    (oh / ih, ow / iw)
                } else {
                    (2, 2) // fallback
                }
            } else {
                (sr, sc)
            };
            let pred = scale_grid(&test_ex.input, actual_sr, actual_sc);
            let (correct, total) = grid_matches(&pred, &test_ex.output);
            total_correct += correct;
            total_cells += total;
            if correct != total { all_exact = false; }
        }
    } else if let Some((tr, tc)) = solve_tiling(task) {
        best_strategy = "tiling";
        for test_ex in &task.test {
            let pred = tile_grid(&test_ex.input, tr, tc);
            let (correct, total) = grid_matches(&pred, &test_ex.output);
            total_correct += correct;
            total_cells += total;
            if correct != total { all_exact = false; }
        }
    } else if solve_fractal_tile(task) {
        best_strategy = "fractal_tile";
        for test_ex in &task.test {
            let pred = apply_fractal_tile(&test_ex.input);
            let (correct, total) = grid_matches(&pred, &test_ex.output);
            total_correct += correct;
            total_cells += total;
            if correct != total { all_exact = false; }
        }
    } else if let Some(compose_ops) = solve_compose_ops(task) {
        best_strategy = "compose";
        for test_ex in &task.test {
            let mut g = test_ex.input.clone();
            for op in &compose_ops {
                if let Some(next) = apply_compose_op(&g, op) { g = next; } else { break; }
            }
            let (correct, total) = grid_matches(&g, &test_ex.output);
            total_correct += correct;
            total_cells += total;
            if correct != total { all_exact = false; }
        }
    } else {
        // Diff-dim: grid-level rotor (fallback) — log size relationships
        best_strategy = "grid";
        let ex0 = &task.train[0];
        let ih = ex0.input.height;
        let iw = ex0.input.width;
        let oh = ex0.output.height;
        let ow = ex0.output.width;
        let shrink = oh <= ih && ow <= iw;
        let rh = if ih > 0 { oh as f32 / ih as f32 } else { 0.0 };
        let rw = if iw > 0 { ow as f32 / iw as f32 } else { 0.0 };
        eprintln!("  GRID-FALLBACK {}: {}x{} -> {}x{}  ratio={:.2}h {:.2}w  {}",
            task.id, ih, iw, oh, ow, rh, rw,
            if shrink { "SHRINK" } else { "GROW" });
        let grid_rule = solve_normal_equations(&train_inputs, &train_outputs);

        for test_ex in &task.test {
            let test_mv = encode_grid(&test_ex.input);
            let pred_mv = grid_rule.geo(&test_mv);
            let pred = decode_grid(&pred_mv, test_ex.output.height, test_ex.output.width);
            let (correct, total) = grid_matches(&pred, &test_ex.output);
            total_correct += correct;
            total_cells += total;
            if correct != total { all_exact = false; }
        }
    }

    TaskDiagnostic {
        id: task.id.clone(),
        n_train: task.train.len(),
        n_test: task.test.len(),
        same_dims,
        rotor_consistency: mean_bv,
        mean_bv_norm: mean_bv,
        solved: all_exact,
        n_correct_cells: total_correct,
        n_total_cells: total_cells,
        strategy: best_strategy,
    }
}

/// Print a small grid to stdout with ANSI colors.
pub fn print_grid(grid: &Grid, indent: &str) {
    const COLORS: [&str; 10] = [
        "\x1b[40m",   // 0: black
        "\x1b[44m",   // 1: blue
        "\x1b[41m",   // 2: red
        "\x1b[42m",   // 3: green
        "\x1b[43m",   // 4: yellow
        "\x1b[100m",  // 5: gray
        "\x1b[45m",   // 6: magenta
        "\x1b[46m",   // 7: cyan
        "\x1b[104m",  // 8: light blue
        "\x1b[101m",  // 9: light red
    ];
    for row in &grid.cells {
        print!("{}", indent);
        for &cell in row {
            let c = (cell as usize).min(9);
            print!("{} {} \x1b[0m", COLORS[c], cell);
        }
        println!();
    }
}
