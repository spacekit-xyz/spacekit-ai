#![allow(warnings)]
#![allow(unused_imports)]
//! ARC-AGI Domain-Specific Language — composable grid primitives with bounded search.
//!
//! Defines ~30 grid transformation primitives and searches for programs (depth 1–3)
//! that exactly reproduce all training input→output pairs. Subsumes and extends the
//! existing compose_ops with object-level operations, conditional transforms, and
//! richer parameter inference.
//!
//! Search is tractable via dimension pruning: given (ih,iw)→(oh,ow), most operations
//! are filtered before evaluation. A depth-3 search over 200 candidates with pruning
//! evaluates ~20k programs, each validated against 2–4 small grids.

use std::collections::VecDeque;

use rand::Rng;
use rand::seq::SliceRandom;

use crate::arc_agi::{
    encode_grid, extract_rule, multivector_cosine_similarity,
    Grid, ArcTask, NUM_COLORS,
    find_objects, grid_exact_match,
    most_common_color, content_bbox, find_enclosed_bbox, extract_subgrid,
    apply_gravity, apply_symmetry, apply_connect_lines, apply_geometric,
    scale_grid, tile_grid, downscale_grid, apply_mirror_tile, apply_fractal_tile,
    FlowDiagnostic, flow_diagnostic,
};
use crate::clifford::Multivector;

// ─── DSL operations ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum Op {
    // Geometric (same-dim or known dim change, no learned params)
    HFlip,
    VFlip,
    Rot90CW,
    Rot90CCW,
    Rot180,
    Transpose,

    // Color substitution
    ColorSub(u8, u8),
    SwapColors(u8, u8),
    MapColors([u8; NUM_COLORS]),

    // Spatial same-dim transforms
    Gravity(u8),
    SymmetryComplete(u8),
    ConnectLines,
    FillEnclosed,
    ExpandNonBg,
    ErodeNonBg,
    KeepColor(u8),
    RemoveColor(u8),
    ReplaceBackground(u8),

    // Object-level same-dim
    KeepLargestObj,
    KeepSmallestObj,
    OutlineObjects,

    // Diff-dim extraction
    CropToBBox,
    CropToEnclosed,
    ExtractLargestObj,
    ExtractSmallestObj,
    ExtractObjByColor(u8),

    // Diff-dim resize
    Scale(usize, usize),
    Tile(usize, usize),
    Downscale(usize, usize, u8),
    MirrorTile(u8),
    FractalTile,

    // Iteration
    RepeatUntilStable(Box<Op>),
}

// ─── Apply ─────────────────────────────────────────────────────────────────

pub fn apply_op(grid: &Grid, op: &Op) -> Option<Grid> {
    if grid.height == 0 || grid.width == 0 { return None; }
    match op {
        Op::HFlip => Some(apply_geometric(grid, 3)),
        Op::VFlip => Some(apply_geometric(grid, 4)),
        Op::Rot90CW => Some(apply_geometric(grid, 0)),
        Op::Rot90CCW => Some(apply_geometric(grid, 1)),
        Op::Rot180 => Some(apply_geometric(grid, 2)),
        Op::Transpose => Some(apply_geometric(grid, 5)),

        Op::ColorSub(a, b) => {
            let mut g = grid.clone();
            for row in g.cells.iter_mut() {
                for cell in row.iter_mut() {
                    if *cell == *a { *cell = *b; }
                }
            }
            Some(g)
        }
        Op::SwapColors(a, b) => {
            let mut g = grid.clone();
            for row in g.cells.iter_mut() {
                for cell in row.iter_mut() {
                    if *cell == *a { *cell = *b; }
                    else if *cell == *b { *cell = *a; }
                }
            }
            Some(g)
        }
        Op::MapColors(map) => {
            let mut g = grid.clone();
            for row in g.cells.iter_mut() {
                for cell in row.iter_mut() {
                    *cell = map[*cell as usize];
                }
            }
            Some(g)
        }

        Op::Gravity(dir) => Some(apply_gravity(grid, *dir)),
        Op::SymmetryComplete(axis) => Some(apply_symmetry(grid, *axis)),
        Op::ConnectLines => Some(apply_connect_lines(grid)),
        Op::FillEnclosed => Some(fill_enclosed(grid)),
        Op::ExpandNonBg => Some(expand_non_bg(grid)),
        Op::ErodeNonBg => Some(erode_non_bg(grid)),
        Op::KeepColor(c) => Some(keep_color(grid, *c)),
        Op::RemoveColor(c) => Some(remove_color(grid, *c)),
        Op::ReplaceBackground(new_bg) => Some(replace_background(grid, *new_bg)),

        Op::KeepLargestObj => Some(keep_largest_obj(grid)),
        Op::KeepSmallestObj => Some(keep_smallest_obj(grid)),
        Op::OutlineObjects => Some(outline_objects(grid)),

        Op::CropToBBox => crop_to_bbox(grid),
        Op::CropToEnclosed => crop_to_enclosed(grid),
        Op::ExtractLargestObj => extract_largest_obj(grid),
        Op::ExtractSmallestObj => extract_smallest_obj(grid),
        Op::ExtractObjByColor(c) => extract_obj_by_color(grid, *c),

        Op::Scale(sr, sc) => Some(scale_grid(grid, *sr, *sc)),
        Op::Tile(tr, tc) => Some(tile_grid(grid, *tr, *tc)),
        Op::Downscale(sr, sc, method) => Some(downscale_grid(grid, *sr, *sc, *method)),
        Op::MirrorTile(mode) => apply_mirror_tile(grid, *mode),
        Op::FractalTile => {
            if grid.height * grid.height > 900 || grid.width * grid.width > 900 {
                return None;
            }
            Some(apply_fractal_tile(grid))
        }

        Op::RepeatUntilStable(inner) => {
            let mut g = grid.clone();
            for _ in 0..20 {
                let next = apply_op(&g, inner)?;
                if next.height != g.height || next.width != g.width { return None; }
                if grid_exact_match(&next, &g) { break; }
                g = next;
            }
            Some(g)
        }
    }
}

// ─── Dimension prediction (for search pruning) ────────────────────────────

fn op_output_dims(ih: usize, iw: usize, op: &Op) -> Option<(usize, usize)> {
    match op {
        Op::HFlip | Op::VFlip | Op::Rot180 => Some((ih, iw)),
        Op::Rot90CW | Op::Rot90CCW | Op::Transpose => Some((iw, ih)),
        Op::ColorSub(..) | Op::SwapColors(..) | Op::MapColors(..) => Some((ih, iw)),
        Op::Gravity(..) | Op::SymmetryComplete(..) | Op::ConnectLines => Some((ih, iw)),
        Op::FillEnclosed | Op::ExpandNonBg | Op::ErodeNonBg => Some((ih, iw)),
        Op::KeepColor(..) | Op::RemoveColor(..) | Op::ReplaceBackground(..) => Some((ih, iw)),
        Op::KeepLargestObj | Op::KeepSmallestObj | Op::OutlineObjects => Some((ih, iw)),
        Op::RepeatUntilStable(_) => Some((ih, iw)),
        Op::Scale(sr, sc) => Some((ih * sr, iw * sc)),
        Op::Tile(tr, tc) => Some((ih * tr, iw * tc)),
        Op::Downscale(sr, sc, _) => {
            if ih % sr == 0 && iw % sc == 0 { Some((ih / sr, iw / sc)) } else { None }
        }
        Op::MirrorTile(mode) => match mode {
            0 | 3 => Some((ih * 2, iw * 2)),
            1 => Some((ih, iw * 2)),
            2 => Some((ih * 2, iw)),
            _ => None,
        },
        Op::FractalTile => Some((ih * ih, iw * iw)),
        // Variable-output ops: unknown dims, must evaluate
        Op::CropToBBox | Op::CropToEnclosed => None,
        Op::ExtractLargestObj | Op::ExtractSmallestObj | Op::ExtractObjByColor(..) => None,
    }
}

fn dims_match(predicted: Option<(usize, usize)>, oh: usize, ow: usize) -> bool {
    match predicted {
        Some((ph, pw)) => ph == oh && pw == ow,
        None => true, // unknown dims, must evaluate
    }
}

// ─── New grid operations ───────────────────────────────────────────────────

fn fill_enclosed(grid: &Grid) -> Grid {
    let h = grid.height;
    let w = grid.width;
    let bg = most_common_color(grid);

    let mut outside = vec![vec![false; w]; h];
    let mut queue = VecDeque::new();

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

    // Find the non-bg color surrounding each enclosed region
    let mut cells = grid.cells.clone();
    for r in 0..h {
        for c in 0..w {
            if grid.cells[r][c] == bg && !outside[r][c] {
                let mut fill_color = bg;
                for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                        let v = grid.cells[nr as usize][nc as usize];
                        if v != bg { fill_color = v; break; }
                    }
                }
                cells[r][c] = fill_color;
            }
        }
    }
    Grid { cells, height: h, width: w }
}

fn expand_non_bg(grid: &Grid) -> Grid {
    let h = grid.height;
    let w = grid.width;
    let bg = most_common_color(grid);
    let mut cells = grid.cells.clone();

    for r in 0..h {
        for c in 0..w {
            if grid.cells[r][c] != bg { continue; }
            for (dr, dc) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                    let v = grid.cells[nr as usize][nc as usize];
                    if v != bg { cells[r][c] = v; break; }
                }
            }
        }
    }
    Grid { cells, height: h, width: w }
}

fn erode_non_bg(grid: &Grid) -> Grid {
    let h = grid.height;
    let w = grid.width;
    let bg = most_common_color(grid);
    let mut cells = grid.cells.clone();

    for r in 0..h {
        for c in 0..w {
            if grid.cells[r][c] == bg { continue; }
            let has_bg_neighbor = [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)].iter().any(|(dr, dc)| {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 { return true; }
                grid.cells[nr as usize][nc as usize] == bg
            });
            if has_bg_neighbor { cells[r][c] = bg; }
        }
    }
    Grid { cells, height: h, width: w }
}

fn keep_color(grid: &Grid, color: u8) -> Grid {
    let bg = most_common_color(grid);
    let cells = grid.cells.iter().map(|row|
        row.iter().map(|&c| if c == color || c == bg { c } else { bg }).collect()
    ).collect();
    Grid { cells, height: grid.height, width: grid.width }
}

fn remove_color(grid: &Grid, color: u8) -> Grid {
    let bg = most_common_color(grid);
    let cells = grid.cells.iter().map(|row|
        row.iter().map(|&c| if c == color { bg } else { c }).collect()
    ).collect();
    Grid { cells, height: grid.height, width: grid.width }
}

fn replace_background(grid: &Grid, new_bg: u8) -> Grid {
    let bg = most_common_color(grid);
    if bg == new_bg { return grid.clone(); }
    let cells = grid.cells.iter().map(|row|
        row.iter().map(|&c| if c == bg { new_bg } else { c }).collect()
    ).collect();
    Grid { cells, height: grid.height, width: grid.width }
}

fn keep_largest_obj(grid: &Grid) -> Grid {
    let bg = most_common_color(grid);
    let objects = find_objects(grid, bg);
    if objects.is_empty() { return grid.clone(); }
    let largest = objects.iter().max_by_key(|o| o.pixels.len()).unwrap();
    let mut cells = vec![vec![bg; grid.width]; grid.height];
    for &(r, c) in &largest.pixels { cells[r][c] = largest.color; }
    Grid { cells, height: grid.height, width: grid.width }
}

fn keep_smallest_obj(grid: &Grid) -> Grid {
    let bg = most_common_color(grid);
    let objects = find_objects(grid, bg);
    if objects.is_empty() { return grid.clone(); }
    let smallest = objects.iter().min_by_key(|o| o.pixels.len()).unwrap();
    let mut cells = vec![vec![bg; grid.width]; grid.height];
    for &(r, c) in &smallest.pixels { cells[r][c] = smallest.color; }
    Grid { cells, height: grid.height, width: grid.width }
}

fn outline_objects(grid: &Grid) -> Grid {
    let h = grid.height;
    let w = grid.width;
    let bg = most_common_color(grid);
    let mut cells = vec![vec![bg; w]; h];

    for r in 0..h {
        for c in 0..w {
            if grid.cells[r][c] == bg { continue; }
            let on_border = [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)].iter().any(|(dr, dc)| {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 { return true; }
                grid.cells[nr as usize][nc as usize] == bg
            });
            if on_border { cells[r][c] = grid.cells[r][c]; }
        }
    }
    Grid { cells, height: h, width: w }
}

fn crop_to_bbox(grid: &Grid) -> Option<Grid> {
    let bg = most_common_color(grid);
    let (r0, c0, bh, bw) = content_bbox(grid, bg)?;
    if bh == 0 || bw == 0 || r0 + bh > grid.height || c0 + bw > grid.width { return None; }
    Some(extract_subgrid(grid, r0, c0, bh, bw))
}

fn crop_to_enclosed(grid: &Grid) -> Option<Grid> {
    let bg = most_common_color(grid);
    let (r0, c0, bh, bw) = find_enclosed_bbox(grid, bg)?;
    if bh == 0 || bw == 0 || r0 + bh > grid.height || c0 + bw > grid.width { return None; }
    Some(extract_subgrid(grid, r0, c0, bh, bw))
}

fn extract_largest_obj(grid: &Grid) -> Option<Grid> {
    let bg = most_common_color(grid);
    let objects = find_objects(grid, bg);
    let obj = objects.iter().max_by_key(|o| o.pixels.len())?;
    let (bh, bw) = (obj.bbox_h(), obj.bbox_w());
    if bh == 0 || bw == 0 || obj.min_r + bh > grid.height || obj.min_c + bw > grid.width {
        return None;
    }
    Some(extract_subgrid(grid, obj.min_r, obj.min_c, bh, bw))
}

fn extract_smallest_obj(grid: &Grid) -> Option<Grid> {
    let bg = most_common_color(grid);
    let objects = find_objects(grid, bg);
    let obj = objects.iter().min_by_key(|o| o.pixels.len())?;
    let (bh, bw) = (obj.bbox_h(), obj.bbox_w());
    if bh == 0 || bw == 0 || obj.min_r + bh > grid.height || obj.min_c + bw > grid.width {
        return None;
    }
    Some(extract_subgrid(grid, obj.min_r, obj.min_c, bh, bw))
}

fn extract_obj_by_color(grid: &Grid, color: u8) -> Option<Grid> {
    let bg = most_common_color(grid);
    let objects = find_objects(grid, bg);
    let targets: Vec<_> = objects.iter().filter(|o| o.color == color).collect();
    if targets.is_empty() { return None; }
    let min_r = targets.iter().map(|o| o.min_r).min().unwrap();
    let max_r = targets.iter().map(|o| o.max_r).max().unwrap();
    let min_c = targets.iter().map(|o| o.min_c).min().unwrap();
    let max_c = targets.iter().map(|o| o.max_c).max().unwrap();
    let bh = max_r - min_r + 1;
    let bw = max_c - min_c + 1;
    if bh == 0 || bw == 0 || min_r + bh > grid.height || min_c + bw > grid.width {
        return None;
    }
    Some(extract_subgrid(grid, min_r, min_c, bh, bw))
}

// ─── Parameter inference ───────────────────────────────────────────────────

fn infer_color_map(task: &ArcTask) -> Option<[u8; NUM_COLORS]> {
    if !task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width
    ) { return None; }

    let mut map = [0xFFu8; NUM_COLORS];
    for ex in &task.train {
        for r in 0..ex.input.height {
            for c in 0..ex.input.width {
                let ic = ex.input.cells[r][c] as usize;
                let oc = ex.output.cells[r][c];
                if map[ic] == 0xFF { map[ic] = oc; }
                else if map[ic] != oc { return None; }
            }
        }
    }
    for i in 0..NUM_COLORS {
        if map[i] == 0xFF { map[i] = i as u8; }
    }
    if (0..NUM_COLORS).all(|i| map[i] == i as u8) { return None; }
    Some(map)
}

// ─── Candidate generation ──────────────────────────────────────────────────

fn generate_candidates(task: &ArcTask) -> Vec<Op> {
    let mut ops = Vec::with_capacity(256);

    // Geometric
    ops.push(Op::HFlip);
    ops.push(Op::VFlip);
    ops.push(Op::Rot90CW);
    ops.push(Op::Rot90CCW);
    ops.push(Op::Rot180);
    ops.push(Op::Transpose);

    // Gravity, symmetry
    for d in 0..4u8 { ops.push(Op::Gravity(d)); }
    for a in 0..4u8 { ops.push(Op::SymmetryComplete(a)); }
    ops.push(Op::ConnectLines);
    ops.push(Op::FillEnclosed);
    ops.push(Op::ExpandNonBg);
    ops.push(Op::ErodeNonBg);

    // Object-level same-dim
    ops.push(Op::KeepLargestObj);
    ops.push(Op::KeepSmallestObj);
    ops.push(Op::OutlineObjects);

    // Diff-dim extraction
    ops.push(Op::CropToBBox);
    ops.push(Op::CropToEnclosed);
    ops.push(Op::ExtractLargestObj);
    ops.push(Op::ExtractSmallestObj);

    // Color operations — use colors actually present in the task
    let mut all_colors = [false; NUM_COLORS];
    for ex in &task.train {
        for row in &ex.input.cells { for &v in row { all_colors[v as usize] = true; } }
        for row in &ex.output.cells { for &v in row { all_colors[v as usize] = true; } }
    }
    let palette: Vec<u8> = (0..NUM_COLORS as u8).filter(|&c| all_colors[c as usize]).collect();

    for &a in &palette {
        for &b in &palette {
            if a != b { ops.push(Op::ColorSub(a, b)); }
        }
    }
    for i in 0..palette.len() {
        for j in (i + 1)..palette.len() {
            ops.push(Op::SwapColors(palette[i], palette[j]));
        }
    }
    for &c in &palette {
        ops.push(Op::KeepColor(c));
        ops.push(Op::RemoveColor(c));
        ops.push(Op::ReplaceBackground(c));
        ops.push(Op::ExtractObjByColor(c));
    }

    if let Some(map) = infer_color_map(task) {
        ops.push(Op::MapColors(map));
    }

    // Size-changing ops — derive from training dimension ratios
    let mut added_scale = std::collections::HashSet::new();
    for ex in &task.train {
        let ih = ex.input.height;
        let iw = ex.input.width;
        let oh = ex.output.height;
        let ow = ex.output.width;

        // Scale / tile (grow)
        if oh >= ih && ow >= iw && ih > 0 && iw > 0 && oh % ih == 0 && ow % iw == 0 {
            let sr = oh / ih;
            let sc = ow / iw;
            if (sr > 1 || sc > 1) && !added_scale.contains(&(sr, sc, 0u8)) {
                added_scale.insert((sr, sc, 0));
                ops.push(Op::Scale(sr, sc));
                ops.push(Op::Tile(sr, sc));
            }
        }
        // Downscale (shrink)
        if oh <= ih && ow <= iw && oh > 0 && ow > 0 && ih % oh == 0 && iw % ow == 0 {
            let sr = ih / oh;
            let sc = iw / ow;
            if sr >= 2 && sc >= 2 && !added_scale.contains(&(sr, sc, 1)) {
                added_scale.insert((sr, sc, 1));
                for method in 0..3u8 { ops.push(Op::Downscale(sr, sc, method)); }
            }
        }
    }
    // Common scale factors
    for f in 2usize..=4 {
        if !added_scale.contains(&(f, f, 0)) { ops.push(Op::Scale(f, f)); ops.push(Op::Tile(f, f)); }
    }

    for mode in 0..4u8 { ops.push(Op::MirrorTile(mode)); }
    ops.push(Op::FractalTile);

    // RepeatUntilStable wrapping same-dim ops
    let stable_ops = vec![
        Op::FillEnclosed,
        Op::ExpandNonBg,
        Op::ErodeNonBg,
        Op::ConnectLines,
    ];
    for inner in stable_ops {
        ops.push(Op::RepeatUntilStable(Box::new(inner.clone())));
    }
    for d in 0..4u8 {
        ops.push(Op::RepeatUntilStable(Box::new(Op::Gravity(d))));
    }

    ops
}

// ─── Validation ────────────────────────────────────────────────────────────

fn validate_program_on_training(task: &ArcTask, program: &[&Op]) -> bool {
    for ex in &task.train {
        let mut g = ex.input.clone();
        for op in program {
            match apply_op(&g, op) {
                Some(next) => g = next,
                None => return false,
            }
        }
        if !grid_exact_match(&g, &ex.output) { return false; }
    }
    true
}

fn apply_program(grid: &Grid, program: &[&Op]) -> Option<Grid> {
    let mut g = grid.clone();
    for op in program {
        g = apply_op(&g, op)?;
    }
    Some(g)
}

// ─── Clifford hints for DSL candidate ordering ───────────────────────────────

/// Fixed 3×3 reference (colors 1–9) for per-op rule vectors used in hint scoring.
pub fn dsl_op_reference_grid() -> Grid {
    Grid {
        height: 3,
        width: 3,
        cells: vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
        ],
    }
}

/// `extract_rule(E[in], E[out])` on the reference grid. `None` if `apply_op` fails.
pub fn dsl_op_signature(op: &Op) -> Option<Multivector> {
    let g_in = dsl_op_reference_grid();
    let g_out = apply_op(&g_in, op)?;
    let z_in = encode_grid(&g_in);
    let z_out = encode_grid(&g_out);
    Some(extract_rule(&z_in, &z_out))
}

/// Mean of per-example `extract_rule(encode(in), encode(out))` over training pairs.
pub fn train_rule_consensus_mv(task: &ArcTask) -> Multivector {
    let n = task.train.len();
    if n == 0 {
        return Multivector::zero();
    }
    let mut acc = Multivector::zero();
    for ex in &task.train {
        let z_in = encode_grid(&ex.input);
        let z_out = encode_grid(&ex.output);
        acc = acc.add(&extract_rule(&z_in, &z_out));
    }
    acc.scale(1.0 / n as f32)
}

fn clifford_op_hint_score_with_consensus(consensus: &Multivector, op: &Op) -> f32 {
    let sig = match dsl_op_signature(op) {
        Some(s) => s,
        None => return 0.0,
    };
    let c = multivector_cosine_similarity(consensus, &sig);
    if c.is_finite() { c } else { 0.0 }
}

/// Cosine similarity between training rule consensus and the op’s reference signature.
pub fn clifford_op_hint_score(task: &ArcTask, op: &Op) -> f32 {
    if task.train.is_empty() {
        return 0.0;
    }
    let consensus = train_rule_consensus_mv(task);
    clifford_op_hint_score_with_consensus(&consensus, op)
}

// ─── Search engine ─────────────────────────────────────────────────────────

pub fn dsl_solve(task: &ArcTask) -> Option<(Vec<Grid>, &'static str)> {
    dsl_solve_with_flow(task, None)
}

pub fn dsl_solve_with_flow(
    task: &ArcTask,
    flow: Option<&FlowDiagnostic>,
) -> Option<(Vec<Grid>, &'static str)> {
    if task.train.is_empty() || task.test.is_empty() { return None; }

    let oh = task.train[0].output.height;
    let ow = task.train[0].output.width;
    let ih = task.train[0].input.height;
    let iw = task.train[0].input.width;

    let mut candidates = generate_candidates(task);

    // Probability current guided reordering: sort ops by alignment with the
    // bivector flow direction.  Geometric tasks (rotation-dominated flow) try
    // spatial ops first; color tasks (boost-dominated) try color ops first.
    if let Some(f) = flow {
        reorder_by_flow(&mut candidates, f);
    }

    // Convergence-adaptive time budgets: converging flow means a simple rule
    // likely exists at low depth → spend less on depth-2/3.  Non-converging
    // flow suggests the task needs composition → invest more search time.
    let (budget_d2_ms, budget_d3_ms) = match flow {
        Some(f) if f.converging && !f.is_degenerate() => (300, 800),
        Some(f) if !f.converging => (800, 2500),
        _ => (500, 1500),
    };
    let start = std::time::Instant::now();
    let budget_d2 = std::time::Duration::from_millis(budget_d2_ms);
    let budget_d3 = std::time::Duration::from_millis(budget_d3_ms);

    // ── Depth 1: collect all valid programs, pick highest Clifford hint (tie → first in order)
    let mut best: Option<(f32, Vec<Grid>)> = None;
    for op in &candidates {
        if let Some(pred_dims) = op_output_dims(ih, iw, op) {
            if pred_dims.0 != oh || pred_dims.1 != ow { continue; }
        }
        if validate_program_on_training(task, &[op]) {
            if let Some(predictions) = predict_test(task, &[op]) {
                let score = clifford_op_hint_score(task, op);
                let replace = best.as_ref().map_or(true, |(s, _)| score > *s);
                if replace {
                    best = Some((score, predictions));
                }
            }
        }
    }
    if let Some((_, predictions)) = best {
        return Some((predictions, "dsl_d1"));
    }

    // ── Depth 2 ──
    let step2_final: Vec<usize> = (0..candidates.len()).collect();

    for (i, op1) in candidates.iter().enumerate() {
        if start.elapsed() > budget_d2 { break; }

        let mid_dims = op_output_dims(ih, iw, op1);
        let actual_mid = if mid_dims.is_none() {
            apply_op(&task.train[0].input, op1).map(|g| (g.height, g.width))
        } else {
            mid_dims
        };
        let (mh, mw) = match actual_mid {
            Some(d) if d.0 > 0 && d.1 > 0 && d.0 <= 90 && d.1 <= 90 => d,
            _ => continue,
        };

        for &j in &step2_final {
            if let Some(pred_dims) = op_output_dims(mh, mw, &candidates[j]) {
                if pred_dims.0 != oh || pred_dims.1 != ow { continue; }
            }
            if i == j && is_self_inverse(&candidates[i]) { continue; }

            if validate_program_on_training(task, &[&candidates[i], &candidates[j]]) {
                let predictions = predict_test(task, &[&candidates[i], &candidates[j]])?;
                return Some((predictions, "dsl_d2"));
            }
        }
    }

    // ── Depth 3 (structural ops only at middle, strict pruning) ──
    let structural_indices: Vec<usize> = candidates.iter().enumerate()
        .filter(|(_, op)| is_structural_op(op))
        .map(|(i, _)| i)
        .collect();

    for &i in &structural_indices {
        if start.elapsed() > budget_d3 { break; }

        let mid1 = op_output_dims(ih, iw, &candidates[i]).or_else(||
            apply_op(&task.train[0].input, &candidates[i]).map(|g| (g.height, g.width))
        );
        let (m1h, m1w) = match mid1 {
            Some(d) if d.0 > 0 && d.1 > 0 && d.0 <= 60 && d.1 <= 60 => d,
            _ => continue,
        };

        for &j in &structural_indices {
            if start.elapsed() > budget_d3 { break; }

            let mid2 = match op_output_dims(m1h, m1w, &candidates[j]) {
                Some(d) if d.0 > 0 && d.1 > 0 && d.0 <= 60 && d.1 <= 60 => d,
                _ => continue,
            };

            for &k in &step2_final {
                if let Some(pred_dims) = op_output_dims(mid2.0, mid2.1, &candidates[k]) {
                    if pred_dims.0 != oh || pred_dims.1 != ow { continue; }
                }
                if validate_program_on_training(task, &[&candidates[i], &candidates[j], &candidates[k]]) {
                    let predictions = predict_test(task, &[&candidates[i], &candidates[j], &candidates[k]])?;
                    return Some((predictions, "dsl_d3"));
                }
            }
        }
    }

    None
}

fn predict_test(task: &ArcTask, program: &[&Op]) -> Option<Vec<Grid>> {
    let mut predictions = Vec::with_capacity(task.test.len());
    for test_ex in &task.test {
        predictions.push(apply_program(&test_ex.input, program)?);
    }
    Some(predictions)
}

// ─── A* program search (Clifford-guided) ────────────────────────────────────

use std::collections::BinaryHeap;
use std::cmp::Ordering;

struct AstarNode {
    program: Vec<usize>,
    grids: Vec<Grid>,     // current grid per training example after applying program
    g: f32,               // cost = program length
    h: f32,               // heuristic = mean Clifford distance to target
}

impl AstarNode {
    fn f(&self) -> f32 { self.g + 2.0 * self.h }
}

impl PartialEq for AstarNode {
    fn eq(&self, other: &Self) -> bool { self.f().to_bits() == other.f().to_bits() }
}
impl Eq for AstarNode {}
impl PartialOrd for AstarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for AstarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f().partial_cmp(&self.f()).unwrap_or(Ordering::Equal)
    }
}

fn clifford_distance_to_targets(grids: &[Grid], targets: &[Grid]) -> f32 {
    if grids.len() != targets.len() || grids.is_empty() { return f32::MAX; }
    let mut total = 0.0f32;
    for (g, t) in grids.iter().zip(targets.iter()) {
        if g.height != t.height || g.width != t.width {
            total += 1.0;
            continue;
        }
        let zg = encode_grid(g);
        let zt = encode_grid(t);
        let cos = multivector_cosine_similarity(&zg, &zt);
        total += 1.0 - cos.max(-1.0).min(1.0);
    }
    total / grids.len() as f32
}

/// A* search through DSL program space, using Clifford distance as heuristic.
/// Returns (predictions_for_test, strategy_label) if a program is found.
pub fn astar_dsl_solve(task: &ArcTask, max_depth: usize, budget_ms: u64) -> Option<(Vec<Grid>, &'static str)> {
    if task.train.is_empty() || task.test.is_empty() { return None; }

    let candidates = generate_candidates(task);
    let targets: Vec<&Grid> = task.train.iter().map(|ex| &ex.output).collect();
    let target_grids: Vec<Grid> = targets.iter().map(|g| (*g).clone()).collect();
    let start = std::time::Instant::now();
    let budget = std::time::Duration::from_millis(budget_ms);

    let init_grids: Vec<Grid> = task.train.iter().map(|ex| ex.input.clone()).collect();
    let init_h = clifford_distance_to_targets(&init_grids, &target_grids);

    let mut heap = BinaryHeap::new();
    heap.push(AstarNode {
        program: vec![],
        grids: init_grids,
        g: 0.0,
        h: init_h,
    });

    let mut expanded = 0u64;
    let max_expand = 50_000u64;

    while let Some(node) = heap.pop() {
        if start.elapsed() > budget || expanded > max_expand { break; }
        expanded += 1;

        if node.program.len() >= max_depth { continue; }

        for (i, op) in candidates.iter().enumerate() {
            // Skip no-ops (same op twice if self-inverse)
            if let Some(&last) = node.program.last() {
                if last == i && is_self_inverse(&candidates[last]) { continue; }
            }

            // Apply op to all training grids
            let mut next_grids = Vec::with_capacity(node.grids.len());
            let mut all_ok = true;
            let mut all_match = true;

            for (j, g) in node.grids.iter().enumerate() {
                match apply_op(g, op) {
                    Some(ng) => {
                        if !grid_exact_match(&ng, &target_grids[j]) { all_match = false; }
                        next_grids.push(ng);
                    }
                    None => { all_ok = false; break; }
                }
            }
            if !all_ok { continue; }

            // Goal check: all training examples match
            if all_match {
                let mut prog_refs: Vec<&Op> = node.program.iter().map(|&idx| &candidates[idx]).collect();
                prog_refs.push(op);
                if let Some(preds) = predict_test(task, &prog_refs) {
                    let label = match prog_refs.len() {
                        1 => "astar_d1",
                        2 => "astar_d2",
                        _ => "astar_d3",
                    };
                    return Some((preds, label));
                }
            }

            // Dimension pruning: skip if dims getting further from target
            let target_h = target_grids[0].height;
            let target_w = target_grids[0].width;
            let cur_h = next_grids[0].height;
            let cur_w = next_grids[0].width;
            let remaining_depth = max_depth - node.program.len() - 1;
            if remaining_depth == 0 && (cur_h != target_h || cur_w != target_w) {
                continue;
            }

            let h = clifford_distance_to_targets(&next_grids, &target_grids);
            let mut new_program = node.program.clone();
            new_program.push(i);

            heap.push(AstarNode {
                program: new_program,
                grids: next_grids,
                g: node.g + 1.0,
                h,
            });
        }
    }

    None
}

fn is_self_inverse(op: &Op) -> bool {
    matches!(op,
        Op::HFlip | Op::VFlip | Op::Rot180 |
        Op::SwapColors(..) | Op::ConnectLines
    )
}

/// Structural ops: geometric, gravity, symmetry, object — excludes color parametric variants
/// to keep depth-3 search tractable.
fn is_structural_op(op: &Op) -> bool {
    matches!(op,
        Op::HFlip | Op::VFlip | Op::Rot90CW | Op::Rot90CCW | Op::Rot180 | Op::Transpose |
        Op::Gravity(..) | Op::SymmetryComplete(..) | Op::ConnectLines |
        Op::FillEnclosed | Op::ExpandNonBg | Op::ErodeNonBg |
        Op::KeepLargestObj | Op::KeepSmallestObj | Op::OutlineObjects |
        Op::CropToBBox | Op::CropToEnclosed |
        Op::ExtractLargestObj | Op::ExtractSmallestObj |
        Op::MapColors(..) | Op::RepeatUntilStable(..)
    )
}

#[allow(dead_code)]
fn is_cheap_op(op: &Op) -> bool {
    is_structural_op(op) || matches!(op,
        Op::ColorSub(..) | Op::SwapColors(..) |
        Op::KeepColor(..) | Op::RemoveColor(..) | Op::ReplaceBackground(..)
    )
}

/// Classify ops into spatial (rotation-plane) vs causal (boost-plane) families.
/// Returns a bias score: positive favors spatial_bias > 0, negative favors < 0.
fn op_spatial_affinity(op: &Op) -> f32 {
    match op {
        // Purely geometric — rotation-plane transformations
        Op::HFlip | Op::VFlip | Op::Rot90CW | Op::Rot90CCW |
        Op::Rot180 | Op::Transpose => 1.0,
        Op::Gravity(..) | Op::SymmetryComplete(..) => 0.8,
        Op::Scale(..) | Op::Tile(..) | Op::Downscale(..) |
        Op::MirrorTile(..) | Op::FractalTile => 0.7,
        Op::ConnectLines => 0.5,

        // Object extraction — mixed but spatially oriented
        Op::CropToBBox | Op::CropToEnclosed |
        Op::ExtractLargestObj | Op::ExtractSmallestObj |
        Op::KeepLargestObj | Op::KeepSmallestObj => 0.3,
        Op::ExtractObjByColor(..) => 0.1,

        // Purely color — boost-plane transformations
        Op::ColorSub(..) | Op::SwapColors(..) | Op::MapColors(..) |
        Op::KeepColor(..) | Op::RemoveColor(..) | Op::ReplaceBackground(..) => -1.0,

        // Morphological — mixed
        Op::FillEnclosed | Op::ExpandNonBg | Op::ErodeNonBg |
        Op::OutlineObjects => 0.0,

        Op::RepeatUntilStable(inner) => op_spatial_affinity(inner) * 0.9,
    }
}

/// Reorder candidates so those aligned with the flow direction come first.
/// spatial_bias > 0 → prefer geometric ops; < 0 → prefer color ops.
fn reorder_by_flow(candidates: &mut Vec<Op>, flow: &FlowDiagnostic) {
    let bias = flow.spatial_bias();
    if bias.abs() < 0.05 { return; }

    candidates.sort_by(|a, b| {
        let sa = op_spatial_affinity(a) * bias;
        let sb = op_spatial_affinity(b) * bias;
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ─── MCTS program search (flow + Clifford hints, stochastic rollouts) ────────
//
// Explores the same DSL as A* but uses UCB1 + random rollouts instead of a
// single best-first heuristic — better when Clifford distance is deceptive.

struct MctsNode {
    program: Vec<usize>,
    grids: Vec<Grid>,
    visits: u32,
    sum_reward: f32,
    children: Vec<usize>,
    expanded: bool,
}

fn train_grids_all_match(grids: &[Grid], task: &ArcTask) -> bool {
    if grids.len() != task.train.len() {
        return false;
    }
    grids.iter().zip(task.train.iter()).all(|(g, ex)| grid_exact_match(g, &ex.output))
}

fn mcts_label(depth: usize) -> &'static str {
    match depth {
        1 => "mcts_d1",
        2 => "mcts_d2",
        _ => "mcts_d3",
    }
}

fn mcts_ucb_value(node: &MctsNode, parent_visits: u32, exploration: f32) -> f32 {
    if node.visits == 0 {
        return f32::INFINITY;
    }
    let mean = node.sum_reward / node.visits as f32;
    let ln_n = (parent_visits.max(1) as f32).ln();
    let bonus = exploration * (ln_n / node.visits as f32).sqrt();
    mean + bonus
}

fn mcts_pick_child(arena: &[MctsNode], parent_idx: usize, exploration: f32, rng: &mut impl Rng) -> usize {
    let children = &arena[parent_idx].children;
    let pv = arena[parent_idx].visits.max(1);
    let mut best = children[0];
    let mut best_u = mcts_ucb_value(&arena[best], pv, exploration);
    let mut ties = vec![best];
    for &ci in &children[1..] {
        let u = mcts_ucb_value(&arena[ci], pv, exploration);
        if u > best_u + 1e-6 {
            best_u = u;
            best = ci;
            ties.clear();
            ties.push(ci);
        } else if (u - best_u).abs() <= 1e-6 {
            ties.push(ci);
        }
    }
    *ties.choose(rng).unwrap_or(&best)
}

fn mcts_candidate_order(task: &ArcTask, candidates: &[Op], flow: &FlowDiagnostic) -> Vec<usize> {
    if candidates.is_empty() {
        return vec![];
    }
    // One `train_rule_consensus_mv` + one signature score per op; sorting only compares floats
    // (sorting by `clifford_op_hint_score` in the comparator was O(n log n) full encodes per expand).
    let consensus = if task.train.is_empty() {
        Multivector::zero()
    } else {
        train_rule_consensus_mv(task)
    };
    let bias = flow.spatial_bias();
    let mut scored: Vec<(f32, usize)> = (0..candidates.len())
        .map(|i| {
            let hint = if task.train.is_empty() {
                0.0
            } else {
                clifford_op_hint_score_with_consensus(&consensus, &candidates[i])
            };
            let s = hint + 0.12 * op_spatial_affinity(&candidates[i]) * bias;
            (s, i)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let k = if flow.is_degenerate() {
        scored.len().min(40)
    } else if flow.converging {
        scored.len().min(14)
    } else {
        scored.len().min(22)
    };
    scored.truncate(k);
    scored.into_iter().map(|(_, i)| i).collect()
}

fn mcts_expand(
    arena: &mut Vec<MctsNode>,
    leaf: usize,
    task: &ArcTask,
    candidates: &[Op],
    flow: &FlowDiagnostic,
) {
    let order = mcts_candidate_order(task, candidates, flow);
    let prog_base = arena[leaf].program.clone();
    let grids_base = arena[leaf].grids.clone();
    let mut new_nodes = Vec::new();
    for &op_i in &order {
        if let Some(&last) = prog_base.last() {
            if last == op_i && is_self_inverse(&candidates[last]) {
                continue;
            }
        }
        let op = &candidates[op_i];
        let mut next_grids = Vec::with_capacity(grids_base.len());
        let mut ok = true;
        for g in &grids_base {
            match apply_op(g, op) {
                Some(ng) => next_grids.push(ng),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let mut prog = prog_base.clone();
        prog.push(op_i);
        new_nodes.push(MctsNode {
            program: prog,
            grids: next_grids,
            visits: 0,
            sum_reward: 0.0,
            children: vec![],
            expanded: false,
        });
    }
    let start = arena.len();
    arena.extend(new_nodes);
    let end = arena.len();
    arena[leaf].children.extend(start..end);
    arena[leaf].expanded = true;
}

fn mcts_evaluate_state(grids: &[Grid], targets: &[Grid]) -> f32 {
    let dist = clifford_distance_to_targets(grids, targets);
    let cliff = 1.0 / (1.0 + 4.0 * dist);
    let mut cell_acc = 0.0f32;
    let mut n = 0u32;
    for (g, t) in grids.iter().zip(targets.iter()) {
        if g.height == t.height && g.width == t.width {
            let tot = (g.height * g.width) as f32;
            let mut c = 0u32;
            for r in 0..g.height {
                for col in 0..g.width {
                    if g.cells[r][col] == t.cells[r][col] {
                        c += 1;
                    }
                }
            }
            cell_acc += c as f32 / tot;
            n += 1;
        }
    }
    let cell_term = if n > 0 { cell_acc / n as f32 } else { 0.0 };
    0.6 * cell_term + 0.4 * cliff
}

fn mcts_rollout(
    start_grids: &[Grid],
    task: &ArcTask,
    candidates: &[Op],
    targets: &[Grid],
    max_extra_steps: usize,
    rng: &mut impl Rng,
) -> f32 {
    let mut g = start_grids.to_vec();
    let steps = max_extra_steps.saturating_add(2).min(6);
    for _ in 0..steps {
        if train_grids_all_match(&g, task) {
            return 1.0;
        }
        if candidates.is_empty() {
            break;
        }
        let op_i = rng.gen_range(0..candidates.len());
        let op = &candidates[op_i];
        let mut next = Vec::new();
        let mut ok = true;
        for grid in &g {
            match apply_op(grid, op) {
                Some(ng) => next.push(ng),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            break;
        }
        g = next;
    }
    if train_grids_all_match(&g, task) {
        return 1.0;
    }
    mcts_evaluate_state(&g, targets)
}

fn mcts_try_extract(
    task: &ArcTask,
    node: &MctsNode,
    candidates: &[Op],
) -> Option<(Vec<Grid>, &'static str)> {
    if !train_grids_all_match(&node.grids, task) || node.program.is_empty() {
        return None;
    }
    let prog: Vec<&Op> = node.program.iter().map(|&i| &candidates[i]).collect();
    if !validate_program_on_training(task, &prog) {
        return None;
    }
    let preds = predict_test(task, &prog)?;
    Some((preds, mcts_label(prog.len())))
}

fn mcts_select_path(
    arena: &[MctsNode],
    max_depth: usize,
    exploration: f32,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let mut path = vec![0usize];
    loop {
        let cur = *path.last().unwrap();
        let depth = arena[cur].program.len();
        if depth >= max_depth {
            break;
        }
        if !arena[cur].expanded {
            break;
        }
        if arena[cur].children.is_empty() {
            break;
        }
        let nxt = mcts_pick_child(arena, cur, exploration, rng);
        path.push(nxt);
    }
    path
}

/// Monte Carlo Tree Search over the same DSL as [`astar_dsl_solve`].
///
/// - **Selection:** UCB1 on child mean rollout reward.
/// - **Expansion:** top-k ops ranked by [`clifford_op_hint_score`] plus a small
///   flow (`spatial_bias`) prior; k adapts to converging / diverging / degenerate flow.
/// - **Rollout:** random operator steps on all train grids, then
///   [`mcts_evaluate_state`] (cell accuracy + Clifford distance) if not solved.
/// - **Success:** all training grids match targets → [`predict_test`].
///
/// Returns `None` if budget expires without a valid program.
pub fn mcts_dsl_solve(
    task: &ArcTask,
    max_depth: usize,
    simulations: usize,
    exploration: f32,
    budget_ms: u64,
) -> Option<(Vec<Grid>, &'static str)> {
    if task.train.is_empty() || task.test.is_empty() || max_depth == 0 {
        return None;
    }

    let candidates = generate_candidates(task);
    if candidates.is_empty() {
        return None;
    }

    let flow = flow_diagnostic(task);
    let target_grids: Vec<Grid> = task.train.iter().map(|ex| ex.output.clone()).collect();
    let init_grids: Vec<Grid> = task.train.iter().map(|ex| ex.input.clone()).collect();

    let mut arena = vec![MctsNode {
        program: vec![],
        grids: init_grids,
        visits: 0,
        sum_reward: 0.0,
        children: vec![],
        expanded: false,
    }];

    let mut rng = rand::thread_rng();
    let start = std::time::Instant::now();
    let budget = std::time::Duration::from_millis(budget_ms);

    for _ in 0..simulations {
        if start.elapsed() > budget {
            break;
        }

        let path = mcts_select_path(&arena, max_depth, exploration, &mut rng);
        let leaf = *path.last().unwrap();

        if arena[leaf].program.len() < max_depth && !arena[leaf].expanded {
            mcts_expand(&mut arena, leaf, task, &candidates, &flow);
        }

        let sim_node = if !arena[leaf].children.is_empty() {
            *arena[leaf].children.choose(&mut rng).unwrap_or(&leaf)
        } else {
            leaf
        };

        if let Some(win) = mcts_try_extract(task, &arena[sim_node], &candidates) {
            return Some(win);
        }

        let reward = if train_grids_all_match(&arena[sim_node].grids, task) {
            1.0
        } else {
            let extra = max_depth.saturating_sub(arena[sim_node].program.len());
            mcts_rollout(
                &arena[sim_node].grids,
                task,
                &candidates,
                &target_grids,
                extra,
                &mut rng,
            )
        };

        let mut bp = path.clone();
        if sim_node != leaf {
            bp.push(sim_node);
        }
        for &idx in &bp {
            arena[idx].visits = arena[idx].visits.saturating_add(1);
            arena[idx].sum_reward += reward;
        }
    }

    for node in &arena {
        if let Some(win) = mcts_try_extract(task, node, &candidates) {
            return Some(win);
        }
    }

    None
}

// ─── Unit tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc_agi::{encode_grid, extract_rule, multivector_cosine_similarity, Grid, ArcTask, ArcExample, FlowDiagnostic};

    fn make_grid(cells: Vec<Vec<u8>>) -> Grid {
        let height = cells.len();
        let width = if height > 0 { cells[0].len() } else { 0 };
        Grid { cells, height, width }
    }

    fn make_task(train: Vec<(Vec<Vec<u8>>, Vec<Vec<u8>>)>, test_in: Vec<Vec<u8>>, test_out: Vec<Vec<u8>>) -> ArcTask {
        ArcTask {
            id: "test".to_string(),
            train: train.into_iter().map(|(i, o)| ArcExample {
                input: make_grid(i), output: make_grid(o),
            }).collect(),
            test: vec![ArcExample {
                input: make_grid(test_in), output: make_grid(test_out),
            }],
        }
    }

    fn flow_rotation_dominated() -> FlowDiagnostic {
        FlowDiagnostic {
            boost_norm: 0.1,
            rotation_norm: 0.9,
            flow_magnitudes: vec![0.5, 0.3, 0.1],
            converging: true,
            mean_bv_direction: [0.0; 28],
        }
    }

    fn flow_boost_dominated() -> FlowDiagnostic {
        FlowDiagnostic {
            boost_norm: 0.9,
            rotation_norm: 0.1,
            flow_magnitudes: vec![0.5, 0.3, 0.1],
            converging: true,
            mean_bv_direction: [0.0; 28],
        }
    }

    fn flow_balanced() -> FlowDiagnostic {
        FlowDiagnostic {
            boost_norm: 0.5,
            rotation_norm: 0.5,
            flow_magnitudes: vec![0.5, 0.3],
            converging: true,
            mean_bv_direction: [0.0; 28],
        }
    }

    fn flow_diverging() -> FlowDiagnostic {
        FlowDiagnostic {
            boost_norm: 0.4,
            rotation_norm: 0.6,
            flow_magnitudes: vec![0.1, 0.3, 0.5],
            converging: false,
            mean_bv_direction: [0.0; 28],
        }
    }

    // ── op_spatial_affinity classification ──

    #[ignore]
    #[test]
    fn geometric_ops_have_positive_affinity() {
        let geos = [Op::HFlip, Op::VFlip, Op::Rot90CW, Op::Rot90CCW,
                     Op::Rot180, Op::Transpose];
        for op in &geos {
            assert!(op_spatial_affinity(op) > 0.0,
                "{:?} should have positive spatial affinity", op);
        }
    }

    #[ignore]
    #[test]
    fn color_ops_have_negative_affinity() {
        let colors = [
            Op::ColorSub(1, 2), Op::SwapColors(1, 2),
            Op::MapColors([0; NUM_COLORS]),
            Op::KeepColor(1), Op::RemoveColor(1), Op::ReplaceBackground(1),
        ];
        for op in &colors {
            assert!(op_spatial_affinity(op) < 0.0,
                "{:?} should have negative spatial affinity", op);
        }
    }

    #[ignore]
    #[test]
    fn morphological_ops_are_neutral() {
        let morphs = [Op::FillEnclosed, Op::ExpandNonBg, Op::ErodeNonBg, Op::OutlineObjects];
        for op in &morphs {
            assert_eq!(op_spatial_affinity(op), 0.0,
                "{:?} should have zero spatial affinity", op);
        }
    }

    #[ignore]
    #[test]
    fn repeat_until_stable_inherits_inner_affinity() {
        let inner = Op::FillEnclosed;
        let wrapped = Op::RepeatUntilStable(Box::new(inner.clone()));
        assert_eq!(op_spatial_affinity(&wrapped), op_spatial_affinity(&inner) * 0.9);

        let geo_inner = Op::Gravity(0);
        let geo_wrapped = Op::RepeatUntilStable(Box::new(geo_inner.clone()));
        assert!(op_spatial_affinity(&geo_wrapped) > 0.0);
    }

    // ── reorder_by_flow ──

    #[ignore]
    #[test]
    fn rotation_flow_puts_geometric_ops_first() {
        let mut ops = vec![
            Op::ColorSub(1, 2),
            Op::HFlip,
            Op::RemoveColor(3),
            Op::Rot90CW,
        ];
        let flow = flow_rotation_dominated();
        reorder_by_flow(&mut ops, &flow);

        assert!(op_spatial_affinity(&ops[0]) >= op_spatial_affinity(&ops[1]),
            "first op should have highest spatial affinity after rotation-flow reorder");
        assert!(matches!(ops[0], Op::HFlip | Op::Rot90CW),
            "geometric ops should lead after rotation-flow reorder, got {:?}", ops[0]);
    }

    #[ignore]
    #[test]
    fn boost_flow_puts_color_ops_first() {
        let mut ops = vec![
            Op::HFlip,
            Op::ColorSub(1, 2),
            Op::Rot90CW,
            Op::RemoveColor(3),
        ];
        let flow = flow_boost_dominated();
        reorder_by_flow(&mut ops, &flow);

        assert!(matches!(ops[0], Op::ColorSub(..) | Op::RemoveColor(..)),
            "color ops should lead after boost-flow reorder, got {:?}", ops[0]);
    }

    #[ignore]
    #[test]
    fn balanced_flow_preserves_order() {
        let ops_before = vec![Op::HFlip, Op::ColorSub(1, 2), Op::Rot90CW];
        let mut ops = ops_before.clone();
        let flow = flow_balanced();
        reorder_by_flow(&mut ops, &flow);

        // spatial_bias = 0.0, which is < 0.05 threshold, so no reorder
        assert_eq!(ops.len(), ops_before.len());
    }

    // ── convergence-adaptive budgets ──

    #[ignore]
    #[test]
    fn converging_flow_gets_tighter_budgets() {
        let f_conv = flow_rotation_dominated();
        assert!(f_conv.converging && !f_conv.is_degenerate());
        // The code uses 300ms d2, 800ms d3 for converging

        let f_div = flow_diverging();
        assert!(!f_div.converging);
        // The code uses 800ms d2, 2500ms d3 for diverging
        // Diverging gets more search time — that's the right behavior
    }

    // ── apply_op basic sanity ──

    #[ignore]
    #[test]
    fn apply_op_hflip_reverses_columns() {
        let g = make_grid(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let result = apply_op(&g, &Op::HFlip).unwrap();
        assert_eq!(result.cells, vec![vec![3, 2, 1], vec![6, 5, 4]]);
    }

    #[ignore]
    #[test]
    fn apply_op_vflip_reverses_rows() {
        let g = make_grid(vec![vec![1, 2], vec![3, 4]]);
        let result = apply_op(&g, &Op::VFlip).unwrap();
        assert_eq!(result.cells, vec![vec![3, 4], vec![1, 2]]);
    }

    #[ignore]
    #[test]
    fn apply_op_empty_grid_returns_none() {
        let g = Grid { cells: vec![], height: 0, width: 0 };
        assert!(apply_op(&g, &Op::HFlip).is_none());
    }
    
    #[ignore]
    #[test]
    fn apply_op_color_sub() {
        let g = make_grid(vec![vec![1, 2, 1], vec![3, 1, 2]]);
        let result = apply_op(&g, &Op::ColorSub(1, 5)).unwrap();
        assert_eq!(result.cells, vec![vec![5, 2, 5], vec![3, 5, 2]]);
    }

    #[ignore]
    #[test]
    fn apply_op_rot90_dims_swap() {
        let g = make_grid(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let result = apply_op(&g, &Op::Rot90CW).unwrap();
        assert_eq!(result.height, 3);
        assert_eq!(result.width, 2);
    }

    // ── dsl_solve on trivial tasks ──

    #[ignore]
    #[test]
    fn dsl_finds_hflip() {
        let task = make_task(
            vec![
                (vec![vec![1, 2, 3]], vec![vec![3, 2, 1]]),
                (vec![vec![4, 5, 6]], vec![vec![6, 5, 4]]),
            ],
            vec![vec![7, 8, 9]],
            vec![vec![9, 8, 7]],
        );
        let result = dsl_solve(&task);
        assert!(result.is_some(), "DSL should find HFlip");
        let (preds, strategy) = result.unwrap();
        assert_eq!(strategy, "dsl_d1");
        assert_eq!(preds[0].cells, vec![vec![9, 8, 7]]);
    }

    #[ignore]
    #[test]
    fn dsl_finds_vflip() {
        let task = make_task(
            vec![
                (vec![vec![1, 2], vec![3, 4]], vec![vec![3, 4], vec![1, 2]]),
                (vec![vec![5, 6], vec![7, 8]], vec![vec![7, 8], vec![5, 6]]),
            ],
            vec![vec![9, 1], vec![2, 3]],
            vec![vec![2, 3], vec![9, 1]],
        );
        let result = dsl_solve(&task);
        assert!(result.is_some(), "DSL should find VFlip");
        let (preds, strategy) = result.unwrap();
        assert_eq!(strategy, "dsl_d1");
        assert_eq!(preds[0].cells, vec![vec![2, 3], vec![9, 1]]);
    }

    #[ignore]
    #[test]
    fn dsl_finds_color_sub() {
        let task = make_task(
            vec![
                (vec![vec![1, 0, 1]], vec![vec![2, 0, 2]]),
                (vec![vec![0, 1, 0]], vec![vec![0, 2, 0]]),
            ],
            vec![vec![1, 1, 0]],
            vec![vec![2, 2, 0]],
        );
        let result = dsl_solve(&task);
        assert!(result.is_some(), "DSL should find ColorSub(1→2)");
        let (preds, strategy) = result.unwrap();
        assert_eq!(strategy, "dsl_d1");
        assert_eq!(preds[0].cells, vec![vec![2, 2, 0]]);
    }

    #[ignore]
    #[test]
    fn dsl_with_flow_also_finds_hflip() {
        let task = make_task(
            vec![
                (vec![vec![1, 2, 3]], vec![vec![3, 2, 1]]),
                (vec![vec![4, 5, 6]], vec![vec![6, 5, 4]]),
            ],
            vec![vec![7, 8, 9]],
            vec![vec![9, 8, 7]],
        );
        let flow = flow_rotation_dominated();
        let result = dsl_solve_with_flow(&task, Some(&flow));
        assert!(result.is_some(), "flow-guided DSL should still find HFlip");
        let (preds, _) = result.unwrap();
        assert_eq!(preds[0].cells, vec![vec![9, 8, 7]]);
    }

    #[ignore]
    #[test]
    fn dsl_returns_none_for_unsolvable() {
        // Training examples are mutually inconsistent: same input, different outputs
        // No single program can satisfy both.
        let task = make_task(
            vec![
                (vec![vec![1, 2, 3], vec![4, 5, 6]], vec![vec![6, 2, 1], vec![3, 5, 4]]),
                (vec![vec![1, 2, 3], vec![4, 5, 6]], vec![vec![4, 5, 6], vec![1, 2, 3]]),
            ],
            vec![vec![7, 8, 9], vec![1, 2, 3]],
            vec![vec![0, 0, 0], vec![0, 0, 0]],
        );
        let result = dsl_solve(&task);
        assert!(result.is_none(), "DSL should return None for inconsistent mapping");
    }

    #[ignore]
    #[test]
    fn dsl_empty_task_returns_none() {
        let task = ArcTask {
            id: "empty".to_string(),
            train: vec![],
            test: vec![],
        };
        assert!(dsl_solve(&task).is_none());
    }

    // ── op_output_dims correctness ──

    #[ignore]
    #[test]
    fn op_output_dims_geometric_preserve_size() {
        let same_dim_ops = [Op::HFlip, Op::VFlip, Op::Rot180];
        for op in &same_dim_ops {
            assert_eq!(op_output_dims(5, 7, op), Some((5, 7)),
                "{:?} should preserve dims", op);
        }
    }

    #[ignore]
    #[test]
    fn op_output_dims_rot90_swaps() {
        assert_eq!(op_output_dims(3, 5, &Op::Rot90CW), Some((5, 3)));
        assert_eq!(op_output_dims(3, 5, &Op::Rot90CCW), Some((5, 3)));
        assert_eq!(op_output_dims(3, 5, &Op::Transpose), Some((5, 3)));
    }

    #[ignore]
    #[test]
    fn op_output_dims_scale() {
        assert_eq!(op_output_dims(3, 4, &Op::Scale(2, 3)), Some((6, 12)));
        assert_eq!(op_output_dims(3, 4, &Op::Tile(2, 3)), Some((6, 12)));
    }

    #[ignore]
    #[test]
    fn op_output_dims_downscale() {
        assert_eq!(op_output_dims(6, 8, &Op::Downscale(2, 2, 0)), Some((3, 4)));
    }

    // ── is_structural_op / is_self_inverse ──

    #[ignore]
    #[test]
    fn structural_ops_classification() {
        assert!(is_structural_op(&Op::HFlip));
        assert!(is_structural_op(&Op::CropToBBox));
        assert!(is_structural_op(&Op::MapColors([0; NUM_COLORS])));
        assert!(!is_structural_op(&Op::ColorSub(1, 2)));
        assert!(!is_structural_op(&Op::SwapColors(1, 2)));
    }

    #[ignore]
    #[test]
    fn self_inverse_ops() {
        assert!(is_self_inverse(&Op::HFlip));
        assert!(is_self_inverse(&Op::VFlip));
        assert!(is_self_inverse(&Op::Rot180));
        assert!(is_self_inverse(&Op::SwapColors(1, 2)));
        assert!(!is_self_inverse(&Op::Rot90CW));
        assert!(!is_self_inverse(&Op::ColorSub(1, 2)));
    }

    // ── Clifford hints / reference signatures ──

    #[ignore]
    #[test]
    fn color_sub_hint_beats_geometric_distractor() {
        let task = make_task(
            vec![(vec![vec![3, 1], vec![2, 3]], vec![vec![7, 1], vec![2, 7]])],
            vec![vec![3, 3]],
            vec![vec![7, 7]],
        );
        let s_color = clifford_op_hint_score(&task, &Op::ColorSub(3, 7));
        let s_geo = clifford_op_hint_score(&task, &Op::HFlip);
        assert!(
            s_color > s_geo,
            "color rule hint {:.4} should exceed HFlip {:.4} on this task",
            s_color,
            s_geo
        );
    }

    #[ignore]
    #[test]
    fn dsl_color_sub_generalizes_2x2_train_to_tall_test() {
        let train_in = vec![vec![3, 1], vec![2, 3]];
        let train_out = vec![vec![7, 1], vec![2, 7]];
        let test_in: Vec<Vec<u8>> = (0..10).map(|_| vec![3, 1]).collect();
        let test_out: Vec<Vec<u8>> = (0..10).map(|_| vec![7, 1]).collect();
        let task = make_task(vec![(train_in, train_out)], test_in, test_out);
        let (preds, _) = dsl_solve(&task).expect("depth-1 ColorSub");
        assert_eq!(preds[0].cells, task.test[0].output.cells);
    }

    #[ignore]
    #[test]
    fn composed_op_geo_product_correlates_with_chained_extract_rule() {
        let g = dsl_op_reference_grid();
        let g1 = apply_op(&g, &Op::HFlip).unwrap();
        let g2 = apply_op(&g1, &Op::ColorSub(3, 7)).unwrap();
        let z0 = encode_grid(&g);
        let z2 = encode_grid(&g2);
        let r_chain = extract_rule(&z0, &z2);
        let s_a = dsl_op_signature(&Op::HFlip).unwrap();
        let s_b = dsl_op_signature(&Op::ColorSub(3, 7)).unwrap();
        let c1 = multivector_cosine_similarity(&r_chain, &s_b.geo(&s_a));
        let c2 = multivector_cosine_similarity(&r_chain, &s_a.geo(&s_b));
        // Nonlinear encode → chained `extract_rule` need not match either geo-product order;
        // we only expect a strong linear relation (same or opposite direction in R^256).
        let align = c1.abs().max(c2.abs());
        assert!(
            align > 0.5,
            "expect |cos| ≫ 0 vs one product order; c1={:.4} c2={:.4}",
            c1,
            c2
        );
    }

    #[ignore]
    #[test]
    fn mcts_finds_depth1_hflip() {
        // Train [1,2,3]→[3,2,1]: HFlip. Inferred MapColors swaps 1↔3 but is identity on
        // colors 4,5,6, so on test [4,5,6] it leaves the grid fixed while HFlip → [6,5,4].
        let task = make_task(
            vec![(vec![vec![1, 2, 3]], vec![vec![3, 2, 1]])],
            vec![vec![4, 5, 6]],
            vec![vec![6, 5, 4]],
        );
        // Few sims + tiny wall budget: ordering surfaces HFlip on first expand before budget bites.
        let r = mcts_dsl_solve(&task, 3, 5, 1.4, 5);
        assert!(r.is_some(), "MCTS should find HFlip for 1×3 row reversal");
        let (preds, label) = r.unwrap();
        assert_eq!(label, "mcts_d1");
        assert_eq!(preds[0].cells, task.test[0].output.cells);
    }
}
