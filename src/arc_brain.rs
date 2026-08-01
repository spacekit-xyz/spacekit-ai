use std::collections::HashMap;
use std::path::Path;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::arc_agi::{
    encode_grid, flow_diagnostic, load_arc_tasks, solve_task, ArcExample, ArcTask, Grid, NUM_COLORS,
};
use crate::arc_dsl::astar_dsl_solve;
use crate::clifford::Multivector;
use crate::dimension::language::LanguageRuntime;
use crate::dimension::manager::DimensionManager;
use crate::micro_brain::{MicroBrain, MicroBrainRole};

const NEIGHBORHOOD: usize = 3;
const HALF_K: usize = NEIGHBORHOOD / 2;

// ─── Clifford cell encoding (translation-invariant) ─────────────────────────

/// Relative position vector — purely SPACELIKE (e₁…e₇), matching arc_agi encoding.
fn relative_position_vector(dr: isize, dc: isize) -> Multivector {
    let pi = std::f32::consts::PI;
    let u = (dr as f32 + HALF_K as f32) / (NEIGHBORHOOD as f32 - 1.0);
    let v = (dc as f32 + HALF_K as f32) / (NEIGHBORHOOD as f32 - 1.0);
    let pv = [
        0.0, // e₀ = 0: no timelike
        (pi * u).sin(),
        (pi * u).cos(),
        (pi * v).sin(),
        (pi * v).cos(),
        (2.0 * pi * u).sin(),
        (2.0 * pi * v).sin(),
        (pi * (u + v)).sin(),
    ];
    let norm: f32 = pv.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let mut normed = [0.0f32; 8];
    for i in 0..8 {
        normed[i] = pv[i] / norm;
    }
    Multivector::vector(&normed)
}

/// Color vector — timelike-dominant (e₀) with spacelike tag, matching arc_agi encoding.
/// Color 0 (background) gets negative e₀ to remain distinguishable from foreground.
fn color_vector_full(color: u8) -> Multivector {
    match color {
        0 => {
            let mut v = [0.0f32; 8];
            v[0] = -1.0;
            Multivector::vector(&v)
        }
        c @ 1..=8 => {
            let mut v = [0.0f32; 8];
            v[0] = 1.0;
            v[(c - 1) as usize] += 0.3;
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= norm;
            }
            Multivector::vector(&v)
        }
        9 => {
            let mut v = [0.0f32; 8];
            v[0] = 1.0;
            v[1] = 0.15;
            v[2] = 0.15;
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= norm;
            }
            Multivector::vector(&v)
        }
        _ => Multivector::zero(),
    }
}

fn encode_cell_neighborhood(grid: &Grid, r: usize, c: usize) -> Vec<f32> {
    let mut mv = Multivector::zero();
    for dr in -(HALF_K as isize)..=(HALF_K as isize) {
        for dc in -(HALF_K as isize)..=(HALF_K as isize) {
            let gr = r as isize + dr;
            let gc = c as isize + dc;
            let color =
                if gr >= 0 && gr < grid.height as isize && gc >= 0 && gc < grid.width as isize {
                    grid.cells[gr as usize][gc as usize]
                } else {
                    0
                };
            let pos = relative_position_vector(dr, dc);
            let col = color_vector_full(color);
            mv = mv.add(&pos.geo(&col));
        }
    }
    let n = mv.component_norm();
    if n > 1e-8 {
        mv = mv.scale(1.0 / n);
    }
    mv.components.to_vec()
}

// ─── Task-level encoding for strategy routing ───────────────────────────────
//
// Two-path encoding:
//   Path A (Clifford): geometric grid encoding + scalar features — 512+25 dims
//   Path B (Language Bridge): task description → GrowformerLanguageEncoder →
//     LanguageBridge → 128d bridged vector (cross-domain knowledge from
//     math/science/coding training data in brain.bin)
//
// Combined: [bridge_128d, clifford_512d, scalars_25d, flow_6d, monopole_*] — see
// [`MONOPOLE_TASK_ENCODING_EXTRA_DIM`] for the tail block.

fn task_to_text(task: &ArcTask) -> String {
    let ex = &task.train[0];
    let ih = ex.input.height;
    let iw = ex.input.width;
    let oh = ex.output.height;
    let ow = ex.output.width;

    let mut in_colors = [0u32; NUM_COLORS];
    for row in &ex.input.cells {
        for &c in row {
            in_colors[c as usize] += 1;
        }
    }
    let mut out_colors = [0u32; NUM_COLORS];
    for row in &ex.output.cells {
        for &c in row {
            out_colors[c as usize] += 1;
        }
    }
    let in_distinct = in_colors.iter().filter(|&&c| c > 0).count();
    let out_distinct = out_colors.iter().filter(|&&c| c > 0).count();
    let total = (ih * iw) as f32;

    let dim_relation = if ih == oh && iw == ow {
        let changed = (0..ih)
            .flat_map(|r| (0..iw).map(move |c| (r, c)))
            .filter(|&(r, c)| ex.input.cells[r][c] != ex.output.cells[r][c])
            .count();
        format!(
            "same dimensions, {:.0}% cells changed",
            changed as f32 / total * 100.0
        )
    } else if oh > ih && ow > iw {
        let rh = oh as f32 / ih as f32;
        let rw = ow as f32 / iw as f32;
        if (rh - rw).abs() < 0.01 {
            format!("output scaled up {:.1}x uniformly", rh)
        } else {
            format!(
                "output grows {}x{} to {}x{}, ratio {:.1}h {:.1}w",
                ih, iw, oh, ow, rh, rw
            )
        }
    } else if oh < ih && ow < iw {
        format!(
            "output shrinks {}x{} to {}x{}, extraction or summary",
            ih, iw, oh, ow
        )
    } else {
        format!("asymmetric resize {}x{} to {}x{}", ih, iw, oh, ow)
    };

    let bg = in_colors
        .iter()
        .enumerate()
        .max_by_key(|(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let bg_frac = in_colors[bg] as f32 / total;

    format!(
        "grid transformation: input {}x{} with {} colors (bg color {} at {:.0}%), \
         output {}x{} with {} colors. {} examples. {}",
        ih,
        iw,
        in_distinct,
        bg,
        bg_frac * 100.0,
        oh,
        ow,
        out_distinct,
        task.train.len(),
        dim_relation,
    )
}

fn encode_task(task: &ArcTask, lang_rt: Option<&LanguageRuntime>) -> Vec<f32> {
    let bridge_dim = 128;
    let raw_dim = 768;
    let mut features =
        Vec::with_capacity(raw_dim + bridge_dim + 256 * 2 + 32 + MONOPOLE_TASK_ENCODING_EXTRA_DIM);

    // Path B: Language bridge (cross-domain knowledge from brain.bin)
    // Include BOTH raw 768d embedding and 128d bridged to minimize information loss
    if let Some(rt) = lang_rt {
        let text = task_to_text(task);
        match rt.encode_and_bridge(&text) {
            Ok((raw, bridged)) => {
                features.extend_from_slice(&raw);
                if raw.len() < raw_dim {
                    features.extend(std::iter::repeat(0.0f32).take(raw_dim - raw.len()));
                }
                features.extend_from_slice(&bridged.routed_vector);
            }
            Err(_) => {
                features.extend(std::iter::repeat(0.0f32).take(raw_dim + bridge_dim));
            }
        }
    } else {
        features.extend(std::iter::repeat(0.0f32).take(raw_dim + bridge_dim));
    }

    // Path A: Clifford grid encoding
    let input_mv = encode_grid(&task.train[0].input);
    let output_mv = encode_grid(&task.train[0].output);
    features.extend_from_slice(&input_mv.components);
    features.extend_from_slice(&output_mv.components);

    let ex = &task.train[0];
    let ih = ex.input.height as f32;
    let iw = ex.input.width as f32;
    let oh = ex.output.height as f32;
    let ow = ex.output.width as f32;

    features.push(ih / 30.0);
    features.push(iw / 30.0);
    features.push(oh / 30.0);
    features.push(ow / 30.0);
    features.push(if ih > 0.0 { oh / ih } else { 0.0 });
    features.push(if iw > 0.0 { ow / iw } else { 0.0 });
    features.push(if (ih - oh).abs() < 0.5 && (iw - ow).abs() < 0.5 {
        1.0
    } else {
        0.0
    });

    let mut color_counts = [0u32; NUM_COLORS];
    let total_cells = (ex.input.height * ex.input.width) as f32;
    for row in &ex.input.cells {
        for &c in row {
            color_counts[c as usize] += 1;
        }
    }
    for c in 0..NUM_COLORS {
        features.push(color_counts[c] as f32 / total_cells.max(1.0));
    }

    let in_colors = color_counts.iter().filter(|&&c| c > 0).count() as f32 / NUM_COLORS as f32;
    let mut out_counts = [0u32; NUM_COLORS];
    for row in &ex.output.cells {
        for &c in row {
            out_counts[c as usize] += 1;
        }
    }
    let out_colors = out_counts.iter().filter(|&&c| c > 0).count() as f32 / NUM_COLORS as f32;
    features.push(in_colors);
    features.push(out_colors);

    if ex.input.height == ex.output.height && ex.input.width == ex.output.width {
        let mut changed = 0u32;
        for r in 0..ex.input.height {
            for c in 0..ex.input.width {
                if ex.input.cells[r][c] != ex.output.cells[r][c] {
                    changed += 1;
                }
            }
        }
        features.push(changed as f32 / total_cells.max(1.0));
    } else {
        features.push(1.0);
    }

    features.push(task.train.len() as f32 / 5.0);

    let bg = color_counts
        .iter()
        .enumerate()
        .max_by_key(|(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0);
    features.push(color_counts[bg] as f32 / total_cells.max(1.0));

    // Path C: Flow diagnostic features (Schrödinger continuity)
    let flow = flow_diagnostic(task);
    features.push(flow.boost_norm);
    features.push(flow.rotation_norm);
    features.push(flow.spatial_bias());
    features.push(if flow.converging { 1.0 } else { 0.0 });
    features.push(if flow.is_degenerate() { 1.0 } else { 0.0 });
    let mean_mag = if flow.flow_magnitudes.is_empty() {
        0.0
    } else {
        flow.flow_magnitudes.iter().sum::<f32>() / flow.flow_magnitudes.len() as f32
    };
    features.push(mean_mag);

    // Path D: Global monopole / Wilson plaquette field (first train pair)
    let m_in = monopole_field_features(&ex.input);
    let m_out = monopole_field_features(&ex.output);
    debug_assert_eq!(m_in.len(), MONOPOLE_GLOBAL_FEATURE_DIM);
    debug_assert_eq!(m_out.len(), MONOPOLE_GLOBAL_FEATURE_DIM);
    features.extend_from_slice(&m_in);
    features.extend_from_slice(&m_out);
    let l2_diff: f32 = m_in
        .iter()
        .zip(m_out.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    features.push(m_out[1] - m_in[1]);
    features.push(m_out[2] - m_in[2]);
    features.push(m_out[19] - m_in[19]);
    features.push(l2_diff);

    features
}

// ─── Per-task one-pass cell learning ────────────────────────────────────────

/// Encode output cell (r,c) using the FULL corresponding input block/region.
///
/// For SHRINK tasks (oh<ih): each output cell summarizes a block of input cells.
///   Features: color histogram of the block, dominant color one-hot, block Clifford
///   encoding, uniformity flag, positional context (output coords, scale).
///
/// For GROW tasks (oh>ih): each output cell is inside an expanded input cell.
///   Features: source input cell's neighborhood encoding, sub-block position (where
///   inside the expanded cell this output cell sits), source cell one-hot color.
///
/// For ARBITRARY changes: fractional mapping with neighborhood + positional context.
fn encode_cell_rich(input: &Grid, out_r: usize, out_c: usize, oh: usize, ow: usize) -> Vec<f32> {
    let ih = input.height;
    let iw = input.width;

    let shrink = oh < ih && ow < iw;
    let grow = oh > ih && ow > iw;

    if shrink && ih % oh == 0 && iw % ow == 0 {
        encode_cell_shrink_block(input, out_r, out_c, oh, ow)
    } else if grow && oh % ih == 0 && ow % iw == 0 {
        encode_cell_grow_subcell(input, out_r, out_c, oh, ow)
    } else {
        encode_cell_fractional(input, out_r, out_c, oh, ow)
    }
}

/// SHRINK: encode the entire input block that maps to this output cell.
fn encode_cell_shrink_block(
    input: &Grid,
    out_r: usize,
    out_c: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    let ih = input.height;
    let iw = input.width;
    let bh = ih / oh;
    let bw = iw / ow;
    let r0 = out_r * bh;
    let c0 = out_c * bw;

    let mut hist = [0u32; NUM_COLORS];
    let mut total = 0u32;
    let mut block_mv = Multivector::zero();

    for br in 0..bh {
        for bc in 0..bw {
            let r = (r0 + br).min(ih - 1);
            let c = (c0 + bc).min(iw - 1);
            let color = input.cells[r][c];
            hist[color as usize] += 1;
            total += 1;
            block_mv = block_mv.add(&color_vector_full(color));
        }
    }

    let n = block_mv.component_norm();
    if n > 1e-8 {
        block_mv = block_mv.scale(1.0 / n);
    }

    let mut features = Vec::with_capacity(NUM_COLORS * 2 + 256 + 16);

    // Color histogram (fraction of each color in the block)
    for c in 0..NUM_COLORS {
        features.push(hist[c] as f32 / total.max(1) as f32);
    }
    // Dominant color one-hot
    let dominant = hist
        .iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0);
    for c in 0..NUM_COLORS {
        features.push(if c == dominant { 1.0 } else { 0.0 });
    }
    // Block uniformity: 1.0 if all same color
    let distinct = hist.iter().filter(|&&v| v > 0).count();
    features.push(if distinct == 1 { 1.0 } else { 0.0 });
    features.push(distinct as f32 / NUM_COLORS as f32);

    // Block Clifford encoding (256 components)
    features.extend_from_slice(&block_mv.components);

    // Neighbor blocks: encode the dominant color of each neighboring output cell's block
    for dr in -1i32..=1 {
        for dc in -1i32..=1 {
            if dr == 0 && dc == 0 {
                continue;
            }
            let nr = out_r as i32 + dr;
            let nc = out_c as i32 + dc;
            if nr >= 0 && nr < oh as i32 && nc >= 0 && nc < ow as i32 {
                let nbr0 = nr as usize * bh;
                let nbc0 = nc as usize * bw;
                let mut nhist = [0u32; NUM_COLORS];
                for br in 0..bh {
                    for bc in 0..bw {
                        let r = (nbr0 + br).min(ih - 1);
                        let c = (nbc0 + bc).min(iw - 1);
                        nhist[input.cells[r][c] as usize] += 1;
                    }
                }
                let ndom = nhist
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &v)| v)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                features.push(ndom as f32 / (NUM_COLORS - 1).max(1) as f32);
            } else {
                features.push(-1.0);
            }
        }
    }

    // Positional context
    features.push(out_r as f32 / oh.max(1) as f32);
    features.push(out_c as f32 / ow.max(1) as f32);
    features.push(bh as f32 / ih.max(1) as f32);
    features.push(bw as f32 / iw.max(1) as f32);
    features.push(oh as f32 / 30.0);
    features.push(ow as f32 / 30.0);

    features
}

/// GROW: encode the source input cell + sub-block position within expanded cell.
fn encode_cell_grow_subcell(
    input: &Grid,
    out_r: usize,
    out_c: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    let ih = input.height;
    let iw = input.width;
    let sh = oh / ih;
    let sw = ow / iw;

    let src_r = out_r / sh;
    let src_c = out_c / sw;
    let sub_r = out_r % sh;
    let sub_c = out_c % sw;

    // Neighborhood encoding of source input cell
    let base = encode_cell_neighborhood(input, src_r.min(ih - 1), src_c.min(iw - 1));

    let mut features = Vec::with_capacity(base.len() + NUM_COLORS + 16);
    features.extend_from_slice(&base);

    // Source cell color one-hot
    let color = input.cells[src_r.min(ih - 1)][src_c.min(iw - 1)];
    for c in 0..NUM_COLORS as u8 {
        features.push(if c == color { 1.0 } else { 0.0 });
    }

    // Sub-block position (where within the expanded cell)
    features.push(sub_r as f32 / sh.max(1) as f32);
    features.push(sub_c as f32 / sw.max(1) as f32);
    features.push(if sub_r == 0 { 1.0 } else { 0.0 }); // top edge
    features.push(if sub_c == 0 { 1.0 } else { 0.0 }); // left edge
    features.push(if sub_r == sh - 1 { 1.0 } else { 0.0 }); // bottom edge
    features.push(if sub_c == sw - 1 { 1.0 } else { 0.0 }); // right edge
    features.push(if sub_r == 0 && sub_c == 0 { 1.0 } else { 0.0 }); // corner: TL
    features.push(if sub_r == 0 && sub_c == sw - 1 {
        1.0
    } else {
        0.0
    }); // corner: TR
    features.push(if sub_r == sh - 1 && sub_c == 0 {
        1.0
    } else {
        0.0
    }); // corner: BL
    features.push(if sub_r == sh - 1 && sub_c == sw - 1 {
        1.0
    } else {
        0.0
    }); // corner: BR

    // Scale factors
    features.push(sh as f32 / 4.0);
    features.push(sw as f32 / 4.0);
    features.push(out_r as f32 / oh.max(1) as f32);
    features.push(out_c as f32 / ow.max(1) as f32);
    features.push(if color == 0 { 1.0 } else { 0.0 }); // is source background

    features
}

/// ARBITRARY dim change: fractional mapping with full neighborhood + positional context.
fn encode_cell_fractional(
    input: &Grid,
    out_r: usize,
    out_c: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    let ih = input.height;
    let iw = input.width;

    // Fractional mapping to input space
    let fr = if oh > 1 {
        out_r as f32 * (ih - 1) as f32 / (oh - 1).max(1) as f32
    } else {
        ih as f32 / 2.0
    };
    let fc = if ow > 1 {
        out_c as f32 * (iw - 1) as f32 / (ow - 1).max(1) as f32
    } else {
        iw as f32 / 2.0
    };
    let ir = (fr.round() as usize).min(ih - 1);
    let ic = (fc.round() as usize).min(iw - 1);

    // Wider neighborhood (5×5) around mapped position
    let half_k = 2usize;
    let mut mv = Multivector::zero();
    for dr in -(half_k as isize)..=(half_k as isize) {
        for dc in -(half_k as isize)..=(half_k as isize) {
            let gr = ir as isize + dr;
            let gc = ic as isize + dc;
            let color = if gr >= 0 && gr < ih as isize && gc >= 0 && gc < iw as isize {
                input.cells[gr as usize][gc as usize]
            } else {
                0
            };
            let pos = relative_position_vector(dr, dc);
            let col = color_vector_full(color);
            mv = mv.add(&pos.geo(&col));
        }
    }
    let n = mv.component_norm();
    if n > 1e-8 {
        mv = mv.scale(1.0 / n);
    }

    let mut features = Vec::with_capacity(256 + NUM_COLORS + 16);
    features.extend_from_slice(&mv.components);

    // Color at mapped position
    let center_color = input.cells[ir][ic];
    for c in 0..NUM_COLORS as u8 {
        features.push(if c == center_color { 1.0 } else { 0.0 });
    }

    // Positional context
    features.push(out_r as f32 / oh.max(1) as f32);
    features.push(out_c as f32 / ow.max(1) as f32);
    features.push(fr / ih.max(1) as f32);
    features.push(fc / iw.max(1) as f32);
    let rh = if ih > 0 { oh as f32 / ih as f32 } else { 1.0 };
    let rw = if iw > 0 { ow as f32 / iw as f32 } else { 1.0 };
    features.push(rh);
    features.push(rw);
    features.push(if ih == oh && iw == ow { 1.0 } else { 0.0 });

    features
}

/// Train one-pass cell Paramecium on `task.train`, then predict for each `predict_on` example.
fn solve_task_onepass_for_examples(task: &ArcTask, predict_on: &[ArcExample]) -> Option<Vec<Grid>> {
    if task.train.is_empty() || predict_on.is_empty() {
        return None;
    }

    let same_dim = task
        .train
        .iter()
        .all(|ex| ex.input.height == ex.output.height && ex.input.width == ex.output.width);

    let class_names: Vec<String> = (0..NUM_COLORS).map(|c| format!("c{}", c)).collect();
    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut targets: Vec<usize> = Vec::new();

    for ex in &task.train {
        let oh = ex.output.height;
        let ow = ex.output.width;
        for r in 0..oh {
            for c in 0..ow {
                if same_dim {
                    embeddings.push(encode_cell_neighborhood(&ex.input, r, c));
                } else {
                    embeddings.push(encode_cell_rich(&ex.input, r, c, oh, ow));
                }
                targets.push(ex.output.cells[r][c] as usize);
            }
        }
    }

    if embeddings.is_empty() {
        return None;
    }

    let max_len = embeddings.iter().map(|e| e.len()).max().unwrap_or(0);
    for emb in &mut embeddings {
        emb.resize(max_len, 0.0);
    }

    let emb_dim = max_len;
    let sample_refs: Vec<(&[f32], usize)> = embeddings
        .iter()
        .zip(targets.iter())
        .map(|(e, &t)| (e.as_slice(), t))
        .collect();

    let brain = MicroBrain::build_from_data(
        MicroBrainRole::Custom("arc_cell".into()),
        emb_dim,
        NUM_COLORS,
        class_names,
        &sample_refs,
    );

    let results: Vec<Grid> = predict_on
        .iter()
        .map(|test_ex| {
            let oh = test_ex.output.height;
            let ow = test_ex.output.width;
            let pred_h = if same_dim { test_ex.input.height } else { oh };
            let pred_w = if same_dim { test_ex.input.width } else { ow };

            let mut cells = vec![vec![0u8; pred_w]; pred_h];
            let mut brain_local = brain.clone();
            for r in 0..pred_h {
                for c in 0..pred_w {
                    let mut emb = if same_dim {
                        encode_cell_neighborhood(&test_ex.input, r, c)
                    } else {
                        encode_cell_rich(&test_ex.input, r, c, pred_h, pred_w)
                    };
                    emb.resize(emb_dim, 0.0);
                    let (cls, _conf, _logits) = brain_local.predict(&emb);
                    cells[r][c] = cls as u8;
                }
            }
            Grid {
                cells,
                height: pred_h,
                width: pred_w,
            }
        })
        .collect();

    Some(results)
}

fn solve_task_onepass(task: &ArcTask) -> Option<Vec<Grid>> {
    if task.test.is_empty() {
        return None;
    }
    solve_task_onepass_for_examples(task, &task.test)
}

// ─── Dirac particle field solver (color-stratified) ──────────────────────────
//
// The grid lives in 3D: (row, col, color_layer). Each color gets its own
// spacetime sheet — particles exist only where that color appears. The spatial
// PATTERN on each layer is a pure position field: no cross-color interference.
//
// For SHRINK: per-block field strength of each color layer determines the output.
//   Particles on the same layer interfere constructively when spatially coherent.
// For GROW: the sub-position pattern of each color layer is learned from training.
// For SAME-DIM / ARBITRARY: per-cell neighborhood field strength per color layer.
//
// Two modes:
//   DIRECT:  argmax_c |Ψ_c(block)| — zero-parameter, spatially-weighted majority
//   LEARNED: [|Ψ_0|, ..., |Ψ_9|, features] → Paramecium → output color

/// Position spinor for a cell, normalized by grid dims. Pure spatial, no color.
fn grid_position_spinor(r: usize, c: usize, h: usize, w: usize) -> Multivector {
    let pi = std::f32::consts::PI;
    let u = if h > 1 {
        r as f32 / (h - 1) as f32
    } else {
        0.5
    };
    let v = if w > 1 {
        c as f32 / (w - 1) as f32
    } else {
        0.5
    };
    let pv = [
        0.0,
        (pi * u).sin(),
        (pi * u).cos(),
        (pi * v).sin(),
        (pi * v).cos(),
        (2.0 * pi * u).sin(),
        (2.0 * pi * v).sin(),
        (pi * (u + v)).sin(),
    ];
    let norm: f32 = pv.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    let mut normed = [0.0f32; 8];
    for i in 0..8 {
        normed[i] = pv[i] / norm;
    }
    Multivector::vector(&normed)
}

// ─── Dirac monopole field (phase 1): bivector spinors + Wilson plaquettes ─────
//
// Cell spinor ψ(r,c) = pos(r,c) ∧ color(cell) — grade-2, couples space and palette.
// Link (a→b): U_ab = ψ_b * ψ_a† (geometric product with reverse).
// Plaquette Wilson loop W = U_ab U_bc U_cd U_da around each 2×2 square.
// The grade-8 (pseudoscalar) part is the ideal monopole charge; in practice it
// often vanishes for bivector-built links, so we use |pseudo| when significant
// and otherwise the loop’s component norm as a curvature / “abelian” proxy.

/// Length of [`monopole_field_features`]. Kept stable for downstream encoders.
pub const MONOPOLE_GLOBAL_FEATURE_DIM: usize = 26;

/// Length of [`cell_monopole_features`].
pub const MONOPOLE_CELL_FEATURE_DIM: usize = 4;

/// Extra floats appended by [`encode_task`] after flow diagnostics: input monopole,
/// output monopole, and four summary deltas (mean/var/rms deltas + L2 between vectors).
pub const MONOPOLE_TASK_ENCODING_EXTRA_DIM: usize = 2 * MONOPOLE_GLOBAL_FEATURE_DIM + 4;

fn cell_bivector_spinor(grid: &Grid, r: usize, c: usize) -> Multivector {
    let pos = grid_position_spinor(r, c, grid.height, grid.width);
    let col = color_vector_full(grid.cells[r][c]);
    pos.wedge(&col)
}

/// Parallel transport between horizontally or vertically adjacent cells.
fn link_variable(grid: &Grid, r1: usize, c1: usize, r2: usize, c2: usize) -> Multivector {
    let dr = (r2 as isize - r1 as isize).abs();
    let dc = (c2 as isize - c1 as isize).abs();
    debug_assert!((dr == 1 && dc == 0) || (dr == 0 && dc == 1));
    let psi_a = cell_bivector_spinor(grid, r1, c1);
    let psi_b = cell_bivector_spinor(grid, r2, c2);
    psi_b.geo(&psi_a.reverse())
}

/// Wilson loop multivector for the elementary plaquette with top-left `(r, c)`.
fn plaquette_wilson_loop(grid: &Grid, r: usize, c: usize) -> Multivector {
    let h = grid.height;
    let w = grid.width;
    if r + 1 >= h || c + 1 >= w {
        return Multivector::zero();
    }
    let u_ab = link_variable(grid, r, c, r, c + 1);
    let u_bc = link_variable(grid, r, c + 1, r + 1, c + 1);
    let u_cd = link_variable(grid, r + 1, c + 1, r + 1, c);
    let u_da = link_variable(grid, r + 1, c, r, c);
    u_ab.geo(&u_bc).geo(&u_cd).geo(&u_da)
}

/// Scalar plaquette signal: signed pseudoscalar (monopole) when large vs norm, else curvature energy.
fn plaquette_charge(grid: &Grid, r: usize, c: usize) -> f32 {
    let w_loop = plaquette_wilson_loop(grid, r, c);
    let p = w_loop.pseudoscalar_part();
    let n = w_loop.component_norm();
    if n < 1e-12 {
        return 0.0;
    }
    if p.abs() > 1e-6 * n {
        p
    } else {
        n
    }
}

fn plaquette_touches_color(grid: &Grid, pr: usize, pc: usize, color: u8) -> bool {
    grid.cells[pr][pc] == color
        || grid.cells[pr][pc + 1] == color
        || grid.cells[pr + 1][pc] == color
        || grid.cells[pr + 1][pc + 1] == color
}

/// Grid-level monopole statistics from all elementary plaquettes.
pub fn monopole_field_features(grid: &Grid) -> Vec<f32> {
    let h = grid.height;
    let w = grid.width;
    let mut out = vec![0.0f32; MONOPOLE_GLOBAL_FEATURE_DIM];
    if h < 2 || w < 2 {
        return out;
    }

    let n = (h - 1) * (w - 1);
    let nf = n as f32;
    let mut qs: Vec<f32> = Vec::with_capacity(n);
    for pr in 0..h - 1 {
        for pc in 0..w - 1 {
            qs.push(plaquette_charge(grid, pr, pc));
        }
    }

    let sum_q: f32 = qs.iter().sum();
    let mean_q = sum_q / nf;
    let var_q: f32 = qs.iter().map(|&q| (q - mean_q).powi(2)).sum::<f32>() / nf;
    let max_abs = qs.iter().map(|q| q.abs()).fold(0.0f32, f32::max);
    let min_q = qs.iter().copied().fold(f32::INFINITY, f32::min);
    let max_q = qs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let m3: f32 = qs.iter().map(|&q| (q - mean_q).powi(3)).sum::<f32>() / nf;
    let skew = m3 / (var_q.powf(1.5) + 1e-8);
    let strong = qs.iter().filter(|&&q| q.abs() > 1e-3).count() as f32 / nf;
    let sign_bal: f32 = qs.iter().map(|&q| q.signum()).sum::<f32>() / nf;

    let sum_abs: f32 = qs.iter().map(|q| q.abs()).sum::<f32>() + 1e-8;
    let mut color_touch = [0.0f32; NUM_COLORS];
    for pr in 0..h - 1 {
        for pc in 0..w - 1 {
            let q = qs[pr * (w - 1) + pc].abs();
            for col in 0..NUM_COLORS as u8 {
                if plaquette_touches_color(grid, pr, pc, col) {
                    color_touch[col as usize] += q;
                }
            }
        }
    }

    let mut rms_acc = 0.0f32;
    let mut w_abs_sum = 0.0f32;
    let mut wc_r = 0.0f32;
    let mut wc_c = 0.0f32;
    let inv_h = 1.0f32 / h.max(1) as f32;
    let inv_w = 1.0f32 / w.max(1) as f32;
    for pr in 0..h - 1 {
        for pc in 0..w - 1 {
            let q = qs[pr * (w - 1) + pc];
            rms_acc += q * q;
            let wgt = q.abs();
            let cr = (pr as f32 + 0.5) * inv_h;
            let cc = (pc as f32 + 0.5) * inv_w;
            wc_r += wgt * cr;
            wc_c += wgt * cc;
            w_abs_sum += wgt;
        }
    }
    let rms = (rms_acc / nf).sqrt();
    let (cent_r, cent_c) = if w_abs_sum > 1e-8 {
        (wc_r / w_abs_sum, wc_c / w_abs_sum)
    } else {
        (0.5, 0.5)
    };
    let mut spread_r = 0.0f32;
    let mut spread_c = 0.0f32;
    if w_abs_sum > 1e-8 {
        for pr in 0..h - 1 {
            for pc in 0..w - 1 {
                let wgt = qs[pr * (w - 1) + pc].abs();
                let cr = (pr as f32 + 0.5) * inv_h;
                let cc = (pc as f32 + 0.5) * inv_w;
                spread_r += wgt * (cr - cent_r).powi(2);
                spread_c += wgt * (cc - cent_c).powi(2);
            }
        }
        spread_r = (spread_r / w_abs_sum).sqrt();
        spread_c = (spread_c / w_abs_sum).sqrt();
    }

    out[0] = sum_q / nf;
    out[1] = mean_q;
    out[2] = var_q;
    out[3] = max_abs;
    out[4] = min_q;
    out[5] = max_q;
    out[6] = skew;
    out[7] = strong;
    out[8] = sign_bal;
    for i in 0..NUM_COLORS {
        out[9 + i] = color_touch[i] / sum_abs;
    }
    out[19] = rms;
    out[20] = cent_r;
    out[21] = cent_c;
    out[22] = spread_r;
    out[23] = spread_c;
    out[24] = (nf + 1.0).ln() / 6.0;
    out[25] = if max_abs > 1e-12 {
        var_q / max_abs
    } else {
        0.0
    };
    out
}

/// Local monopole context for a cell from adjacent plaquette charges.
pub fn cell_monopole_features(grid: &Grid, r: usize, c: usize) -> Vec<f32> {
    let h = grid.height;
    let w = grid.width;
    let mut adj: Vec<f32> = Vec::with_capacity(4);
    if r + 1 < h && c + 1 < w {
        adj.push(plaquette_charge(grid, r, c));
    }
    if c > 0 && r + 1 < h {
        adj.push(plaquette_charge(grid, r, c - 1));
    }
    if r > 0 && c + 1 < w {
        adj.push(plaquette_charge(grid, r - 1, c));
    }
    if r > 0 && c > 0 {
        adj.push(plaquette_charge(grid, r - 1, c - 1));
    }
    if adj.is_empty() {
        return vec![0.0f32; MONOPOLE_CELL_FEATURE_DIM];
    }
    let n = adj.len() as f32;
    let mean: f32 = adj.iter().sum::<f32>() / n;
    let max_abs = adj.iter().map(|q| q.abs()).fold(0.0f32, f32::max);
    let var: f32 = adj.iter().map(|&q| (q - mean).powi(2)).sum::<f32>() / n;
    vec![mean, max_abs, var.sqrt(), n / 4.0]
}

// ─── Particle topology: per-color spatial structure of the entire grid ────────

/// Flood-fill to measure a connected component. Returns component size.
fn flood_fill_size(
    grid: &Grid,
    visited: &mut Vec<Vec<bool>>,
    sr: usize,
    sc: usize,
    color: u8,
) -> usize {
    let mut stack = vec![(sr, sc)];
    let mut size = 0;
    while let Some((r, c)) = stack.pop() {
        if r >= grid.height || c >= grid.width {
            continue;
        }
        if visited[r][c] || grid.cells[r][c] != color {
            continue;
        }
        visited[r][c] = true;
        size += 1;
        if r > 0 {
            stack.push((r - 1, c));
        }
        if r + 1 < grid.height {
            stack.push((r + 1, c));
        }
        if c > 0 {
            stack.push((r, c - 1));
        }
        if c + 1 < grid.width {
            stack.push((r, c + 1));
        }
    }
    size
}

/// Full topology of a grid: per-color spatial statistics.
/// Returns a fixed-size feature vector capturing the particle field topology.
fn grid_topology_features(grid: &Grid) -> Vec<f32> {
    let h = grid.height;
    let w = grid.width;
    let total = (h * w) as f32;
    let mut features = Vec::with_capacity(NUM_COLORS * 8 + 20);

    let mut visited = vec![vec![false; w]; h];

    for color in 0..NUM_COLORS as u8 {
        let mut count = 0u32;
        let mut sum_r = 0.0f32;
        let mut sum_c = 0.0f32;
        let mut sum_r2 = 0.0f32;
        let mut sum_c2 = 0.0f32;
        let mut min_r = h;
        let mut max_r = 0usize;
        let mut min_c = w;
        let mut max_c = 0usize;

        for r in 0..h {
            for c in 0..w {
                if grid.cells[r][c] == color {
                    count += 1;
                    let rf = r as f32 / h.max(1) as f32;
                    let cf = c as f32 / w.max(1) as f32;
                    sum_r += rf;
                    sum_c += cf;
                    sum_r2 += rf * rf;
                    sum_c2 += cf * cf;
                    if r < min_r {
                        min_r = r;
                    }
                    if r > max_r {
                        max_r = r;
                    }
                    if c < min_c {
                        min_c = c;
                    }
                    if c > max_c {
                        max_c = c;
                    }
                }
            }
        }

        let frac = count as f32 / total.max(1.0);
        features.push(frac);

        if count > 0 {
            let n = count as f32;
            let centroid_r = sum_r / n;
            let centroid_c = sum_c / n;
            let var_r = (sum_r2 / n - centroid_r * centroid_r).max(0.0);
            let var_c = (sum_c2 / n - centroid_c * centroid_c).max(0.0);
            features.push(centroid_r);
            features.push(centroid_c);
            features.push(var_r.sqrt()); // spread_r
            features.push(var_c.sqrt()); // spread_c
                                         // Bbox span (normalized)
            features.push((max_r - min_r) as f32 / h.max(1) as f32);
            features.push((max_c - min_c) as f32 / w.max(1) as f32);
        } else {
            features.extend_from_slice(&[0.0; 6]);
        }

        // Connected components
        let mut n_components = 0u32;
        let mut largest_component = 0usize;
        // Reset visited for this color
        for r in 0..h {
            for c in 0..w {
                visited[r][c] = false;
            }
        }
        for r in 0..h {
            for c in 0..w {
                if grid.cells[r][c] == color && !visited[r][c] {
                    let sz = flood_fill_size(grid, &mut visited, r, c, color);
                    n_components += 1;
                    if sz > largest_component {
                        largest_component = sz;
                    }
                }
            }
        }
        features.push(n_components as f32 / 10.0);
        let largest_frac = if count > 0 {
            largest_component as f32 / count as f32
        } else {
            0.0
        };
        features.push(largest_frac);
    }

    // Color adjacency: for each color, how many other colors touch it?
    let mut adj_matrix = [[false; NUM_COLORS]; NUM_COLORS];
    for r in 0..h {
        for c in 0..w {
            let color = grid.cells[r][c] as usize;
            if r + 1 < h {
                let nb = grid.cells[r + 1][c] as usize;
                if nb != color {
                    adj_matrix[color][nb] = true;
                    adj_matrix[nb][color] = true;
                }
            }
            if c + 1 < w {
                let nb = grid.cells[r][c + 1] as usize;
                if nb != color {
                    adj_matrix[color][nb] = true;
                    adj_matrix[nb][color] = true;
                }
            }
        }
    }
    for color in 0..NUM_COLORS {
        let n_adj = adj_matrix[color].iter().filter(|&&v| v).count();
        features.push(n_adj as f32 / (NUM_COLORS - 1) as f32);
    }

    // Grid-level symmetry
    let h_sym = (0..h).all(|r| (0..w).all(|c| grid.cells[r][c] == grid.cells[r][w - 1 - c]));
    let v_sym = (0..h).all(|r| (0..w).all(|c| grid.cells[r][c] == grid.cells[h - 1 - r][c]));
    let rot_sym =
        h == w && (0..h).all(|r| (0..w).all(|c| grid.cells[r][c] == grid.cells[w - 1 - c][r]));
    features.push(if h_sym { 1.0 } else { 0.0 });
    features.push(if v_sym { 1.0 } else { 0.0 });
    features.push(if rot_sym { 1.0 } else { 0.0 });

    // Distinct color count
    let distinct = (0..NUM_COLORS as u8)
        .filter(|&c| grid.cells.iter().any(|row| row.contains(&c)))
        .count();
    features.push(distinct as f32 / NUM_COLORS as f32);

    features
}

/// Compute per-cell charges: boundary (borders different color), connectivity (same-color neighbors).
fn cell_charges(grid: &Grid, r: usize, c: usize) -> (f32, f32) {
    let color = grid.cells[r][c];
    let h = grid.height;
    let w = grid.width;
    let mut same = 0u8;
    let mut diff = 0u8;
    if r > 0 {
        if grid.cells[r - 1][c] == color {
            same += 1;
        } else {
            diff += 1;
        }
    }
    if r + 1 < h {
        if grid.cells[r + 1][c] == color {
            same += 1;
        } else {
            diff += 1;
        }
    }
    if c > 0 {
        if grid.cells[r][c - 1] == color {
            same += 1;
        } else {
            diff += 1;
        }
    }
    if c + 1 < w {
        if grid.cells[r][c + 1] == color {
            same += 1;
        } else {
            diff += 1;
        }
    }
    let boundary = if diff > 0 { 1.0 } else { -1.0 }; // +1 boundary, -1 interior
    let connectivity = same as f32 / 4.0;
    (boundary, connectivity)
}

/// Per-color charged field in a block. Each particle's position spinor is scaled
/// by its boundary charge: boundary particles add positively, interior negatively.
/// Returns (field_strength, charged_strength, count, boundary_count) per color.
struct ColorBlockField {
    strength: [f32; NUM_COLORS],
    charged_strength: [f32; NUM_COLORS],
    count: [u32; NUM_COLORS],
    boundary_count: [u32; NUM_COLORS],
    connectivity_sum: [f32; NUM_COLORS],
}

fn color_layer_block_charged(
    grid: &Grid,
    r0: usize,
    c0: usize,
    bh: usize,
    bw: usize,
) -> ColorBlockField {
    let mut accums = [
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
    ];
    let mut charged_accums = [
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
    ];
    let mut counts = [0u32; NUM_COLORS];
    let mut boundary_counts = [0u32; NUM_COLORS];
    let mut conn_sums = [0.0f32; NUM_COLORS];

    for dr in 0..bh {
        for dc in 0..bw {
            let r = (r0 + dr).min(grid.height - 1);
            let c = (c0 + dc).min(grid.width - 1);
            let color = grid.cells[r][c] as usize;
            if color < NUM_COLORS {
                let pos = grid_position_spinor(r, c, grid.height, grid.width);
                let (boundary_charge, connectivity) = cell_charges(grid, r, c);
                accums[color] = accums[color].add(&pos);
                charged_accums[color] = charged_accums[color].add(&pos.scale(boundary_charge));
                counts[color] += 1;
                if boundary_charge > 0.0 {
                    boundary_counts[color] += 1;
                }
                conn_sums[color] += connectivity;
            }
        }
    }

    let mut strength = [0.0f32; NUM_COLORS];
    let mut charged_strength = [0.0f32; NUM_COLORS];
    for c in 0..NUM_COLORS {
        if counts[c] > 0 {
            strength[c] = accums[c].component_norm();
            charged_strength[c] = charged_accums[c].component_norm();
        }
    }

    ColorBlockField {
        strength,
        charged_strength,
        count: counts,
        boundary_count: boundary_counts,
        connectivity_sum: conn_sums,
    }
}

/// Per-color charged field in a neighborhood.
fn color_layer_neighborhood_charged(
    grid: &Grid,
    cr: usize,
    cc: usize,
    radius: usize,
) -> ColorBlockField {
    let r_start = if cr >= radius { cr - radius } else { 0 };
    let c_start = if cc >= radius { cc - radius } else { 0 };
    let r_end = (cr + radius + 1).min(grid.height);
    let c_end = (cc + radius + 1).min(grid.width);

    let mut accums = [
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
    ];
    let mut charged_accums = [
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
        Multivector::zero(),
    ];
    let mut counts = [0u32; NUM_COLORS];
    let mut boundary_counts = [0u32; NUM_COLORS];
    let mut conn_sums = [0.0f32; NUM_COLORS];

    for r in r_start..r_end {
        for c in c_start..c_end {
            let color = grid.cells[r][c] as usize;
            if color < NUM_COLORS {
                let pos = grid_position_spinor(r, c, grid.height, grid.width);
                let (boundary_charge, connectivity) = cell_charges(grid, r, c);
                accums[color] = accums[color].add(&pos);
                charged_accums[color] = charged_accums[color].add(&pos.scale(boundary_charge));
                counts[color] += 1;
                if boundary_charge > 0.0 {
                    boundary_counts[color] += 1;
                }
                conn_sums[color] += connectivity;
            }
        }
    }

    let mut strength = [0.0f32; NUM_COLORS];
    let mut charged_strength = [0.0f32; NUM_COLORS];
    for c in 0..NUM_COLORS {
        if counts[c] > 0 {
            strength[c] = accums[c].component_norm();
            charged_strength[c] = charged_accums[c].component_norm();
        }
    }

    ColorBlockField {
        strength,
        charged_strength,
        count: counts,
        boundary_count: boundary_counts,
        connectivity_sum: conn_sums,
    }
}

/// Build a COMPACT feature vector from charged particle fields.
/// Designed for few-shot learning: keep features low-dimensional but maximally
/// discriminative via particle charges and local Wilson-plaquette context.
fn dirac_cell_features(
    grid: &Grid,
    out_r: usize,
    out_c: usize,
    oh: usize,
    ow: usize,
    mode: DiracMode,
) -> Vec<f32> {
    let ih = grid.height;
    let iw = grid.width;
    // Base ~42 / ~17 / ~26 + [`MONOPOLE_CELL_FEATURE_DIM`] monopole locals
    let mut features = Vec::with_capacity(50 + MONOPOLE_CELL_FEATURE_DIM);

    match mode {
        DiracMode::Shrink { bh, bw } => {
            let r0 = out_r * bh;
            let c0 = out_c * bw;
            let field = color_layer_block_charged(grid, r0, c0, bh, bw);
            let block_size = (bh * bw) as f32;

            // Core: per-color fraction in block (10 features — the essential signal)
            for c in 0..NUM_COLORS {
                features.push(field.count[c] as f32 / block_size);
            }

            // Charged: per-color boundary fraction (how much of this color is at edges?)
            for c in 0..NUM_COLORS {
                features.push(if field.count[c] > 0 {
                    field.boundary_count[c] as f32 / field.count[c] as f32
                } else {
                    0.0
                });
            }

            // Connectivity: mean connectivity per color (clustered vs scattered)
            for c in 0..NUM_COLORS {
                features.push(if field.count[c] > 0 {
                    field.connectivity_sum[c] / field.count[c] as f32
                } else {
                    0.0
                });
            }

            // Charged field polarity: sign of charged vs uncharged strength
            // Positive = boundary-dominated, negative = interior-dominated
            for c in 0..NUM_COLORS {
                if field.strength[c] > 1e-8 {
                    features.push(field.charged_strength[c] / field.strength[c]);
                } else {
                    features.push(0.0);
                }
            }

            // Position (2 features)
            features.push(out_r as f32 / oh.max(1) as f32);
            features.push(out_c as f32 / ow.max(1) as f32);

            // Monopole / Wilson curvature at block center on the input grid
            let mon_r = r0 + (bh.saturating_sub(1)) / 2;
            let mon_c = c0 + (bw.saturating_sub(1)) / 2;
            features.extend(cell_monopole_features(grid, mon_r, mon_c));
        }

        DiracMode::Grow { sh, sw } => {
            let src_r = (out_r / sh).min(ih - 1);
            let src_c = (out_c / sw).min(iw - 1);
            let sub_r = out_r % sh;
            let sub_c = out_c % sw;

            let src_color = grid.cells[src_r][src_c];
            let (boundary, connectivity) = cell_charges(grid, src_r, src_c);

            // Source cell: color + charges (3 features)
            features.push(src_color as f32 / 9.0);
            features.push(boundary);
            features.push(connectivity);

            // Neighborhood charged field (compact: just the source color's charges)
            let nbr = color_layer_neighborhood_charged(grid, src_r, src_c, 1);
            features.push(nbr.count[src_color as usize] as f32 / 9.0);
            features.push(if nbr.count[src_color as usize] > 0 {
                nbr.boundary_count[src_color as usize] as f32 / nbr.count[src_color as usize] as f32
            } else {
                0.0
            });
            // How many distinct colors in neighborhood
            let distinct_nbr = nbr.count.iter().filter(|&&v| v > 0).count();
            features.push(distinct_nbr as f32 / NUM_COLORS as f32);

            // Sub-position (6 features)
            features.push(sub_r as f32 / sh.max(1) as f32);
            features.push(sub_c as f32 / sw.max(1) as f32);
            features.push(if sub_r == 0 { 1.0 } else { 0.0 });
            features.push(if sub_c == 0 { 1.0 } else { 0.0 });
            features.push(if sub_r == sh - 1 { 1.0 } else { 0.0 });
            features.push(if sub_c == sw - 1 { 1.0 } else { 0.0 });

            features.push(if src_color == 0 { 1.0 } else { 0.0 });

            features.extend(cell_monopole_features(grid, src_r, src_c));
        }

        DiracMode::SameDim | DiracMode::Arbitrary => {
            let (ir, ic) = if oh == ih && ow == iw {
                (out_r, out_c)
            } else {
                let fr = if oh > 1 {
                    out_r * (ih - 1) / (oh - 1).max(1)
                } else {
                    ih / 2
                };
                let fc = if ow > 1 {
                    out_c * (iw - 1) / (ow - 1).max(1)
                } else {
                    iw / 2
                };
                (fr.min(ih - 1), fc.min(iw - 1))
            };

            let center = grid.cells[ir][ic];
            let (boundary, connectivity) = cell_charges(grid, ir, ic);

            // Center cell charges (3 features)
            features.push(center as f32 / 9.0);
            features.push(boundary);
            features.push(connectivity);

            // Neighborhood charged field (radius 2)
            let nbr = color_layer_neighborhood_charged(grid, ir, ic, 2);
            for c in 0..NUM_COLORS {
                features.push(nbr.count[c] as f32 / 25.0); // 5x5 neighborhood
            }
            for c in 0..NUM_COLORS {
                features.push(if nbr.count[c] > 0 {
                    nbr.connectivity_sum[c] / nbr.count[c] as f32
                } else {
                    0.0
                });
            }

            // Position (3 features)
            features.push(out_r as f32 / oh.max(1) as f32);
            features.push(out_c as f32 / ow.max(1) as f32);
            features.push(if ih == oh && iw == ow { 1.0 } else { 0.0 });

            features.extend(cell_monopole_features(grid, ir, ic));
        }
    }

    features
}

#[derive(Clone, Copy)]
enum DiracMode {
    Shrink { bh: usize, bw: usize },
    Grow { sh: usize, sw: usize },
    SameDim,
    Arbitrary,
}

/// Train Dirac cell field on `task.train`, then predict outputs for each grid in `inputs`
/// (same output shape `(oh,ow)` as the first train pair).
pub fn solve_task_dirac_predict_inputs(task: &ArcTask, inputs: &[Grid]) -> Option<Vec<Grid>> {
    if task.train.is_empty() || inputs.is_empty() {
        return None;
    }

    let ex0 = &task.train[0];
    let ih = ex0.input.height;
    let iw = ex0.input.width;
    let oh = ex0.output.height;
    let ow = ex0.output.width;

    if ih == 0 || iw == 0 || oh == 0 || ow == 0 {
        return None;
    }

    let mode = if ih == oh && iw == ow {
        DiracMode::SameDim
    } else if oh < ih && ow < iw && ih % oh == 0 && iw % ow == 0 {
        DiracMode::Shrink {
            bh: ih / oh,
            bw: iw / ow,
        }
    } else if oh > ih && ow > iw && oh % ih == 0 && ow % iw == 0 {
        DiracMode::Grow {
            sh: oh / ih,
            sw: ow / iw,
        }
    } else {
        DiracMode::Arbitrary
    };

    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut targets: Vec<usize> = Vec::new();

    for ex in &task.train {
        for r in 0..ex.output.height {
            for c in 0..ex.output.width {
                embeddings.push(dirac_cell_features(&ex.input, r, c, oh, ow, mode));
                targets.push(ex.output.cells[r][c] as usize);
            }
        }
    }

    if embeddings.is_empty() {
        return None;
    }

    let max_len = embeddings.iter().map(|e| e.len()).max().unwrap_or(0);
    for emb in &mut embeddings {
        emb.resize(max_len, 0.0);
    }

    let class_names: Vec<String> = (0..NUM_COLORS).map(|c| format!("c{}", c)).collect();
    let sample_refs: Vec<(&[f32], usize)> = embeddings
        .iter()
        .zip(targets.iter())
        .map(|(e, &t)| (e.as_slice(), t))
        .collect();

    let brain = MicroBrain::build_from_data(
        MicroBrainRole::Custom("dirac_field".into()),
        max_len,
        NUM_COLORS,
        class_names,
        &sample_refs,
    );

    let results: Vec<Grid> = inputs
        .iter()
        .map(|inp| {
            let mut cells = vec![vec![0u8; ow]; oh];
            let mut brain_local = brain.clone();
            for r in 0..oh {
                for c in 0..ow {
                    let mut emb = dirac_cell_features(inp, r, c, oh, ow, mode);
                    emb.resize(max_len, 0.0);
                    let (cls, _conf, _logits) = brain_local.predict(&emb);
                    cells[r][c] = cls as u8;
                }
            }
            Grid {
                cells,
                height: oh,
                width: ow,
            }
        })
        .collect();

    Some(results)
}

/// Dirac solver: train on `task.train`, predict `task.test`.
pub fn solve_task_dirac(task: &ArcTask) -> Option<Vec<Grid>> {
    if task.test.is_empty() {
        return None;
    }
    let inputs: Vec<Grid> = task.test.iter().map(|e| e.input.clone()).collect();
    solve_task_dirac_predict_inputs(task, &inputs)
}

// ─── Learned solver labels for ArcBrain (no hand-coded `solve_task` routing) ──

const ASTAR_ROUTER_MS: u64 = 3000;

fn train_cell_accuracy_fraction(preds: &[Grid], task: &ArcTask) -> Option<f32> {
    if preds.len() != task.train.len() {
        return None;
    }
    let mut correct = 0usize;
    let mut total = 0usize;
    for (ex, g) in task.train.iter().zip(preds.iter()) {
        let oh = ex.output.height;
        let ow = ex.output.width;
        if g.height != oh || g.width != ow {
            total += oh * ow;
            continue;
        }
        for r in 0..oh {
            for c in 0..ow {
                total += 1;
                if g.cells[r][c] == ex.output.cells[r][c] {
                    correct += 1;
                }
            }
        }
    }
    if total == 0 {
        return None;
    }
    Some(correct as f32 / total as f32)
}

fn learned_solver_tie_priority(name: &str) -> u8 {
    match name {
        "astar" => 0,
        "dirac" => 1,
        "onepass" => 2,
        _ => 3,
    }
}

/// Best learned solver by fit on training examples (resubstitution). Ties: astar > dirac > onepass.
/// Returns `"none"` if every candidate scores 0 on train.
pub fn pick_learned_solver_label(task: &ArcTask) -> String {
    if task.train.is_empty() {
        return "none".into();
    }

    let s_astar = if !task.test.is_empty() && astar_dsl_solve(task, 3, ASTAR_ROUTER_MS).is_some() {
        1.0f32
    } else {
        0.0f32
    };

    let train_inputs: Vec<Grid> = task.train.iter().map(|e| e.input.clone()).collect();
    let s_dirac = solve_task_dirac_predict_inputs(task, &train_inputs)
        .and_then(|preds| train_cell_accuracy_fraction(&preds, task))
        .unwrap_or(0.0f32);

    let s_one = solve_task_onepass_for_examples(task, &task.train)
        .and_then(|preds| train_cell_accuracy_fraction(&preds, task))
        .unwrap_or(0.0f32);

    let mut best_name = "astar";
    let mut best_score = f32::NEG_INFINITY;
    for &(name, sc) in &[("astar", s_astar), ("dirac", s_dirac), ("onepass", s_one)] {
        if sc > best_score + 1e-7
            || ((sc - best_score).abs() <= 1e-7
                && learned_solver_tie_priority(name) < learned_solver_tie_priority(best_name))
        {
            best_score = sc;
            best_name = name;
        }
    }

    if best_score <= 0.0 {
        "none".into()
    } else {
        best_name.into()
    }
}

/// Run a named learned solver on `task.test` (`astar` | `dirac` | `onepass`).
pub fn apply_learned_solver(task: &ArcTask, name: &str) -> Option<Vec<Grid>> {
    if task.test.is_empty() {
        return None;
    }
    match name {
        "astar" => astar_dsl_solve(task, 3, ASTAR_ROUTER_MS).map(|(g, _)| g),
        "dirac" => solve_task_dirac(task),
        "onepass" => solve_task_onepass(task),
        _ => None,
    }
}

/// Zero-parameter Dirac: try multiple physics-based decode rules on SHRINK blocks.
/// Rules: (1) max field strength, (2) max non-background field strength,
/// (3) majority count, (4) majority non-background count, (5) min non-zero field.
/// Validates each rule on training, returns first that passes.
pub fn solve_task_dirac_direct(task: &ArcTask) -> Option<Vec<Grid>> {
    if task.train.is_empty() || task.test.is_empty() {
        return None;
    }

    let ex0 = &task.train[0];
    let ih = ex0.input.height;
    let iw = ex0.input.width;
    let oh = ex0.output.height;
    let ow = ex0.output.width;

    if oh >= ih || ow >= iw || ih % oh != 0 || iw % ow != 0 {
        return None;
    }

    // Detect background color (most common in input)
    let mut global_counts = [0u32; NUM_COLORS];
    for row in &ex0.input.cells {
        for &v in row {
            global_counts[v as usize] += 1;
        }
    }
    let bg = global_counts
        .iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Decode rules: each takes the full charged field + bg color, returns predicted color.
    // Enumerate many physics-based observables of the particle field.
    type DecodeRule = fn(&ColorBlockField, usize) -> u8;

    let rules: Vec<(DecodeRule, &str)> = vec![
        // Count-based
        (
            |f, _| {
                f.count
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &v)| v)
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0)
            },
            "majority",
        ),
        (
            |f, bg| {
                f.count
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != bg)
                    .max_by_key(|(_, &v)| v)
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0)
            },
            "majority_nobg",
        ),
        (
            |f, bg| {
                f.count
                    .iter()
                    .enumerate()
                    .filter(|&(i, &v)| i != bg && v > 0)
                    .min_by_key(|(_, &v)| v)
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0)
            },
            "minority_nobg",
        ),
        // Field strength (spatial coherence weighted)
        (
            |f, _| {
                f.strength
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0)
            },
            "max_field",
        ),
        (
            |f, bg| {
                f.strength
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != bg)
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0)
            },
            "max_field_nobg",
        ),
        (
            |f, bg| {
                f.strength
                    .iter()
                    .enumerate()
                    .filter(|&(i, v)| i != bg && *v > 0.0)
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0)
            },
            "min_field_nobg",
        ),
        // Charged field (boundary-dominated particles)
        (
            |f, bg| {
                f.charged_strength
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != bg)
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0)
            },
            "max_charged_nobg",
        ),
        // Boundary particles only
        (
            |f, bg| {
                f.boundary_count
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != bg)
                    .max_by_key(|(_, &v)| v)
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0)
            },
            "max_boundary_nobg",
        ),
        // Interior particles only (total - boundary)
        (
            |f, bg| {
                let mut best = 0u8;
                let mut best_v = 0u32;
                for i in 0..NUM_COLORS {
                    if i == bg {
                        continue;
                    }
                    let interior = f.count[i].saturating_sub(f.boundary_count[i]);
                    if interior > best_v {
                        best_v = interior;
                        best = i as u8;
                    }
                }
                best
            },
            "max_interior_nobg",
        ),
        // Most connected color (highest mean connectivity)
        (
            |f, bg| {
                let mut best = 0u8;
                let mut best_v = 0.0f32;
                for i in 0..NUM_COLORS {
                    if i == bg || f.count[i] == 0 {
                        continue;
                    }
                    let mean_conn = f.connectivity_sum[i] / f.count[i] as f32;
                    if mean_conn > best_v {
                        best_v = mean_conn;
                        best = i as u8;
                    }
                }
                best
            },
            "most_connected_nobg",
        ),
        // Unique color: appears exactly once in block
        (
            |f, bg| {
                for i in 0..NUM_COLORS {
                    if i != bg && f.count[i] == 1 {
                        return i as u8;
                    }
                }
                f.count
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != bg)
                    .max_by_key(|(_, &v)| v)
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0)
            },
            "unique_nobg",
        ),
        // Second most common (after bg)
        (
            |f, bg| {
                let mut sorted: Vec<(usize, u32)> = f
                    .count
                    .iter()
                    .enumerate()
                    .filter(|&(i, &v)| i != bg && v > 0)
                    .map(|(i, &v)| (i, v))
                    .collect();
                sorted.sort_by(|a, b| b.1.cmp(&a.1));
                if sorted.len() >= 2 {
                    sorted[1].0 as u8
                } else if !sorted.is_empty() {
                    sorted[0].0 as u8
                } else {
                    bg as u8
                }
            },
            "second_most_nobg",
        ),
    ];

    for &(rule_fn, _rule_name) in &rules {
        let mut valid = true;
        for ex in &task.train {
            let ebh = ex.input.height / ex.output.height;
            let ebw = ex.input.width / ex.output.width;
            for r in 0..ex.output.height {
                for c in 0..ex.output.width {
                    let field = color_layer_block_charged(&ex.input, r * ebh, c * ebw, ebh, ebw);
                    let pred = rule_fn(&field, bg);
                    if pred != ex.output.cells[r][c] {
                        valid = false;
                        break;
                    }
                }
                if !valid {
                    break;
                }
            }
            if !valid {
                break;
            }
        }

        if valid {
            let results: Vec<Grid> = task
                .test
                .iter()
                .map(|test_ex| {
                    let t_bh = test_ex.input.height / oh;
                    let t_bw = test_ex.input.width / ow;
                    let mut cells = vec![vec![0u8; ow]; oh];
                    for r in 0..oh {
                        for c in 0..ow {
                            let field = color_layer_block_charged(
                                &test_ex.input,
                                r * t_bh,
                                c * t_bw,
                                t_bh,
                                t_bw,
                            );
                            cells[r][c] = rule_fn(&field, bg);
                        }
                    }
                    Grid {
                        cells,
                        height: oh,
                        width: ow,
                    }
                })
                .collect();
            return Some(results);
        }
    }

    None
}

// ─── Load language runtime from trained brain.bin (read-only) ───────────────

fn load_language_runtime(brain_path: &str) -> Option<LanguageRuntime> {
    let bytes = match std::fs::read(brain_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "  [lang] Cannot read {}: {} — falling back to Clifford-only",
                brain_path, e
            );
            return None;
        }
    };
    let dm: DimensionManager =
        match crate::systems::checkpoint::deserialize_checkpoint_from_bytes(&bytes) {
            Ok(dm) => dm,
            Err(e) => {
                eprintln!(
                    "  [lang] Cannot deserialize {}: {} — falling back to Clifford-only",
                    brain_path, e
                );
                return None;
            }
        };
    let rt = dm.language_runtime;
    if !rt.bridge.calibrated {
        eprintln!(
            "  [lang] Bridge in {} not calibrated — falling back to Clifford-only",
            brain_path
        );
        return None;
    }
    println!(
        "  [lang] Loaded calibrated LanguageRuntime from {} (bridge {}→{}d)",
        brain_path, rt.bridge.input_dim, rt.bridge.output_dim
    );
    Some(rt)
}

// ─── ArcBrain ──────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct ArcBrain {
    pub strategy_router: MicroBrain,
    pub strategy_names: Vec<String>,
    pub neighborhood_size: usize,
    pub has_language_bridge: bool,
}

impl ArcBrain {
    pub fn train(tasks: &[ArcTask], lang_brain_path: &str) -> Self {
        println!("=== Training ArcBrain ===");

        // Load language runtime from brain.bin (read-only, never overwritten)
        println!(
            "\n--- Loading GrowformerLanguageEncoder from {} ---",
            lang_brain_path
        );
        let lang_rt = load_language_runtime(lang_brain_path);
        let has_bridge = lang_rt.is_some();
        if has_bridge {
            println!("  Language bridge: ACTIVE — cross-domain embeddings enabled");
        } else {
            println!("  Language bridge: INACTIVE — Clifford-only mode");
        }

        if tasks.is_empty() {
            println!("  WARNING: no tasks — returning stub ArcBrain");
            let mut rng = rand::thread_rng();
            return ArcBrain {
                strategy_router: MicroBrain::new(
                    MicroBrainRole::Custom("arc_strategy_router".into()),
                    256,
                    1,
                    64,
                    vec!["none".to_string()],
                    &mut rng,
                ),
                strategy_names: vec!["none".to_string()],
                neighborhood_size: NEIGHBORHOOD,
                has_language_bridge: has_bridge,
            };
        }

        // Phase 1: Label each task by best learned solver on training (A* / Dirac / one-pass)
        println!("\n--- Phase 1: Learned solver labels (train fit, no hand-coded pipeline) ---");
        let mut strategy_set: Vec<String> = Vec::new();
        let mut strategy_map: HashMap<String, usize> = HashMap::new();
        let mut router_embeddings: Vec<Vec<f32>> = Vec::new();
        let mut router_targets: Vec<usize> = Vec::new();
        let mut strategy_counts: HashMap<String, usize> = HashMap::new();

        for task in tasks {
            let strat = pick_learned_solver_label(task);
            *strategy_counts.entry(strat.clone()).or_default() += 1;

            let strat_idx = if let Some(&idx) = strategy_map.get(&strat) {
                idx
            } else {
                let idx = strategy_set.len();
                strategy_set.push(strat.clone());
                strategy_map.insert(strat, idx);
                idx
            };

            let task_emb = encode_task(task, lang_rt.as_ref());
            router_embeddings.push(task_emb);
            router_targets.push(strat_idx);
        }

        let n_useful = strategy_counts
            .iter()
            .filter(|(k, _)| *k != "none")
            .map(|(_, v)| v)
            .sum::<usize>();
        println!(
            "  Tasks labeled: {} (non-none best solver: {})",
            tasks.len(),
            n_useful
        );
        println!("  Strategy classes: {}", strategy_set.len());
        println!(
            "  Embedding dim: {}",
            router_embeddings.first().map_or(0, |e| e.len())
        );
        for (strat, count) in &strategy_counts {
            println!("    {}: {} tasks", strat, count);
        }

        // Phase 2: Build strategy router via one-pass Paramecium lattice
        println!("\n--- Phase 2: Strategy router (one-pass) ---");
        let sample_refs: Vec<(&[f32], usize)> = router_embeddings
            .iter()
            .zip(router_targets.iter())
            .map(|(e, &t)| (e.as_slice(), t))
            .collect();

        let router = MicroBrain::build_from_data(
            MicroBrainRole::Custom("arc_strategy_router".into()),
            router_embeddings.first().map_or(256, |e| e.len()),
            strategy_set.len(),
            strategy_set.clone(),
            &sample_refs,
        );

        println!(
            "  Router lattice programs: {}",
            router.lattice.program_count()
        );

        // Phase 3: Validate router on training set
        println!("\n--- Phase 3: Router self-validation ---");
        let mut router_clone = router.clone();
        let mut router_correct = 0;
        for (emb, &target) in router_embeddings.iter().zip(router_targets.iter()) {
            let (cls, _conf, _logits) = router_clone.predict(emb);
            if cls == target {
                router_correct += 1;
            }
        }
        println!(
            "  Router accuracy on labeled tasks: {}/{} ({:.1}%)",
            router_correct,
            router_embeddings.len(),
            router_correct as f32 / router_embeddings.len().max(1) as f32 * 100.0
        );

        println!("\n  Training complete.");

        ArcBrain {
            strategy_router: router,
            strategy_names: strategy_set,
            neighborhood_size: NEIGHBORHOOD,
            has_language_bridge: has_bridge,
        }
    }

    pub fn route(
        &mut self,
        task: &ArcTask,
        lang_rt: Option<&LanguageRuntime>,
    ) -> (usize, f32, String) {
        let emb = encode_task(task, lang_rt);
        let (cls, conf, _logits) = self.strategy_router.predict(&emb);
        let name = self
            .strategy_names
            .get(cls)
            .cloned()
            .unwrap_or_else(|| "unknown".into());
        (cls, conf, name)
    }

    /// Predict best learned solver from task embedding, then run it on `task.test`.
    pub fn solve_routed(
        &mut self,
        task: &ArcTask,
        lang_rt: Option<&LanguageRuntime>,
    ) -> Option<Vec<Grid>> {
        let (_cls, _conf, name) = self.route(task, lang_rt);
        apply_learned_solver(task, name.as_str())
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let bytes = crate::systems::checkpoint::serialize_checkpoint_to_bytes(self)?;
        std::fs::write(path, bytes).map_err(|e| format!("save failed: {}", e))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read failed: {}", e))?;
        crate::systems::checkpoint::deserialize_checkpoint_from_bytes(&bytes)
    }
}

// ─── Benchmark ─────────────────────────────────────────────────────────────

fn grids_match(a: &Grid, b: &Grid) -> bool {
    if a.height != b.height || a.width != b.width {
        return false;
    }
    for r in 0..a.height {
        for c in 0..a.width {
            if a.cells[r][c] != b.cells[r][c] {
                return false;
            }
        }
    }
    true
}

pub fn benchmark(brain: &mut ArcBrain, tasks: &[ArcTask], lang_rt: Option<&LanguageRuntime>) {
    let mut solved_handcoded = 0;
    let mut solved_pertask = 0;
    let mut solved_combined = 0;
    let mut total = 0;

    let mut router_correct = 0;
    let mut router_total = 0;

    // Flow diagnostic accumulators
    let mut flow_converging_solved = 0u32;
    let mut flow_converging_total = 0u32;
    let mut flow_diverging_solved = 0u32;
    let mut flow_diverging_total = 0u32;
    let mut flow_degenerate_count = 0u32;
    let mut spatial_bias_sum_solved = 0.0f32;
    let mut spatial_bias_sum_unsolved = 0.0f32;
    let mut n_solved_flow = 0u32;
    let mut n_unsolved_flow = 0u32;

    // Expert solver accumulators
    let mut verified_high = 0u32;
    let mut verified_low = 0u32;
    let mut verified_sum = 0.0f32;
    let mut verified_count = 0u32;
    let mut decomp_geometric = 0u32;
    let mut decomp_causal = 0u32;
    let mut decomp_compositional = 0u32;
    let mut decomp_unknown = 0u32;

    let same_dim_count = tasks
        .iter()
        .filter(|t| {
            t.train
                .iter()
                .all(|ex| ex.input.height == ex.output.height && ex.input.width == ex.output.width)
        })
        .count();

    println!("\n=== ArcBrain Benchmark ===");
    println!("Total tasks: {}, same-dim: {}", tasks.len(), same_dim_count);
    println!(
        "Language bridge: {}",
        if lang_rt.is_some() {
            "ACTIVE"
        } else {
            "INACTIVE"
        }
    );
    println!("Strategies known: {:?}", brain.strategy_names);

    for task in tasks {
        total += 1;

        let diag = solve_task(task);
        let hc_solved = diag.solved;
        if hc_solved {
            solved_handcoded += 1;
        }

        if let Some(ref f) = diag.flow {
            if f.converging {
                flow_converging_total += 1;
                if hc_solved {
                    flow_converging_solved += 1;
                }
            } else {
                flow_diverging_total += 1;
                if hc_solved {
                    flow_diverging_solved += 1;
                }
            }
            if f.is_degenerate() {
                flow_degenerate_count += 1;
            }
            if hc_solved {
                spatial_bias_sum_solved += f.spatial_bias();
                n_solved_flow += 1;
            } else {
                spatial_bias_sum_unsolved += f.spatial_bias();
                n_unsolved_flow += 1;
            }
        }

        if let Some(ref v) = diag.verification {
            verified_count += 1;
            verified_sum += v.confidence;
            if v.confidence > 0.6 {
                verified_high += 1;
            } else {
                verified_low += 1;
            }
        }
        match diag.decomposition {
            Some(crate::arc_agi::TransformationType::Geometric) => decomp_geometric += 1,
            Some(crate::arc_agi::TransformationType::Causal) => decomp_causal += 1,
            Some(crate::arc_agi::TransformationType::Compositional) => decomp_compositional += 1,
            Some(crate::arc_agi::TransformationType::Unknown) => decomp_unknown += 1,
            None => {}
        }

        let (_cls, conf, predicted_strategy) = brain.route(task, lang_rt);

        let oracle_learned = pick_learned_solver_label(task);
        router_total += 1;
        if predicted_strategy == oracle_learned {
            router_correct += 1;
        }

        let lattice_solved = if let Some(preds) = solve_task_onepass(task) {
            task.test
                .iter()
                .enumerate()
                .all(|(i, test_ex)| i < preds.len() && grids_match(&preds[i], &test_ex.output))
        } else {
            false
        };

        if lattice_solved {
            solved_pertask += 1;
        }

        let either = hc_solved || lattice_solved;
        if either {
            solved_combined += 1;
        }

        if lattice_solved && !hc_solved {
            println!(
                "  NEW (lattice only): {} [router predicted: {} conf={:.2}]",
                task.id, predicted_strategy, conf
            );
        }
    }

    let router_acc = if router_total > 0 {
        router_correct as f32 / router_total as f32 * 100.0
    } else {
        0.0
    };

    println!("\n--- Results ---");
    println!("Tasks evaluated: {}", total);
    println!(
        "Hand-coded pipeline: {}/{} solved ({:.1}%)",
        solved_handcoded,
        total,
        solved_handcoded as f32 / total.max(1) as f32 * 100.0
    );
    println!(
        "Per-task lattice:    {}/{} solved ({:.1}%)",
        solved_pertask,
        total,
        solved_pertask as f32 / total.max(1) as f32 * 100.0
    );
    println!(
        "Combined (either):   {}/{} solved ({:.1}%)",
        solved_combined,
        total,
        solved_combined as f32 / total.max(1) as f32 * 100.0
    );
    println!(
        "Strategy router:     {}/{} match oracle learned label ({:.1}%)",
        router_correct, router_total, router_acc
    );

    // Probability flow analysis
    println!("\n--- Flow Diagnostic (Schrödinger continuity) ---");
    println!(
        "Converging flow:  {}/{} solved ({:.1}%)",
        flow_converging_solved,
        flow_converging_total,
        flow_converging_solved as f32 / flow_converging_total.max(1) as f32 * 100.0
    );
    println!(
        "Diverging flow:   {}/{} solved ({:.1}%)",
        flow_diverging_solved,
        flow_diverging_total,
        flow_diverging_solved as f32 / flow_diverging_total.max(1) as f32 * 100.0
    );
    println!(
        "Degenerate (|B|≈0): {}/{} tasks",
        flow_degenerate_count, total
    );
    if n_solved_flow > 0 {
        println!(
            "Mean spatial bias (solved):   {:.3}",
            spatial_bias_sum_solved / n_solved_flow as f32
        );
    }
    if n_unsolved_flow > 0 {
        println!(
            "Mean spatial bias (unsolved): {:.3}",
            spatial_bias_sum_unsolved / n_unsolved_flow as f32
        );
    }

    // Expert solver analysis
    println!("\n--- Expert Solver ---");
    println!(
        "Verified solutions: {}/{} (high conf: {}, low conf: {})",
        verified_count, total, verified_high, verified_low
    );
    if verified_count > 0 {
        println!(
            "Mean verification confidence: {:.3}",
            verified_sum / verified_count as f32
        );
    }
    println!(
        "Decomposition: geometric={}, causal={}, compositional={}, unknown={}",
        decomp_geometric, decomp_causal, decomp_compositional, decomp_unknown
    );
}

#[cfg(test)]
mod monopole_phase1_tests {
    use super::*;
    use crate::arc_agi::{ArcExample, ArcTask};

    fn grid_2x2(cells: [[u8; 2]; 2]) -> Grid {
        Grid {
            cells: vec![cells[0].to_vec(), cells[1].to_vec()],
            height: 2,
            width: 2,
        }
    }

    #[test]
    fn monopole_global_dim_matches_vector() {
        let g = grid_2x2([[1, 2], [3, 4]]);
        let f = monopole_field_features(&g);
        assert_eq!(f.len(), MONOPOLE_GLOBAL_FEATURE_DIM);
    }

    #[test]
    fn monopole_cell_dim_matches_vector() {
        let g = grid_2x2([[1, 1], [1, 1]]);
        let f = cell_monopole_features(&g, 0, 0);
        assert_eq!(f.len(), MONOPOLE_CELL_FEATURE_DIM);
    }

    #[test]
    fn monopole_wilson_loop_is_finite() {
        let g = grid_2x2([[5, 5], [5, 5]]);
        let w = plaquette_wilson_loop(&g, 0, 0);
        assert!(w.component_norm().is_finite());
        let q = plaquette_charge(&g, 0, 0);
        assert!(q.is_finite());
    }

    #[test]
    fn encode_task_appends_monopole_block() {
        let task = ArcTask {
            id: "test".into(),
            train: vec![ArcExample {
                input: grid_2x2([[1, 0], [0, 1]]),
                output: grid_2x2([[1, 1], [1, 1]]),
            }],
            test: vec![],
        };
        let emb = encode_task(&task, None);
        assert!(
            emb.len() >= MONOPOLE_TASK_ENCODING_EXTRA_DIM,
            "embedding should include monopole tail"
        );
        for &x in emb.iter().rev().take(4) {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn monopole_blank_vs_center_dot_differs() {
        let blank = Grid {
            cells: vec![vec![0u8; 3], vec![0u8; 3], vec![0u8; 3]],
            height: 3,
            width: 3,
        };
        let marked = Grid {
            cells: vec![
                vec![0u8, 0u8, 0u8],
                vec![0u8, 8u8, 0u8],
                vec![0u8, 0u8, 0u8],
            ],
            height: 3,
            width: 3,
        };
        let f0 = monopole_field_features(&blank);
        let f1 = monopole_field_features(&marked);
        let l1: f32 = f0.iter().zip(f1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            l1 > 1e-5,
            "blank vs single foreground cell should change features (L1={})",
            l1
        );
    }
}

// ─── CLI entry point ───────────────────────────────────────────────────────

pub fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut do_train = false;
    let mut do_test = false;
    let mut brain_path = "arc-agi-brain.bin".to_string();
    let mut lang_brain_path = "brain.bin".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--train" => do_train = true,
            "--test" => do_test = true,
            "--brain" => {
                i += 1;
                if i < args.len() {
                    brain_path = args[i].clone();
                }
            }
            "--lang-brain" => {
                i += 1;
                if i < args.len() {
                    lang_brain_path = args[i].clone();
                }
            }
            _ => {}
        }
        i += 1;
    }

    if !do_train && !do_test {
        do_train = true;
        do_test = true;
    }

    let data_dir = std::path::PathBuf::from(
        std::env::var("ARC_AGI_ROOT").unwrap_or_else(|_| "data/arc-agi/data/training".to_string()),
    );
    if !data_dir.exists() {
        eprintln!("ARC-AGI data not found at {:?}", data_dir);
        eprintln!("Clone https://github.com/fchollet/ARC-AGI into data/arc-agi/");
        std::process::exit(1);
    }
    let tasks = load_arc_tasks(&data_dir);
    println!("Loaded {} ARC tasks from {:?}", tasks.len(), data_dir);

    if do_train {
        let brain = ArcBrain::train(&tasks, &lang_brain_path);
        brain
            .save(Path::new(&brain_path))
            .expect("Failed to save brain");
        println!(
            "\nARC brain saved to {} (lang brain read from {})",
            brain_path, lang_brain_path
        );

        if do_test {
            let mut brain = brain;
            let lang_rt = load_language_runtime(&lang_brain_path);
            benchmark(&mut brain, &tasks, lang_rt.as_ref());
        }
    } else if do_test {
        println!("\n=== Loading ArcBrain from {} ===", brain_path);
        let mut brain = ArcBrain::load(Path::new(&brain_path)).expect("Failed to load brain");
        let lang_rt = load_language_runtime(&lang_brain_path);
        benchmark(&mut brain, &tasks, lang_rt.as_ref());
    }
}
