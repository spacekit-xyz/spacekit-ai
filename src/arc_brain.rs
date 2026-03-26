use std::path::Path;
use std::collections::HashMap;

use serde::{Serialize, Deserialize};

use crate::arc_agi::{load_arc_tasks, ArcTask, Grid, NUM_COLORS, encode_grid, solve_task};
use crate::clifford::Multivector;
use crate::dimension::language::LanguageRuntime;
use crate::dimension::manager::DimensionManager;
use crate::micro_brain::{MicroBrain, MicroBrainRole};

const NEIGHBORHOOD: usize = 3;
const HALF_K: usize = NEIGHBORHOOD / 2;

// ─── Clifford cell encoding (translation-invariant) ─────────────────────────

fn relative_position_vector(dr: isize, dc: isize) -> Multivector {
    let pi = std::f32::consts::PI;
    let u = (dr as f32 + HALF_K as f32) / (NEIGHBORHOOD as f32 - 1.0);
    let v = (dc as f32 + HALF_K as f32) / (NEIGHBORHOOD as f32 - 1.0);
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

fn encode_cell_neighborhood(grid: &Grid, r: usize, c: usize) -> Vec<f32> {
    let mut mv = Multivector::zero();
    for dr in -(HALF_K as isize)..=(HALF_K as isize) {
        for dc in -(HALF_K as isize)..=(HALF_K as isize) {
            let gr = r as isize + dr;
            let gc = c as isize + dc;
            let color = if gr >= 0 && gr < grid.height as isize && gc >= 0 && gc < grid.width as isize {
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
    if n > 1e-8 { mv = mv.scale(1.0 / n); }
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
// Combined: [bridge_128d, clifford_512d, scalars_25d] = 665d

fn task_to_text(task: &ArcTask) -> String {
    let ex = &task.train[0];
    let ih = ex.input.height;
    let iw = ex.input.width;
    let oh = ex.output.height;
    let ow = ex.output.width;

    let mut in_colors = [0u32; NUM_COLORS];
    for row in &ex.input.cells {
        for &c in row { in_colors[c as usize] += 1; }
    }
    let mut out_colors = [0u32; NUM_COLORS];
    for row in &ex.output.cells {
        for &c in row { out_colors[c as usize] += 1; }
    }
    let in_distinct = in_colors.iter().filter(|&&c| c > 0).count();
    let out_distinct = out_colors.iter().filter(|&&c| c > 0).count();
    let total = (ih * iw) as f32;

    let dim_relation = if ih == oh && iw == ow {
        let changed = (0..ih).flat_map(|r| (0..iw).map(move |c| (r, c)))
            .filter(|&(r, c)| ex.input.cells[r][c] != ex.output.cells[r][c])
            .count();
        format!("same dimensions, {:.0}% cells changed", changed as f32 / total * 100.0)
    } else if oh > ih && ow > iw {
        let rh = oh as f32 / ih as f32;
        let rw = ow as f32 / iw as f32;
        if (rh - rw).abs() < 0.01 {
            format!("output scaled up {:.1}x uniformly", rh)
        } else {
            format!("output grows {}x{} to {}x{}, ratio {:.1}h {:.1}w", ih, iw, oh, ow, rh, rw)
        }
    } else if oh < ih && ow < iw {
        format!("output shrinks {}x{} to {}x{}, extraction or summary", ih, iw, oh, ow)
    } else {
        format!("asymmetric resize {}x{} to {}x{}", ih, iw, oh, ow)
    };

    let bg = in_colors.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0);
    let bg_frac = in_colors[bg] as f32 / total;

    format!(
        "grid transformation: input {}x{} with {} colors (bg color {} at {:.0}%), \
         output {}x{} with {} colors. {} examples. {}",
        ih, iw, in_distinct, bg, bg_frac * 100.0,
        oh, ow, out_distinct,
        task.train.len(),
        dim_relation,
    )
}

fn encode_task(task: &ArcTask, lang_rt: Option<&LanguageRuntime>) -> Vec<f32> {
    let bridge_dim = 128;
    let mut features = Vec::with_capacity(bridge_dim + 256 * 2 + 32);

    // Path B: Language bridge (cross-domain knowledge from brain.bin)
    if let Some(rt) = lang_rt {
        let text = task_to_text(task);
        match rt.encode_and_bridge(&text) {
            Ok((_raw, bridged)) => {
                features.extend_from_slice(&bridged.routed_vector);
            }
            Err(_) => {
                features.extend(std::iter::repeat(0.0f32).take(bridge_dim));
            }
        }
    } else {
        features.extend(std::iter::repeat(0.0f32).take(bridge_dim));
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
    features.push(if (ih - oh).abs() < 0.5 && (iw - ow).abs() < 0.5 { 1.0 } else { 0.0 });

    let mut color_counts = [0u32; NUM_COLORS];
    let total_cells = (ex.input.height * ex.input.width) as f32;
    for row in &ex.input.cells {
        for &c in row { color_counts[c as usize] += 1; }
    }
    for c in 0..NUM_COLORS {
        features.push(color_counts[c] as f32 / total_cells.max(1.0));
    }

    let in_colors = color_counts.iter().filter(|&&c| c > 0).count() as f32 / NUM_COLORS as f32;
    let mut out_counts = [0u32; NUM_COLORS];
    for row in &ex.output.cells {
        for &c in row { out_counts[c as usize] += 1; }
    }
    let out_colors = out_counts.iter().filter(|&&c| c > 0).count() as f32 / NUM_COLORS as f32;
    features.push(in_colors);
    features.push(out_colors);

    if ex.input.height == ex.output.height && ex.input.width == ex.output.width {
        let mut changed = 0u32;
        for r in 0..ex.input.height {
            for c in 0..ex.input.width {
                if ex.input.cells[r][c] != ex.output.cells[r][c] { changed += 1; }
            }
        }
        features.push(changed as f32 / total_cells.max(1.0));
    } else {
        features.push(1.0);
    }

    features.push(task.train.len() as f32 / 5.0);

    let bg = color_counts.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0);
    features.push(color_counts[bg] as f32 / total_cells.max(1.0));

    features
}

// ─── Per-task one-pass cell learning ────────────────────────────────────────

fn solve_task_onepass(task: &ArcTask) -> Option<Vec<Grid>> {
    let same_dim = task.train.iter().all(|ex|
        ex.input.height == ex.output.height && ex.input.width == ex.output.width
    );
    if !same_dim { return None; }

    let class_names: Vec<String> = (0..NUM_COLORS).map(|c| format!("c{}", c)).collect();
    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut targets: Vec<usize> = Vec::new();

    for ex in &task.train {
        for r in 0..ex.input.height {
            for c in 0..ex.input.width {
                embeddings.push(encode_cell_neighborhood(&ex.input, r, c));
                targets.push(ex.output.cells[r][c] as usize);
            }
        }
    }

    let sample_refs: Vec<(&[f32], usize)> = embeddings.iter()
        .zip(targets.iter())
        .map(|(e, &t)| (e.as_slice(), t))
        .collect();

    let brain = MicroBrain::build_from_data(
        MicroBrainRole::Custom("arc_cell".into()),
        256,
        NUM_COLORS,
        class_names,
        &sample_refs,
    );

    let results: Vec<Grid> = task.test.iter().map(|test_ex| {
        let h = test_ex.input.height;
        let w = test_ex.input.width;
        let mut cells = vec![vec![0u8; w]; h];
        let mut brain_local = brain.clone();
        for r in 0..h {
            for c in 0..w {
                let emb = encode_cell_neighborhood(&test_ex.input, r, c);
                let (cls, _conf, _logits) = brain_local.predict(&emb);
                cells[r][c] = cls as u8;
            }
        }
        Grid { cells, height: h, width: w }
    }).collect();

    Some(results)
}

// ─── Load language runtime from trained brain.bin (read-only) ───────────────

fn load_language_runtime(brain_path: &str) -> Option<LanguageRuntime> {
    let bytes = match std::fs::read(brain_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  [lang] Cannot read {}: {} — falling back to Clifford-only", brain_path, e);
            return None;
        }
    };
    let dm: DimensionManager = match crate::systems::checkpoint::deserialize_checkpoint_from_bytes(&bytes) {
        Ok(dm) => dm,
        Err(e) => {
            eprintln!("  [lang] Cannot deserialize {}: {} — falling back to Clifford-only", brain_path, e);
            return None;
        }
    };
    let rt = dm.language_runtime;
    if !rt.bridge.calibrated {
        eprintln!("  [lang] Bridge in {} not calibrated — falling back to Clifford-only", brain_path);
        return None;
    }
    println!("  [lang] Loaded calibrated LanguageRuntime from {} (bridge {}→{}d)",
        brain_path, rt.bridge.input_dim, rt.bridge.output_dim);
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
        println!("\n--- Loading GrowformerLanguageEncoder from {} ---", lang_brain_path);
        let lang_rt = load_language_runtime(lang_brain_path);
        let has_bridge = lang_rt.is_some();
        if has_bridge {
            println!("  Language bridge: ACTIVE — cross-domain embeddings enabled");
        } else {
            println!("  Language bridge: INACTIVE — Clifford-only mode");
        }

        // Phase 1: Run hand-coded pipeline to discover strategy labels
        println!("\n--- Phase 1: Strategy discovery (hand-coded pipeline) ---");
        let mut strategy_set: Vec<String> = Vec::new();
        let mut strategy_map: HashMap<String, usize> = HashMap::new();
        let mut router_embeddings: Vec<Vec<f32>> = Vec::new();
        let mut router_targets: Vec<usize> = Vec::new();
        let mut solved_count = 0;
        let mut strategy_counts: HashMap<String, usize> = HashMap::new();

        for task in tasks {
            let diag = solve_task(task);
            if !diag.solved { continue; }
            solved_count += 1;

            let strat = diag.strategy.to_string();
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

        println!("  Solved by hand-coded: {}/{}", solved_count, tasks.len());
        println!("  Strategy classes: {}", strategy_set.len());
        println!("  Embedding dim: {}", router_embeddings.first().map_or(0, |e| e.len()));
        for (strat, count) in &strategy_counts {
            println!("    {}: {} tasks", strat, count);
        }

        // Phase 2: Build strategy router via one-pass Paramecium lattice
        println!("\n--- Phase 2: Strategy router (one-pass) ---");
        let sample_refs: Vec<(&[f32], usize)> = router_embeddings.iter()
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

        println!("  Router lattice programs: {}", router.lattice.program_count());

        // Phase 3: Validate router on training set
        println!("\n--- Phase 3: Router self-validation ---");
        let mut router_clone = router.clone();
        let mut router_correct = 0;
        for (emb, &target) in router_embeddings.iter().zip(router_targets.iter()) {
            let (cls, _conf, _logits) = router_clone.predict(emb);
            if cls == target { router_correct += 1; }
        }
        println!("  Router accuracy on solved tasks: {}/{} ({:.1}%)",
            router_correct, router_embeddings.len(),
            router_correct as f32 / router_embeddings.len().max(1) as f32 * 100.0);

        println!("\n  Training complete.");

        ArcBrain {
            strategy_router: router,
            strategy_names: strategy_set,
            neighborhood_size: NEIGHBORHOOD,
            has_language_bridge: has_bridge,
        }
    }

    pub fn route(&mut self, task: &ArcTask, lang_rt: Option<&LanguageRuntime>) -> (usize, f32, String) {
        let emb = encode_task(task, lang_rt);
        let (cls, conf, _logits) = self.strategy_router.predict(&emb);
        let name = self.strategy_names.get(cls).cloned().unwrap_or_else(|| "unknown".into());
        (cls, conf, name)
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
    if a.height != b.height || a.width != b.width { return false; }
    for r in 0..a.height {
        for c in 0..a.width {
            if a.cells[r][c] != b.cells[r][c] { return false; }
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

    let same_dim_count = tasks.iter().filter(|t|
        t.train.iter().all(|ex|
            ex.input.height == ex.output.height && ex.input.width == ex.output.width
        )
    ).count();

    println!("\n=== ArcBrain Benchmark ===");
    println!("Total tasks: {}, same-dim: {}", tasks.len(), same_dim_count);
    println!("Language bridge: {}", if lang_rt.is_some() { "ACTIVE" } else { "INACTIVE" });
    println!("Strategies known: {:?}", brain.strategy_names);

    for task in tasks {
        total += 1;

        let diag = solve_task(task);
        let hc_solved = diag.solved;
        if hc_solved { solved_handcoded += 1; }

        let (_cls, conf, predicted_strategy) = brain.route(task, lang_rt);

        if hc_solved {
            router_total += 1;
            if predicted_strategy == diag.strategy {
                router_correct += 1;
            }
        }

        let same_dim = task.train.iter().all(|ex|
            ex.input.height == ex.output.height && ex.input.width == ex.output.width
        );
        let lattice_solved = if same_dim {
            if let Some(preds) = solve_task_onepass(task) {
                task.test.iter().enumerate().all(|(i, test_ex)| {
                    i < preds.len() && grids_match(&preds[i], &test_ex.output)
                })
            } else { false }
        } else { false };

        if lattice_solved { solved_pertask += 1; }

        let either = hc_solved || lattice_solved;
        if either { solved_combined += 1; }

        if lattice_solved && !hc_solved {
            println!("  NEW (lattice only): {} [router predicted: {} conf={:.2}]",
                task.id, predicted_strategy, conf);
        }
    }

    let router_acc = if router_total > 0 {
        router_correct as f32 / router_total as f32 * 100.0
    } else { 0.0 };

    println!("\n--- Results ---");
    println!("Tasks evaluated: {}", total);
    println!("Hand-coded pipeline: {}/{} solved ({:.1}%)",
        solved_handcoded, total, solved_handcoded as f32 / total.max(1) as f32 * 100.0);
    println!("Per-task lattice:    {}/{} same-dim solved ({:.1}% of same-dim)",
        solved_pertask, same_dim_count, solved_pertask as f32 / same_dim_count.max(1) as f32 * 100.0);
    println!("Combined (either):   {}/{} solved ({:.1}%)",
        solved_combined, total, solved_combined as f32 / total.max(1) as f32 * 100.0);
    println!("Strategy router:     {}/{} correct on solved tasks ({:.1}%)",
        router_correct, router_total, router_acc);
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
            "--brain" => { i += 1; if i < args.len() { brain_path = args[i].clone(); } }
            "--lang-brain" => { i += 1; if i < args.len() { lang_brain_path = args[i].clone(); } }
            _ => {}
        }
        i += 1;
    }

    if !do_train && !do_test {
        do_train = true;
        do_test = true;
    }

    let data_dir = std::path::PathBuf::from(
        std::env::var("ARC_AGI_ROOT")
            .unwrap_or_else(|_| "data/arc-agi/data/training".to_string())
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
        brain.save(Path::new(&brain_path)).expect("Failed to save brain");
        println!("\nARC brain saved to {} (lang brain read from {})", brain_path, lang_brain_path);

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
