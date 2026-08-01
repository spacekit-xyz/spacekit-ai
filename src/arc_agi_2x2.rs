//! ARC solver benchmark: A*, Dirac (learned / direct), and optional **ArcBrain**
//! routing (`encode_task` + monopole tail → Paramecium → learned solver:
//! astar | dirac | onepass). No hand-coded structural pattern shortcuts.
//!
//! Usage:
//!   cargo run --release --bin growformer-arc-agi-2x2
//!   cargo run --release --bin growformer-arc-agi-2x2 -- --all
//!   cargo run --release --bin growformer-arc-agi-2x2 -- --count 50
//!   cargo run --release --bin growformer-arc-agi-2x2 -- --3x3
//!   cargo run --release --bin growformer-arc-agi-2x2 -- --3x3 --count 30
//!   cargo run --release --bin growformer-arc-agi-2x2 -- --train-router [save.bin]
//!   cargo run --release --bin growformer-arc-agi-2x2 -- --router arc-router.bin
//!   cargo run --release --bin growformer-arc-agi-2x2 -- --lang-brain path/to/brain.bin
//!
//! Default task set: training tasks that touch a **2×2** grid (small curated slice).
//! Pass **`--3x3`** to benchmark tasks that touch a **3×3** grid instead.
//! `--train-router` fits the router on the **same** subset you benchmark (memorization-style);
//! use `--all` for the full training corpus (slow: labels run every solver per task).

use growformer::arc_agi::{load_arc_tasks, ArcTask, Grid};
use growformer::arc_brain::{
    apply_learned_solver, solve_task_dirac, solve_task_dirac_direct, ArcBrain,
};
use growformer::arc_dsl::astar_dsl_solve;
use std::path::{Path, PathBuf};

fn touches_2x2(task: &ArcTask) -> bool {
    task.train.iter().any(|ex| {
        (ex.input.height == 2 && ex.input.width == 2)
            || (ex.output.height == 2 && ex.output.width == 2)
    }) || task.test.iter().any(|ex| {
        (ex.input.height == 2 && ex.input.width == 2)
            || (ex.output.height == 2 && ex.output.width == 2)
    })
}

fn touches_3x3(task: &ArcTask) -> bool {
    task.train.iter().any(|ex| {
        (ex.input.height == 3 && ex.input.width == 3)
            || (ex.output.height == 3 && ex.output.width == 3)
    }) || task.test.iter().any(|ex| {
        (ex.input.height == 3 && ex.input.width == 3)
            || (ex.output.height == 3 && ex.output.width == 3)
    })
}

fn grid_cell_matches(pred: &Grid, expected: &Grid) -> (usize, usize) {
    if pred.height != expected.height || pred.width != expected.width {
        return (0, expected.height * expected.width);
    }
    let mut correct = 0;
    let total = expected.height * expected.width;
    for r in 0..expected.height {
        for c in 0..expected.width {
            if pred.cells[r][c] == expected.cells[r][c] {
                correct += 1;
            }
        }
    }
    (correct, total)
}

fn check_solution(task: &ArcTask, preds: &[Grid]) -> (bool, usize, usize) {
    let mut correct = 0;
    let mut total = 0;
    let mut all_match = true;
    for (i, test_ex) in task.test.iter().enumerate() {
        if i < preds.len() {
            let (c, t) = grid_cell_matches(&preds[i], &test_ex.output);
            correct += c;
            total += t;
            if c != t {
                all_match = false;
            }
        } else {
            total += test_ex.output.height * test_ex.output.width;
            all_match = false;
        }
    }
    (all_match, correct, total)
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    let v = args.get(i + 1)?;
    if v.starts_with('-') {
        return None;
    }
    Some(v.clone())
}

fn main() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/arc-agi/data/training");
    if !dir.is_dir() {
        eprintln!("ARC data not found at {:?}", dir);
        eprintln!("Clone https://github.com/fchollet/ARC-AGI into data/arc-agi/");
        std::process::exit(1);
    }

    let all_tasks = load_arc_tasks(&dir);

    let args: Vec<String> = std::env::args().collect();
    let count_flag = args
        .iter()
        .position(|a| a == "--count")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok());
    let run_all = args.iter().any(|a| a == "--all");
    let subset_3x3 = args.iter().any(|a| a == "--3x3");

    let lang_brain_path = arg_value(&args, "--lang-brain").unwrap_or_else(|| "brain.bin".into());

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let default_router_bin = manifest.join("data/arc-router.bin");
    let train_router = args.iter().any(|a| a == "--train-router");
    let router_save_path = if train_router {
        arg_value(&args, "--train-router")
            .map(PathBuf::from)
            .unwrap_or(default_router_bin)
    } else {
        default_router_bin.clone()
    };
    let router_load_path = arg_value(&args, "--router").map(PathBuf::from);

    let mut brain_opt: Option<ArcBrain> = None;

    let subset: Vec<ArcTask> = if run_all {
        all_tasks
    } else if let Some(n) = count_flag {
        if subset_3x3 {
            all_tasks
                .into_iter()
                .filter(|t| touches_3x3(t))
                .take(n)
                .collect()
        } else {
            all_tasks.into_iter().take(n).collect()
        }
    } else if subset_3x3 {
        all_tasks.into_iter().filter(|t| touches_3x3(t)).collect()
    } else {
        all_tasks.into_iter().filter(|t| touches_2x2(t)).collect()
    };

    let label = if run_all {
        "all".into()
    } else if count_flag.is_some() {
        if subset_3x3 {
            format!("first {} (3×3-touching)", subset.len())
        } else {
            format!("first {}", subset.len())
        }
    } else if subset_3x3 {
        "3×3 subset".into()
    } else {
        "2×2 subset".into()
    };

    if train_router {
        eprintln!(
            "Training ArcBrain on {} tasks (learned solver labels; lang: {})…",
            subset.len(),
            lang_brain_path
        );
        let brain = ArcBrain::train(&subset, &lang_brain_path);
        if let Some(parent) = router_save_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match brain.save(&router_save_path) {
            Ok(()) => eprintln!("Saved router to {:?}", router_save_path),
            Err(e) => eprintln!(
                "Could not save router ({}): {}",
                router_save_path.display(),
                e
            ),
        }
        brain_opt = Some(brain);
    } else if let Some(ref p) = router_load_path {
        match ArcBrain::load(Path::new(p)) {
            Ok(b) => {
                eprintln!("Loaded ArcBrain router from {:?}", p);
                brain_opt = Some(b);
            }
            Err(e) => eprintln!("Could not load --router {:?}: {}", p, e),
        }
    }

    eprintln!("╔══════════════════════════════════════════════════════════════");
    eprintln!("║ Solver Benchmark — {} tasks ({})", subset.len(), label);
    if brain_opt.is_some() {
        eprintln!("║ ArcBrain routed: ON");
    } else {
        eprintln!("║ ArcBrain routed: OFF (use --train-router or --router PATH)");
    }
    eprintln!("╠══════════════════════════════════════════════════════════════\n");

    let mut astar_solved = 0usize;
    let mut dirac_solved = 0usize;
    let mut direct_solved = 0usize;
    let mut routed_solved = 0usize;
    let mut combined_solved = 0usize;

    struct Row {
        id: String,
        dims: String,
        astar: String,
        dirac: String,
        direct: String,
        routed: String,
        best: String,
    }
    let mut rows: Vec<Row> = Vec::new();

    for (i, task) in subset.iter().enumerate() {
        let ih = task.train[0].input.height;
        let iw = task.train[0].input.width;
        let oh = task.train[0].output.height;
        let ow = task.train[0].output.width;
        let dims = format!("{}x{}→{}x{}", ih, iw, oh, ow);

        eprint!("  [{:>2}/{}] {} ({}) ", i + 1, subset.len(), task.id, dims);

        // A* DSL search
        let astar_result = astar_dsl_solve(task, 3, 3000);
        let (a_ok, a_c, a_t) = match &astar_result {
            Some((preds, _)) => check_solution(task, preds),
            None => (
                false,
                0,
                task.test
                    .iter()
                    .map(|e| e.output.height * e.output.width)
                    .sum(),
            ),
        };
        let a_strat = astar_result.as_ref().map(|(_, s)| *s).unwrap_or("—");

        // Dirac learned field
        let dirac_result = solve_task_dirac(task);
        let (d_ok, d_c, d_t) = match &dirac_result {
            Some(preds) => check_solution(task, preds),
            None => (
                false,
                0,
                task.test
                    .iter()
                    .map(|e| e.output.height * e.output.width)
                    .sum(),
            ),
        };

        // Dirac direct (zero-parameter, shrink only)
        let direct_result = solve_task_dirac_direct(task);
        let (dd_ok, dd_c, dd_t) = match &direct_result {
            Some(preds) => check_solution(task, preds),
            None => (
                false,
                0,
                task.test
                    .iter()
                    .map(|e| e.output.height * e.output.width)
                    .sum(),
            ),
        };

        let (r_ok, r_c, r_t, routed_cell) = if let Some(ref mut brain) = brain_opt {
            let (_cls, conf, name) = brain.route(task, None);
            let routed_result = apply_learned_solver(task, name.as_str());
            let (ok, c, t) = match &routed_result {
                Some(preds) => check_solution(task, preds),
                None => (
                    false,
                    0,
                    task.test
                        .iter()
                        .map(|e| e.output.height * e.output.width)
                        .sum(),
                ),
            };
            let cell = if routed_result.is_some() {
                format!("{} {:.2} {}/{}", name, conf, c, t)
            } else {
                format!("{} {:.2} —", name, conf)
            };
            (ok, c, t, cell)
        } else {
            (false, 0, 0, "—".into())
        };

        if a_ok {
            astar_solved += 1;
        }
        if d_ok {
            dirac_solved += 1;
        }
        if dd_ok {
            direct_solved += 1;
        }
        if r_ok {
            routed_solved += 1;
        }
        let any = a_ok || d_ok || dd_ok || r_ok;
        if any {
            combined_solved += 1;
        }

        let best = if r_ok {
            "ROUTED"
        } else if d_ok {
            "DIRAC"
        } else if dd_ok {
            "DIRECT"
        } else if a_ok {
            "A*"
        } else {
            "MISS"
        };

        eprintln!(
            "{:<8} A*:{}/{} Dirac:{}/{} Direct:{}/{} R:{}",
            best, a_c, a_t, d_c, d_t, dd_c, dd_t, routed_cell
        );

        rows.push(Row {
            id: task.id.clone(),
            dims,
            astar: if a_ok {
                format!("✓ {}", a_strat)
            } else {
                format!("✗ {}/{}", a_c, a_t)
            },
            dirac: if d_ok {
                "✓".into()
            } else {
                format!("✗ {}/{}", d_c, d_t)
            },
            direct: if dd_ok {
                "✓".into()
            } else if direct_result.is_some() {
                format!("✗ {}/{}", dd_c, dd_t)
            } else {
                "n/a".into()
            },
            routed: if brain_opt.is_some() {
                if r_ok {
                    "✓".into()
                } else {
                    format!("✗ {}/{}", r_c, r_t)
                }
            } else {
                "—".into()
            },
            best: best.into(),
        });
    }

    let n = subset.len();
    eprintln!("\n╔══════════════════════════════════════════════════════════════════════════");
    eprintln!("║ RESULTS");
    eprintln!("╠══════════════════════════════════════════════════════════════════════════");
    eprintln!(
        "║ A* DSL search:     {:>2}/{} ({:.1}%)",
        astar_solved,
        n,
        astar_solved as f32 / n as f32 * 100.0
    );
    eprintln!(
        "║ Dirac (learned):   {:>2}/{} ({:.1}%)",
        dirac_solved,
        n,
        dirac_solved as f32 / n as f32 * 100.0
    );
    eprintln!(
        "║ Dirac (direct):    {:>2}/{} ({:.1}%)",
        direct_solved,
        n,
        direct_solved as f32 / n as f32 * 100.0
    );
    eprintln!(
        "║ ArcBrain routed:   {:>2}/{} ({:.1}%)",
        routed_solved,
        n,
        routed_solved as f32 / n as f32 * 100.0
    );
    eprintln!(
        "║ Combined (any):    {:>2}/{} ({:.1}%)",
        combined_solved,
        n,
        combined_solved as f32 / n as f32 * 100.0
    );
    eprintln!("╠══════════════════════════════════════════════════════════════════════════");
    eprintln!(
        "║ {:>3} │ {:<10} │ {:<10} │ {:<12} │ {:<8} │ {:<6} │ {:<6} │ {}",
        "#", "task_id", "dims", "A*", "Dirac", "Direct", "Routed", "best"
    );
    eprintln!("╟─────┼────────────┼────────────┼──────────────┼────────┼────────┼────────┼──────");
    for (i, r) in rows.iter().enumerate() {
        eprintln!(
            "║ {:>3} │ {:<10} │ {:<10} │ {:<12} │ {:<8} │ {:<6} │ {:<6} │ {}",
            i + 1,
            &r.id[..r.id.len().min(10)],
            r.dims,
            r.astar,
            r.dirac,
            r.direct,
            r.routed,
            r.best
        );
    }
    eprintln!("╚══════════════════════════════════════════════════════════════════════════");
}
