//! `gf-system1` — train, evaluate, distil and serve System 1 brains.

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
#[cfg(feature = "server")]
use serde_json::json;
use serde_json::Value;

use growformer_system1::brain::{LmArch, LmConfig, System1Brain, TrainConfig};
use growformer_system1::decide::{Backend, DecideRequest, QuestionRef};
use growformer_system1::eval::{distill, evaluate, DistillSource, EvalOptions, EvalReport};
use growformer_system1::project::{
    label_counts, load_decider, read_project, read_questions, read_rows, relative_to, EmbCache,
};
use growformer_system1::prompt::TemplatePreset;

#[derive(Parser)]
#[command(
    name = "gf-system1",
    version,
    about = "System 1 decision brains: typed Choice / Score / Noul answers with calibrated probabilities"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Train fast heads from labelled JSONL (writes the brain file).
    Train {
        /// Growformer project manifest with a [system1] section (supplies the paths below).
        #[arg(long)]
        project: Option<PathBuf>,
        /// Question schema (TOML or JSON).
        #[arg(long)]
        questions: Option<PathBuf>,
        /// Labelled JSONL files or folders (repeatable).
        #[arg(long)]
        data: Vec<PathBuf>,
        #[arg(long, short)]
        out: Option<PathBuf>,
        #[arg(long)]
        name: Option<String>,
        #[arg(long, default_value_t = 0.2)]
        holdout: f64,
        #[arg(long, default_value_t = 8)]
        min_rows: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    /// Point a brain at an open model (GGUF + tokenizer.json) for questions the heads can't answer.
    AttachLm {
        #[arg(long)]
        brain: PathBuf,
        /// llama | qwen2 | qwen3
        #[arg(long)]
        arch: String,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        /// chatml | chatml-nothink | llama3 | plain
        #[arg(long)]
        template: Option<String>,
        #[arg(long, default_value_t = 2048)]
        max_state_tokens: usize,
    },
    /// Answer questions about one state (prints the response JSON).
    Decide {
        #[arg(long)]
        brain: PathBuf,
        /// State text.
        #[arg(long, conflicts_with_all = ["state_json", "request"])]
        state: Option<String>,
        /// State from a JSON file (object/array/string).
        #[arg(long)]
        state_json: Option<PathBuf>,
        /// Full request JSON file (`{"state":…, "questions":[…]}`).
        #[arg(long)]
        request: Option<PathBuf>,
        /// Ask only these question ids (repeatable; default: all).
        #[arg(long = "question", short = 'q')]
        questions: Vec<String>,
        #[arg(long)]
        min_confidence: Option<f32>,
        /// head | probe | lm
        #[arg(long)]
        force: Option<String>,
        /// Don't load the brain's open model.
        #[arg(long)]
        no_lm: bool,
    },
    /// Accuracy, Brier, calibration error and latency on labelled rows.
    Eval {
        #[arg(long)]
        project: Option<PathBuf>,
        #[arg(long)]
        brain: Option<PathBuf>,
        #[arg(long)]
        data: Vec<PathBuf>,
        /// Also score with the brain's open model.
        #[arg(long)]
        lm: bool,
        /// Fit per-question open-model temperatures (on half the rows; report on the other half).
        #[arg(long)]
        fit_lm_temperature: bool,
        /// Pick per-question head-vs-model thresholds (same half split).
        #[arg(long)]
        tune_router: bool,
        /// Write fitted temperatures / thresholds back into the brain.
        #[arg(long)]
        save: bool,
        #[arg(long)]
        json_out: Option<PathBuf>,
        /// Evaluate at most N rows.
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Keep rows the heads were trained on (their head scores are then optimistic).
        #[arg(long)]
        include_train_rows: bool,
        /// Tune a question's router threshold only with at least this many fitting rows.
        #[arg(long, default_value_t = 50)]
        min_router_rows: usize,
    },
    /// Label states with the open model → JSONL with soft labels for `train`.
    Distill {
        #[arg(long)]
        brain: PathBuf,
        /// JSONL of `{"state": …}` / `{"text": …}` rows, or a `.txt` file with
        /// one state per line (repeatable; folders read their *.jsonl / *.txt).
        #[arg(long)]
        states: Vec<PathBuf>,
        #[arg(long, short)]
        out: PathBuf,
        #[arg(long = "question", short = 'q')]
        questions: Vec<String>,
        /// lm = ask the open model every question; best = what `decide` would
        /// answer (probe / confident head where trained, else the open model).
        #[arg(long, default_value = "lm")]
        source: String,
        /// Label at most N states (0 = all).
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Skip states the brain was trained on.
        #[arg(long)]
        skip_trained: bool,
    },
    /// Train probe heads on the attached open model's hidden state (Qwen3).
    /// One model pass per row; features are cached next to the brain.
    TrainProbe {
        #[arg(long)]
        project: Option<PathBuf>,
        #[arg(long)]
        brain: Option<PathBuf>,
        /// Labelled JSONL files or folders (default: the project's [system1].data).
        #[arg(long)]
        data: Vec<PathBuf>,
        /// Depth of the tapped layer, 0 = first … 1 = last.
        #[arg(long, default_value_t = 0.66)]
        layer: f32,
        /// Feature cache (default: <brain>.emb).
        #[arg(long)]
        cache: Option<PathBuf>,
        #[arg(long, default_value_t = 150)]
        epochs: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    /// Label counts per question and outcome, flagging thin outcomes.
    Stats {
        #[arg(long)]
        project: Option<PathBuf>,
        #[arg(long)]
        questions: Option<PathBuf>,
        #[arg(long)]
        data: Vec<PathBuf>,
        /// Flag outcomes with fewer rows than this.
        #[arg(long, default_value_t = 30)]
        min: usize,
    },
    /// HTTP server: POST /v1/decide, GET /v1/questions, GET /health.
    #[cfg(feature = "server")]
    Serve {
        #[arg(long)]
        project: Option<PathBuf>,
        #[arg(long)]
        brain: Option<PathBuf>,
        #[arg(long, default_value = "127.0.0.1:8787")]
        addr: String,
        #[arg(long)]
        no_lm: bool,
    },
}

fn main() {
    if let Err(e) = run(Cli::parse()) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn brain_path(project: &Option<PathBuf>, brain: &Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(b) = brain {
        return Ok(b.clone());
    }
    match project {
        Some(p) => Ok(read_project(p)?.brain_output),
        None => Err("pass --brain or --project".into()),
    }
}

fn parse_arch(s: &str) -> Result<LmArch, String> {
    match s.to_ascii_lowercase().as_str() {
        "llama" => Ok(LmArch::Llama),
        "qwen2" | "qwen2.5" => Ok(LmArch::Qwen2),
        "qwen3" => Ok(LmArch::Qwen3),
        o => Err(format!("unknown --arch {o:?} (llama | qwen2 | qwen3)")),
    }
}

fn parse_template(s: &str) -> Result<TemplatePreset, String> {
    match s {
        "chatml" => Ok(TemplatePreset::Chatml),
        "chatml-nothink" => Ok(TemplatePreset::ChatmlNothink),
        "llama3" => Ok(TemplatePreset::Llama3),
        "plain" => Ok(TemplatePreset::Plain),
        o => Err(format!(
            "unknown --template {o:?} (chatml | chatml-nothink | llama3 | plain)"
        )),
    }
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "–".into())
}

fn print_train_report(b: &System1Brain, n_rows: usize, skipped: usize) {
    println!("rows: {n_rows} used, {skipped} skipped (no usable label)");
    println!(
        "{:<18} {:>6} {:>6} {:>7} {:>9} {:>9} {:>7} {:>7}",
        "question", "train", "test", "temp", "acc", "acc@cal", "ece", "brier"
    );
    for q in &b.questions.questions {
        match (b.heads.get(&q.id), b.reports.get(&q.id)) {
            (Some(_), Some(r)) => println!(
                "{:<18} {:>6} {:>6} {:>7.2} {:>9.3} {:>9.3} {:>7.3} {:>7.3}{}",
                q.id,
                r.n_train,
                r.n_holdout,
                r.temperature,
                r.raw.accuracy,
                r.calibrated.accuracy,
                r.calibrated.ece,
                r.calibrated.brier,
                r.calibrated
                    .mae
                    .map(|m| format!("  mae {m:.2}"))
                    .unwrap_or_default()
            ),
            (Some(h), None) => println!(
                "{:<18} {:>6}   (too few rows for a held-out report)",
                q.id, h.n_train
            ),
            (None, _) => println!(
                "{:<18}   no head (too few labels) — answered by the open model if attached",
                q.id
            ),
        }
    }
}

fn print_eval(rep: &EvalReport) {
    println!(
        "{:<16} {:>4} {:>7} {:>6} {:>7} {:>6} {:>7} {:>6} {:>7} {:>6} {:>6}",
        "question",
        "n",
        "head",
        "h.ece",
        "probe",
        "p.ece",
        "lm",
        "lm.ece",
        "routed",
        "r.thr",
        "head%"
    );
    for q in &rep.questions {
        println!(
            "{:<16} {:>4} {:>7} {:>6} {:>7} {:>6} {:>7} {:>6} {:>7} {:>6} {:>6}",
            q.id,
            q.n,
            fmt_opt(q.head.as_ref().map(|m| m.accuracy)),
            fmt_opt(q.head.as_ref().map(|m| m.ece)),
            fmt_opt(q.probe.as_ref().map(|m| m.accuracy)),
            fmt_opt(q.probe.as_ref().map(|m| m.ece)),
            fmt_opt(q.lm.as_ref().map(|m| m.accuracy)),
            fmt_opt(q.lm.as_ref().map(|m| m.ece)),
            fmt_opt(q.routed.as_ref().map(|m| m.accuracy)),
            q.router_threshold
                .map(|t| format!("{t:.2}"))
                .unwrap_or_else(|| "–".into()),
            q.head_share
                .map(|s| format!("{:.0}%", s * 100.0))
                .unwrap_or_else(|| "–".into()),
        );
    }
    println!(
        "routed = head when its confidence ≥ r.thr, else the probe (if trained) or the open model"
    );
    let l = &rep.head_latency;
    println!(
        "head latency: p50 {:.3} ms  p90 {:.3} ms  p99 {:.3} ms  (n={})",
        l.p50_ms, l.p90_ms, l.p99_ms, l.n
    );
    if let Some(l) = &rep.probe_latency {
        println!(
            "probe latency (one state pass, all questions): p50 {:.1} ms  p90 {:.1} ms  p99 {:.1} ms",
            l.p50_ms, l.p90_ms, l.p99_ms
        );
    }
    if let Some(l) = &rep.lm_latency {
        println!(
            "open-model latency (all questions, one shared state pass): p50 {:.1} ms  p90 {:.1} ms  p99 {:.1} ms",
            l.p50_ms, l.p90_ms, l.p99_ms
        );
    }
}

/// States for `distill`: JSONL rows (`state` / `text` / the row itself) or
/// `.txt` lines; folders contribute their *.jsonl and *.txt files.
fn read_states(paths: &[PathBuf]) -> Result<Vec<Value>, String> {
    let mut files = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p)
                .map_err(|e| format!("read_dir {}: {e}", p.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| {
                    matches!(
                        f.extension().and_then(|x| x.to_str()),
                        Some("jsonl") | Some("txt")
                    )
                })
                .collect();
            v.sort();
            files.extend(v);
        } else {
            files.push(p.clone());
        }
    }
    let mut out = Vec::new();
    for f in files {
        let raw = std::fs::read_to_string(&f).map_err(|e| format!("read {}: {e}", f.display()))?;
        let is_txt = f.extension().and_then(|x| x.to_str()) == Some("txt");
        for (i, l) in raw.lines().enumerate() {
            let l = l.trim();
            if l.is_empty() || (is_txt && l.starts_with('#')) {
                continue;
            }
            if is_txt {
                out.push(Value::String(l.to_string()));
            } else {
                let v: Value = serde_json::from_str(l)
                    .map_err(|e| format!("{}:{}: {e}", f.display(), i + 1))?;
                out.push(
                    v.get("state")
                        .or_else(|| v.get("text"))
                        .cloned()
                        .unwrap_or(v),
                );
            }
        }
    }
    Ok(out)
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.cmd {
        Cmd::Train {
            project,
            questions,
            data,
            out,
            name,
            holdout,
            min_rows,
            seed,
        } => {
            let proj = project.as_ref().map(|p| read_project(p)).transpose()?;
            let qpath = questions
                .or(proj.as_ref().map(|p| p.questions.clone()))
                .ok_or("pass --questions or --project")?;
            let data = if data.is_empty() {
                proj.as_ref().map(|p| p.data.clone()).unwrap_or_default()
            } else {
                data
            };
            if data.is_empty() {
                return Err("no --data (or [system1].data)".into());
            }
            let out = out
                .or(proj.as_ref().map(|p| p.brain_output.clone()))
                .ok_or("pass --out or --project")?;
            let qs = read_questions(&qpath)?;
            let (rows, skipped) = read_rows(&data, &qs)?;
            let name = name
                .or(proj.as_ref().map(|p| p.name.clone()))
                .unwrap_or_else(|| "system1".into());
            let mut b = System1Brain::new(name, qs)?;
            let mut kept_probe = None;
            // Keep an existing brain's open-model settings and calibration.
            if let Ok(old) = System1Brain::load(&out) {
                b.lm = old.lm;
                // Probe heads don't depend on the fast heads; keep those whose
                // question is unchanged (and the rows they were trained on).
                if let Some(mut p) = old.probe {
                    p.heads
                        .retain(|id, _| old.questions.get(id) == b.questions.get(id));
                    if !p.heads.is_empty() {
                        kept_probe = Some((p, old.train_rows));
                    }
                }
                // Open-model temperatures stay valid only while the question
                // (and so its prompt) is unchanged.
                b.lm_temperature = old
                    .lm_temperature
                    .into_iter()
                    .filter(|(id, _)| old.questions.get(id) == b.questions.get(id))
                    .collect();
                // Per-question router thresholds were tuned against the old
                // heads; keep only the global default.
                if !old.router.per_question.is_empty() {
                    eprintln!(
                        "note: cleared {} tuned router threshold(s) from the previous brain; re-run `eval --tune-router` on held-out rows",
                        old.router.per_question.len()
                    );
                }
                b.router.head_min_confidence = old.router.head_min_confidence;
            }
            if let Some(p) = &proj {
                if let Some(mut lm) = p.lm.clone() {
                    let dir = out.parent().unwrap_or(Path::new("."));
                    lm.model = relative_to(Path::new(&lm.model), dir);
                    lm.tokenizer = relative_to(Path::new(&lm.tokenizer), dir);
                    b.lm = Some(lm);
                }
            }
            let cfg = TrainConfig {
                holdout_frac: holdout,
                min_rows,
                seed,
                ..Default::default()
            };
            let t0 = std::time::Instant::now();
            b.train_heads(&rows, &cfg)?;
            if let Some((p, old_rows)) = kept_probe {
                // Prefer the probe wherever it beat the new head on held-out rows.
                for (id, pr) in &p.reports {
                    let head_acc = b.reports.get(id).map(|r| r.calibrated.accuracy);
                    if head_acc
                        .map(|h| pr.calibrated.accuracy >= h)
                        .unwrap_or(true)
                    {
                        b.router.per_question.insert(id.clone(), 1.01);
                    }
                }
                b.train_rows.extend(old_rows);
                b.train_rows.sort_unstable();
                b.train_rows.dedup();
                b.probe = Some(p);
            }
            b.save(&out)?;
            print_train_report(&b, rows.len(), skipped);
            println!(
                "trained in {:.1}s → {} ({} heads, {} features)",
                t0.elapsed().as_secs_f32(),
                out.display(),
                b.heads.len(),
                b.featurizer.as_ref().map(|f| f.dim()).unwrap_or(0)
            );
        }
        Cmd::AttachLm {
            brain,
            arch,
            model,
            tokenizer,
            template,
            max_state_tokens,
        } => {
            let mut b = System1Brain::load(&brain)?;
            let arch = parse_arch(&arch)?;
            let template = match template {
                Some(t) => parse_template(&t)?,
                None => match arch {
                    LmArch::Llama => TemplatePreset::Llama3,
                    LmArch::Qwen2 => TemplatePreset::Chatml,
                    LmArch::Qwen3 => TemplatePreset::ChatmlNothink,
                },
            };
            let dir = brain.parent().unwrap_or(Path::new("."));
            b.lm = Some(LmConfig {
                arch,
                model: relative_to(&model, dir),
                tokenizer: relative_to(&tokenizer, dir),
                template,
                system: None,
                max_state_tokens,
            });
            b.lm_temperature.clear();
            b.save(&brain)?;
            println!("attached {:?} model to {}", arch, brain.display());
            if b.probe.is_some() {
                println!("note: the brain's probe heads only work with the model they were trained on; if this is a different model, re-run `gf-system1 train-probe`");
            }
            println!("next: gf-system1 eval --brain {} --data <labelled.jsonl> --lm --fit-lm-temperature --tune-router --save", brain.display());
        }
        Cmd::Decide {
            brain,
            state,
            state_json,
            request,
            questions,
            min_confidence,
            force,
            no_lm,
        } => {
            let mut d = load_decider(&brain, !no_lm)?;
            let req: DecideRequest = if let Some(r) = request {
                let raw = std::fs::read_to_string(&r).map_err(|e| e.to_string())?;
                serde_json::from_str(&raw).map_err(|e| format!("request: {e}"))?
            } else {
                let state: Value = match (state, state_json) {
                    (Some(s), _) => Value::String(s),
                    (None, Some(p)) => {
                        let raw = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
                        serde_json::from_str(&raw).map_err(|e| format!("state json: {e}"))?
                    }
                    (None, None) => return Err("pass --state, --state-json or --request".into()),
                };
                DecideRequest {
                    state,
                    questions: questions.into_iter().map(QuestionRef::Id).collect(),
                    min_confidence,
                    force: match force.as_deref() {
                        None => None,
                        Some("head") => Some(Backend::Head),
                        Some("probe") => Some(Backend::Probe),
                        Some("lm") => Some(Backend::Lm),
                        Some(o) => return Err(format!("--force {o:?}: head | probe | lm")),
                    },
                }
            };
            let resp = d.decide(&req)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&resp).map_err(|e| e.to_string())?
            );
        }
        Cmd::Eval {
            project,
            brain,
            data,
            lm,
            fit_lm_temperature,
            tune_router,
            save,
            json_out,
            limit,
            include_train_rows,
            min_router_rows,
        } => {
            let bpath = brain_path(&project, &brain)?;
            let data = if data.is_empty() {
                project
                    .as_ref()
                    .map(|p| read_project(p).map(|p| p.data))
                    .transpose()?
                    .unwrap_or_default()
            } else {
                data
            };
            let has_probe = System1Brain::load(&bpath)?.probe.is_some();
            let mut d = load_decider(&bpath, lm || fit_lm_temperature || tune_router || has_probe)?;
            let (mut rows, _) = read_rows(&data, &d.brain.questions)?;
            let total = rows.len();
            let overlap = rows.iter().filter(|r| d.brain.trained_on(&r.text)).count();
            if !include_train_rows && overlap > 0 {
                rows.retain(|r| !d.brain.trained_on(&r.text));
                eprintln!(
                    "left out {overlap} of {total} rows the heads were trained on ({} remain; --include-train-rows keeps them)",
                    rows.len()
                );
                if rows.is_empty() {
                    return Err("every row was used to train the heads, so nothing is left to evaluate. \
Hold some rows out: split the data (e.g. `gf-llm jsonl-split`), `train` on one part and `eval --data` the other."
                        .into());
                }
            }
            if limit > 0 && rows.len() > limit {
                rows.truncate(limit);
            }
            let optimistic = d.brain.train_rows.is_empty() || (include_train_rows && overlap > 0);
            let fitting = fit_lm_temperature || tune_router;
            let mut router_notes: Vec<String> = Vec::new();
            let report = if fitting {
                // Fit on one half, report on the other (no peeking).
                let (fit_idx, rep_idx) = growformer_system1::data::split_holdout(&rows, 0.5, 11);
                let fit_rows: Vec<_> = fit_idx.iter().map(|&i| rows[i].clone()).collect();
                let rep_rows: Vec<_> = rep_idx.iter().map(|&i| rows[i].clone()).collect();
                eprintln!(
                    "fitting on {} rows, reporting on {}",
                    fit_rows.len(),
                    rep_rows.len()
                );
                let fit_rep = evaluate(
                    &mut d,
                    &fit_rows,
                    &EvalOptions {
                        use_lm: lm || fit_lm_temperature,
                        fit_lm_temperature,
                        tune_router,
                        min_router_rows,
                    },
                )?;
                for q in fit_rep.questions.iter().filter(|q| q.router_not_tuned) {
                    router_notes.push(format!(
                        "router not tuned for `{}`: {} fitting rows < --min-router-rows {min_router_rows}; kept threshold {:.2}",
                        q.id,
                        q.n,
                        d.brain.router.threshold(&q.id)
                    ));
                }
                evaluate(
                    &mut d,
                    &rep_rows,
                    &EvalOptions {
                        use_lm: lm || fit_lm_temperature,
                        ..Default::default()
                    },
                )?
            } else {
                evaluate(
                    &mut d,
                    &rows,
                    &EvalOptions {
                        use_lm: lm,
                        ..Default::default()
                    },
                )?
            };
            if optimistic {
                println!(
                    "rows: {}  (some may be rows the heads were trained on: head scores are optimistic)",
                    report.rows
                );
            } else {
                println!("rows: {}  (none used to train the heads)", report.rows);
            }
            print_eval(&report);
            for n in &router_notes {
                println!("{n}");
            }
            if let Some(p) = json_out {
                std::fs::write(&p, serde_json::to_string_pretty(&report).unwrap())
                    .map_err(|e| e.to_string())?;
            }
            if save {
                d.brain.save(&bpath)?;
                println!(
                    "saved temperatures / router thresholds → {}",
                    bpath.display()
                );
            }
        }
        Cmd::Distill {
            brain,
            states,
            out,
            questions,
            source,
            limit,
            skip_trained,
        } => {
            let source = match source.as_str() {
                "lm" => DistillSource::Lm,
                "best" => DistillSource::Best,
                o => return Err(format!("--source {o:?}: lm | best")),
            };
            let mut d = load_decider(&brain, true)?;
            let mut st = read_states(&states)?;
            let total = st.len();
            if skip_trained {
                st.retain(|s| {
                    !d.brain
                        .trained_on(&growformer_system1::state::state_to_text(s))
                });
            }
            if limit > 0 && st.len() > limit {
                st.truncate(limit);
            }
            eprintln!("labelling {} of {total} states ({source:?})", st.len());
            let t0 = std::time::Instant::now();
            let mut f = std::fs::File::create(&out).map_err(|e| e.to_string())?;
            let mut n = 0usize;
            for (i, chunk) in st.chunks(25).enumerate() {
                for r in distill(&mut d, chunk, &questions, source)? {
                    writeln!(f, "{r}").map_err(|e| e.to_string())?;
                    n += 1;
                }
                let done = (i + 1) * 25;
                if done < st.len() {
                    let el = t0.elapsed().as_secs_f32();
                    eprintln!(
                        "  {done}/{}  ({:.0}s, ~{:.0}s left)",
                        st.len(),
                        el,
                        el / done as f32 * (st.len() - done) as f32
                    );
                }
            }
            println!(
                "labelled {n} states in {:.1}s → {}\nnext: gf-system1 train --questions <q.toml> --data <your labelled data> --data {} --out {}",
                t0.elapsed().as_secs_f32(),
                out.display(),
                out.display(),
                brain.display()
            );
        }
        Cmd::TrainProbe {
            project,
            brain,
            data,
            layer,
            cache,
            epochs,
            seed,
        } => {
            use growformer_system1::probe::{Probe, ProbeTrainConfig};
            let bpath = brain_path(&project, &brain)?;
            let data = if data.is_empty() {
                project
                    .as_ref()
                    .map(|p| read_project(p).map(|p| p.data))
                    .transpose()?
                    .unwrap_or_default()
            } else {
                data
            };
            if data.is_empty() {
                return Err("pass --data (labelled JSONL) or --project".into());
            }
            let mut d = load_decider(&bpath, true)?;
            if !d.has_scorer() {
                return Err(
                    "train-probe needs an open model: run `gf-system1 attach-lm` first".into(),
                );
            }
            let key = d.scorer_model_key().expect("scorer attached");
            let (rows, skipped) = read_rows(&data, &d.brain.questions)?;
            if rows.is_empty() {
                return Err("no labelled rows".into());
            }
            let cache_path = cache.unwrap_or_else(|| {
                let mut p = bpath.clone().into_os_string();
                p.push(".emb");
                PathBuf::from(p)
            });
            let mut c = EmbCache::load(&cache_path, &format!("{key}@{layer}"));
            let cached = rows.iter().filter(|r| c.get(&r.text).is_some()).count();
            eprintln!(
                "{} rows ({skipped} skipped); {cached} features cached, computing {}",
                rows.len(),
                rows.len() - cached
            );
            let t0 = std::time::Instant::now();
            let mut feats = Vec::with_capacity(rows.len());
            let mut computed = 0usize;
            let todo = rows.len() - cached;
            for r in &rows {
                if let Some(v) = c.get(&r.text) {
                    feats.push(v.clone());
                    continue;
                }
                let v = d.embed_raw(&r.text, layer)?;
                c.insert(&r.text, v.clone());
                feats.push(v);
                computed += 1;
                if computed.is_multiple_of(25) {
                    let el = t0.elapsed().as_secs_f32();
                    eprintln!(
                        "  {computed}/{todo}  ({:.0}s, ~{:.0}s left)",
                        el,
                        el / computed as f32 * (todo - computed) as f32
                    );
                    c.save(&cache_path)?;
                }
            }
            c.save(&cache_path)?;
            eprintln!(
                "features: {} × {} in {:.1}s; training probe heads…",
                feats.len(),
                feats[0].len(),
                t0.elapsed().as_secs_f32()
            );
            let t1 = std::time::Instant::now();
            let probe = Probe::train(
                &d.brain.questions,
                &rows,
                &feats,
                key,
                layer,
                &ProbeTrainConfig {
                    seed,
                    epochs,
                    ..Default::default()
                },
            )?;
            let b = &mut d.brain;
            println!(
                "{:<16} {:>6} {:>6} {:>7} {:>6} {:>9} {:>9} {:>7}   routing",
                "question", "train", "test", "C", "temp", "head acc", "probe acc", "p.ece"
            );
            for q in &b.questions.questions {
                let Some(h) = probe.heads.get(&q.id) else {
                    println!("{:<16}   no probe (too few labels)", q.id);
                    continue;
                };
                let pr = probe.reports.get(&q.id);
                let hr = b.reports.get(&q.id);
                // Prefer the probe unless the head did better on held-out rows.
                let probe_wins = match (pr, hr) {
                    (Some(p), Some(h)) => p.calibrated.accuracy >= h.calibrated.accuracy,
                    _ => true,
                };
                if probe_wins {
                    b.router.per_question.insert(q.id.clone(), 1.01);
                } else {
                    b.router.per_question.remove(&q.id);
                }
                println!(
                    "{:<16} {:>6} {:>6} {:>7} {:>6} {:>9} {:>9} {:>7}   {}",
                    q.id,
                    pr.map(|r| r.n_train.to_string())
                        .unwrap_or_else(|| h.n_train.to_string()),
                    pr.map(|r| r.n_holdout.to_string())
                        .unwrap_or_else(|| "–".into()),
                    h.c,
                    pr.map(|r| format!("{:.2}", r.temperature))
                        .unwrap_or_else(|| "–".into()),
                    fmt_opt(hr.map(|r| r.calibrated.accuracy)),
                    fmt_opt(pr.map(|r| r.calibrated.accuracy)),
                    fmt_opt(pr.map(|r| r.calibrated.ece)),
                    if probe_wins {
                        "probe".to_string()
                    } else {
                        format!(
                            "head if conf ≥ {:.2}, else probe",
                            b.router.threshold(&q.id)
                        )
                    }
                );
            }
            b.mark_trained(rows.iter().map(|r| r.text.clone()));
            b.probe = Some(probe);
            b.save(&bpath)?;
            println!(
                "trained in {:.1}s → {} (features cached in {})\nnext: gf-system1 eval --brain {} --data <held-out rows>",
                t1.elapsed().as_secs_f32(),
                bpath.display(),
                cache_path.display(),
                bpath.display()
            );
        }
        Cmd::Stats {
            project,
            questions,
            data,
            min,
        } => {
            let proj = project.as_deref().map(read_project).transpose()?;
            let qpath = questions
                .or(proj.as_ref().map(|p| p.questions.clone()))
                .ok_or("pass --questions or --project")?;
            let data = if data.is_empty() {
                proj.map(|p| p.data).unwrap_or_default()
            } else {
                data
            };
            let qs = read_questions(&qpath)?;
            let (rows, skipped) = read_rows(&data, &qs)?;
            println!(
                "{} labelled rows ({skipped} without a usable label)",
                rows.len()
            );
            let mut thin = 0usize;
            for (qid, counts) in label_counts(&rows, &qs) {
                let total: usize = counts.iter().map(|c| c.1).sum();
                println!("\n{qid}  ({total} rows)");
                for (o, n) in counts {
                    let flag = if n < min {
                        thin += 1;
                        format!("  ← below {min}")
                    } else {
                        String::new()
                    };
                    println!("  {o:<24} {n:>5}{flag}");
                }
            }
            if thin > 0 {
                println!("\n{thin} outcome(s) below {min} rows: add examples or merge them into a broader outcome.");
            }
        }
        #[cfg(feature = "server")]
        Cmd::Serve {
            project,
            brain,
            addr,
            no_lm,
        } => {
            let bpath = brain_path(&project, &brain)?;
            let d = load_decider(&bpath, !no_lm)?;
            serve(d, &addr)?;
        }
    }
    Ok(())
}

#[cfg(feature = "server")]
fn serve(d: growformer_system1::Decider, addr: &str) -> Result<(), String> {
    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use std::sync::{Arc, Mutex};

    type Shared = Arc<Mutex<growformer_system1::Decider>>;
    let info = json!({
        "brain": d.brain.name,
        "questions": d.brain.questions.questions.iter().map(|q| q.id.clone()).collect::<Vec<_>>(),
        "heads": d.brain.heads.keys().cloned().collect::<Vec<_>>(),
        "open_model": d.scorer_description(),
    });
    let shared: Shared = Arc::new(Mutex::new(d));

    async fn decide(
        State(s): State<Shared>,
        Json(req): Json<DecideRequest>,
    ) -> Result<Json<growformer_system1::DecideResponse>, (StatusCode, Json<Value>)> {
        tokio::task::spawn_blocking(move || {
            let mut d = s.lock().map_err(|_| "decider lock poisoned".to_string())?;
            d.decide(&req)
        })
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        })?
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))
    }

    async fn questions(State(s): State<Shared>) -> Json<Value> {
        let d = s.lock().unwrap();
        Json(serde_json::to_value(&d.brain.questions).unwrap_or(Value::Null))
    }

    let health = Json(info.clone());
    let app = Router::new()
        .route("/v1/decide", post(decide))
        .route("/v1/questions", get(questions))
        .route("/health", get(move || async move { health }))
        .with_state(shared);

    let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| format!("bind {addr}: {e}"))?;
        eprintln!("gf-system1 serving {info} on http://{addr}  (POST /v1/decide)");
        axum::serve(listener, app).await.map_err(|e| e.to_string())
    })
}
