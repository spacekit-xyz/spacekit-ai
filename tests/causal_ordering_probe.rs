use growformer::clifford::{
    causal_block_bivectors, causal_block_interval, causal_block_vector, classify_interval,
    embed_bridge_vector, temporal_ordering_loss, temporal_ordering_score,
    causal_forward_energy, causal_retro_energy, causal_intervention_energy,
    causal_grade_logits, causal_grade_loss, combined_causal_loss,
    causal_contrastive_repulsion, CausalGrade, IntervalType,
    CAUSAL_BLADE_COUNT, CAUSAL_BLOCK_DIM,
};
use growformer::dimension::language::{
    CausalAnnotation, EncoderPreset, HashingLanguageEncoder, LanguageEncoder,
};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct CausalRow {
    task_id: String,
    text: String,
    #[serde(default)]
    causal: Option<CausalAnnotation>,
}

fn load_causal_rows() -> Vec<CausalRow> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/fintech/train_sentiment_causal.jsonl"
    );
    let data = fs::read_to_string(path).expect("read causal JSONL");
    data.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("parse causal row"))
        .collect()
}

fn embed_text(encoder: &HashingLanguageEncoder, text: &str) -> growformer::clifford::Multivector {
    let vec = encoder.encode(text);
    embed_bridge_vector(&vec)
}

#[test]
fn causal_block_dimensions_correct() {
    assert_eq!(CAUSAL_BLOCK_DIM, 4);
    assert_eq!(CAUSAL_BLADE_COUNT, 6);
}

#[test]
fn temporal_ordering_sign_consistency() {
    let encoder = HashingLanguageEncoder::new(EncoderPreset::MiniLmL6V2);
    let rows = load_causal_rows();

    let mut forward_scores: Vec<(String, f32)> = Vec::new();
    let mut retro_scores: Vec<(String, f32)> = Vec::new();

    for row in &rows {
        let Some(ref causal) = row.causal else {
            continue;
        };
        let (Some(ref cause), Some(ref effect)) = (&causal.cause_span, &causal.effect_span) else {
            continue;
        };

        let cause_mv = embed_text(&encoder, cause);
        let effect_mv = embed_text(&encoder, effect);
        let score = temporal_ordering_score(&cause_mv, &effect_mv);

        let is_retro = causal
            .causal_subtype
            .as_deref()
            .map_or(false, |s| s == "retrospective_framing");

        if is_retro {
            retro_scores.push((row.task_id.clone(), score));
        } else {
            forward_scores.push((row.task_id.clone(), score));
        }
    }

    assert!(
        !forward_scores.is_empty(),
        "should have forward causal pairs"
    );
    assert!(!retro_scores.is_empty(), "should have retrospective pairs");

    println!("--- Forward causal pairs ({}) ---", forward_scores.len());
    let mut fwd_positive = 0;
    for (id, score) in &forward_scores {
        let dir = if *score > 0.0 { "+" } else { "-" };
        println!("  {id}: {dir}{:.4}", score.abs());
        if *score > 0.0 {
            fwd_positive += 1;
        }
    }

    println!("--- Retrospective pairs ({}) ---", retro_scores.len());
    for (id, score) in &retro_scores {
        let dir = if *score > 0.0 { "+" } else { "-" };
        println!("  {id}: {dir}{:.4}", score.abs());
    }

    let fwd_rate = fwd_positive as f32 / forward_scores.len() as f32;
    println!(
        "\nForward positive rate: {:.1}% ({}/{})",
        fwd_rate * 100.0,
        fwd_positive,
        forward_scores.len()
    );
    println!("(At this stage, the hashing encoder is not trained on causal structure,");
    println!(" so sign consistency is a baseline — improvement comes from training.)");
}

#[test]
fn causal_block_interval_varies_across_rows() {
    let encoder = HashingLanguageEncoder::new(EncoderPreset::MiniLmL6V2);
    let rows = load_causal_rows();

    let mut intervals: Vec<(String, f32, IntervalType)> = Vec::new();
    for row in &rows {
        let Some(ref causal) = row.causal else {
            continue;
        };
        let (Some(ref cause), Some(ref effect)) = (&causal.cause_span, &causal.effect_span) else {
            continue;
        };
        let cause_mv = embed_text(&encoder, cause);
        let effect_mv = embed_text(&encoder, effect);
        let s2 = causal_block_interval(&cause_mv, &effect_mv);
        let itype = classify_interval(s2);
        intervals.push((row.task_id.clone(), s2, itype));
    }

    assert!(!intervals.is_empty());

    let timelike = intervals.iter().filter(|(_, _, t)| *t == IntervalType::Timelike).count();
    let spacelike = intervals.iter().filter(|(_, _, t)| *t == IntervalType::Spacelike).count();
    let lightlike = intervals.iter().filter(|(_, _, t)| *t == IntervalType::Lightlike).count();

    println!("--- Causal block intervals ({} pairs) ---", intervals.len());
    println!(
        "  Timelike: {timelike}, Spacelike: {spacelike}, Lightlike: {lightlike}"
    );
    for (id, s2, itype) in &intervals {
        println!("  {id}: s²={s2:.4} ({itype:?})");
    }
}

#[test]
fn temporal_ordering_loss_nonzero_on_untrained_encoder() {
    let encoder = HashingLanguageEncoder::new(EncoderPreset::MiniLmL6V2);
    let rows = load_causal_rows();

    let mut total_loss = 0.0f32;
    let mut count = 0;
    for row in &rows {
        let Some(ref causal) = row.causal else {
            continue;
        };
        let (Some(ref cause), Some(ref effect)) = (&causal.cause_span, &causal.effect_span) else {
            continue;
        };
        let is_retro = causal
            .causal_subtype
            .as_deref()
            .map_or(false, |s| s == "retrospective_framing");
        let cause_mv = embed_text(&encoder, cause);
        let effect_mv = embed_text(&encoder, effect);
        let loss = temporal_ordering_loss(&cause_mv, &effect_mv, !is_retro, 0.5);
        total_loss += loss;
        count += 1;
    }

    assert!(count > 0);
    let avg = total_loss / count as f32;
    println!(
        "Average temporal ordering loss (untrained, margin=0.5): {avg:.4} over {count} pairs"
    );
    println!("(This is the baseline that training should reduce.)");
}

#[test]
fn causal_grade_distribution_baseline() {
    let encoder = HashingLanguageEncoder::new(EncoderPreset::MiniLmL6V2);
    let rows = load_causal_rows();

    let mut forward_count = 0usize;
    let mut retro_count = 0usize;
    let mut interv_count = 0usize;
    let mut grade_loss_sum = 0.0f32;
    let mut grade_correct = 0usize;
    let mut total = 0usize;

    for row in &rows {
        let Some(ref causal) = row.causal else { continue };
        let (Some(ref cause), Some(ref effect)) = (&causal.cause_span, &causal.effect_span) else {
            continue;
        };
        let grade = CausalGrade::from_labels(
            &causal.causal_type,
            causal.causal_subtype.as_deref(),
        );
        match grade {
            CausalGrade::Forward => forward_count += 1,
            CausalGrade::Retrospective => retro_count += 1,
            CausalGrade::Interventional => interv_count += 1,
        }

        let cause_mv = embed_text(&encoder, cause);
        let effect_mv = embed_text(&encoder, effect);
        let logits = causal_grade_logits(&cause_mv, &effect_mv);
        let loss = causal_grade_loss(&cause_mv, &effect_mv, grade);
        grade_loss_sum += loss;

        let predicted = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if predicted == grade.class_index() {
            grade_correct += 1;
        }
        total += 1;
    }

    println!("--- Causal grade distribution ({total} rows) ---");
    println!("  Forward: {forward_count}, Retrospective: {retro_count}, Interventional: {interv_count}");
    let accuracy = if total > 0 { grade_correct as f32 / total as f32 } else { 0.0 };
    let avg_loss = if total > 0 { grade_loss_sum / total as f32 } else { 0.0 };
    println!("  Grade accuracy (untrained): {:.1}% ({grade_correct}/{total})", accuracy * 100.0);
    println!("  Average grade CE loss: {avg_loss:.4}");
    println!("(Baseline: random would be ~33.3%. Improvement comes from training.)");

    assert!(total >= 37, "should have at least 37 causal rows with spans");
}

#[test]
fn causal_energy_profiles_differ_by_grade() {
    let encoder = HashingLanguageEncoder::new(EncoderPreset::MiniLmL6V2);
    let rows = load_causal_rows();

    let mut fwd_energies: Vec<f32> = Vec::new();
    let mut ret_energies: Vec<f32> = Vec::new();
    let mut int_energies: Vec<f32> = Vec::new();

    for row in &rows {
        let Some(ref causal) = row.causal else { continue };
        let (Some(ref cause), Some(ref effect)) = (&causal.cause_span, &causal.effect_span) else {
            continue;
        };
        let grade = CausalGrade::from_labels(
            &causal.causal_type,
            causal.causal_subtype.as_deref(),
        );
        let cause_mv = embed_text(&encoder, cause);
        let effect_mv = embed_text(&encoder, effect);

        let fwd_e = causal_forward_energy(&cause_mv, &effect_mv);
        let ret_e = causal_retro_energy(&cause_mv, &effect_mv);
        let int_e = causal_intervention_energy(&cause_mv, &effect_mv);

        match grade {
            CausalGrade::Forward => fwd_energies.push(fwd_e[2]),
            CausalGrade::Retrospective => ret_energies.push(ret_e[0].abs()),
            CausalGrade::Interventional => int_energies.push(int_e),
        }
    }

    let avg = |v: &[f32]| -> f32 {
        if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 }
    };

    println!("--- Causal energy profiles (untrained baseline) ---");
    println!("  Forward boost-plane norm (avg): {:.4} ({} rows)", avg(&fwd_energies), fwd_energies.len());
    println!("  Retro |e_03| (avg):             {:.4} ({} rows)", avg(&ret_energies), ret_energies.len());
    println!("  Interventional bivector mag (avg): {:.4} ({} rows)", avg(&int_energies), int_energies.len());
    println!("(Training should separate these profiles further.)");

    assert!(!fwd_energies.is_empty());
    assert!(!ret_energies.is_empty());
    assert!(!int_energies.is_empty());
}

#[test]
fn contrastive_repulsion_baseline_on_contrast_groups() {
    let encoder = HashingLanguageEncoder::new(EncoderPreset::MiniLmL6V2);
    let rows = load_causal_rows();

    struct GradeEntry {
        cause_mv: growformer::clifford::Multivector,
        effect_mv: growformer::clifford::Multivector,
        grade: CausalGrade,
    }
    let mut groups: std::collections::HashMap<String, Vec<GradeEntry>> = std::collections::HashMap::new();

    for row in &rows {
        let Some(ref causal) = row.causal else { continue };
        let (Some(ref cause), Some(ref effect)) = (&causal.cause_span, &causal.effect_span) else {
            continue;
        };
        let Some(ref cg) = causal.contrast_group else { continue };
        let grade = CausalGrade::from_labels(
            &causal.causal_type,
            causal.causal_subtype.as_deref(),
        );
        let cause_mv = embed_text(&encoder, cause);
        let effect_mv = embed_text(&encoder, effect);
        groups.entry(cg.clone()).or_default().push(GradeEntry {
            cause_mv, effect_mv, grade,
        });
    }

    let mut total_repulsion = 0.0f32;
    let mut pair_count = 0u32;

    for (group_name, entries) in &groups {
        for (i, ei) in entries.iter().enumerate() {
            for ej in entries.iter().skip(i + 1) {
                if ei.grade == ej.grade { continue; }
                let (fwd, ret) = if ei.grade == CausalGrade::Forward
                    || (ei.grade != CausalGrade::Retrospective
                        && ej.grade == CausalGrade::Retrospective)
                {
                    ((&ei.cause_mv, &ei.effect_mv), (&ej.cause_mv, &ej.effect_mv))
                } else {
                    ((&ej.cause_mv, &ej.effect_mv), (&ei.cause_mv, &ei.effect_mv))
                };
                let rep = causal_contrastive_repulsion(fwd, ret, 0.3);
                total_repulsion += rep;
                pair_count += 1;
                println!(
                    "  {}: {:?} vs {:?} -> repulsion={:.4}",
                    group_name, ei.grade, ej.grade, rep
                );
            }
        }
    }

    let avg_rep = if pair_count > 0 { total_repulsion / pair_count as f32 } else { 0.0 };
    println!("\n--- Contrastive repulsion baseline ---");
    println!("  Pairs: {pair_count}, Avg repulsion: {avg_rep:.4}");
    println!("(Training should push repulsion toward 0 as grades separate.)");

    assert!(pair_count >= 5, "should have at least 5 cross-grade pairs, got {pair_count}");
}
