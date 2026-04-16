use growformer::clifford::{
    causal_block_bivectors, causal_block_interval, causal_block_vector, classify_interval,
    embed_bridge_vector, temporal_ordering_loss, temporal_ordering_score, IntervalType,
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
