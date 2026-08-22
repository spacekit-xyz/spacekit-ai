//! Train the categorical sentiment scaffold on `data/sentiment` JSONL.
//!
//! ```text
//! cargo run --example categorical_sentiment_train --features categorical -- [DATA_DIR] [STEPS] [embed_dim]
//! ```
//!
//! Defaults: `DATA_DIR=data/sentiment`, `STEPS=400`, `embed_dim=128`.
//! Uses token-hash embeddings and optional plural heuristic. Match `embed_dim` to the parse node.
//!
//! Optional: `GROWFORMER_BRANCH_STATS=3` logs norms + cosine for the first 3 batch rows every 50 steps
//! (stderr) to confirm branch vectors are stable when the parse tree is untrained (not a cache bug).

use growformer::category::curriculum::CurriculumScheduler;
use growformer::category::embedding::TokenHashEmbedder;
use growformer::category::growformer::{GrowformerNode, GrowformerTrainer, TrainerConfig};
use growformer::category::training::{SentimentJsonlSelection, TrainingBatch};
use growformer::category::NodeId;

fn main() {
    let mut args = std::env::args().skip(1);
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_dir = args
        .next()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| manifest.join("data/sentiment"));
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(400);
    let embed_dim: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let branch_dim: usize = (embed_dim / 2).max(32);

    if !data_dir.is_dir() {
        eprintln!(
            "Not a directory: {} (pass path to data/sentiment)",
            data_dir.display()
        );
        std::process::exit(1);
    }

    let mut batch =
        TrainingBatch::from_sentiment_jsonl_dir(&data_dir, SentimentJsonlSelection::TrainFilesOnly)
            .unwrap_or_else(|e| panic!("load {:?}: {}", data_dir, e));

    println!("Loaded {} training rows from {:?}", batch.len(), data_dir);

    let embedder = TokenHashEmbedder::new(embed_dim);
    batch.fill_missing_embeddings(&embedder);
    batch.reinforce_plural_with_heuristic();

    let curriculum = CurriculumScheduler::new(120, 280);
    let cfg = TrainerConfig {
        parse_node_id: 0,
        embed_dim,
        branch_dim,
        lr: 0.06,
        head_seed: 42,
        branch_stats_sample_count: std::env::var("GROWFORMER_BRANCH_STATS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        ..Default::default()
    };
    let mut trainer = GrowformerTrainer::with_config(curriculum, cfg);
    trainer.add_node(GrowformerNode::new(
        NodeId::new(0),
        "parse",
        embed_dim,
        vec![0.35f32; embed_dim],
    ));

    trainer.train(&batch, steps);
    trainer.stage_summary();

    let sample = batch
        .records
        .first()
        .map(|r| r.input.as_str())
        .unwrap_or("I hate mondays");
    if let Ok(out) = trainer.infer_head_detail(sample) {
        println!("\nSample detail for: {:?}", sample);
        out.display();
    }
}
