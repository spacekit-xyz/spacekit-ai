#![cfg(feature = "categorical")]

use growformer::category::curriculum::CurriculumScheduler;
use growformer::category::forward::char_hash_embed;
use growformer::category::growformer::{GrowformerNode, GrowformerTrainer, TrainerConfig};
use growformer::category::training::example_training_batch;
use growformer::category::{bifunctor_branch_vectors, record_embedding, NodeId};

#[test]
fn trainer_reduces_sentiment_ce_on_batch() {
    let batch = example_training_batch();
    let curriculum = CurriculumScheduler::new(5, 10);
    let embed_dim = 64usize;
    let branch_dim = 24usize;
    let cfg = TrainerConfig {
        parse_node_id: 0,
        embed_dim,
        branch_dim,
        lr: 0.2,
        head_seed: 7,
        ..Default::default()
    };
    let mut trainer = GrowformerTrainer::with_config(curriculum, cfg);
    trainer.add_node(GrowformerNode::new(
        NodeId::new(0),
        "parse",
        embed_dim,
        vec![0.5f32; embed_dim],
    ));

    let first = &batch.records[0];
    let loss_before = {
        let emb = record_embedding(first, embed_dim);
        let comp = &trainer.nodes[&0].node.composition;
        let (s, _) = bifunctor_branch_vectors(comp, &emb, branch_dim);
        trainer.sentiment_head.cross_entropy(&s, first.sentiment.class_index())
    };

    for _ in 0..80 {
        trainer.step(&batch);
    }

    let loss_after = {
        let emb = record_embedding(first, embed_dim);
        let comp = &trainer.nodes[&0].node.composition;
        let (s, _) = bifunctor_branch_vectors(comp, &emb, branch_dim);
        trainer.sentiment_head.cross_entropy(&s, first.sentiment.class_index())
    };

    assert!(
        loss_after < loss_before,
        "expected CE to drop: before={} after={}",
        loss_before,
        loss_after
    );
}

#[test]
fn infer_head_runs_after_registering_parse() {
    let curriculum = CurriculumScheduler::new(10, 20);
    let mut trainer = GrowformerTrainer::new(curriculum);
    let d = 32usize;
    trainer.add_node(GrowformerNode::new(NodeId::new(0), "parse", d, vec![0.3f32; d]));
    let r = trainer.infer_head("I hate mondays");
    assert!(r.is_ok());
}

#[test]
fn infer_head_detail_has_probs_and_aux_head() {
    let curriculum = CurriculumScheduler::new(10, 20);
    let mut trainer = GrowformerTrainer::new(curriculum);
    let d = 16usize;
    trainer.add_node(GrowformerNode::new(NodeId::new(0), "parse", d, vec![0.2f32; d]));
    let det = trainer.infer_head_detail("I love fridays").expect("detail");
    assert_eq!(det.sentiment_probs.len(), 7);
    assert_eq!(det.aux_probs.len(), 6);
    let s: f32 = det.sentiment_probs.iter().sum();
    assert!((s - 1.0).abs() < 1e-3);
    let compact = det.to_result();
    assert_eq!(compact.inferred_category, det.aux_predicted);
}

#[test]
fn infer_head_with_embedding_matches_explicit_hash() {
    let curriculum = CurriculumScheduler::new(10, 20);
    let d = 24usize;
    let cfg = TrainerConfig {
        parse_node_id: 0,
        embed_dim: d,
        branch_dim: 16,
        lr: 0.08,
        head_seed: 42,
        ..Default::default()
    };
    let mut trainer = GrowformerTrainer::with_config(curriculum, cfg);
    trainer.add_node(GrowformerNode::new(NodeId::new(0), "parse", d, vec![0.15f32; d]));
    let text = "snow is cold";
    let emb = char_hash_embed(text, d);
    let a = trainer.infer_head_with_embedding(text, &emb).expect("a");
    let b = trainer.infer_head(text).expect("b");
    assert_eq!(a.sentiment, b.sentiment);
    assert_eq!(a.inferred_category, b.inferred_category);
}

#[test]
fn infer_head_batch_len_matches() {
    let curriculum = CurriculumScheduler::new(10, 20);
    let mut trainer = GrowformerTrainer::new(curriculum);
    let d = 12usize;
    trainer.add_node(GrowformerNode::new(NodeId::new(0), "parse", d, vec![0.4f32; d]));
    let lines = ["a", "b c", "d e f"];
    let out = trainer.infer_head_batch(&lines);
    assert_eq!(out.len(), 3);
    assert!(out.iter().all(|r| r.is_ok()));
}
