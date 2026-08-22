// examples/sentiment_generalization.rs
// Full end-to-end demo: three-stage curriculum training + generalization test.

use growformer::{
    training::example_training_batch,
    sentiment::SentimentFunctor,
    curriculum::CurriculumScheduler,
    growformer::{GrowformerNode, GrowformerTrainer},
    category::NodeId,
};

fn main() {
    let batch = example_training_batch();

    // Curriculum: scaffold=80 steps, loosen=80–200, harden=200+
    let curriculum = CurriculumScheduler::new(80, 200);
    let mut trainer = GrowformerTrainer::new(curriculum);

    // Three nodes: parse (bifunctor root), sentiment (left), entity (right)
    trainer.add_node(GrowformerNode::new(NodeId::new(0), "parse",     512, vec![0.01; 512]));
    trainer.add_node(GrowformerNode::new(NodeId::new(1), "sentiment", 360, vec![0.01; 360]));
    trainer.add_node(GrowformerNode::new(NodeId::new(2), "entity",    362, vec![0.01; 362]));

    trainer.train(&batch, 300);
    trainer.stage_summary();

    // Inference tests
    let functor = SentimentFunctor::new(vec![0.3; 64], vec![0.1; 64], 0.05);
    println!("── Generalization Inference ──\n");
    for input in [
        "I hate mondays",       // seen
        "I love fridays",       // seen
        "I hate wednesdays",    // day substitution
        "I hate sundays",       // day substitution
        "I hate tuesdays",      // day substitution
        "I hate rain",          // cross-category: weather
        "I love coffee",        // cross-category: object
        "I hate deadlines",     // cross-category: event-adjacent
    ] {
        trainer.infer(input, &functor).display();
    }
}
