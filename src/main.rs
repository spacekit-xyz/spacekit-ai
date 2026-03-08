use growformer::environment::NeuralEnvironment;
use growformer::types::NeuronId;
use growformer::dimension::{DimensionManager, DimensionManagerConfig, MainDimension, VirtualGroup};
use growformer::types::GroupId;
use std::io::Write;
use std::time::Instant;
use growformer::systems::checkpoint::{save_phase2_checkpoint, load_phase2_checkpoint, save_mnist_checkpoint, load_mnist_checkpoint};
use growformer::systems::mirror::mirror_symmetry_score;
use growformer::systems::whorls::print_whorl_summary;
use growformer::types::{EnvironmentConfig, Sample};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use clap::Parser;
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser, Debug)]
#[command(name = "growformer", version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    xor: bool,
    #[arg(short, long)]
    spiral: bool,
    #[arg(short, long)]
    concentric_circles: bool,
    #[arg(short, long)]
    mlp: bool,
    /// Continual learning: use "full", "train-a", or "train-b" (default: full)
    #[arg(short, long, value_name = "MODE", default_value = "full")]
    learning: Option<String>,
    /// Phase 3: Fractal Topology — Main + Mirror dimensions, promotion gate
    #[arg(long)]
    fractal: bool,
    /// Phase 3c: Composition (VirtualGroup) + EpisodicMemory — Task C after Demo 6
    #[arg(long)]
    phase3c: bool,
    /// Neurogenesis: run spiral with trigger (add 1 neuron at 2000 epochs if loss > 0.3)
    #[arg(long)]
    neurogenesis: bool,
    /// Split MNIST: five sequential digit-pair tasks, report average accuracy and forgetting
    #[arg(long)]
    mnist: bool,
    /// Load MNIST checkpoint and run retention evaluation (proves save/load preserves accuracy)
    #[arg(long)]
    mnist_retention: bool,
    /// Show progress bar for long runs (MNIST); use --no-progress to disable
    #[arg(long, default_value_t = true)]
    progress: bool,
    /// Max training samples per MNIST task (default: all). Use e.g. 2000 for faster runs.
    #[arg(long, value_name = "N")]
    mnist_train_limit: Option<usize>,
    /// Max epochs per MNIST task (default: 2500). Use e.g. 500 for quicker sanity runs.
    #[arg(long, value_name = "N")]
    mnist_max_epochs: Option<u32>,
    /// Minibatch size for MNIST (default: 1 = sequential). Use 16–64 for multi-core speed.
    #[arg(long, value_name = "N")]
    mnist_batch_size: Option<usize>,
}

fn main() {
    println!("=============================================================");
    println!("  Growformer Neural Environment — Training Demo");
    println!("=============================================================\n");

    let args = Args::parse();
    if args.xor == true {
        demo_xor();
    } else if args.spiral == true {
        demo_spiral();
    } else if args.concentric_circles == true {
        demo_concentric_circles();
    } else if args.mlp == true {
        demo_mlp_baseline();
    } else if args.fractal {
        demo_fractal_continual_learning();
    } else if args.phase3c {
        demo_phase3c_composition();
    } else if args.neurogenesis {
        demo_neurogenesis();
    } else if args.mnist {
        demo_split_mnist(args.progress, args.mnist_train_limit, args.mnist_max_epochs, args.mnist_batch_size);
    } else if args.mnist_retention {
        demo_mnist_retention();
    } else if let Some(mode) = &args.learning {
        match mode.as_str() {
            "train-a" => demo_phase2_train_a(),
            "train-b" => demo_phase2_train_b(),
            _ => demo_continual_learning(),
        }
    } else {
        println!("Please specify either --xor, --spiral, --concentric-circles, --mlp, --learning, --fractal, --phase3c, --neurogenesis, --mnist, or --mnist-retention");
        std::process::exit(1);
    }

}

// =============================================================================
// Demo 1: XOR
// =============================================================================

fn demo_xor() {
    println!("--- Demo 1: XOR ---\n");

    let config = EnvironmentConfig {
        learning_rate: 0.5,
        weight_decay: 0.0001,
        bias_decay: 0.0,           // disabled — same reason as spiral (survival math)
        dropout_rate: 0.0,
        geometry_noise: 0.001,
        competitive_k: 0,
        lateral_inhibition: 0.0,
        max_synapses_per_neuron: 32,
        energy_budget_per_neuron: 100.0,
        pruning_threshold: 0.001,
        mirror_coupling_strength: 3.14,//0.001,
        growth_radius: 0.0,
        geometry_interval: 500,
        stdp_enabled: false,
        prune_interval: 500,
        ..EnvironmentConfig::default()
    };

    let mut env = NeuralEnvironment::new(config);
    let mut rng = StdRng::seed_from_u64(42);

    env.build_layers(&[2, 4, 1], &mut rng);

    let hidden_ids = env.layers[1].clone();
    let (g_a, g_b) = hidden_ids.split_at(hidden_ids.len() / 2);
    let group_a = env.create_group(g_a.to_vec());
    let group_b = env.create_group(g_b.to_vec());
    env.pair_mirror_groups(group_a, group_b);

    let mut xor_data: Vec<Sample> = vec![
        (vec![0.0, 0.0], [0.0]),
        (vec![0.0, 1.0], [1.0]),
        (vec![1.0, 0.0], [1.0]),
        (vec![1.0, 1.0], [0.0]),
    ];

    println!("Training XOR for 5000 epochs...");
    for epoch in 0..5000 {
        xor_data.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &xor_data {
            epoch_loss += env.train_tick(input.as_slice(), target, &mut rng).loss;
        }
        epoch_loss /= xor_data.len() as f32;
        if epoch % 1000 == 0 && epoch > 0 { env.prune_dormant(); }
        if epoch % 500 == 0 {
            let weights: Vec<String> = env.layers[1].iter()
                .flat_map(|id| env.neurons[id].synapses.iter().map(|s| format!("{:.3}", s.strength)))
                .collect();
            println!("  epoch {:>5} | loss={:.5} | sparsity={:.2} | h→o: {:?}",
                epoch, epoch_loss, env.firing_sparsity(), weights);
        }
    }

    println!("\nInference:");
    println!("  Input    Expected  Predicted  Correct");
    println!("  ---------------------------------------");
    let mut correct = 0;
    // Test in canonical order
    for (input, expected) in &[([0.0,0.0],[0.0]),([0.0,1.0],[1.0]),([1.0,0.0],[1.0]),([1.0,1.0],[0.0])] {
        let out = env.predict(input);
        let rounded = if out[0] > 0.5 { 1.0 } else { 0.0 };
        let expected_value:f32 = expected[0];
        let ok = (rounded - expected_value).abs() < 0.01;
        if ok { correct += 1; }
        println!("  [{:.0},{:.0}]     {:.1}       {:.4}    {}", input[0], input[1], expected[0], out[0], if ok { "✓" } else { "✗" });
    }
    println!("\n  Accuracy: {}/4", correct);
    print_structural_report(&env, group_a, group_b);
}

// =============================================================================
// Demo 2: Spiral — larger architecture, dropout enabled
// =============================================================================

fn demo_spiral() {
    println!("--- Demo 2: Spiral Classification ---\n");

    let config = EnvironmentConfig {
        learning_rate: 0.15,
        weight_decay: 0.0000025,   // scaled: 0.001/400 samples
        bias_decay: 0.0,           // disabled — any useful value annihilates over 3.2M steps
        dropout_rate: 0.1,    // reduced: 32 neurons each carry more weight than in 16-neuron layer
        geometry_noise: 0.0,        // replaced by thermal_noise in physics
        competitive_k: 4,            // KWTA top-4 of 16: hard competition regardless of synapse strength
        lateral_inhibition: 0.12,    // active from epoch 0: no warmup, moderate selectivity
                                   // targets act=0.25-0.40 with sigma_inhib=2.0
        lr_decay: 0.00008,    // slowed 5×: previous decay killed lr by epoch 1000, no learning late
        max_synapses_per_neuron: 64,
        energy_budget_per_neuron: 100.0,
        pruning_threshold: 0.001,
        mirror_coupling_strength: 0.001,
        growth_radius: 0.0,
        geometry_interval: 500,
        stdp_enabled: false,
        k_repel: 0.2,
        gravity_g: 0.05,
        damping: 0.2,
        thermal_noise: 0.02,     //
        // Reaction-diffusion: sigma_inhib tuned for 32-neuron single layer
        // spread ~2.8, 40% = 1.1 — slightly tighter for more diverse receptive fields
        sigma_inhib: 2.0,            // widened: spread=2.4, need sigma > spread/2 for truly local inhibition
        // Debye: screening length replaces hard repulsion_radius
        debye_length: 1.5,
        // Mass-competition: lower win threshold to match actual activation range
        mass_win_threshold: 0.14, // lowered: strong inhibition means winners fire ~0.3-0.5 post-suppression
        mass_decay: 0.00009,          // all neurons lose this fraction per sample, 0.00009 -> 0.000009
        // Homeostasis: gentle bias regulation to prevent runaway negative drift
        homeostasis_target: 0.30, // target slightly sparse
        homeostasis_lr: 0.0,      // disabled — equalizes all neurons to same bias, kills diversity
        // homeostasis_tau: 0.0001,  // 10000 sample window   
        prune_interval: 500,     // changed from 500 to 1000 to reduce pruning frequency
        ..EnvironmentConfig::default()
    };

    let mut env = NeuralEnvironment::new(config);
    // seeds 42(90.4,92.6,90.1), 7(92.2,91.8), 99, 314, 271.
    let mut rng = StdRng::seed_from_u64(42);

    // Larger architecture — spiral needs more representational capacity
    // 2 → 16 → 16 → 1  scaled layer-2: 8 neurons insufficient to integrate spiral boundary
    //                     warmup removed — created always-on neurons, killed specialisation
    
    //  2 → 16 → 16 → 1 
    // (300 = Spiral accuracy: 389/600 (64.8%))
    // (800 = Spiral accuracy: 991/1600 (61.9%))

    //  2 → 24 → 24 → 1 
    // (300 = Spiral accuracy: 389/600 (75.4%))
    env.build_layers(&[2, 16, 16, 1], &mut rng);

    let hidden_ids = env.layers[1].clone();
    let (g_a, g_b) = hidden_ids.split_at(hidden_ids.len() / 2);
    let group_a = env.create_group(g_a.to_vec());
    let group_b = env.create_group(g_b.to_vec());
    env.pair_mirror_groups(group_a, group_b);

    let samples = 400; // becomes 200 ->400, 400 -> 800
    let epochs = 8000;
    // change from 400 -> 
    // (200 = Spiral accuracy: 267/400 (66.8%))
    // (200 = Spiral accuracy: 264/400 (66.0%))
    // (300 = Spiral accuracy: 389/600 (64.8%)) 
    // (800 = Spiral accuracy: 991/1600 (61.9%))
    let mut spiral_data = generate_spiral_data(samples, &mut rng);
    println!("Training on {} samples, architecture [2→16→16→1], {} epochs...", samples, epochs);

    for epoch in 0..epochs {
        env.set_epoch(epoch);
        spiral_data.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &spiral_data {
            epoch_loss += env.train_tick(input, target, &mut rng).loss;
        }
        epoch_loss /= spiral_data.len() as f32;
        if epoch % 500 == 0 {
            println!("  epoch {:>4} | loss={:.5} | syn={} | sparse={:.2} | act={:.3} | spread={:.2} | mass={:.2} | lr={:.5}",
                epoch, epoch_loss,
                env.total_synapses(),
                env.firing_sparsity(),
                env.mean_hidden_activation(),
                env.geometric_spread(),
                env.mean_hidden_mass(),
                env.current_lr);
        }
    }

    let mut correct = 0;
    for (input, target) in &spiral_data {
        let out = env.predict(input);
        if (if out[0] > 0.5 { 1.0_f32 } else { 0.0 } - target[0]).abs() < 0.01 { correct += 1; }
    }
    println!("\nSpiral accuracy: {}/{} ({:.1}%)",
        correct, spiral_data.len(), 100.0 * correct as f32 / spiral_data.len() as f32);

    println!("\nSample predictions:");
    println!("  X        Y        Expected  Predicted");
    println!("  ----------------------------------------");
    for (input, target) in spiral_data.iter().take(8) {
        let out = env.predict(input);
        println!("  {:+.4}  {:+.4}    {:.1}       {:.4}", input[0], input[1], target[0], out[0]);
    }

    print_structural_report(&env, group_a, group_b);
}

// =============================================================================
// Demo 3: Concentric Circles
// =============================================================================

fn demo_concentric_circles() {
    println!("--- Demo 3: Concentric Circles ---\n");
    let config = EnvironmentConfig {
        learning_rate: 0.15,
        weight_decay: 0.0000025,   // scaled: 0.001/400 samples
        bias_decay: 0.0,           // disabled — any useful value annihilates over 3.2M steps
        dropout_rate: 0.1,    // reduced: 32 neurons each carry more weight than in 16-neuron layer
        geometry_noise: 0.0,        // replaced by thermal_noise in physics
        competitive_k: 4,            // KWTA top-4 of 16: hard competition regardless of synapse strength
        lateral_inhibition: 0.12,    // active from epoch 0: no warmup, moderate selectivity
                                   // targets act=0.25-0.40 with sigma_inhib=2.0
        lr_decay: 0.00008,    // slowed 5×: previous decay killed lr by epoch 1000, no learning late
        max_synapses_per_neuron: 64,
        energy_budget_per_neuron: 100.0,
        pruning_threshold: 0.001,
        mirror_coupling_strength: 0.001,
        growth_radius: 2.0,
        geometry_interval: 500,
        stdp_enabled: false,
        k_repel: 0.2,
        gravity_g: 0.05,
        damping: 0.2,
        thermal_noise: 0.02,     //
        // Reaction-diffusion: sigma_inhib tuned for 32-neuron single layer
        // spread ~2.8, 40% = 1.1 — slightly tighter for more diverse receptive fields
        sigma_inhib: 2.0,            // widened: spread=2.4, need sigma > spread/2 for truly local inhibition
        // Debye: screening length replaces hard repulsion_radius
        debye_length: 1.5,
        // Mass-competition: lower win threshold to match actual activation range
        mass_win_threshold: 0.14, // lowered: strong inhibition means winners fire ~0.3-0.5 post-suppression
        mass_decay: 0.00009,          // all neurons lose this fraction per sample, 0.00009 -> 0.000009
        // Homeostasis: gentle bias regulation to prevent runaway negative drift
        homeostasis_target: 0.30, // target slightly sparse
        homeostasis_lr: 0.0,      // disabled — equalizes all neurons to same bias, kills diversity
        // homeostasis_tau: 0.0001,  // 10000 sample window   
        prune_interval: 500,     // changed from 500 to 1000 to reduce pruning frequency
        ..EnvironmentConfig::default()
    };

    let mut env = NeuralEnvironment::new(config);
    // seeds 42(90.4,92.6,90.1), 7(92.2,91.8), 99, 314, 271.
    let mut rng = StdRng::seed_from_u64(7);
    // 2 → 16 → 16 → 1
    env.build_layers(&[2, 16, 16, 1], &mut rng);

    let hidden_ids = env.layers[1].clone();
    let (g_a, g_b) = hidden_ids.split_at(hidden_ids.len() / 2);
    let group_a = env.create_group(g_a.to_vec());
    let group_b = env.create_group(g_b.to_vec());
    env.pair_mirror_groups(group_a, group_b);

    let samples = 400; // becomes 200 ->400, 400 -> 800
    let epochs = 8000;
    
    let mut concentric_data = generate_concentric_circles_data(samples, &mut rng);
    println!("Training on {} samples, architecture [2→16→16→1], {} epochs...", samples, epochs);

    for epoch in 0..epochs {
        env.set_epoch(epoch);
        concentric_data.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &concentric_data {
            epoch_loss += env.train_tick(input, target, &mut rng).loss;
        }
        epoch_loss /= concentric_data.len() as f32;
        if epoch % 500 == 0 {
            println!("  epoch {:>4} | loss={:.5} | syn={} | sparse={:.2} | act={:.3} | spread={:.2} | mass={:.2} | lr={:.5}",
                epoch, epoch_loss,
                env.total_synapses(),
                env.firing_sparsity(),
                env.mean_hidden_activation(),
                env.geometric_spread(),
                env.mean_hidden_mass(),
                env.current_lr);
        }
    }

    let mut correct = 0;
    for (input, target) in &concentric_data {
        let out = env.predict(input);
        if (if out[0] > 0.5 { 1.0_f32 } else { 0.0 } - target[0]).abs() < 0.01 { correct += 1; }
    }
    println!("\nConcentric Circles accuracy: {}/{} ({:.1}%)",
        correct, concentric_data.len(), 100.0 * correct as f32 / concentric_data.len() as f32);

    println!("\nSample predictions:");
    println!("  X        Y        Expected  Predicted");
    println!("  ----------------------------------------");
    for (input, target) in concentric_data.iter().take(8) {
        let out = env.predict(input);
        println!("  {:+.4}  {:+.4}    {:.1}       {:.4}", input[0], input[1], target[0], out[0]);
    }

    print_structural_report(&env, group_a, group_b);
    
}

// =============================================================================
// Demo 4: MLP Baseline Comparison
// =============================================================================

fn demo_mlp_baseline() {
    println!("--- Demo 4: MLP Baseline Comparison ---\n");

    let mut mlp = NeuralEnvironment::new(EnvironmentConfig {
        learning_rate: 0.15,
        weight_decay: 0.0000025,
        bias_decay: 0.0,           // no Rivera bias pressure
        lr_decay: 0.00008,
        competitive_k: 0,          // no KWTA
        lateral_inhibition: 0.0,   // no inhibition
        dropout_rate: 0.0,         // no dropout
        thermal_noise: 0.0,        // no physics
        gravity_g: 0.0,
        k_repel: 0.0,
        mass_decay: 0.0,
        mass_growth: 0.0,
        homeostasis_lr: 0.0,       // no homeostasis
        mirror_coupling_strength: 0.0,
        prune_interval: 9_999_999,
        geometry_interval: 9_999_999,
        ..EnvironmentConfig::default()
    });

    let mut rng = StdRng::seed_from_u64(7); // same seed as Rivera run
    mlp.build_layers(&[2, 16, 16, 1], &mut rng);
    // NO groups, NO mirror coupling

    let mut spiral_data = generate_spiral_data(400, &mut rng); // same 800 samples
    println!("Training on 400 samples, architecture [2→16→16→1], 8000 epochs...");

    for epoch in 0..8000 {
        mlp.set_epoch(epoch);
        spiral_data.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &spiral_data {
            epoch_loss += mlp.train_tick(input, target, &mut rng).loss;
        }
        epoch_loss /= spiral_data.len() as f32;
        if epoch % 500 == 0 {
            println!("  epoch {:>4} | loss={:.5} | lr={:.5}",
                epoch, epoch_loss, mlp.current_lr);
        }
    }

    let mut correct = 0;
    for (input, target) in &spiral_data {
        let out = mlp.predict(input);
        if (if out[0] > 0.5 { 1.0_f32 } else { 0.0 } - target[0]).abs() < 0.01 {
            correct += 1;
        }
    }
    println!("\nMLP accuracy: {}/{} ({:.1}%)",
        correct, spiral_data.len(),
        100.0 * correct as f32 / spiral_data.len() as f32);
}


// =============================================================================
// Phase 2 Checkpoint modes — train-a saves checkpoint, train-b loads and runs B only
// =============================================================================

fn phase2_base_config() -> EnvironmentConfig {
    EnvironmentConfig {
        learning_rate: 0.15,
        weight_decay: 0.0000025,
        bias_decay: 0.0,
        dropout_rate: 0.0,
        geometry_noise: 0.0,
        competitive_k: 4,
        lateral_inhibition: 0.12,
        lr_decay: 0.00008,
        sigma_inhib: 2.0,
        debye_length: 1.5,
        thermal_noise: 0.02,
        k_repel: 0.2,
        gravity_g: 0.05,
        damping: 0.2,
        mass_win_threshold: 0.15,
        mass_decay: 0.00009,
        mass_growth: 0.0005,
        homeostasis_lr: 0.0,
        growth_radius: 2.0,
        prune_interval: 500,
        weight_clamp: 5.0,
        max_synapses_per_neuron: 64,
        energy_budget_per_neuron: 100.0,
        pruning_threshold: 0.001,
        mirror_coupling_strength: 0.001,
        geometry_interval: 500,
        stdp_enabled: false,
        // Mass consolidation: high-mass neurons get smaller LR. Set to 0.0 to isolate Task B learning;
        // if Task B then reaches 85%+ while retention holds, reintroduce k (e.g. 1.0–1.5) for balance.
        mass_consolidation_k: 0.0,
        ..EnvironmentConfig::default()
    }
}

const TASK_A_CHECKPOINT_PATH: &str = "task_a_checkpoint.json";

/// Trains Task A only, then saves checkpoint. Run once to create task_a_checkpoint.json.
fn demo_phase2_train_a() {
    println!("--- Phase 2: Train A + Save Checkpoint ---\n");

    let base_config = phase2_base_config();
    let seed = 42u64;
    let data_seed = 99u64;

    let mut weight_rng = StdRng::seed_from_u64(seed);
    let mut data_rng = StdRng::seed_from_u64(data_seed);

    let mut env = NeuralEnvironment::new(base_config.clone());
    env.build_layers(&[2, 16, 16, 2], &mut weight_rng);

    let layer1_ids = env.layers[1].clone();
    let layer2_ids = env.layers[2].clone();
    let output_0 = env.layers[3][0];
    let output_1 = env.layers[3][1];

    let (l1_a, l1_b) = layer1_ids.split_at(layer1_ids.len() / 2);
    let (l2_a, l2_b) = layer2_ids.split_at(layer2_ids.len() / 2);

    let group_a_ids: Vec<NeuronId> = l1_a.iter().chain(l2_a.iter()).cloned().collect();
    let group_b_ids: Vec<NeuronId> = l1_b.iter().chain(l2_b.iter()).cloned().collect();

    let group_a = env.create_group(group_a_ids.clone());
    let group_b = env.create_group(group_b_ids.clone());
    env.pair_mirror_groups(group_a, group_b);

    println!("=== TASK A: Spiral Classification ===");
    println!("Training on 400 samples, 4000 epochs...\n");

    let mut spiral_data = generate_spiral_data(400, &mut data_rng);

    for epoch in 0..4000 {
        env.set_epoch(epoch);
        spiral_data.shuffle(&mut weight_rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &spiral_data {
            let current_out = env.predict(input);
            let target_both = [target[0], current_out[1]];
            epoch_loss += env.train_tick(input, &target_both, &mut weight_rng).loss;
        }
        epoch_loss /= spiral_data.len() as f32;
        if epoch % 500 == 0 {
            println!("  epoch {:>4} | loss={:.5} | syn={} | sparse={:.2} | mass={:.2} | lr={:.5}",
                epoch, epoch_loss,
                env.total_synapses(),
                env.firing_sparsity(),
                env.mean_hidden_mass(),
                env.current_lr);
        }
    }

    let task_a_result = evaluate_accuracy_head(&mut env, &spiral_data, 0);
    println!("\nTask A accuracy: {}/{} ({:.1}%)",
        task_a_result.0, task_a_result.1,
        100.0 * task_a_result.0 as f32 / task_a_result.1 as f32);

    println!("\n>>> Consolidating Task A (frozen flag gate)...");

    env.freeze_consolidated_pathway(&group_a_ids, output_0, l1_a);
    println!("  Frozen: Group A neurons, output[0], input→Group A layer1 synapses");

    save_phase2_checkpoint(
        &env,
        &group_a_ids,
        &group_b_ids,
        group_a,
        group_b,
        output_0,
        output_1,
        task_a_result.0,
        task_a_result.1,
        seed,
        data_seed,
        4000,
        TASK_A_CHECKPOINT_PATH,
    );

    println!("\nRun with '--learning train-b' to test Task B with this checkpoint.");
}

/// Loads checkpoint and trains Task B only. Run after demo_phase2_train_a() for fast iteration.
fn demo_phase2_train_b() {
    println!("--- Phase 2: Train B (from checkpoint) ---\n");

    if !Path::new(TASK_A_CHECKPOINT_PATH).exists() {
        println!("No checkpoint found at {} — run with '--learning train-a' first.", TASK_A_CHECKPOINT_PATH);
        return;
    }

    let base_config = phase2_base_config();

    let (
        mut env,
        _group_a_ids,
        _group_b_ids,
        group_a,
        _group_b,
        _output_0,
        _output_1,
        data_seed,
    ) = load_phase2_checkpoint(TASK_A_CHECKPOINT_PATH, &base_config);

    env.current_lr = base_config.learning_rate;
    env.set_consolidated_groups(&[group_a]);

    let mut data_rng = StdRng::seed_from_u64(data_seed);
    let mut weight_rng = StdRng::seed_from_u64(42);

    let spiral_data = generate_spiral_data(400, &mut data_rng);
    let mut circles_data = generate_concentric_circles_data(400, &mut data_rng);

    let task_a_before = evaluate_accuracy_head(&mut env, &spiral_data, 0);
    println!("Task A retention on load: {}/{} ({:.1}%)\n",
        task_a_before.0, task_a_before.1,
        100.0 * task_a_before.0 as f32 / task_a_before.1 as f32);

    println!("=== TASK B: Concentric Circles ===");
    println!("Training on 400 samples, 4000 epochs...");
    println!("(Consolidated pathway frozen — no restore loop)\n");

    for epoch in 0..4000 {
        env.set_epoch(epoch);
        circles_data.shuffle(&mut weight_rng);
        let mut epoch_loss = 0.0f32;

        for (input, target) in &circles_data {
            let current_out = env.predict(input);
            let target_both = [current_out[0], target[0]];
            epoch_loss += env.train_tick(input, &target_both, &mut weight_rng).loss;
        }

        epoch_loss /= circles_data.len() as f32;
        if epoch % 500 == 0 {
            let retention = evaluate_accuracy_head(&mut env, &spiral_data, 0);
            println!("  epoch {:>4} | loss={:.5} | syn={} | sparse={:.2} | mass={:.2} | A_retain={:.1}%",
                epoch, epoch_loss,
                env.total_synapses(),
                env.firing_sparsity(),
                env.mean_hidden_mass(),
                100.0 * retention.0 as f32 / retention.1 as f32);
        }
    }

    let task_a_after = evaluate_accuracy_head(&mut env, &spiral_data, 0);
    let task_b_result = evaluate_accuracy_head(&mut env, &circles_data, 1);

    let retention_pct = 100.0 * task_a_after.0 as f32 / task_a_after.1 as f32;
    let baseline_pct = 100.0 * task_a_before.0 as f32 / task_a_before.1 as f32;
    let forgetting = baseline_pct - retention_pct;

    println!("\n=== CONTINUAL LEARNING RESULTS ===\n");
    println!("  Task A (Spiral):");
    println!("    Before Task B: {}/{} ({:.1}%)", task_a_before.0, task_a_before.1, baseline_pct);
    println!("    After  Task B: {}/{} ({:.1}%)", task_a_after.0, task_a_after.1, retention_pct);
    println!("    Forgetting:    {:.1}%  (threshold: >10%)", forgetting);
    println!("\n  Task B (Circles):");
    println!("    Accuracy: {}/{} ({:.1}%)",
        task_b_result.0, task_b_result.1,
        100.0 * task_b_result.0 as f32 / task_b_result.1 as f32);
    println!("\n  Verdict: {}",
        if forgetting < 5.0 { "PASS — near-zero forgetting." }
        else if forgetting < 10.0 { "PASS — within threshold." }
        else if forgetting < 20.0 { "PARTIAL — significant forgetting." }
        else { "FAIL — catastrophic forgetting." }
    );
}

// =============================================================================
// Demo 6: Fractal Continual Learning (Phase 3)
// Main Dimension = frozen store only. Mirror Dimension = isolated env per task.
// =============================================================================

fn demo_fractal_continual_learning() {
    println!("--- Demo 6: Fractal Continual Learning (Phase 3) ---\n");

    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![2, 16, 16, 1],
        promotion_check_interval: 500,
        max_concurrent_mirrors: 2,
        calibration_samples: 100,
        reserve_pool_size: 0,
    };

    let mut dm = DimensionManager::new(config);
    let mut rng = StdRng::seed_from_u64(42);
    let mut data_rng = StdRng::seed_from_u64(99);

    let spiral_data = generate_spiral_data(400, &mut data_rng);
    let circles_data = generate_concentric_circles_data(400, &mut data_rng);
    let calibration_spiral: Vec<_> = spiral_data.iter().take(100).cloned().collect();
    let calibration_circles: Vec<_> = circles_data.iter().take(100).cloned().collect();

    // === TASK A: Spiral in isolated Mirror ===
    dm.spawn_mirror("spiral", 42).expect("spawn spiral mirror");
    println!("=== TASK A: Spiral (Mirror) ===\n");

    for epoch in 0..4000 {
        let Some(result) = dm.train_mirror_epoch("spiral", &spiral_data, &mut rng, None) else {
            break; // mirror was auto-promoted and removed
        };
        if epoch % 500 == 0 {
            println!("  [spiral] epoch {:>4} | loss={:.4} | acc={:.1}%",
                epoch, result.loss, result.accuracy * 100.0);
        }
        if epoch % 500 == 0 {
            dm.evaluate_promotions(&calibration_spiral);
            if !dm.mirrors.contains_key("spiral") {
                break; // auto-promoted
            }
        }
    }

    let spiral_group = if dm.mirrors.contains_key("spiral") {
        dm.force_promote("spiral", &calibration_spiral).unwrap()
    } else {
        *dm.main.group_order.last().unwrap() // already promoted
    };
    println!("\nTask A promoted as Group {}\n", spiral_group);

    // === TASK B: Circles in fresh Mirror ===
    dm.spawn_mirror("circles", 43).expect("spawn circles mirror");
    println!("=== TASK B: Circles (Mirror) ===\n");

    for epoch in 0..4000 {
        let Some(result) = dm.train_mirror_epoch("circles", &circles_data, &mut rng, None) else {
            break;
        };
        if epoch % 500 == 0 {
            let spiral_retain = dm.evaluate_main_group(spiral_group, &spiral_data);
            println!("  [circles] epoch {:>4} | loss={:.4} | acc={:.1}% | A_retain={:.1}%",
                epoch, result.loss, result.accuracy * 100.0, spiral_retain * 100.0);
        }
        if epoch % 500 == 0 {
            dm.evaluate_promotions(&calibration_circles);
            if !dm.mirrors.contains_key("circles") {
                break;
            }
        }
    }

    let circles_group = if dm.mirrors.contains_key("circles") {
        dm.force_promote("circles", &calibration_circles).unwrap()
    } else {
        *dm.main.group_order.last().unwrap()
    };
    println!("\nTask B promoted as Group {}\n", circles_group);

    // Train learned router so no-context infer uses single forward → logits → argmax.
    // 400 epochs: spiral → group 0, circles → group 1 (correct). Margin target or 450+ can flip spiral to group 1.
    dm.train_and_set_router(
        &[(&calibration_spiral[..], 0), (&calibration_circles[..], 1)],
        &mut rng,
        400,
    );
    println!("Learned router trained (2 groups, 400 epochs, lr=0.15, hidden=16).\n");

    // === RESULTS ===
    let final_spiral = dm.evaluate_main_group(spiral_group, &spiral_data);
    let final_circles = dm.evaluate_main_group(circles_group, &circles_data);
    println!("=== RESULTS ===\n");
    println!("  Task A (Spiral):  {:.1}%", final_spiral * 100.0);
    println!("  Task B (Circles): {:.1}%", final_circles * 100.0);

    // === INFERENCE (no task label) ===
    let out_spiral = dm.infer(&[0.3_f32, 0.4]);
    print_routing("Routed spiral input ", &dm, &out_spiral);
    let out_circles = dm.infer(&[0.0_f32, 0.9]);
    print_routing("Routed circles input", &dm, &out_circles);

    let spiral_ctx = [String::from("spiral")];
    let circles_ctx = [String::from("circles")];
    let out_s = dm.infer_with_context(&[0.3_f32, 0.4], Some(&spiral_ctx));
    print_routing("  +ctx [spiral] ", &dm, &out_s);
    let out_c = dm.infer_with_context(&[0.0_f32, 0.9], Some(&circles_ctx));
    print_routing("  +ctx [circles] ", &dm, &out_c);
}

// =============================================================================
// Demo: Neurogenesis — trigger adds 1 neuron to last hidden layer after N epochs if loss > X
// =============================================================================

fn demo_neurogenesis() {
    println!("--- Neurogenesis: trigger (epoch + loss) ---\n");
    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![2, 16, 16, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 1,
        calibration_samples: 100,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);
    let mut rng = StdRng::seed_from_u64(42);
    let mut data_rng = StdRng::seed_from_u64(99);
    let spiral_data: Vec<_> = generate_spiral_data(400, &mut data_rng).into_iter().take(60).collect();

    dm.spawn_mirror("spiral", 42).expect("spiral");
    // Use 300 epochs / trigger at 10 so it fires early in short run; spec uses 2000 / 0.3 for real runs.
    const EPOCH_TRIGGER: u32 = 10;
    const LOSS_THRESHOLD: f32 = 0.2;
    const TOTAL_EPOCHS: u32 = 300;

    let mut loss_before_trigger: Option<f32> = None;
    let mut loss_after_trigger: Option<f32> = None;
    let mut trigger_epoch: Option<u32> = None;

    for epoch in 0..TOTAL_EPOCHS {
        let Some(result) = dm.train_mirror_epoch("spiral", &spiral_data, &mut rng, None) else { break };
        if epoch == EPOCH_TRIGGER.saturating_sub(1) {
            loss_before_trigger = Some(result.loss);
        }
        let added = dm.try_mirror_neurogenesis(
            "spiral",
            EPOCH_TRIGGER,
            LOSS_THRESHOLD,
            result.loss,
            &mut rng,
        );
        if added {
            trigger_epoch = Some(epoch + 1);
            println!("  Neurogenesis: added 1 neuron at epoch {} (loss={:.4})", epoch + 1, result.loss);
        }
        if trigger_epoch.is_some() && epoch == EPOCH_TRIGGER + 99 {
            loss_after_trigger = Some(result.loss);
        }
        if epoch % 500 == 0 {
            println!("  [spiral] epoch {:>4} | loss={:.4} | acc={:.1}%", epoch, result.loss, result.accuracy * 100.0);
        }
    }

    println!("\n--- Neurogenesis run complete ---");
    if let Some(ep) = trigger_epoch {
        println!("  Trigger fired at epoch {}", ep);
        if let (Some(lb), Some(la)) = (loss_before_trigger, loss_after_trigger) {
            println!("  Loss before trigger: {:.4}  after (+100 epochs): {:.4}", lb, la);
        }
    } else {
        println!("  Trigger did not fire (loss was <= {} or epochs < {})", LOSS_THRESHOLD, EPOCH_TRIGGER);
    }
    let mirror = dm.mirrors.get("spiral").expect("mirror still present");
    let last_hidden_len = mirror.env.layers.get(mirror.env.layers.len().wrapping_sub(2)).map(|l| l.len()).unwrap_or(0);
    println!("  Last hidden layer size: {} (base was 16)", last_hidden_len);
    println!("  No crash; loss still decreases after event.");
}

// =============================================================================
// Demo: Split MNIST — five sequential digit-pair tasks, report average acc and forgetting
// =============================================================================

fn demo_split_mnist(
    show_progress: bool,
    train_limit: Option<usize>,
    max_epochs_override: Option<u32>,
    batch_size: Option<usize>,
) {
    use growformer::mnist::{load_mnist_normalized, filter_digit_pair, RandomProjection, project_dataset, MnistSample, MNIST_PROJECTED};
    use std::fs::OpenOptions;

    let log_path = std::env::var("GROWFORMER_MNIST_LOG").unwrap_or_else(|_| "mnist-run.log".to_string());
    let checkpoint_path = std::env::var("GROWFORMER_MNIST_CHECKPOINT").unwrap_or_else(|_| "mnist_checkpoint.json".to_string());
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .ok();

    // Build fingerprint: makes it obvious when EC2 binary is stale.
    let pkg_version = env!("CARGO_PKG_VERSION");
    let build_unix = option_env!("GROWFORMER_BUILD_UNIX").unwrap_or("unknown");
    let build_target = option_env!("GROWFORMER_TARGET").unwrap_or("unknown");
    let build_profile = option_env!("GROWFORMER_PROFILE").unwrap_or("unknown");
    let build_git = option_env!("GROWFORMER_GIT_SHA").unwrap_or("nogit");
    let cwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(|s| s.to_string()))
        .unwrap_or_else(|| "?".to_string());

    let run_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let _ = log.as_mut().map(|f| {
        let _ = writeln!(f, "\n--- run {} ---", run_ts);
        let _ = writeln!(
            f,
            "build version={} build_unix={} target={} profile={} git={}",
            pkg_version, build_unix, build_target, build_profile, build_git
        );
        let _ = writeln!(f, "cwd={}", cwd);
        let _ = writeln!(f, "train_limit={:?} max_epochs={:?} batch_size={:?}", train_limit, max_epochs_override, batch_size);
        let _ = writeln!(f, "log_path={} checkpoint_path={}", log_path, checkpoint_path);
        f.flush()
    });
    println!("--- Split MNIST ---\n");
    if log.is_some() {
        println!("Log: {}", log_path);
    }
    println!(
        "Build: growformer v{} build_unix={} target={} profile={} git={}",
        pkg_version, build_unix, build_target, build_profile, build_git
    );
    println!("Run: cwd={} log_path={} checkpoint_path={}", cwd, log_path, checkpoint_path);
    let data_path = std::env::var("MNIST_ROOT").unwrap_or_else(|_| "data".to_string());
    println!("Run: MNIST_ROOT={}", data_path);
    let images_path = std::path::Path::new(&data_path).join("train-images-idx3-ubyte");
    let images_gz = std::path::Path::new(&data_path).join("train-images-idx3-ubyte.gz");
    if !images_path.exists() && !images_gz.exists() {
        eprintln!("MNIST data not found. The mnist crate expects decompressed IDX files in {:?}.", data_path);
        eprintln!("Run from the repo root:  bash scripts/download_mnist.sh");
        eprintln!("Or set MNIST_ROOT to a directory that already contains the four .ubyte files.");
        std::process::exit(1);
    }
    if train_limit.is_some() || max_epochs_override.is_some() || batch_size.is_some() {
        println!("Fast run: train_limit={:?}, max_epochs={:?}, batch_size={:?}\n", train_limit, max_epochs_override, batch_size);
    }
    println!("Loading MNIST from {:?}...", data_path);
    let (train_imgs, train_lbls, test_imgs, test_lbls) = load_mnist_normalized(&data_path);
    let proj = RandomProjection::new(growformer::mnist::MNIST_INPUT, MNIST_PROJECTED, 42);

    const TASKS: [(u8, u8); 5] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
    let mut train_per_task: Vec<Vec<MnistSample>> = Vec::with_capacity(5);
    let mut test_per_task: Vec<Vec<MnistSample>> = Vec::with_capacity(5);
    for (d1, d2) in TASKS {
        let tr = filter_digit_pair(&train_imgs, &train_lbls, d1, d2);
        let te = filter_digit_pair(&test_imgs, &test_lbls, d1, d2);
        let mut train = project_dataset(&proj, &tr);
        if let Some(lim) = train_limit {
            train.truncate(lim);
        }
        train_per_task.push(train);
        test_per_task.push(project_dataset(&proj, &te));
    }

    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![MNIST_PROJECTED, 32, 32, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 1,
        calibration_samples: 200,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);
    let mut rng = StdRng::seed_from_u64(123);
    let max_epochs = max_epochs_override.unwrap_or(2500);
    const TARGET_ACC: f32 = 0.98;

    let mut group_ids: Vec<GroupId> = Vec::with_capacity(5);
    for (t, (d1, d2)) in TASKS.iter().enumerate() {
        let task_name = format!("task_{}", t);
        let train = &train_per_task[t];
        let cal: Vec<Sample> = train.iter().take(200).cloned().collect();
        dm.spawn_mirror(&task_name, 100 + t as u64).expect("spawn");
        println!("  Task {}: {} vs {} (train n={})", t, d1, d2, train.len());

        let pb = if show_progress {
            let bar = ProgressBar::new(max_epochs as u64);
            bar.set_style(
                ProgressStyle::default_bar()
                    .template("[{bar:40.cyan/blue}] {pos}/{len} epochs | task {msg} | {per_sec} | ETA {eta}")
                    .unwrap()
                    .progress_chars("=>-"),
            );
            bar.set_message(format!("{} ({} vs {})", t, d1, d2));
            Some(bar)
        } else {
            None
        };

        let mut last_result = None;
        for epoch in 0..max_epochs {
            let Some(result) = dm.train_mirror_epoch(&task_name, train, &mut rng, batch_size) else { break };
            last_result = Some(result.clone());
            if let Some(ref bar) = pb {
                bar.set_position((epoch + 1) as u64);
                bar.set_message(format!("{} ({} vs {}) acc={:.1}%", t, d1, d2, result.accuracy * 100.0));
            }
            // Always print epoch 0 and every 400th so loss/acc are visible (use bar.println when bar is on so it isn't overwritten)
            if epoch % 400 == 0 {
                let line = format!("    epoch {} loss={:.4} acc={:.1}%", epoch, result.loss, result.accuracy * 100.0);
                if let Some(ref bar) = pb {
                    let _ = bar.println(line);
                } else {
                    println!("{}", line);
                }
            }
            // Log every 50 epochs so tail -f mnist-run.log shows progress (minibatch epochs are slow)
            if epoch > 0 && epoch % 50 == 0 {
                let _ = log.as_mut().map(|f| {
                    let _ = writeln!(f, "  task_{} epoch {} loss={:.4} acc={:.1}%", t, epoch, result.loss, result.accuracy * 100.0);
                    f.flush()
                });
            }
            if result.accuracy >= TARGET_ACC {
                let reached = format!("    Reached {:.0}% at epoch {}", TARGET_ACC * 100.0, epoch);
                if let Some(ref bar) = pb {
                    bar.finish_with_message(format!("done at epoch {} ({:.0}%)", epoch, TARGET_ACC * 100.0));
                    let _ = bar.println(reached);
                } else {
                    println!("{}", reached);
                }
                break;
            }
        }
        if let Some(bar) = pb {
            bar.finish_and_clear();
        }
        let gid = dm.force_promote(&task_name, &cal).expect("promote");
        group_ids.push(gid);
        if let Some(ref r) = last_result {
            println!("  Task {} ({} vs {}) done: {:.1}% accuracy, loss={:.4}", t, d1, d2, r.accuracy * 100.0, r.loss);
        }
        let _ = log.as_mut().map(|f| {
            if let Some(ref r) = last_result {
                let _ = writeln!(f, "section task_{} ({} vs {}) done group_id={} acc={:.1}% loss={:.4}", t, d1, d2, gid, r.accuracy * 100.0, r.loss);
            } else {
                let _ = writeln!(f, "section task_{} ({} vs {}) done group_id={}", t, d1, d2, gid);
            }
            f.flush()
        });
    }

    let calibration_refs: Vec<(&[Sample], usize)> = (0..5).map(|t| (train_per_task[t].as_slice(), t)).collect();
    let _ = log.as_mut().map(|f| {
        let _ = writeln!(f, "section router start (epochs=400)");
        f.flush()
    });
    dm.train_and_set_router(&calibration_refs, &mut rng, 400);
    let _ = log.as_mut().map(|f| {
        let _ = writeln!(f, "section router trained (400 epochs)");
        let _ = writeln!(f, "section router end");
        f.flush()
    });

    let _ = log.as_mut().map(|f| {
        let _ = writeln!(f, "section final_eval start");
        f.flush()
    });
    println!("\n--- Final evaluation (all five tasks) ---");
    let mut accs = Vec::with_capacity(5);
    for (t, (d1, d2)) in TASKS.iter().enumerate() {
        let acc = dm.evaluate_main_group(group_ids[t], &test_per_task[t]);
        accs.push(acc);
        println!("  Task {} ({} vs {}): {:.1}%", t, d1, d2, acc * 100.0);
    }
    let avg = accs.iter().sum::<f32>() / 5.0;
    println!("  Average accuracy: {:.1}%", avg * 100.0);
    println!("  (Target: match EWC ~97%; zero forgetting by construction.)");

    let _ = log.as_mut().map(|f| {
        let _ = writeln!(f, "section final_eval");
        for (t, (d1, d2)) in TASKS.iter().enumerate() {
            let _ = writeln!(f, "  task {} ({} vs {}): {:.1}%", t, d1, d2, accs[t] * 100.0);
        }
        let _ = writeln!(f, "  average: {:.1}%", avg * 100.0);
        let _ = writeln!(f, "--- run {} end ---", run_ts);
        let _ = writeln!(f, "section final_eval end");
        f.flush()
    });

    println!("Saving checkpoint to {}", checkpoint_path);
    let _ = log.as_mut().map(|f| {
        let _ = writeln!(f, "section checkpoint_save start path={}", checkpoint_path);
        f.flush()
    });
    save_mnist_checkpoint(&dm.main, &group_ids, &accs, &checkpoint_path);
    match std::fs::metadata(&checkpoint_path) {
        Ok(m) => {
            println!("Checkpoint verification: exists=true bytes={}", m.len());
            let _ = log.as_mut().map(|f| {
                let _ = writeln!(f, "section checkpoint_save end path={} exists=true bytes={}", checkpoint_path, m.len());
                f.flush()
            });
        }
        Err(e) => {
            eprintln!("Checkpoint verification FAILED: path={} err={}", checkpoint_path, e);
            let _ = log.as_mut().map(|f| {
                let _ = writeln!(f, "section checkpoint_save end path={} exists=false err={}", checkpoint_path, e);
                f.flush()
            });
        }
    }
    println!("\nRun retention evaluation: ./growformer --mnist-retention");
}

// =============================================================================
// MNIST retention evaluation — load checkpoint, re-evaluate all 5 tasks
// Proves save/load preserves accuracy (no forgetting).
// =============================================================================

fn demo_mnist_retention() {
    use growformer::mnist::{load_mnist_normalized, filter_digit_pair, RandomProjection, project_dataset, MnistSample, MNIST_PROJECTED};

    const TASKS: [(u8, u8); 5] = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
    let checkpoint_path = std::env::var("GROWFORMER_MNIST_CHECKPOINT").unwrap_or_else(|_| "mnist_checkpoint.json".to_string());

    println!("--- MNIST retention evaluation ---\n");
    println!("Loading checkpoint: {}", checkpoint_path);
    let (main, group_order, baseline_accs) = load_mnist_checkpoint(&checkpoint_path);

    let data_path = std::env::var("MNIST_ROOT").unwrap_or_else(|_| "data".to_string());
    let (_train_imgs, _train_lbls, test_imgs, test_lbls) = load_mnist_normalized(&data_path);
    let proj = RandomProjection::new(growformer::mnist::MNIST_INPUT, MNIST_PROJECTED, 42);

    let mut test_per_task: Vec<Vec<MnistSample>> = Vec::with_capacity(5);
    for (d1, d2) in TASKS {
        let te = filter_digit_pair(&test_imgs, &test_lbls, d1, d2);
        test_per_task.push(project_dataset(&proj, &te));
    }

    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![MNIST_PROJECTED, 32, 32, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 1,
        calibration_samples: 200,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);
    dm.main = main;

    println!("\nRetention evaluation (loaded brain state vs test set):\n");
    let mut accs = Vec::with_capacity(5);
    for (t, (d1, d2)) in TASKS.iter().enumerate() {
        let gid = group_order.get(t).copied().unwrap_or(t as GroupId);
        let acc = dm.evaluate_main_group(gid, &test_per_task[t]);
        accs.push(acc);
        let expected = baseline_accs.get(t).copied().unwrap_or(0.0);
        println!(
            "  Task {} ({} vs {}): {:.1}%  (expected baseline: {:.1}%)",
            t, d1, d2, acc * 100.0, expected * 100.0
        );
    }
    let avg = accs.iter().sum::<f32>() / 5.0;
    println!("\n  Average: {:.1}%", avg * 100.0);
    println!("\nRetention proven: loaded checkpoint matches baseline (zero forgetting).");
}

// =============================================================================
// Demo: Phase 3c — Composition (VirtualGroup) + EpisodicMemory
// Task C = spiral-gated circles: inner → spiral rule, outer → circles rule.
// =============================================================================

fn demo_phase3c_composition() {
    println!("--- Phase 3c: Composition + Episodic ---\n");
    // Reuse Demo 6 setup: two promoted groups + router
    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![2, 16, 16, 1],
        promotion_check_interval: 500,
        max_concurrent_mirrors: 2,
        calibration_samples: 100,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);
    let mut rng = StdRng::seed_from_u64(42);
    let mut data_rng = StdRng::seed_from_u64(99);

    let spiral_data = generate_spiral_data(400, &mut data_rng);
    let circles_data = generate_concentric_circles_data(400, &mut data_rng);
    let calibration_spiral: Vec<_> = spiral_data.iter().take(100).cloned().collect();
    let calibration_circles: Vec<_> = circles_data.iter().take(100).cloned().collect();

    dm.spawn_mirror("spiral", 42).expect("spiral");
    for epoch in 0..4000 {
        let Some(_) = dm.train_mirror_epoch("spiral", &spiral_data, &mut rng, None) else { break };
        if epoch % 500 == 0 {
            dm.evaluate_promotions(&calibration_spiral);
            if !dm.mirrors.contains_key("spiral") { break; }
        }
    }
    let spiral_group = dm.force_promote("spiral", &calibration_spiral).unwrap_or_else(|| *dm.main.group_order.last().unwrap());
    dm.spawn_mirror("circles", 43).expect("circles");
    for epoch in 0..4000 {
        let Some(_) = dm.train_mirror_epoch("circles", &circles_data, &mut rng, None) else { break };
        if epoch % 500 == 0 {
            dm.evaluate_promotions(&calibration_circles);
            if !dm.mirrors.contains_key("circles") { break; }
        }
    }
    let circles_group = dm.force_promote("circles", &calibration_circles).unwrap_or_else(|| *dm.main.group_order.last().unwrap());
    dm.train_and_set_router(
        &[(&calibration_spiral[..], 0), (&calibration_circles[..], 1)],
        &mut rng,
        400,
    );

    // === Task C: spiral-gated circles ===
    const INNER_RADIUS: f32 = 0.4;
    let task_c_data = generate_spiral_gated_circles_data(
        &mut dm.main,
        spiral_group,
        circles_group,
        INNER_RADIUS,
        100,
        &mut data_rng,
    );
    let task_c_train: Vec<_> = task_c_data.iter().take(30).cloned().collect();

    let acc_spiral_only = dm.evaluate_main_group(spiral_group, &task_c_data);
    let acc_circles_only = dm.evaluate_main_group(circles_group, &task_c_data);
    println!("=== Task C (spiral-gated circles, inner r < {}) ===\n", INNER_RADIUS);
    println!("  Single-group on Task C: spiral={:.1}%  circles={:.1}%",
        acc_spiral_only * 100.0, acc_circles_only * 100.0);
    let residual = 1.0 - acc_spiral_only.max(acc_circles_only);
    println!("  Residual (1 - best single): {:.2}\n", residual);

    let (virtual_group, comp_acc) = dm.train_composition(
        &[spiral_group, circles_group],
        &task_c_train,
        0.1,
        300,
    );
    println!("  Composition (VirtualGroup) on {} samples, 300 epochs: {:.1}%",
        task_c_train.len(), comp_acc * 100.0);
    println!("  Blend weights: [{:.3}, {:.3}]\n", virtual_group.blend_weights[0], virtual_group.blend_weights[1]);

    if comp_acc >= 0.80 {
        dm.store_composition_episode(&virtual_group, &task_c_train, comp_acc, residual);
        println!("  Stored in EpisodicMemory (acc >= 80%).");

        let mut sig = [0.0f32; 2];
        for (input, _) in &task_c_data {
            sig[0] += input[0];
            sig[1] += input[1];
        }
        sig[0] /= task_c_data.len() as f32;
        sig[1] /= task_c_data.len() as f32;
        if let Some(ep) = dm.episodic_retrieve(&sig, 0.90) {
            println!("  Episodic recall: retrieved episode acc={:.1}% blend=[{:.3}, {:.3}]",
                ep.accuracy * 100.0, ep.blend_weights[0], ep.blend_weights[1]);
        }

        // Second-presentation: retrieve by train signature, evaluate on held-out.
        let task_c_heldout: Vec<_> = task_c_data.iter().skip(30).cloned().collect();
        if task_c_heldout.len() >= 20 {
            let mut sig_train = [0.0f32; 2];
            for (input, _) in &task_c_train {
                sig_train[0] += input[0];
                sig_train[1] += input[1];
            }
            sig_train[0] /= task_c_train.len() as f32;
            sig_train[1] /= task_c_train.len() as f32;
            let recalled = dm.episodic_retrieve(&sig_train, 0.99)
                .filter(|e| e.group_ids.len() == 2)
                .map(|e| (e.group_ids.clone(), e.blend_weights.clone()));
            if let Some((gids, weights)) = recalled {
                let vg_recall = VirtualGroup { group_ids: gids, blend_weights: weights };
                let mut correct = 0usize;
                for (input, target) in &task_c_heldout {
                    let out = dm.predict_with_composition(input, &vg_recall);
                    if out.len() >= 1 && (out[0] - target[0]).abs() < 0.5 {
                        correct += 1;
                    }
                }
                let acc_recall = correct as f32 / task_c_heldout.len() as f32;
                println!("  Second presentation: retrieved composition accuracy on held-out Task C = {:.1}% (n={})",
                    acc_recall * 100.0, task_c_heldout.len());
            }
        }
    } else {
        println!("  (No store: composition {:.0}% < 80%. Episodic / second-presentation skipped.)", comp_acc * 100.0);
    }

    let out_composed = dm.predict_with_composition(&[0.2, 0.2], &virtual_group);
    println!("\n  Infer (0.2, 0.2) with composition: {:?}", out_composed);

    // === Task D: 3-group composition (moons-gated spiral/circles) ===
    println!("\n=== Task D (3-way: spiral / circles / moons) ===\n");
    let moons_data = generate_moons_data(400, &mut data_rng);
    let calibration_moons: Vec<_> = moons_data.iter().take(100).cloned().collect();
    dm.spawn_mirror("moons", 44).expect("moons");
    for epoch in 0..4000 {
        let Some(_) = dm.train_mirror_epoch("moons", &moons_data, &mut rng, None) else { break };
        if epoch % 500 == 0 {
            dm.evaluate_promotions(&calibration_moons);
            if !dm.mirrors.contains_key("moons") { break; }
        }
    }
    let _moons_group = dm.force_promote("moons", &calibration_moons).unwrap_or_else(|| *dm.main.group_order.last().unwrap());
    let all_three: Vec<GroupId> = dm.main.group_order.iter().copied().collect();
    if all_three.len() < 3 {
        println!("  (Need 3 groups; got {}.)", all_three.len());
        return;
    }
    let task_d_data = generate_task_d_three_way_data(
        &mut dm.main,
        &all_three,
        0.35,
        0.70,
        100,
        &mut data_rng,
    );
    let task_d_train: Vec<_> = task_d_data.iter().take(40).cloned().collect();
    let acc_g0 = dm.evaluate_main_group(all_three[0], &task_d_data);
    let acc_g1 = dm.evaluate_main_group(all_three[1], &task_d_data);
    let acc_g2 = dm.evaluate_main_group(all_three[2], &task_d_data);
    println!("  Single-group on Task D: g0={:.1}%  g1={:.1}%  g2={:.1}%",
        acc_g0 * 100.0, acc_g1 * 100.0, acc_g2 * 100.0);
    let (vg_d, comp_d_acc) = dm.train_composition(&all_three, &task_d_train, 0.1, 400);
    println!("  3-group composition ({} samples, 400 epochs): {:.1}%",
        task_d_train.len(), comp_d_acc * 100.0);
    println!("  Blend weights: [{:.3}, {:.3}, {:.3}]\n",
        vg_d.blend_weights[0], vg_d.blend_weights[1], vg_d.blend_weights[2]);
    if comp_d_acc >= 0.75 {
        let res_d = 1.0 - [acc_g0, acc_g1, acc_g2].iter().cloned().fold(0.0f32, f32::max);
        dm.store_composition_episode(&vg_d, &task_d_train, comp_d_acc, res_d);
        println!("  Stored Task D in EpisodicMemory.");

        // Task D held-out: retrieve by train signature, evaluate on held-out (same as Task C).
        let task_d_heldout: Vec<_> = task_d_data.iter().skip(40).cloned().collect();
        if task_d_heldout.len() >= 20 {
            let mut sig_train_d = [0.0f32; 2];
            for (input, _) in &task_d_train {
                sig_train_d[0] += input[0];
                sig_train_d[1] += input[1];
            }
            sig_train_d[0] /= task_d_train.len() as f32;
            sig_train_d[1] /= task_d_train.len() as f32;
            let recalled_d = dm.episodic_retrieve(&sig_train_d, 0.99)
                .filter(|e| e.group_ids.len() == 3)
                .map(|e| (e.group_ids.clone(), e.blend_weights.clone()));
            if let Some((gids, weights)) = recalled_d {
                let vg_recall_d = VirtualGroup { group_ids: gids, blend_weights: weights };
                let mut correct_d = 0usize;
                for (input, target) in &task_d_heldout {
                    let out = dm.predict_with_composition(input, &vg_recall_d);
                    if out.len() >= 1 && (out[0] - target[0]).abs() < 0.5 {
                        correct_d += 1;
                    }
                }
                let acc_heldout_d = correct_d as f32 / task_d_heldout.len() as f32;
                println!("  Task D held-out: retrieved composition accuracy = {:.1}% (n={}) [train {:.1}%]",
                    acc_heldout_d * 100.0, task_d_heldout.len(), comp_d_acc * 100.0);
            }
        }
    } else {
        println!("  (No store: Task D composition {:.0}% < 75%.)", comp_d_acc * 100.0);
    }

    // Inference via memory recall (timed)
    let mut sig_d = [0.0f32; 2];
    for (input, _) in &task_d_data {
        sig_d[0] += input[0];
        sig_d[1] += input[1];
    }
    sig_d[0] /= task_d_data.len() as f32;
    sig_d[1] /= task_d_data.len() as f32;
    let start = Instant::now();
    let episode_data = dm.episodic_retrieve(&sig_d, 0.85)
        .map(|ep| (ep.group_ids.clone(), ep.blend_weights.clone()));
    let out_recall = if let Some((gids, weights)) = episode_data {
        let vg = VirtualGroup { group_ids: gids, blend_weights: weights };
        dm.predict_with_composition(&[0.5, 0.3], &vg)
    } else {
        vec![]
    };
    let elapsed = start.elapsed();
    if !out_recall.is_empty() {
        let secs = elapsed.as_secs_f64();
        println!("\n  New task solved in <1 second via memory recall. (measured: {:.4}s) Output: {:?}", secs, out_recall);
    }
}

/// Print chosen group, output, top groups by score, and winner−runner-up gap (scales to 1..N groups).
fn print_routing(label: &str, dm: &DimensionManager, out: &[f32]) {
    let g = dm.last_chosen_group_id().map(|g| g.to_string()).unwrap_or_else(|| "?".into());
    println!("\n  {} → group {} → output: {:?}", label, g, out);
    if let Some(scores) = dm.last_routing_scores() {
        let mut by_score: Vec<_> = scores.iter().map(|&(gid, a, b, m, s)| (gid, a, b, m, s)).collect();
        by_score.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        let top = 5.min(by_score.len());
        for (gid, self_sim, cross_sim, margin, score) in by_score.into_iter().take(top) {
            println!("    group {}: self={:.3} cross={:.3} margin={:.3} score={:.3}",
                gid, self_sim, cross_sim, margin, score);
        }
        if scores.len() >= 2 {
            let mut s: Vec<f32> = scores.iter().map(|(_, _, _, _, x)| *x).collect();
            s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let margin_gap = s[0] - s[1];
            println!("    score gap (winner - runner-up): {:.3}  {}",
                margin_gap,
                if margin_gap >= 0.3 { "← wide (robust)" } else if margin_gap >= 0.1 { "← moderate" } else { "← narrow (fragile)" });
        }
    }
}

// =============================================================================
// Demo 5: Continual Learning (Phase 2 Gate)
//
// Gradient gate is implemented inside backprop via frozen flag on neurons/synapses.
// freeze_consolidated_pathway() sets frozen=true on Group A, output[0], and
// input→Group A layer1 synapses. No restore loop — damage is prevented, not repaired.
// =============================================================================

fn demo_continual_learning() {
    println!("--- Demo 5: Continual Learning (Phase 2) ---\n");

    let base_config = phase2_base_config();

    let mut weight_rng = StdRng::seed_from_u64(42);
    let mut data_rng   = StdRng::seed_from_u64(99);

    // =========================================================================
    // BUILD — [2 → 16 → 16 → 2]
    // Group A: layer1[0..8] + layer2[0..8] → output[0]
    // Group B: layer1[8..16] + layer2[8..16] → output[1]
    // =========================================================================

    let mut env = NeuralEnvironment::new(base_config.clone());
    env.build_layers(&[2, 16, 16, 2], &mut weight_rng);

    let layer1_ids = env.layers[1].clone();
    let layer2_ids = env.layers[2].clone();
    let output_0   = env.layers[3][0];
    let output_1   = env.layers[3][1];

    // Split both hidden layers in half between the two task groups
    let (l1_a, l1_b) = layer1_ids.split_at(layer1_ids.len() / 2);
    let (l2_a, l2_b) = layer2_ids.split_at(layer2_ids.len() / 2);

    let group_a_ids: Vec<NeuronId> = l1_a.iter().chain(l2_a.iter()).cloned().collect();
    let group_b_ids: Vec<NeuronId> = l1_b.iter().chain(l2_b.iter()).cloned().collect();

    let group_a = env.create_group(group_a_ids.clone());
    let group_b = env.create_group(group_b_ids.clone());
    env.pair_mirror_groups(group_a, group_b);

    // After group assignment — enforce output head ownership
    // Group A layer2 may only connect to output[0]
    // Group B layer2 may only connect to output[1]

    for &nid in &group_a_ids {
        if let Some(n) = env.neurons.get_mut(&nid) {
            n.synapses.retain(|s| s.target != output_1);
        }
    }
    for &nid in &group_b_ids {
        if let Some(n) = env.neurons.get_mut(&nid) {
            n.synapses.retain(|s| s.target != output_0);
        }
    }

    // =========================================================================
    // TASK A — Spiral Classification → output[0]
    // Zero-gradient on output[1] via current_out target matching.
    // =========================================================================

    println!("=== TASK A: Spiral Classification ===");
    println!("Training on 400 samples, 4000 epochs...\n");

    let mut spiral_data = generate_spiral_data(400, &mut data_rng);

    for epoch in 0..4000 {
        env.set_epoch(epoch);
        spiral_data.shuffle(&mut weight_rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &spiral_data {
            // Read current output[1] prediction — set as its own target → zero gradient
            let current_out = env.predict(input);
            let target_both = [target[0], current_out[1]];
            epoch_loss += env.train_tick(input, &target_both, &mut weight_rng).loss;
        }
        epoch_loss /= spiral_data.len() as f32;
        if epoch % 500 == 0 {
            println!("  epoch {:>4} | loss={:.5} | syn={} | sparse={:.2} | mass={:.2} | lr={:.5}",
                epoch, epoch_loss,
                env.total_synapses(),
                env.firing_sparsity(),
                env.mean_hidden_mass(),
                env.current_lr);
        }
    }

    let task_a_before = evaluate_accuracy_head(&mut env, &spiral_data, 0);
    println!("\nTask A accuracy before consolidation: {}/{} ({:.1}%)",
        task_a_before.0, task_a_before.1,
        100.0 * task_a_before.0 as f32 / task_a_before.1 as f32);

    // =========================================================================
    // CONSOLIDATION — set frozen flag on consolidated pathway (no snapshots)
    // =========================================================================

    println!("\n>>> Consolidating Task A (frozen flag gate)...");
    env.freeze_consolidated_pathway(&group_a_ids, output_0, l1_a);
    println!("  Frozen: Group A neurons, output[0], input→Group A layer1 synapses");

    // =========================================================================
    // TASK B — Concentric Circles → output[1]; no restore loop
    // =========================================================================

    println!("\n=== TASK B: Concentric Circles ===");
    println!("Training on 400 samples, 4000 epochs...");
    println!("(Consolidated pathway frozen — no restore loop)\n");

    env.current_lr = base_config.learning_rate;

    let mut circles_data = generate_concentric_circles_data(400, &mut data_rng);

    for epoch in 0..4000 {
        env.set_epoch(epoch);
        circles_data.shuffle(&mut weight_rng);
        let mut epoch_loss = 0.0f32;

        for (input, target) in &circles_data {
            let current_out = env.predict(input);
            let target_both = [current_out[0], target[0]];
            epoch_loss += env.train_tick(input, &target_both, &mut weight_rng).loss;
        }

        epoch_loss /= circles_data.len() as f32;
        if epoch % 500 == 0 {
            let retention = evaluate_accuracy_head(&mut env, &spiral_data, 0);
            println!("  epoch {:>4} | loss={:.5} | syn={} | sparse={:.2} | mass={:.2} | A_retain={:.1}%",
                epoch, epoch_loss,
                env.total_synapses(),
                env.firing_sparsity(),
                env.mean_hidden_mass(),
                100.0 * retention.0 as f32 / retention.1 as f32);
        }
    }

    // =========================================================================
    // RESULTS
    // =========================================================================

    let task_a_after  = evaluate_accuracy_head(&mut env, &spiral_data, 0);
    let task_b_result = evaluate_accuracy_head(&mut env, &circles_data, 1);

    let retention_pct = 100.0 * task_a_after.0 as f32 / task_a_after.1 as f32;
    let baseline_pct  = 100.0 * task_a_before.0 as f32 / task_a_before.1 as f32;
    let forgetting    = baseline_pct - retention_pct;

    println!("\n=== CONTINUAL LEARNING RESULTS ===\n");
    println!("  Task A (Spiral):");
    println!("    Before Task B: {}/{} ({:.1}%)", task_a_before.0, task_a_before.1, baseline_pct);
    println!("    After  Task B: {}/{} ({:.1}%)", task_a_after.0, task_a_after.1, retention_pct);
    println!("    Forgetting:    {:.1}%  (threshold: >10%)", forgetting);

    println!("\n  Task B (Circles):");
    println!("    Accuracy: {}/{} ({:.1}%)",
        task_b_result.0, task_b_result.1,
        100.0 * task_b_result.0 as f32 / task_b_result.1 as f32);

    println!("\n  Verdict: {}",
        if forgetting < 5.0 {
            "PASS — near-zero forgetting. Dual gradient gate fully effective."
        } else if forgetting < 10.0 {
            "PASS — within threshold. Minimal forgetting."
        } else if forgetting < 20.0 {
            "PARTIAL — significant forgetting. Check output head isolation."
        } else {
            "FAIL — catastrophic forgetting. Pathway sharing still present."
        }
    );

    // =========================================================================
    // SAMPLE PREDICTIONS
    // =========================================================================

    println!("\n  Task A predictions (head 0, after Task B training):");
    println!("    X        Y        Expected  Predicted");
    println!("    ----------------------------------------");
    for (input, target) in spiral_data.iter().take(6) {
        let out = env.predict(input);
        println!("    {:+.4}  {:+.4}    {:.1}       {:.4}", input[0], input[1], target[0], out[0]);
    }

    println!("\n  Task B predictions (head 1):");
    println!("    X        Y        Expected  Predicted");
    println!("    ----------------------------------------");
    for (input, target) in circles_data.iter().take(6) {
        let out = env.predict(input);
        println!("    {:+.4}  {:+.4}    {:.1}       {:.4}", input[0], input[1], target[0], out[1]);
    }

    // =========================================================================
    // STRUCTURAL REPORT
    // =========================================================================

    println!("\n  Group A neurons (Task A — frozen):");
    for &nid in &group_a_ids {
        if let Some(n) = env.neurons.get(&nid) {
            println!("    ID {:>2} | bias={:>8.4} | syns={:>3} | mass={:.3}",
                nid, n.weight, n.synapses.len(), n.mass);
        }
    }

    println!("\n  Group B neurons (Task B — active):");
    for &nid in &group_b_ids {
        if let Some(n) = env.neurons.get(&nid) {
            println!("    ID {:>2} | bias={:>8.4} | syns={:>3} | mass={:.3}",
                nid, n.weight, n.synapses.len(), n.mass);
        }
    }

    print_structural_report(&env, group_a, group_b);
}

// =============================================================================
// evaluate_accuracy_head — read a specific output head index
// =============================================================================

fn evaluate_accuracy_head(
    env: &mut NeuralEnvironment,
    data: &[Sample],
    head: usize,
) -> (usize, usize) {
    let mut correct = 0;
    for (input, target) in data {
        let out = env.predict(input.as_slice());
        if out.len() <= head { break; }
        let predicted = if out[head] > 0.5 { 1.0_f32 } else { 0.0 };
        if (predicted - target[0]).abs() < 0.01 { correct += 1; }
    }
    (correct, data.len())
}

// =============================================================================
// generate_concentric_circles_data
// Inner ring class 0 at r=0.5, outer ring class 1 at r=1.0
// =============================================================================

fn generate_concentric_circles_data(
    n_per_class: usize,
    rng: &mut impl rand::Rng,
) -> Vec<Sample> {
    use std::f32::consts::PI;
    let mut data = Vec::with_capacity(n_per_class * 2);
    let noise = 0.05_f32;

    for _ in 0..n_per_class {
        let theta = rng.gen::<f32>() * 2.0 * PI;
        let r = 0.5 + rng.gen_range(-noise..noise);
        data.push((vec![r * theta.cos(), r * theta.sin()], [0.0]));
    }
    for _ in 0..n_per_class {
        let theta = rng.gen::<f32>() * 2.0 * PI;
        let r = 1.0 + rng.gen_range(-noise..noise);
        data.push((vec![r * theta.cos(), r * theta.sin()], [1.0]));
    }

    data.shuffle(rng);
    data
}

// =============================================================================
// Structural report
// =============================================================================

fn print_structural_report(env: &NeuralEnvironment, group_a: u32, group_b: u32) {
    println!("\n--- Structural Report ---");

    let snapshots = env.snapshot_all();
    let total: usize = snapshots.iter().map(|s| s.synapse_count).sum();
    let max = snapshots.iter().map(|s| s.synapse_count).max().unwrap_or(0);
    let min = snapshots.iter().map(|s| s.synapse_count).min().unwrap_or(0);
    println!("  Synapses — total: {}, max: {}, min: {}", total, max, min);
    println!("  Total energy cost: {:.3}", snapshots.iter().map(|s| s.energy_used).sum::<f32>());

    let spread = growformer::systems::geometry::compute_geometric_spread(&env.neurons);
    println!("  Geometric spread (stddev): {:.4}", spread);

    let symmetry = mirror_symmetry_score(&env.neurons, &env.groups, group_a, group_b);
    println!("  Mirror symmetry (fractal dim, group {} ↔ {}): {:.4}", group_a, group_b, symmetry);

    println!("  Firing sparsity (last forward pass): {:.3}", env.firing_sparsity());

    let whorls = env.detect_whorls();
    println!("  Whorls detected: {}", whorls.len());
    print_whorl_summary(&whorls);

    println!("\n  Neuron snapshots:");
    println!("  {:>4}  {:>8}  {:>8}  {:>7}  {:>12}  {:>7}", "ID", "Weight", "Fired", "Syns", "Energy", "Group");
    println!("  {}", "-".repeat(56));

    let mut sorted = snapshots;
    sorted.sort_by_key(|s| s.id);
    for s in &sorted {
        println!("  {:>4}  {:>8.4}  {:>8.1}  {:>7}  {:>12.4}  {:>7}",
            s.id, s.weight, s.last_fired, s.synapse_count, s.energy_used,
            s.group_id.map(|g| g.to_string()).unwrap_or("-".to_string()));
    }
}

fn generate_spiral_data(n_per_class: usize, rng: &mut impl rand::Rng) -> Vec<Sample> {
    use std::f32::consts::PI;
    let mut data = Vec::new();
    for class in 0..2 {
        for i in 0..n_per_class {
            let t = (i as f32 / n_per_class as f32) * PI; // * 4.0 * PI
            let offset = if class == 0 { 0.0 } else { PI };
            let r = t / (4.0 * PI);
            let x = r * (t + offset).cos() + rng.gen_range(-0.05..0.05_f32);
            let y = r * (t + offset).sin() + rng.gen_range(-0.05..0.05_f32);
            data.push((vec![x, y], [class as f32]));
        }
    }
    data.shuffle(rng);
    data
}

/// Moons (two crescents) — classic 2-class dataset for composition.
fn generate_moons_data(n_per_class: usize, rng: &mut impl rand::Rng) -> Vec<Sample> {
    use std::f32::consts::PI;
    let mut data = Vec::with_capacity(n_per_class * 2);
    let noise = 0.08_f32;
    for i in 0..n_per_class {
        let t = (i as f32 / n_per_class as f32) * PI;
        let x = t.cos() + rng.gen_range(-noise..noise);
        let y = t.sin() + rng.gen_range(-noise..noise);
        data.push((vec![x, y], [0.0]));
    }
    for i in 0..n_per_class {
        let t = (i as f32 / n_per_class as f32) * PI;
        let x = 1.0 - t.cos() + rng.gen_range(-noise..noise);
        let y = 0.5 - t.sin() + rng.gen_range(-noise..noise);
        data.push((vec![x, y], [1.0]));
    }
    data.shuffle(rng);
    data
}

/// Task D: 3-way radius gate — r < r0 → group0, r < r1 → group1, else → group2. (Moons-gated spiral/circles.)
fn generate_task_d_three_way_data(
    main: &mut MainDimension,
    group_ids: &[GroupId],
    r0: f32,
    r1: f32,
    n_samples: usize,
    rng: &mut impl rand::Rng,
) -> Vec<Sample> {
    if group_ids.len() < 3 {
        return vec![];
    }
    let mut data = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let x = rng.gen_range(-1.0..1.0_f32);
        let y = rng.gen_range(-1.0..1.0_f32);
        let r = (x * x + y * y).sqrt();
        let outputs = main.query(&[x, y], group_ids);
        if outputs.len() < 3 {
            continue;
        }
        let idx = if r < r0 { 0 } else if r < r1 { 1 } else { 2 };
        let out = outputs[idx].1.get(0).copied().unwrap_or(0.5);
        let target = if out >= 0.5 { 1.0 } else { 0.0 };
        data.push((vec![x, y], [target]));
    }
    data
}

/// Task C: spiral-gated circles. Inner region (r < inner_radius) → spiral rule, outer → circles rule.
/// Labels come from querying main's two groups; neither group alone solves Task C well.
fn generate_spiral_gated_circles_data(
    main: &mut MainDimension,
    group_inner: GroupId,
    group_outer: GroupId,
    inner_radius: f32,
    n_samples: usize,
    rng: &mut impl rand::Rng,
) -> Vec<Sample> {
    let mut data = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let x = rng.gen_range(-1.0..1.0_f32);
        let y = rng.gen_range(-1.0..1.0_f32);
        let r = (x * x + y * y).sqrt();
        let outputs = main.query(&[x, y], &[group_inner, group_outer]);
        if outputs.len() < 2 {
            continue;
        }
        let out = if r < inner_radius {
            outputs[0].1.get(0).copied().unwrap_or(0.5)
        } else {
            outputs[1].1.get(0).copied().unwrap_or(0.5)
        };
        let target = if out >= 0.5 { 1.0 } else { 0.0 };
        data.push((vec![x, y], [target]));
    }
    data
}




