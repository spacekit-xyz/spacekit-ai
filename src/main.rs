use neuro::environment::NeuralEnvironment;
use neuro::systems::mirror::mirror_symmetry_score;
use neuro::systems::whorls::print_whorl_summary;
use neuro::types::EnvironmentConfig;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "spacekit-neuro", version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    xor: bool,
    #[arg(short, long)]
    spiral: bool,
}

fn main() {
    println!("=============================================================");
    println!("  Multidimensional Neural Environment — Training Demo");
    println!("=============================================================\n");

    let args = Args::parse();
    if args.xor == true {
        demo_xor();
    } else if args.spiral == true {
        demo_spiral();
    } else {
        println!("Please specify either --xor or --spiral");
        std::process::exit(1);
    }


    // demo_xor();
    // println!();
    // demo_spiral();
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

    let mut xor_data: Vec<([f32; 2], [f32; 1])> = vec![
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    println!("Training XOR for 5000 epochs...");
    for epoch in 0..5000 {
        xor_data.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        for (input, target) in &xor_data {
            epoch_loss += env.train_tick(input, target, &mut rng).loss;
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
    // seeds 42, 7, 99, 314, 271.
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

    let spread = neuro::systems::geometry::compute_geometric_spread(&env.neurons);
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

fn generate_spiral_data(n_per_class: usize, rng: &mut impl rand::Rng) -> Vec<([f32; 2], [f32; 1])> {
    use std::f32::consts::PI;
    let mut data = Vec::new();
    for class in 0..2 {
        for i in 0..n_per_class {
            let t = (i as f32 / n_per_class as f32) * PI; // * 4.0 * PI
            let offset = if class == 0 { 0.0 } else { PI };
            let r = t / (4.0 * PI);
            let x = r * (t + offset).cos() + rng.gen_range(-0.05..0.05_f32);
            let y = r * (t + offset).sin() + rng.gen_range(-0.05..0.05_f32);
            data.push(([x, y], [class as f32]));
        }
    }
    data.shuffle(rng);
    data
}