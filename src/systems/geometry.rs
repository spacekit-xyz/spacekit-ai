use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use crate::types::*;
use crate::neuron::Neuron;
use std::collections::HashMap;

/// System 4: Physics-Based Geometry — N-body Particle Integrator
///
/// Frozen neurons are skipped in the integration step (pinned in space).
/// Their activations still participate in lateral inhibition via exp(-dist/sigma);
/// as Group B geometry drifts and Group A stays fixed, inhibition geometry
/// between groups diverges — expected and desirable (natural separation).
///
/// Three physics mechanisms, each solving a distinct problem:
///
/// ## Force Model (geometry update)
///
/// 1. LAMINAR GRAVITY — linear restoring force toward layer centroid.
///    Keeps neurons in their laminar plane without inverse-square collapse.
///    F_gravity = G * mass * (centroid - pos) / dist
///
/// 2. DEBYE-SCREENED REPULSION — same-layer repulsion with exponential falloff.
///    F_repel = k_repel * exp(-dist / debye_length) * (pos_i - pos_j) / (dist² + ε)
///    Replaces the hard cutoff radius with adaptive density-dependent screening.
///    Dense regions: effective range shrinks (more screening).
///    Sparse regions: effective range grows (less screening).
///    Result: neurons self-regulate spacing without manual radius tuning.
///
/// 3. HEBBIAN ATTRACTION — correlated synaptic partners attract.
///    F_hebbian = hebbian_attraction * correlation * syn_cost * direction / dist
///
/// 4. VELOCITY DAMPING + THERMAL NOISE — semi-implicit Euler integration.
///    v *= (1 - damping), then v += a * dt + noise
///
/// 5. GROUP BOUNDARY REPULSION — neurons belonging to different task groups
///    receive extra repulsion (3× multiplier), causing task groups to drift
///    into spatially separated clusters. This prevents cross-task interference
///    in synapse growth and lateral inhibition.
///    Neurons with group_id=None are unaffected (input/output layers).
///
/// ## Reaction-Diffusion Lateral Inhibition (activation update, called from environment)
///
/// Replaces uniform layer-mean inhibition with spatially-local inhibition:
///    inhib_i = Σ_j exp(-dist_ij / sigma_inhib) * act_j  /  normalizer
///    act_i -= lateral_inhibition * inhib_i
///
/// The exp(-dist/sigma) kernel means nearby neurons inhibit strongly,
/// distant neurons inhibit weakly. This creates Turing instability:
/// with appropriate sigma_inhib, neurons self-organise into stable
/// non-overlapping receptive fields (spots/patches in input space).
/// The pattern scale is controlled by sigma_inhib relative to the
/// geometry spread — smaller sigma → smaller, more numerous fields.

pub fn update_geometry(
    neurons: &mut HashMap<NeuronId, Neuron>,
    config: &EnvironmentConfig,
    layer_of: &HashMap<NeuronId, usize>,
    rng: &mut impl Rng,
) {
    let n_layers = layer_of.values().max().copied().unwrap_or(0) + 1;

    // Per-layer centroids
    let mut layer_centroids: Vec<Vec3> = vec![Vec3::zero(); n_layers];
    let mut layer_counts: Vec<f32> = vec![0.0; n_layers];
    for (id, neuron) in neurons.iter() {
        if let Some(&layer) = layer_of.get(id) {
            layer_centroids[layer] = layer_centroids[layer] + neuron.geometry;
            layer_counts[layer] += 1.0;
        }
    }
    for i in 0..n_layers {
        if layer_counts[i] > 0.0 {
            layer_centroids[i] = layer_centroids[i] * (1.0 / layer_counts[i]);
        }
    }

    // Snapshot: (pos, activation, mass, layer, group_id)
    // group_id included so force computation can apply group boundary penalty
    let snapshot: HashMap<NeuronId, (Vec3, f32, f32, usize, Option<GroupId>)> = neurons.iter()
        .map(|(&id, n)| {
            let layer = *layer_of.get(&id).unwrap_or(&0);
            (id, (n.geometry, n.activation, n.mass, layer, n.group_id))
        })
        .collect();

    let synapse_snapshot: HashMap<NeuronId, Vec<(NeuronId, f32)>> = neurons.iter()
        .map(|(&id, n)| (id, n.synapses.iter().map(|s| (s.target, s.metabolic_cost())).collect()))
        .collect();

    let forces: Vec<(NeuronId, Vec3)> = crate::maybe_par_iter!(snapshot)
        .map(|(&id, &(pos, act, mass, layer, group_id))| {
            let centroid = layer_centroids.get(layer).copied().unwrap_or(Vec3::zero());
            let mut f = Vec3::zero();

            // --- Force 1: Laminar gravity ---
            let to_centroid = centroid - pos;
            let dist_to_centroid = to_centroid.magnitude().max(0.01);
            f = f + to_centroid * (config.gravity_g * mass / dist_to_centroid);

            // --- Force 2: Debye-screened same-layer repulsion ---
            // Group boundary: neurons from different task groups repel 3× harder,
            // causing task groups to self-organise into spatially separate clusters.
            // Only applies when both neurons have an assigned group (not input/output).
            let max_repulsion_dist = config.debye_length * 4.0;
            for (&other_id, &(other_pos, _, other_mass, other_layer, other_group)) in snapshot.iter() {
                if other_id == id || other_layer != layer { continue; }
                let diff = pos - other_pos;
                let dist_sq = diff.magnitude_sq();
                let dist = dist_sq.sqrt();
                if dist > max_repulsion_dist { continue; }

                // Group boundary penalty: 3× repulsion between different task groups.
                // Neurons with no group (input/output, ungrouped hidden) are unaffected.
                let group_penalty = match (group_id, other_group) {
                    (Some(g1), Some(g2)) if g1 != g2 => 3.0,
                    _ => 1.0,
                };

                // Debye screening: repulsion decays as exp(-dist/lambda_D)
                let screening = (-dist / config.debye_length).exp();
                let magnitude = (config.k_repel * mass * other_mass * group_penalty * screening
                    / (dist_sq + 0.01)).min(5.0);
                f = f + diff * (magnitude / dist.max(0.001));
            }

            // --- Force 3: Hebbian attraction to correlated partners ---
            if let Some(syns) = synapse_snapshot.get(&id) {
                for &(target_id, syn_cost) in syns {
                    if let Some(&(target_pos, target_act, _, _, _)) = snapshot.get(&target_id) {
                        let correlation = (act * target_act).max(0.0);
                        if correlation > 0.0 && syn_cost > 0.0 {
                            let diff = target_pos - pos;
                            let dist = diff.magnitude().max(0.01);
                            let magnitude = config.hebbian_attraction * correlation * syn_cost;
                            f = f + diff * (magnitude / dist);
                        }
                    }
                }
            }

            (id, f)
        })
        .collect();

    // Integrate
    let dt = config.physics_dt;
    for (id, force) in forces {
        if let Some(neuron) = neurons.get_mut(&id) {
            if neuron.frozen { continue; }
            let noise = Vec3::new(
                gaussian_sample(rng) * config.thermal_noise,
                gaussian_sample(rng) * config.thermal_noise,
                gaussian_sample(rng) * config.thermal_noise,
            );
            let accel = force * (1.0 / neuron.mass);
            neuron.velocity = neuron.velocity * (1.0 - config.damping) + accel * dt + noise;
            // Velocity clamp prevents instability at close range
            let speed = neuron.velocity.magnitude();
            if speed > 2.0 { neuron.velocity = neuron.velocity * (2.0 / speed); }
            neuron.geometry = neuron.geometry + neuron.velocity * dt;
            neuron.geometry.x = neuron.geometry.x.clamp(-10.0, 10.0);
            neuron.geometry.y = neuron.geometry.y.clamp(-10.0, 10.0);
            neuron.geometry.z = neuron.geometry.z.clamp(-10.0, 10.0);
        }
    }
}

/// Reaction-diffusion lateral inhibition — Turing pattern mechanism.
///
/// Called from environment.rs forward pass for layer_idx == 1.
/// Replaces uniform layer-mean inhibition with spatially-local inhibition.
///
/// Each neuron i receives inhibition from all same-layer neurons j,
/// weighted by exp(-dist_ij / sigma_inhib). Near neighbors inhibit
/// strongly; distant neurons inhibit weakly or not at all.
///
/// The two-scale structure (local activation, wider inhibition) is
/// the condition for Turing instability — it drives spontaneous
/// pattern formation without explicit engineering.
///
/// sigma_inhib should be 30–60% of geometric spread for stable patterns.
/// Too small → many tiny disconnected fields (under-inhibited).
/// Too large → degenerates to uniform mean inhibition (no patterns).
pub fn reaction_diffusion_inhibition(
    layer_ids: &[NeuronId],
    neurons: &mut HashMap<NeuronId, Neuron>,
    dropout_mask: &HashMap<NeuronId, bool>,
    lateral_inhibition: f32,
    sigma_inhib: f32,
) {
    if lateral_inhibition <= 0.0 || sigma_inhib <= 0.0 { return; }

    // Snapshot positions + activations for this layer
    let layer_state: Vec<(NeuronId, Vec3, f32)> = layer_ids.iter()
        .filter(|id| dropout_mask.get(id) != Some(&true))
        .filter_map(|&id| {
            neurons.get(&id).map(|n| (id, n.geometry, n.activation))
        })
        .collect();

    if layer_state.is_empty() { return; }

    // Compute spatially-weighted inhibitory input for each neuron
    // inhib_i = Σ_j kernel(dist_ij) * act_j  normalized by Σ_j kernel(dist_ij)
    let inhibitions: Vec<(NeuronId, f32)> = layer_state.iter()
        .map(|&(id, pos, _)| {
            let mut weighted_sum = 0.0f32;
            let mut weight_total = 0.0f32;
            for &(other_id, other_pos, other_act) in &layer_state {
                if other_id == id { continue; }
                let dist = pos.distance(&other_pos);
                let kernel = (-dist / sigma_inhib).exp();
                weighted_sum += kernel * other_act;
                weight_total += kernel;
            }
            let inhib = if weight_total > 1e-6 {
                weighted_sum / weight_total
            } else {
                0.0
            };
            (id, inhib)
        })
        .collect();

    // Apply inhibition, floor at 0 (skip frozen — consolidated pathway keeps pre-inhibition activation)
    for (id, inhib) in inhibitions {
        if let Some(n) = neurons.get_mut(&id) {
            if n.frozen { continue; }
            n.activation = (n.activation - lateral_inhibition * inhib).max(0.0);
        }
    }
}

/// Box-Muller transform: two uniform samples → one standard normal sample
fn gaussian_sample(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.gen_range(1e-10_f32..1.0);
    let u2: f32 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

pub fn compute_geometric_spread(neurons: &HashMap<NeuronId, Neuron>) -> f32 {
    if neurons.is_empty() { return 0.0; }
    let sum = neurons.values().fold(Vec3::zero(), |acc, n| acc + n.geometry);
    let count = neurons.len() as f32;
    let centroid = Vec3::new(sum.x / count, sum.y / count, sum.z / count);
    let variance: f32 = neurons.values()
        .map(|n| { let d = n.geometry.distance(&centroid); d * d })
        .sum::<f32>() / count;
    variance.sqrt()
}