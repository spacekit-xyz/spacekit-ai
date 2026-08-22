use crate::neuron::Neuron;
use crate::types::*;
use rand::Rng;
use std::collections::{HashMap, HashSet};

/// System 2: Dynamic Synapse Growth
///
/// Grows new synapses between neurons in adjacent layers based on geometric
/// proximity. Only neurons within growth_radius of each other are candidates.
///
/// GROUP BOUNDARY GATE: synapses are never grown between neurons belonging
/// to different task groups. This keeps each task group a closed subgraph
/// internally — cross-task interference can only occur through the shared
/// output layer, which is intentional and controlled.
///
/// Neurons with group_id=None (input layer, output layer, ungrouped hidden)
/// are exempt from the gate and can form synapses freely.
pub fn grow_synapses(
    neurons: &mut HashMap<NeuronId, Neuron>,
    config: &EnvironmentConfig,
    rng: &mut impl Rng,
    layer_of: &HashMap<NeuronId, usize>,
) -> usize {
    let ids: Vec<NeuronId> = neurons.keys().cloned().collect();
    let positions: HashMap<NeuronId, Vec3> =
        neurons.iter().map(|(id, n)| (*id, n.geometry)).collect();

    // Snapshot group_ids for gate check — avoids borrow conflict inside loop
    let group_ids: HashMap<NeuronId, Option<GroupId>> =
        neurons.iter().map(|(&id, n)| (id, n.group_id)).collect();

    let mut formed = 0;

    for &id in &ids {
        let synapse_count = neurons[&id].synapses.len();
        let over_budget = neurons[&id].over_budget();
        let src_layer = *layer_of.get(&id).unwrap_or(&usize::MAX);
        let src_group = group_ids[&id];

        if neurons[&id].frozen {
            continue;
        }
        if synapse_count >= config.max_synapses_per_neuron || over_budget {
            continue;
        }

        let pos = positions[&id];

        let mut candidates: Vec<(NeuronId, f32)> = ids
            .iter()
            .filter(|&&other| {
                let tgt_layer = *layer_of.get(&other).unwrap_or(&usize::MAX);
                if other == id || tgt_layer != src_layer + 1 {
                    return false;
                }

                // GROUP BOUNDARY GATE
                // Block cross-group connections when both neurons have an assigned group.
                // Neurons without a group (input/output) can connect freely.
                let tgt_group = group_ids[&other];
                match (src_group, tgt_group) {
                    (Some(g1), Some(g2)) if g1 != g2 => return false,
                    _ => {}
                }

                true
            })
            .map(|&other| (other, pos.distance(&positions[&other])))
            .filter(|(_, dist)| *dist < config.growth_radius)
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let n_branches = config.dendritic_branches;
        for (target, dist) in candidates.iter().take(3) {
            if neurons[&id].has_synapse_to(*target) {
                continue;
            }
            let base = (1.0 - dist / config.growth_radius).max(0.1);
            let noise: f32 = rng.gen_range(-0.05..0.05);
            let w = (base + noise).clamp(0.05, 0.5);
            if n_branches > 1 {
                // Count incoming synapses per branch for this target (read-only scan)
                let mut branch_counts = vec![0u32; n_branches];
                for src in neurons.values() {
                    for s in &src.synapses {
                        if s.target == *target {
                            let bi = (s.branch_id as usize) % n_branches;
                            branch_counts[bi] += 1;
                        }
                    }
                }
                let best_branch = branch_counts
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &c)| c)
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0);
                let n = neurons.get_mut(&id).unwrap();
                if n.add_synapse_to_branch(*target, w, config.max_synapses_per_neuron, best_branch)
                {
                    formed += 1;
                    break;
                }
            } else {
                let n = neurons.get_mut(&id).unwrap();
                if n.add_synapse(*target, w, config.max_synapses_per_neuron) {
                    formed += 1;
                    break;
                }
            }
        }
    }
    formed
}

/// Three-phase pruning — mirrors LTP consolidation windows.
/// output_protected: synapses targeting these neurons are never pruned.
pub fn prune_three_phase(
    neurons: &mut HashMap<NeuronId, Neuron>,
    config: &EnvironmentConfig,
    current_tick: u64,
    output_protected: &HashSet<NeuronId>,
    input_protected: &HashSet<NeuronId>,
) -> (usize, usize, usize) {
    let mut p1 = 0usize;
    let mut p2 = 0usize;
    let mut p3 = 0usize;

    // Collect input neuron IDs to avoid borrowing issue inside values_mut
    let input_ids: Vec<NeuronId> = input_protected.iter().cloned().collect();

    for neuron in neurons.values_mut() {
        if neuron.frozen {
            continue;
        }
        // Input neurons' synapses carry raw signal to every hidden neuron.
        // Pruning them means hidden neurons become blind to one input coordinate.
        // Structurally protect all outgoing synapses from input layer neurons.
        if input_ids.contains(&neuron.id) {
            continue;
        }

        // Minimum synapse protection: never prune a neuron's last connection.
        // Without this, KWTA losers accumulate low facilitation → all synapses
        // pruned → neuron permanently disconnected → zero gradient → dead weight.
        // A neuron with 1 synapse can still learn; a neuron with 0 cannot recover.
        let min_synapses = 2;
        if neuron.synapses.len() <= min_synapses {
            continue;
        }

        neuron.synapses.retain(|s| {
            if s.frozen {
                return true;
            }
            // Output-adjacent synapses are structurally protected — never pruned
            if output_protected.contains(&s.target) {
                return true;
            }
            // Engram-consolidated synapses are memory traces — never pruned
            if config.engram_enabled && s.consolidation >= config.engram_prune_threshold {
                return true;
            }

            let cost = s.metabolic_cost();
            let dormancy = current_tick.saturating_sub(s.last_active as u64);

            if s.age < config.prune_early_age {
                let keep = cost >= config.prune_early_threshold;
                if !keep {
                    p1 += 1;
                }
                keep
            } else if s.age < config.prune_long_age {
                let consolidated = s.facilitation > config.prune_mid_facilitation_floor;
                let strong = cost >= config.prune_mid_threshold;
                let keep = strong || consolidated;
                if !keep {
                    p2 += 1;
                }
                keep
            } else {
                let recently_active = dormancy < config.prune_long_dormancy as u64;
                let structurally_strong = cost >= config.prune_long_threshold;
                let keep = recently_active || structurally_strong;
                if !keep {
                    p3 += 1;
                }
                keep
            }
        });
    }

    (p1, p2, p3)
}

/// Activity-dependent potentiation — the positive half of synaptic plasticity.
///
/// Pruning removes unused connections. This function strengthens frequently-used ones.
/// Synapses that survived phase-2 pruning because of high facilitation receive
/// a small strength bonus, completing the Hebbian plasticity loop:
///
///   "Neurons that fire together wire together — and those wires get thicker."
///
/// Biological analog: LTP consolidation. Synapses with high activity grow
/// larger dendritic spines and insert more AMPA receptors, increasing their
/// influence on postsynaptic firing.
pub fn potentiate_active_synapses(
    neurons: &mut HashMap<NeuronId, Neuron>,
    config: &EnvironmentConfig,
    _output_protected: &HashSet<NeuronId>,
) {
    if config.facilitation_bonus <= 0.0 {
        return;
    }

    let high_facilitation = config.prune_mid_facilitation_floor * 1.5;

    for neuron in neurons.values_mut() {
        if neuron.frozen {
            continue;
        }
        for syn in neuron.synapses.iter_mut() {
            if syn.frozen {
                continue;
            }
            if syn.age >= config.prune_early_age && syn.facilitation > high_facilitation {
                let excess = (syn.facilitation - high_facilitation).min(1.0);
                let bonus = config.facilitation_bonus * excess;
                syn.strength = (syn.strength.abs() + bonus).min(5.0) * syn.strength.signum();
            }
        }
    }
}

pub fn prune_dormant_synapses(
    neurons: &mut HashMap<NeuronId, Neuron>,
    min_age: u32,
    min_strength: f32,
) -> usize {
    let mut pruned = 0;
    for neuron in neurons.values_mut() {
        if neuron.frozen {
            continue;
        }
        let before = neuron.synapses.len();
        neuron
            .synapses
            .retain(|s| s.frozen || !(s.age >= min_age && s.metabolic_cost() < min_strength));
        pruned += before - neuron.synapses.len();
    }
    pruned
}
