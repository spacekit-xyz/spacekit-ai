use crate::neuron::Neuron;
use crate::types::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// System 1: Metabolic Pruning (parallelised with rayon)
///
/// Frozen neurons are excluded from over-budget pruning and from the synapse
/// aging/depression pass — no strength or structure changes touch frozen state.
/// output_protected: set of neuron IDs in the output layer.
/// Synapses targeting these neurons are never pruned metabolically —
/// the output neuron's fan-in is architecturally critical and must
/// survive regardless of early-training weight magnitude.
pub fn apply_metabolic_pressure(
    neurons: &mut HashMap<NeuronId, Neuron>,
    config: &EnvironmentConfig,
    output_protected: &HashSet<NeuronId>,
    input_protected: &HashSet<NeuronId>,
) -> usize {
    let pruned_per: Vec<(NeuronId, Vec<usize>)> = crate::maybe_par_iter!(neurons)
        .filter_map(|(&id, neuron)| {
            if neuron.frozen {
                return None;
            }
            if !neuron.over_budget() {
                return None;
            }
            // Never prune synapses from input neurons — they carry raw signal.
            // Without both input coordinates, hidden neurons are geometrically blind.
            if input_protected.contains(&id) {
                return None;
            }

            let mut indexed: Vec<(usize, f32)> = neuron
                .synapses
                .iter()
                .enumerate()
                // Never prune synapses targeting the output layer
                .filter(|(_, s)| !output_protected.contains(&s.target) && !s.frozen)
                // Engram-consolidated synapses are memory traces — never pruned
                .filter(|(_, s)| {
                    !(config.engram_enabled && s.consolidation >= config.engram_prune_threshold)
                })
                .map(|(i, s)| (i, s.metabolic_cost()))
                .collect();

            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let to_prune: Vec<usize> = indexed
                .iter()
                .filter(|(_, cost)| *cost < config.pruning_threshold)
                .map(|(i, _)| *i)
                .collect();

            if to_prune.is_empty() {
                None
            } else {
                Some((id, to_prune))
            }
        })
        .collect();

    let mut total_pruned = 0;
    for (id, mut indices) in pruned_per {
        if let Some(neuron) = neurons.get_mut(&id) {
            indices.sort_unstable_by(|a, b| b.cmp(a));
            for idx in indices {
                if idx < neuron.synapses.len() {
                    neuron.synapses.remove(idx);
                    total_pruned += 1;
                }
                if !neuron.over_budget() {
                    break;
                }
            }
            if neuron.over_budget() {
                neuron.energy_budget *= 1.02;
            }
        }
    }

    // Age all synapses each tick (skip frozen — they are not updated)
    for neuron in neurons.values_mut() {
        if neuron.frozen {
            continue;
        }
        for s in neuron.synapses.iter_mut() {
            if s.frozen {
                continue;
            }
            s.age += 1;
            if s.age > 1000 && s.facilitation > 1.5 {
                s.depression *= 0.999;
            }
        }
    }

    total_pruned
}
