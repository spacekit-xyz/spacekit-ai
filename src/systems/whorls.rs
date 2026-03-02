use crate::types::*;
use crate::neuron::Neuron;
use std::collections::{HashMap, HashSet};

/// System 5: Whorl Detection
///
/// In the Harvard mapping, some axons had coiled into tight whorls
/// for completely unknown reasons. We model this as geometric self-reference:
/// a neuron whose downstream connections loop back into its own spatial neighborhood.
///
/// Whorls are not necessarily pathological — they may represent a stable
/// attractor or memory loop. We detect them, flag them, and expose them
/// as a diagnostic signal rather than eliminating them.

#[derive(Debug, Clone)]
pub struct WhorlReport {
    pub neuron_id: NeuronId,
    pub loop_depth: usize,
    pub spatial_density: f32, // how tightly packed the loop is
    pub involved_neurons: Vec<NeuronId>,
}

/// Detect neurons involved in cycles within a spatial radius
/// Uses DFS to find loops, filtered by spatial proximity
pub fn detect_whorls(
    neurons: &HashMap<NeuronId, Neuron>,
    spatial_radius: f32,
) -> Vec<WhorlReport> {
    let mut reports = Vec::new();

    for (&start_id, _start_neuron) in neurons {
        let cycles = find_cycles_from(neurons, start_id, 8); // max depth 8

        for cycle in cycles {
            // Check if the cycle is spatially coiled
            let positions: Vec<Vec3> = cycle
                .iter()
                .filter_map(|id| neurons.get(id).map(|n| n.geometry))
                .collect();

            let density = compute_spatial_density(&positions, spatial_radius);

            if density > 0.5 {
                reports.push(WhorlReport {
                    neuron_id: start_id,
                    loop_depth: cycle.len(),
                    spatial_density: density,
                    involved_neurons: cycle,
                });
            }
        }
    }

    // Deduplicate (same cycle detected from multiple starting points)
    dedup_whorls(reports)
}

/// DFS to find cycles starting from a given neuron
fn find_cycles_from(
    neurons: &HashMap<NeuronId, Neuron>,
    start: NeuronId,
    max_depth: usize,
) -> Vec<Vec<NeuronId>> {
    let mut cycles = Vec::new();
    let mut path = vec![start];
    let mut visited = HashSet::new();
    visited.insert(start);

    dfs_cycles(neurons, start, start, max_depth, &mut path, &mut visited, &mut cycles);

    cycles
}

fn dfs_cycles(
    neurons: &HashMap<NeuronId, Neuron>,
    current: NeuronId,
    start: NeuronId,
    depth_remaining: usize,
    path: &mut Vec<NeuronId>,
    visited: &mut HashSet<NeuronId>,
    cycles: &mut Vec<Vec<NeuronId>>,
) {
    if depth_remaining == 0 {
        return;
    }

    if let Some(neuron) = neurons.get(&current) {
        for synapse in &neuron.synapses {
            let target = synapse.target;

            if target == start && path.len() > 2 {
                // Found a cycle back to start
                cycles.push(path.clone());
                continue;
            }

            if !visited.contains(&target) {
                visited.insert(target);
                path.push(target);

                dfs_cycles(neurons, target, start, depth_remaining - 1, path, visited, cycles);

                path.pop();
                visited.remove(&target);
            }
        }
    }
}

fn compute_spatial_density(positions: &[Vec3], radius: f32) -> f32 {
    if positions.len() < 2 {
        return 0.0;
    }

    let mut close_pairs = 0;
    let total_pairs = positions.len() * (positions.len() - 1) / 2;

    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            if positions[i].distance(&positions[j]) < radius {
                close_pairs += 1;
            }
        }
    }

    if total_pairs == 0 {
        0.0
    } else {
        close_pairs as f32 / total_pairs as f32
    }
}

fn dedup_whorls(reports: Vec<WhorlReport>) -> Vec<WhorlReport> {
    // Canonical form of a cycle: rotate so the smallest ID is first,
    // then use that as the dedup key. This catches the same cycle
    // detected from different starting neurons.
    let mut seen: std::collections::HashSet<Vec<NeuronId>> = std::collections::HashSet::new();
    let mut unique = Vec::new();

    for report in reports {
        let canonical = canonical_cycle(&report.involved_neurons);
        if seen.insert(canonical) {
            unique.push(report);
        }
    }

    unique
}

/// Rotate the cycle so the smallest ID is at position 0 (canonical form)
fn canonical_cycle(cycle: &[NeuronId]) -> Vec<NeuronId> {
    if cycle.is_empty() {
        return vec![];
    }
    let min_pos = cycle
        .iter()
        .enumerate()
        .min_by_key(|(_, &id)| id)
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut canonical: Vec<NeuronId> = cycle[min_pos..].to_vec();
    canonical.extend_from_slice(&cycle[..min_pos]);
    canonical
}

/// Print a summary of detected whorls
pub fn print_whorl_summary(whorls: &[WhorlReport]) {
    if whorls.is_empty() {
        println!("  No whorls detected.");
        return;
    }
    for w in whorls {
        println!(
            "  Whorl at neuron {:>3}: depth={}, density={:.2}, involved={:?}",
            w.neuron_id, w.loop_depth, w.spatial_density, w.involved_neurons
        );
    }
}