use crate::types::*;
use crate::neuron::Neuron;
use std::collections::HashMap;

/// System 6: IFS Mirror Coupling
///
/// Replaces the previous centroid-averaging approach which collapsed all
/// mirror-group neurons to identical weights.
///
/// An Iterated Function System (IFS) defines a self-similar attractor via
/// a set of contracting affine transformations. Crucially the attractor is
/// NOT a point — it has internal structure at every scale. Each neuron in a
/// mirror group has a specific counterpart in the partner group, and is pulled
/// toward the *individual reflection* of that counterpart across the midplane
/// between the two groups, rather than toward the group centroid.
///
/// This means:
///   - Diversity within each group is preserved (no collapse to mean)
///   - The two groups develop structurally complementary geometry
///   - Different neurons maintain different weights while still coupling
///
/// Weight modification has been fully removed — mirror coupling expresses
/// itself purely through geometry, which then biases synapse growth.

/// Reflect a point across the midplane between two group centroids.
/// The reflection axis is the x-axis (layer axis) only; y and z are preserved.
fn reflect_across_midplane(point: Vec3, centroid_a: Vec3, centroid_b: Vec3) -> Vec3 {
    let midplane_x = (centroid_a.x + centroid_b.x) / 2.0;
    Vec3::new(2.0 * midplane_x - point.x, point.y, point.z)
}

/// Box-counting approximation of fractal dimension for a set of 3D positions.
/// Uses three scales: coarse, medium, fine.
/// D ≈ -slope of log(N) vs log(scale) where N = number of occupied boxes.
pub fn fractal_dimension(positions: &[Vec3]) -> f32 {
    if positions.len() < 3 {
        return 1.0;
    }

    let scales = [2.0_f32, 1.0, 0.5];
    let counts: Vec<f32> = scales
        .iter()
        .map(|&s| box_count(positions, s) as f32)
        .collect();

    // Linear regression of log(count) ~ log(scale)
    let log_s: Vec<f32> = scales.iter().map(|s| s.ln()).collect();
    let log_c: Vec<f32> = counts.iter().map(|c| c.max(1.0).ln()).collect();

    -linear_slope(&log_s, &log_c)
}

fn box_count(positions: &[Vec3], scale: f32) -> usize {
    let mut occupied = std::collections::HashSet::new();
    for p in positions {
        let ix = (p.x / scale).floor() as i32;
        let iy = (p.y / scale).floor() as i32;
        let iz = (p.z / scale).floor() as i32;
        occupied.insert((ix, iy, iz));
    }
    occupied.len()
}

fn linear_slope(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    if n < 2.0 { return 0.0; }
    let sx: f32 = x.iter().sum();
    let sy: f32 = y.iter().sum();
    let sxy: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sxx: f32 = x.iter().map(|a| a * a).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-10 { return 0.0; }
    (n * sxy - sx * sy) / denom
}

/// IFS mirror coupling: pull each neuron toward the individual reflection
/// of its paired counterpart, NOT toward the group centroid.
pub fn apply_ifs_mirror_coupling(
    neurons: &mut HashMap<NeuronId, Neuron>,
    groups: &HashMap<GroupId, NeuronGroup>,
    config: &EnvironmentConfig,
) {
    for group in groups.values() {
        let Some(mirror_id) = group.mirror_group else { continue };
        let Some(mirror) = groups.get(&mirror_id) else { continue };

        let centroid_a = group.centroid;
        let centroid_b = mirror.centroid;

        // Pair neurons by position in members list.
        // Each neuron in group A targets the reflection of its counterpart in group B.
        for (i, &id_a) in group.members.iter().enumerate() {
            let mirror_idx = i % mirror.members.len();
            let id_b = mirror.members[mirror_idx];

            // Get partner's position (immutable borrow)
            let partner_pos = neurons.get(&id_b).map(|n| n.geometry);

            if let (Some(partner), Some(neuron)) = (partner_pos, neurons.get_mut(&id_a)) {
                if neuron.frozen { continue; }
                // Target: the reflection of the partner across the midplane
                let target = reflect_across_midplane(partner, centroid_a, centroid_b);
                let delta = target - neuron.geometry;
                neuron.geometry = neuron.geometry
                    + delta * (config.mirror_coupling_strength * 0.01);
            }
        }

        // Symmetric operation: group B neurons target reflections of group A neurons
        for (i, &id_b) in mirror.members.iter().enumerate() {
            let group_idx = i % group.members.len();
            let id_a = group.members[group_idx];

            let partner_pos = neurons.get(&id_a).map(|n| n.geometry);

            if let (Some(partner), Some(neuron)) = (partner_pos, neurons.get_mut(&id_b)) {
                if neuron.frozen { continue; }
                let target = reflect_across_midplane(partner, centroid_b, centroid_a);
                let delta = target - neuron.geometry;
                neuron.geometry = neuron.geometry
                    + delta * (config.mirror_coupling_strength);
            }
        }
    }
}

/// Structural symmetry score using fractal dimension comparison
/// rather than weight averaging. Two groups can be structurally similar
/// (high score) while having completely different individual neuron weights.
pub fn mirror_symmetry_score(
    neurons: &HashMap<NeuronId, Neuron>,
    groups: &HashMap<GroupId, NeuronGroup>,
    a: GroupId,
    b: GroupId,
) -> f32 {
    let Some(ga) = groups.get(&a) else { return 0.0 };
    let Some(gb) = groups.get(&b) else { return 0.0 };

    let pos_a: Vec<Vec3> = ga.members.iter()
        .filter_map(|id| neurons.get(id).map(|n| n.geometry))
        .collect();
    let pos_b: Vec<Vec3> = gb.members.iter()
        .filter_map(|id| neurons.get(id).map(|n| n.geometry))
        .collect();

    let dim_a = fractal_dimension(&pos_a);
    let dim_b = fractal_dimension(&pos_b);

    // Score: 1 when dimensions match, decays as they diverge
    let diff = (dim_a - dim_b).abs();
    1.0 / (1.0 + diff)
}

pub fn update_group_centroids(
    groups: &mut HashMap<GroupId, NeuronGroup>,
    neurons: &HashMap<NeuronId, Neuron>,
) {
    for group in groups.values_mut() {
        if group.members.is_empty() {
            group.centroid = Vec3::zero();
            continue;
        }
        let sum = group.members.iter()
            .filter_map(|id| neurons.get(id))
            .fold(Vec3::zero(), |acc, n| acc + n.geometry);
        let count = group.members.len() as f32;
        group.centroid = Vec3::new(sum.x / count, sum.y / count, sum.z / count);
    }
}

pub fn pair_mirror_groups(groups: &mut HashMap<GroupId, NeuronGroup>, a: GroupId, b: GroupId) {
    if let Some(g) = groups.get_mut(&a) { g.mirror_group = Some(b); }
    if let Some(g) = groups.get_mut(&b) { g.mirror_group = Some(a); }
}