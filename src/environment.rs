use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::neuron::Neuron;
use crate::types::*;
use crate::systems::{
    geometry::{update_geometry, reaction_diffusion_inhibition, compute_geometric_spread},
    growth::{grow_synapses, prune_dormant_synapses, prune_three_phase, potentiate_active_synapses},
    metabolic::apply_metabolic_pressure,
    mirror::{apply_ifs_mirror_coupling, update_group_centroids},
    stdp::{get_fired_neurons, record_firing, update_stdp_layer},
    whorls::{detect_whorls, WhorlReport},
};

#[derive(Clone, Serialize, Deserialize)]
pub struct NeuralEnvironment {
    pub neurons: HashMap<NeuronId, Neuron>,
    pub groups: HashMap<GroupId, NeuronGroup>,
    pub time: f64,
    pub config: EnvironmentConfig,
    pub layers: Vec<Vec<NeuronId>>,
    pub layer_of: HashMap<NeuronId, usize>,

    next_neuron_id: NeuronId,
    next_group_id: GroupId,
    tick_count: u64,
    dropout_mask: HashMap<NeuronId, bool>,
    output_ids: HashSet<NeuronId>,
    input_ids: HashSet<NeuronId>,   // neurons in input layer — their outgoing synapses are protected

    /// Group IDs that must receive zero gradient (continual learning: consolidated tasks).
    /// Set by Brain before each train_tick so backprop does not update these neurons' weights/synapses.
    consolidated_group_ids: HashSet<GroupId>,

    /// Current effective learning rate — starts at config.learning_rate,
    /// annealed each epoch via set_epoch() when lr_decay > 0.
    pub current_lr: f32,

    /// Ephaptic field: per-hidden-layer EMA of activations.
    /// Index i corresponds to self.layers[i] (only populated for hidden layers).
    /// Provides immediate pattern availability without synaptic changes.
    #[serde(default)]
    ephaptic_fields: Vec<Vec<f32>>,
}

impl NeuralEnvironment {
    pub fn new(config: EnvironmentConfig) -> Self {
        let lr = config.learning_rate;
        Self {
            neurons: HashMap::new(),
            groups: HashMap::new(),
            time: 0.0,
            config,
            layers: Vec::new(),
            layer_of: HashMap::new(),
            next_neuron_id: 0,
            next_group_id: 0,
            tick_count: 0,
            dropout_mask: HashMap::new(),
            output_ids: HashSet::new(),
            input_ids: HashSet::new(),
            consolidated_group_ids: HashSet::new(),
            current_lr: lr,
            ephaptic_fields: Vec::new(),
        }
    }

    /// Set which groups receive zero gradient in backprop (consolidated / frozen tasks).
    /// Call before each train_tick when using continual learning.
    pub fn set_consolidated_groups(&mut self, ids: &[GroupId]) {
        self.consolidated_group_ids = ids.iter().copied().collect();
    }

    /// Current simulation tick (number of steps run).
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Set the current tick (e.g. when restoring from snapshot).
    pub fn set_tick_count(&mut self, v: u64) {
        self.tick_count = v;
    }

    /// Neuron ids in the input layer (read-only).
    pub fn input_ids(&self) -> &HashSet<NeuronId> {
        &self.input_ids
    }

    /// Neuron ids in the output layer (read-only).
    pub fn output_ids(&self) -> &HashSet<NeuronId> {
        &self.output_ids
    }

    /// After loading a checkpoint, sync input/output sets from the restored layers
    /// so they match the checkpoint topology (required for correct forward and freeze logic).
    pub fn sync_input_output_ids_from_layers(&mut self) {
        if let Some(first) = self.layers.first() {
            self.input_ids = first.iter().copied().collect();
        }
        if let Some(last) = self.layers.last() {
            self.output_ids = last.iter().copied().collect();
        }
    }

    /// After loading a checkpoint, set next_neuron_id to max(ids)+1 so new growth
    /// does not reuse existing IDs.
    pub fn sync_next_neuron_id_from_neurons(&mut self) {
        self.next_neuron_id = self.neurons.keys().max().copied().unwrap_or(0).wrapping_add(1);
    }

    /// Number of neurons in the input layer (first layer), or `None` if no layers built.
    pub fn input_layer_size(&self) -> Option<usize> {
        self.layers.first().map(|l| l.len())
    }

    pub fn build_layers(&mut self, layer_sizes: &[usize], rng: &mut impl Rng) {
        self.layers.clear();
        self.layer_of.clear();

        for (layer_idx, &size) in layer_sizes.iter().enumerate() {
            let mut layer_ids = Vec::new();
            for pos_idx in 0..size {
                let id = self.next_neuron_id;
                self.next_neuron_id += 1;

                let x = layer_idx as f32 * 3.0;
                // Normalize y-spread by sqrt(size) so layers of 4, 16, 32 neurons
                // all initialize with similar geometric spread (~2–3 units).
                // Without this: 32-neuron layer spans y=-16..+16 (spread=7+),
                // physics starts in explosion mode rather than equilibrium.
                let y_raw = pos_idx as f32 - (size as f32 / 2.0);
                let y = y_raw / (size as f32).sqrt() * 2.5; // normalize to ±2.5 range
                let z: f32 = rng.gen_range(-0.5..0.5);

                let mut neuron = Neuron::new(id, Vec3::new(x, y, z), &self.config);

                // Bias initialisation:
                // - Input layer: ±0.1 (standard small random init)
                // - ALL hidden layers: +0.3 to +0.8 positive offset
                //   Hidden neurons drift negative over 3.2M steps regardless of depth.
                //   For 2→32→1, layer_idx==1 IS the only hidden layer — was incorrectly
                //   getting ±0.1 init (the `layer_idx > 1` condition missed it entirely).
                //   Starting at +0.55 average gives sigmoid(-1.9 drift + 0.55) = 0.25
                //   after full training — in healthy activation range.
                // - Output layer: ±0.1 (standard)
                let n_layers = layer_sizes.len();
                let is_hidden = layer_idx > 0 && layer_idx < n_layers - 1;
                neuron.weight = if is_hidden {
                    rng.gen_range(0.0_f32..0.4)  // moderate positive: warm-up phase handles early learning
                    // no need for extreme +1.5 init — neurons learn before inhibition activates
                } else {
                    rng.gen_range(-0.1_f32..0.1)  // input/output: standard small random
                };

                self.neurons.insert(id, neuron);
                self.layer_of.insert(id, layer_idx);
                layer_ids.push(id);
            }
            self.layers.push(layer_ids);
        }

        // Xavier uniform init — CES-informed scaling for attenuated layers
        // Standard Xavier assumes activations average ~0.5 across all layers.
        // Layer-1 after reaction-diffusion inhibition averages ~0.35 — 70% of expected.
        // Layer1→layer2 synapses are initialised too small: layer-2 gets weak input,
        // produces weak gradient, drifts to silence.
        // Fix: scale up layer1→layer2 by 1/attenuation ≈ 1/0.7 ≈ 1.43×
        // This is the CES "experience-informed initialization" principle applied to
        // a known environmental property (lateral inhibition attenuation).
        for layer_idx in 0..(self.layers.len() - 1) {
            let sources: Vec<NeuronId> = self.layers[layer_idx].clone();
            let targets: Vec<NeuronId> = self.layers[layer_idx + 1].clone();
            let base_scale = (6.0_f32 / (sources.len() + targets.len()) as f32).sqrt();

            // Compensate for lateral inhibition attenuation on layer 1 output
            // attenuation ≈ (1 - lateral_inhibition * 0.7) based on empirical mean activation
            let attenuation_compensation = if layer_idx == 0
                && self.config.lateral_inhibition > 0.0 {
                1.0 / (1.0 - self.config.lateral_inhibition * 0.7).max(0.3)
            } else {
                1.0
            };
            let scale = base_scale * attenuation_compensation;

            let n_branches = self.config.dendritic_branches;
            let is_to_hidden = layer_idx < self.layers.len() - 2;
            for &src in &sources {
                for &tgt in &targets {
                    let w: f32 = rng.gen_range(-scale..=scale);
                    let neuron = self.neurons.get_mut(&src).unwrap();
                    if n_branches > 1 && is_to_hidden {
                        let br = rng.gen_range(0..n_branches) as u8;
                        neuron.add_synapse_to_branch(tgt, w, self.config.max_synapses_per_neuron, br);
                    } else {
                        neuron.add_synapse(tgt, w, self.config.max_synapses_per_neuron);
                    }
                }
            }
        }
        // Record output layer IDs for pruning protection
        if let Some(output_layer) = self.layers.last() {
            self.output_ids = output_layer.iter().cloned().collect();
        }
        // Record input layer IDs — their outgoing synapses carry raw signal
        // and must not be pruned by weight/age alone. Without both input coordinates
        // reaching every hidden neuron, spiral classification is geometrically impossible.
        if let Some(input_layer) = self.layers.first() {
            self.input_ids = input_layer.iter().cloned().collect();
        }
    }

    /// Neurogenesis: insert one neuron into the given hidden layer. Creates the neuron (or promotes
    /// from `reserve_pool` if provided and non-empty), appends to that layer, adds synapses
    /// to/from adjacent layers with Xavier-style init. Returns the new neuron id, or None if
    /// layer_idx is invalid (not a hidden layer).
    pub fn insert_neuron_at_layer(
        &mut self,
        layer_idx: usize,
        rng: &mut impl Rng,
        reserve_pool: Option<&mut Vec<Neuron>>,
    ) -> Option<NeuronId> {
        let n_layers = self.layers.len();
        if n_layers < 3 || layer_idx == 0 || layer_idx >= n_layers - 1 {
            return None;
        }
        let id = self.next_neuron_id;
        self.next_neuron_id += 1;

        let prev_layer = self.layers[layer_idx - 1].clone();
        let next_layer = self.layers[layer_idx + 1].clone();
        let size_after = self.layers[layer_idx].len() + 1;

        let x = layer_idx as f32 * 3.0;
        let y_raw = (size_after - 1) as f32 - (size_after as f32 / 2.0);
        let y = y_raw / (size_after as f32).sqrt() * 2.5;
        let z: f32 = rng.gen_range(-0.5..0.5);
        let weight = rng.gen_range(0.0_f32..0.4);

        let neuron = if let Some(pool) = reserve_pool {
            if let Some(mut n) = pool.pop() {
                n.id = id;
                n.geometry = Vec3::new(x, y, z);
                n.weight = weight;
                n.synapses.clear();
                n
            } else {
                let mut n = Neuron::new(id, Vec3::new(x, y, z), &self.config);
                n.weight = weight;
                n
            }
        } else {
            let mut n = Neuron::new(id, Vec3::new(x, y, z), &self.config);
            n.weight = weight;
            n
        };

        self.neurons.insert(id, neuron);
        self.layer_of.insert(id, layer_idx);
        self.layers[layer_idx].push(id);
        self.dropout_mask.insert(id, false);

        let fan_in = prev_layer.len();
        let fan_out = next_layer.len();
        let base_scale = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
        let attenuation_compensation = if layer_idx == 1 && self.config.lateral_inhibition > 0.0 {
            1.0 / (1.0 - self.config.lateral_inhibition * 0.7).max(0.3)
        } else {
            1.0
        };
        let scale = base_scale * attenuation_compensation;

        let n_br = self.config.dendritic_branches;
        let is_hidden_target = layer_idx > 0 && layer_idx < self.layers.len() - 1;
        for &src in &prev_layer {
            let w: f32 = rng.gen_range(-scale..=scale);
            if let Some(n) = self.neurons.get_mut(&src) {
                if n_br > 1 && is_hidden_target {
                    let br = rng.gen_range(0..n_br) as u8;
                    n.add_synapse_to_branch(id, w, self.config.max_synapses_per_neuron, br);
                } else {
                    n.add_synapse(id, w, self.config.max_synapses_per_neuron);
                }
            }
        }
        for &tgt in &next_layer {
            let w: f32 = rng.gen_range(-scale..=scale);
            if let Some(n) = self.neurons.get_mut(&id) {
                n.add_synapse(tgt, w, self.config.max_synapses_per_neuron);
            }
        }

        Some(id)
    }

    pub fn create_group(&mut self, member_ids: Vec<NeuronId>) -> GroupId {
        let id = self.next_group_id;
        self.next_group_id += 1;
        let mut group = NeuronGroup::new(id);
        group.members = member_ids.clone();
        for nid in &member_ids {
            if let Some(n) = self.neurons.get_mut(nid) { n.group_id = Some(id); }
        }
        self.groups.insert(id, group);
        id
    }

    pub fn pair_mirror_groups(&mut self, a: GroupId, b: GroupId) {
        crate::systems::mirror::pair_mirror_groups(&mut self.groups, a, b);
        let members_a: Vec<NeuronId> = self.groups[&a].members.clone();
        let members_b: Vec<NeuronId> = self.groups[&b].members.clone();
        for (na, nb) in members_a.iter().zip(members_b.iter()) {
            if let Some(n) = self.neurons.get_mut(na) { n.mirror_partner = Some(*nb); }
            if let Some(n) = self.neurons.get_mut(nb) { n.mirror_partner = Some(*na); }
        }
    }

    /// Freeze the consolidated pathway so no gradient or plasticity can modify it.
    /// Sets frozen=true on: all neurons in group_a_ids (Group A layer1+layer2), output_0,
    /// every synapse from input layer that targets group_a_layer1_ids, and every synapse
    /// that targets output_0 (so Group B cannot rewrite Task A's readout).
    pub fn freeze_consolidated_pathway(
        &mut self,
        group_a_ids: &[NeuronId],
        output_0: NeuronId,
        group_a_layer1_ids: &[NeuronId],
    ) {
        for &nid in group_a_ids {
            if let Some(n) = self.neurons.get_mut(&nid) {
                n.frozen = true;
            }
        }
        if let Some(n) = self.neurons.get_mut(&output_0) {
            n.frozen = true;
        }
        for &input_nid in &self.input_ids {
            if let Some(n) = self.neurons.get_mut(&input_nid) {
                for syn in n.synapses.iter_mut() {
                    if group_a_layer1_ids.contains(&syn.target) {
                        syn.frozen = true;
                    }
                }
            }
        }
        // Freeze every synapse that targets output_0 (incl. from Group B) so Task B cannot corrupt Task A's readout.
        for n in self.neurons.values_mut() {
            for syn in n.synapses.iter_mut() {
                if syn.target == output_0 {
                    syn.frozen = true;
                }
            }
        }
    }

    /// Freeze every neuron and synapse (e.g. when promoting a Mirror env to Main).
    pub fn freeze_all(&mut self) {
        for n in self.neurons.values_mut() {
            n.frozen = true;
            for syn in &mut n.synapses {
                syn.frozen = true;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Forward pass — frozen blocks writes only, not reads
    // -------------------------------------------------------------------------
    // Activations are always computed for every neuron (including frozen) so that
    // Group A's signal flows to output[0] and Group B can read from shared input.
    // Frozen only skips plasticity writes: mass/homeostasis below, and backprop.
    // -------------------------------------------------------------------------

    /// Full forward pass. `training` enables dropout for symmetry breaking.
    fn forward_pass(&mut self, input: &[f32], training: bool, rng: &mut impl Rng) -> Vec<f32> {
        let input_layer = self.layers[0].clone();
        for (i, &nid) in input_layer.iter().enumerate() {
            let val = *input.get(i).unwrap_or(&0.0);
            if let Some(n) = self.neurons.get_mut(&nid) {
                n.activation = val;
                n.last_fired = self.time;
            }
        }

        for layer_idx in 1..self.layers.len() {
            let prev_layer = self.layers[layer_idx - 1].clone();
            let layer_ids  = self.layers[layer_idx].clone();
            let is_output  = layer_idx == self.layers.len() - 1;

            // Read-only snapshots for parallel sum (prev activations and outgoing synapses to this layer)
            let prev_act: HashMap<NeuronId, f32> = prev_layer
                .iter()
                .filter(|id| self.dropout_mask.get(id) != Some(&true))
                .filter_map(|id| self.neurons.get(id).map(|n| (*id, n.activation)))
                .collect();
            let n_branches = self.config.dendritic_branches;
            let prev_synapses: HashMap<NeuronId, Vec<(NeuronId, f32, u8)>> = prev_layer
                .iter()
                .filter_map(|src_id| {
                    let n = self.neurons.get(src_id)?;
                    let targets: Vec<_> = n
                        .synapses
                        .iter()
                        .filter(|s| layer_ids.contains(&s.target))
                        .map(|s| (s.target, s.effective_strength(), s.branch_id))
                        .collect();
                    if targets.is_empty() {
                        None
                    } else {
                        Some((*src_id, targets))
                    }
                })
                .collect();

            // Per-neuron summation: if dendritic_branches > 1, sum per-branch and pick winner
            let sums: Vec<(NeuronId, f32, u8)> = crate::maybe_par_iter!(layer_ids)
                .map(|&nid| {
                    if n_branches <= 1 || is_output {
                        let sum: f32 = prev_layer
                            .iter()
                            .filter_map(|&src_id| {
                                let act = *prev_act.get(&src_id)?;
                                let targets = prev_synapses.get(&src_id)?;
                                targets
                                    .iter()
                                    .find(|(t, _, _)| *t == nid)
                                    .map(|(_, eff_str, _)| act * eff_str)
                            })
                            .sum();
                        (nid, sum, 0u8)
                    } else {
                        let mut branch_sums = vec![0.0f32; n_branches];
                        for &src_id in &prev_layer {
                            if let Some(act) = prev_act.get(&src_id) {
                                if let Some(targets) = prev_synapses.get(&src_id) {
                                    for &(t, eff_str, br) in targets {
                                        if t == nid {
                                            let bi = (br as usize) % n_branches;
                                            branch_sums[bi] += act * eff_str;
                                        }
                                    }
                                }
                            }
                        }
                        let (best_branch, &best_sum) = branch_sums
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap_or((0, &0.0));
                        (nid, best_sum, best_branch as u8)
                    }
                })
                .collect();

            // Sequential: dropout (needs rng) and activate (mutates neurons)
            let dropout_rate = if layer_idx == 1 {
                self.config.dropout_rate
            } else {
                self.config.dropout_rate * 0.5
            };
            for (nid, sum, winning_br) in sums {
                let would_drop = training
                    && !is_output
                    && dropout_rate > 0.0
                    && rng.gen::<f32>() < dropout_rate;
                let frozen = self.neurons.get(&nid).map_or(false, |n| n.frozen);
                let dropped = would_drop && !frozen;

                self.dropout_mask.insert(nid, dropped);

                if dropped {
                    if let Some(n) = self.neurons.get_mut(&nid) {
                        n.activation = 0.0;
                    }
                } else if let Some(n) = self.neurons.get_mut(&nid) {
                    let field_bias = if !is_output && self.config.ephaptic_field_strength > 0.0 {
                        self.ephaptic_fields.get(layer_idx)
                            .and_then(|fv| {
                                let idx = layer_ids.iter().position(|&id| id == nid)?;
                                fv.get(idx).copied()
                            })
                            .unwrap_or(0.0) * self.config.ephaptic_field_strength
                    } else { 0.0 };
                    n.activate(sum + field_bias);
                    n.winning_branch = winning_br;
                    if n.activation > 0.6 {
                        n.last_fired = self.time;
                    }
                }
            }

            // --- Reaction-Diffusion Lateral Inhibition ---
            // Replaces uniform layer-mean inhibition with spatially-local inhibition.
            // Each neuron receives inhibition weighted by exp(-dist/sigma_inhib):
            // near neighbors inhibit strongly, distant neurons weakly or not at all.
            //
            // This creates Turing instability: the two-scale structure (local activation,
            // wider inhibition) drives spontaneous receptive field formation without
            // explicit engineering. Pattern scale set by sigma_inhib / geometric_spread.
            //
            // Applied only to layer 1 (primary feature extraction).
            // Deeper layers need unattenuated signal for integration.
            if self.config.lateral_inhibition > 0.0 && layer_idx == 1 {
                reaction_diffusion_inhibition(
                    &layer_ids,
                    &mut self.neurons,
                    &self.dropout_mask,
                    self.config.lateral_inhibition,
                    self.config.sigma_inhib,
                );
            }

            // --- Mass–Competition Coupling + Homeostatic Plasticity ---
            // Two biological mechanisms that together prevent runaway bias drift:
            //
            // MASS-COMPETITION: winners gain mass (become gravitational hubs),
            // losers lose mass (become exploratory drifters). Geometry becomes
            // a live map of learned feature specialisation.
            //
            // HOMEOSTASIS: each neuron nudges its own bias toward a target activation.
            // Biological analog: synaptic scaling — neurons regulate excitability
            // to maintain stable firing rates regardless of input statistics.
            // This is the missing mechanism that prevented bias collapse in previous runs.
            // Gradient descent pushes bias toward task solution; homeostasis provides
            // a gentle restoring force that prevents runaway negative drift.
            if training && !is_output {
                for &nid in &layer_ids {
                    if self.dropout_mask.get(&nid) == Some(&true) { continue; }
                    if let Some(n) = self.neurons.get_mut(&nid) {
                        if n.frozen { continue; }
                        // --- Mass update ---
                        if self.config.mass_growth > 0.0 {
                            n.mass *= 1.0 - self.config.mass_decay;
                            if n.activation > self.config.mass_win_threshold {
                                let win_strength = (n.activation - self.config.mass_win_threshold)
                                    / (1.0 - self.config.mass_win_threshold);
                                n.mass += self.config.mass_growth * win_strength;
                            }
                            n.mass = n.mass.clamp(self.config.mass_min, self.config.mass_max);
                        }

                        // --- Homeostatic bias adjustment (running average) ---
                        // Updates running_activation EMA first, then nudges bias
                        // toward target based on slow average — not instantaneous value.
                        // This prevents all neurons converging to same bias:
                        // fast gradient changes are invisible to homeostasis (smoothed out),
                        // only slow multi-epoch drift gets corrected.
                        if self.config.homeostasis_lr > 0.0 {
                            // Update slow EMA
                            n.running_activation = (1.0 - self.config.homeostasis_tau)
                                * n.running_activation
                                + self.config.homeostasis_tau * n.activation;
                            // Correct based on slow average vs target
                            let drift = n.running_activation - self.config.homeostasis_target;
                            n.weight -= self.config.homeostasis_lr * drift;
                        }
                    }
                }
            }

            // --- Local KWTA — Neighborhood Competition ---
            // Each neuron competes only with geometric neighbors within kwta_radius.
            // A neuron survives iff it is the local maximum — no neighbor within
            // kwta_radius has higher activation.
            //
            // This produces non-overlapping territorial winners:
            // - neurons in different spatial regions don't suppress each other
            // - each region of input space gets its own champion
            // - sharp tilings emerge instead of soft global clusters
            //
            // Gradient is preserved for all losers (dropout_mask unchanged).
            // Losers can still win on different samples → specialisation via competition.
            //
            // Falls back to global competitive_k when kwta_radius == 0.0.
            let kwta_r = self.config.kwta_radius;
            let k = self.config.competitive_k;

            if kwta_r > 0.0 && !is_output {
                // Soft local KWTA — neighborhood-based suppression with partial signal preserved.
                //
                // Hard zero (previous): losers contribute nothing, gradient starved.
                // Soft suppression (×0.2): losers still carry 20% signal and gradient,
                // allowing them to learn toward winning on different input regions.
                //
                // For each neuron: count how many neighbors within kwta_radius beat it.
                // Winners (beaten_count == 0): activation unchanged — full signal.
                // Losers (beaten_count > 0): activation *= kwta_suppression — partial signal.
                //
                // This creates graded territorial competition:
                // - neurons far from any competitor fire freely
                // - neurons with one stronger neighbor are attenuated but not silenced
                // - only neurons surrounded by stronger neighbors are heavily suppressed
                let suppression = self.config.kwta_suppression;

                let snap: Vec<(NeuronId, f32, [f32; 3])> = layer_ids.iter()
                    .filter(|id| self.dropout_mask.get(id) != Some(&true))
                    .filter_map(|id| self.neurons.get(id).map(|n| (
                        *id,
                        n.activation,
                        [n.geometry.x, n.geometry.y, n.geometry.z],
                    )))
                    .collect();

                for &(nid, act, pos) in &snap {
                    let beaten = snap.iter().any(|&(other_id, other_act, other_pos)| {
                        if other_id == nid { return false; }
                        let dx = pos[0] - other_pos[0];
                        let dy = pos[1] - other_pos[1];
                        let dz = pos[2] - other_pos[2];
                        let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                        dist < kwta_r && other_act > act
                    });
                    if beaten {
                        if let Some(n) = self.neurons.get_mut(&nid) {
                            n.activation *= suppression;  // soft: partial signal preserved
                        }
                    }
                }
            } else if k > 0 && !is_output {
                // Global KWTA fallback
                let mut acts: Vec<(NeuronId, f32)> = layer_ids.iter()
                    .filter(|id| self.dropout_mask.get(id) != Some(&true))
                    .filter_map(|id| self.neurons.get(id).map(|n| (*id, n.activation)))
                    .collect();
                acts.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (i, (nid, _)) in acts.iter().enumerate() {
                    if i >= k {
                        if let Some(n) = self.neurons.get_mut(nid) {
                            n.activation *= 0.02; // losers keep 2% — enough for gradient, not enough to win
                        }
                    }
                }
            }
        }

        // --- Ephaptic field update ---
        // After the full forward pass, update each hidden layer's field from activations.
        // The field is an EMA that captures "what this layer just saw" and persists
        // across inputs, providing immediate pattern availability on subsequent passes.
        if self.config.ephaptic_field_alpha > 0.0 && self.layers.len() > 2 {
            let alpha = self.config.ephaptic_field_alpha;
            if self.ephaptic_fields.len() < self.layers.len() {
                self.ephaptic_fields.resize(self.layers.len(), Vec::new());
            }
            for layer_idx in 1..self.layers.len() - 1 {
                let layer_ids = &self.layers[layer_idx];
                let field = &mut self.ephaptic_fields[layer_idx];
                if field.len() != layer_ids.len() {
                    field.resize(layer_ids.len(), 0.0);
                }
                for (pos, &nid) in layer_ids.iter().enumerate() {
                    if let Some(n) = self.neurons.get(&nid) {
                        field[pos] = alpha * field[pos] + (1.0 - alpha) * n.activation;
                    }
                }
            }
        }

        let output_layer = self.layers.last().unwrap().clone();
        output_layer.iter().map(|id| self.neurons[id].activation).collect()
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut dummy_rng = rand::thread_rng();
        self.forward_pass(input, false, &mut dummy_rng)
    }

    // -------------------------------------------------------------------------
    // Backprop
    // -------------------------------------------------------------------------

    pub fn backprop(&mut self, output: &[f32], target: &[f32]) -> f32 {
        let loss = if self.config.output_bce {
            bce_loss(output, target)
        } else {
            mse_loss(output, target)
        };
        let n_layers = self.layers.len();
        let mut deltas: HashMap<NeuronId, f32> = HashMap::new();
        let use_bce = self.config.output_bce;

        // Output layer — update bias immediately
        for (i, &nid) in self.layers[n_layers - 1].iter().enumerate() {
            let o = output.get(i).copied().unwrap_or(0.0);
            let t = target.get(i).copied().unwrap_or(0.0);
            let delta = if use_bce {
                o - t
            } else {
                (o - t) * o * (1.0 - o)
            };
            deltas.insert(nid, delta);
            if let Some(n) = self.neurons.get_mut(&nid) {
                if n.frozen { continue; }
                let is_consolidated = n.group_id.map_or(false, |gid| self.consolidated_group_ids.contains(&gid));
                let mut eff_lr = if is_consolidated { 0.0 } else { self.current_lr };
                if eff_lr > 0.0 && self.config.mass_consolidation_k > 0.0 {
                    eff_lr /= 1.0 + self.config.mass_consolidation_k * n.mass;
                }
                let upd = (eff_lr * delta).clamp(-1.0, 1.0);
                n.weight = (n.weight * (1.0 - self.config.bias_decay) - upd).clamp(-self.config.weight_clamp, self.config.weight_clamp);
            }
        }

        // Hidden layers backwards: compute gradient updates in parallel, then apply
        let tick_count = self.tick_count;
        let lr = self.current_lr;
        let weight_decay = self.config.weight_decay;
        let bias_decay = self.config.bias_decay;
        let weight_clamp = self.config.weight_clamp;
        let mass_k = self.config.mass_consolidation_k;
        let consolidated_ids = self.consolidated_group_ids.clone();
        let layer_of = self.layer_of.clone();
        let dropout_mask = self.dropout_mask.clone();
        let engram_enabled = self.config.engram_enabled;
        let engram_lr_scale = self.config.engram_lr_scale;
        let n_branches_bp = self.config.dendritic_branches;

        // Snapshot of winning branch per target neuron (for dendritic backprop gating)
        let winning_branches: HashMap<NeuronId, u8> = if n_branches_bp > 1 {
            self.neurons.iter().map(|(&id, n)| (id, n.winning_branch)).collect()
        } else {
            HashMap::new()
        };

        for layer_idx in (1..n_layers).rev() {
            let curr_layer = self.layers[layer_idx].clone();
            let prev_layer = self.layers[layer_idx - 1].clone();
            let mut prev_deltas: HashMap<NeuronId, f32> = HashMap::new();

            // Snapshot: (src_id, src_act, mass, frozen, is_consolidated, weight, syn_snap)
            // syn_snap: (target, strength, consolidation, branch_id)
            let snapshot: Vec<(NeuronId, f32, f32, bool, bool, f32, Vec<(NeuronId, f32, f32, u8)>)> = prev_layer
                .iter()
                .filter(|id| dropout_mask.get(id) != Some(&true))
                .filter_map(|&src_id| {
                    let n = self.neurons.get(&src_id)?;
                    let syn_snap: Vec<_> = n
                        .synapses
                        .iter()
                        .filter(|s| curr_layer.contains(&s.target))
                        .map(|s| (s.target, s.strength, s.consolidation, s.branch_id))
                        .collect();
                    let is_cons = n
                        .group_id
                        .map_or(false, |gid| consolidated_ids.contains(&gid));
                    Some((
                        src_id,
                        n.activation,
                        n.mass,
                        n.frozen,
                        is_cons,
                        n.weight,
                        syn_snap,
                    ))
                })
                .collect();

            type SynUpdate = (NeuronId, f32, bool);
            let updates: Vec<(NeuronId, f32, Option<f32>, Vec<SynUpdate>)> = crate::maybe_par_iter!(snapshot)
                .map(|(src_id, src_act, mass, frozen, is_cons, weight, syn_snap)| {
                    let is_input = layer_of.get(src_id) == Some(&0);
                    let eff_lr = if *frozen || *is_cons {
                        0.0
                    } else if mass_k > 0.0 {
                        lr / (1.0 + mass_k * mass)
                    } else {
                        lr
                    };
                    // Gradient sum: only through winning branch synapses when dendritic
                    let grad_sum: f32 = syn_snap
                        .iter()
                        .filter_map(|(tgt_id, str, _, branch_id)| {
                            if dropout_mask.get(tgt_id) == Some(&true) {
                                return None;
                            }
                            if n_branches_bp > 1 {
                                let wb = winning_branches.get(tgt_id).copied().unwrap_or(0);
                                if *branch_id != wb { return None; }
                            }
                            let d = deltas.get(tgt_id).copied().unwrap_or(0.0);
                            Some(d * str)
                        })
                        .sum();
                    let delta = if is_input {
                        None
                    } else {
                        Some(
                            (grad_sum * src_act * (1.0 - src_act)).clamp(-1.0, 1.0),
                        )
                    };
                    let prev_delta_val =
                        delta.unwrap_or_else(|| grad_sum.clamp(-1.0, 1.0));
                    let new_bias = if let Some(d) = delta {
                        if *frozen {
                            *weight
                        } else {
                            (weight * (1.0 - bias_decay) - eff_lr * d)
                                .clamp(-weight_clamp, weight_clamp)
                        }
                    } else {
                        *weight
                    };
                    // Synapse weight updates: only update winning branch synapses
                    let syn_updates: Vec<SynUpdate> = syn_snap
                        .iter()
                        .filter_map(|(tgt_id, str, consolidation, branch_id)| {
                            if dropout_mask.get(tgt_id) == Some(&true) {
                                return None;
                            }
                            if n_branches_bp > 1 {
                                let wb = winning_branches.get(tgt_id).copied().unwrap_or(0);
                                if *branch_id != wb {
                                    return Some((*tgt_id, *str, false));
                                }
                            }
                            let d = deltas.get(tgt_id).copied().unwrap_or(0.0);
                            let clipped = (d * src_act).clamp(-1.0, 1.0);
                            if *frozen {
                                return Some((*tgt_id, *str, false));
                            }
                            let syn_eff_lr = if engram_enabled {
                                eff_lr * (1.0 - engram_lr_scale * consolidation).max(0.2)
                            } else {
                                eff_lr
                            };
                            let new_str = (str * (1.0 - weight_decay)
                                - syn_eff_lr * clipped)
                                .clamp(-weight_clamp, weight_clamp);
                            Some((*tgt_id, new_str, clipped.abs() > 0.001))
                        })
                        .collect();
                    (
                        *src_id,
                        prev_delta_val,
                        if is_input { None } else { Some(new_bias) },
                        syn_updates,
                    )
                })
                .collect();

            for (src_id, prev_delta_val, new_bias_opt, syn_updates) in updates {
                prev_deltas.insert(src_id, prev_delta_val);
                if let Some(n) = self.neurons.get_mut(&src_id) {
                    if let Some(b) = new_bias_opt {
                        n.weight = b;
                    }
                    for (tgt_id, new_str, set_active) in syn_updates {
                        for syn in n.synapses.iter_mut() {
                            if syn.frozen {
                                continue;
                            }
                            if syn.target == tgt_id {
                                syn.strength = new_str;
                                if set_active {
                                    syn.last_active = tick_count as u32;
                                }
                                break;
                            }
                        }
                    }
                }
            }

            deltas = prev_deltas;
        }

        loss
    }

    // -------------------------------------------------------------------------
    // Engram consolidation — protect memory traces
    // -------------------------------------------------------------------------

    /// Update consolidation for synapses where pre and post both fired this tick.
    /// Engram-engram synapses accumulate consolidation; high consolidation
    /// reduces effective LR and grants pruning immunity.
    fn update_engram_consolidation(&mut self) {
        if !self.config.engram_enabled {
            return;
        }
        let thresh = self.config.engram_activation_threshold.max(0.01);
        let inc = self.config.engram_increment;
        let cap = self.config.engram_cap;

        let fired: std::collections::HashSet<NeuronId> = self.neurons
            .iter()
            .filter(|(_, n)| n.activation >= thresh)
            .map(|(&id, _)| id)
            .collect();

        for neuron in self.neurons.values_mut() {
            if neuron.frozen {
                continue;
            }
            let pre_fired = fired.contains(&neuron.id);
            for syn in neuron.synapses.iter_mut() {
                if syn.frozen {
                    continue;
                }
                if pre_fired && fired.contains(&syn.target) {
                    syn.consolidation = (syn.consolidation + inc).min(cap);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Training tick (full: forward + backprop + plasticity)
    // -------------------------------------------------------------------------

    /// Forward + backprop only; no STDP, prune, grow, or geometry.
    /// Used by minibatch SGD: run on B clones in parallel, then average params.
    pub fn train_tick_gradient_only(
        &mut self,
        input: &[f32],
        target: &[f32],
        rng: &mut impl Rng,
    ) -> f32 {
        let output = self.forward_pass(input, true, rng);
        self.backprop(&output, target)
    }

    pub fn train_tick(
        &mut self, input: &[f32], target: &[f32], rng: &mut impl Rng,
    ) -> TickResult {
        // Forward with dropout enabled
        let output = self.forward_pass(input, true, rng);

        // Engram consolidation: tag synapses where pre and post both fired this tick.
        // Must run before backprop so we use current activations.
        self.update_engram_consolidation();

        let loss   = self.backprop(&output, target);

        let fired = get_fired_neurons(&self.neurons, 0.6);
        record_firing(&mut self.neurons, &fired, self.time);

        if self.config.stdp_enabled {
            let pairs: Vec<(NeuronId, NeuronId)> = self.layers.windows(2)
                .flat_map(|w| w[0].iter().flat_map(|&pre| w[1].iter().map(move |&post| (pre, post))))
                .collect();
            update_stdp_layer(&mut self.neurons, &pairs, self.time, &self.config);
        }

        let pruned = apply_metabolic_pressure(&mut self.neurons, &self.config, &self.output_ids, &self.input_ids);

        let prune_interval = self.config.prune_interval.max(1) as u64;
        let prune_stop = self.config.prune_stop_tick;
        let pruning_active = prune_stop == 0 || self.tick_count < prune_stop;
        if pruning_active && self.tick_count % prune_interval == 0 {
            let (p1, p2, p3) = prune_three_phase(
                &mut self.neurons, &self.config, self.tick_count, &self.output_ids, &self.input_ids
            );
            let _ = (p1, p2, p3);
            // Positive half of activity-dependent plasticity: strengthen well-used synapses.
            // Runs on same cadence as pruning — removes weak, strengthens strong.
            potentiate_active_synapses(&mut self.neurons, &self.config, &self.output_ids);
        }

        let interval = self.config.geometry_interval.max(1) as u64;
        if self.tick_count % interval == 0 {
            update_geometry(&mut self.neurons, &self.config, &self.layer_of, rng);
        }

        let layer_of = self.layer_of.clone();
        let grown = grow_synapses(&mut self.neurons, &self.config, rng, &layer_of);

        update_group_centroids(&mut self.groups, &self.neurons);
        apply_ifs_mirror_coupling(&mut self.neurons, &self.groups, &self.config);

        self.tick_count += 1;
        self.time += 1.0;

        TickResult { loss, output, synapses_pruned: pruned, synapses_grown: grown, neurons_fired: fired.len() }
    }

    // -------------------------------------------------------------------------
    // Inference — no dropout, no weight updates
    // -------------------------------------------------------------------------

    pub fn predict(&mut self, input: &[f32]) -> Vec<f32> {
        self.forward(input)
    }

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    pub fn total_synapses(&self) -> usize {
        self.neurons.values().map(|n| n.synapses.len()).sum()
    }

    pub fn detect_whorls(&self) -> Vec<WhorlReport> {
        detect_whorls(&self.neurons, 1.5)
    }

    pub fn snapshot_all(&self) -> Vec<NeuronSnapshot> {
        self.neurons.values().map(|n| n.snapshot()).collect()
    }

    pub fn prune_dormant(&mut self) {
        prune_dormant_synapses(&mut self.neurons, 2000, 0.001);
    }

    pub fn run_three_phase_pruning(&mut self) -> (usize, usize, usize) {
        let output_ids = self.output_ids.clone();
        prune_three_phase(&mut self.neurons, &self.config, self.tick_count, &output_ids, &self.input_ids)
    }

    /// Call at the start of each epoch to anneal the learning rate.
    /// effective_lr = base_lr * exp(-lr_decay * epoch)
    /// With lr_decay=0.0005 and 8000 epochs: lr drops to ~2% of initial.
    /// Enables aggressive early exploration and precise late convergence.
    pub fn set_epoch(&mut self, epoch: usize) {
        if self.config.lr_decay > 0.0 {
            self.current_lr = self.config.learning_rate
                * (-self.config.lr_decay * epoch as f32).exp();
        }
    }

    /// Fraction of hidden neurons that actually fired on the last forward pass.
    /// Healthy range: 0.2–0.6. Near 1.0 = no sparsity, network is a homogeneous soup.
    pub fn firing_sparsity(&self) -> f32 {
        let hidden: Vec<&Neuron> = self.layers[1..self.layers.len()-1]
            .iter().flat_map(|l| l.iter().filter_map(|id| self.neurons.get(id)))
            .collect();
        if hidden.is_empty() { return 0.0; }
        let fired = hidden.iter().filter(|n| n.activation > 0.6).count();
        fired as f32 / hidden.len() as f32
    }

    /// Mean activation across all hidden neurons — tracks whether inhibition
    /// is suppressing too much (→ 0.0) or too little (→ 0.5+).
    /// Healthy range: 0.2–0.45 with reaction-diffusion inhibition active.
    pub fn mean_hidden_activation(&self) -> f32 {
        let hidden: Vec<&Neuron> = self.layers[1..self.layers.len()-1]
            .iter().flat_map(|l| l.iter().filter_map(|id| self.neurons.get(id)))
            .collect();
        if hidden.is_empty() { return 0.0; }
        hidden.iter().map(|n| n.activation).sum::<f32>() / hidden.len() as f32
    }

    /// Current geometric spread of all neurons.
    /// Should stay 1.5–3.0 with physics active.
    /// Collapse below 0.5 → repulsion too weak; explosion above 5.0 → repulsion too strong.
    pub fn geometric_spread(&self) -> f32 {
        compute_geometric_spread(&self.neurons)
    }

    /// Mean mass of hidden neurons — tracks mass-competition coupling.
    /// Winners gain mass, losers lose mass.
    /// Healthy: spread from ~0.5 (losers) to ~2.0 (hubs) after convergence.
    pub fn mean_hidden_mass(&self) -> f32 {
        let hidden: Vec<&Neuron> = self.layers[1..self.layers.len()-1]
            .iter().flat_map(|l| l.iter().filter_map(|id| self.neurons.get(id)))
            .collect();
        if hidden.is_empty() { return 1.0; }
        hidden.iter().map(|n| n.mass).sum::<f32>() / hidden.len() as f32
    }

    /// Average trainable parameters (neuron weights, synapse strengths) across envs.
    /// Structure is taken from the first env. Used by minibatch SGD after parallel gradient steps.
    pub fn average_params_from(envs: &[Self]) -> Self {
        if envs.is_empty() {
            panic!("average_params_from requires at least one env");
        }
        let mut out = envs[0].clone();
        let n = envs.len() as f32;
        for (nid, neuron) in out.neurons.iter_mut() {
            neuron.weight = envs.iter().map(|e| e.neurons[nid].weight).sum::<f32>() / n;
            for (idx, syn) in neuron.synapses.iter_mut().enumerate() {
                syn.strength = envs
                    .iter()
                    .map(|e| e.neurons[nid].synapses[idx].strength)
                    .sum::<f32>()
                    / n;
            }
        }
        out
    }
}

pub struct TickResult {
    pub loss: f32,
    pub output: Vec<f32>,
    pub synapses_pruned: usize,
    pub synapses_grown: usize,
    pub neurons_fired: usize,
}

pub fn mse_loss(output: &[f32], target: &[f32]) -> f32 {
    output.iter().zip(target.iter())
        .map(|(o, t)| (o - t).powi(2))
        .sum::<f32>() / output.len() as f32
}

pub fn bce_loss(output: &[f32], target: &[f32]) -> f32 {
    output.iter().zip(target.iter())
        .map(|(o, t)| {
            let p = o.clamp(1e-7, 1.0 - 1e-7);
            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
        })
        .sum::<f32>() / output.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_insert_neuron_at_layer_adds_one_neuron_and_synapses() {
        let mut rng = StdRng::seed_from_u64(123);
        let mut config = EnvironmentConfig::default();
        config.max_synapses_per_neuron = 64;
        let mut env = NeuralEnvironment::new(config);
        env.build_layers(&[2, 4, 1], &mut rng);
        let layer_1_len_before = env.layers[1].len();
        assert_eq!(layer_1_len_before, 4);

        let new_id = env.insert_neuron_at_layer(1, &mut rng, None);
        assert!(new_id.is_some());
        let new_id = new_id.unwrap();

        assert_eq!(env.layers[1].len(), 5);
        assert!(env.neurons.contains_key(&new_id));
        assert_eq!(env.layer_of.get(&new_id), Some(&1));

        let n = &env.neurons[&new_id];
        let incoming = env.layers[0].iter().filter(|&&src| {
            env.neurons.get(&src).map_or(false, |s| s.synapses.iter().any(|syn| syn.target == new_id))
        }).count();
        let outgoing = n.synapses.len();
        assert_eq!(incoming, 2, "new neuron should have 2 incoming synapses from input layer");
        assert_eq!(outgoing, 1, "new neuron should have 1 outgoing synapse to output layer");
    }

    #[test]
    fn test_insert_neuron_at_layer_rejects_input_and_output_layers() {
        let mut rng = StdRng::seed_from_u64(123);
        let mut env = NeuralEnvironment::new(EnvironmentConfig::default());
        env.build_layers(&[2, 4, 1], &mut rng);
        assert!(env.insert_neuron_at_layer(0, &mut rng, None).is_none());
        assert!(env.insert_neuron_at_layer(2, &mut rng, None).is_none());
    }
}