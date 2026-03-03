use rand::Rng;
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

            for &src in &sources {
                for &tgt in &targets {
                    let w: f32 = rng.gen_range(-scale..=scale);
                    let neuron = self.neurons.get_mut(&src).unwrap();
                    neuron.add_synapse(tgt, w, self.config.max_synapses_per_neuron);
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

            for &nid in &layer_ids {
                // Dropout: randomly zero activations in hidden layers during training
                // This is the primary symmetry-breaking mechanism — different neurons
                // are knocked out on different samples, forcing specialisation
                // Dropout: higher rate on first hidden layer (feature extraction),
                // lower rate on deeper layers (integration — needs more stable signal)
                let dropout_rate = if layer_idx == 1 {
                    self.config.dropout_rate
                } else {
                    self.config.dropout_rate * 0.5  // half rate on deeper layers
                };
                let would_drop = training
                    && !is_output
                    && dropout_rate > 0.0
                    && rng.gen::<f32>() < dropout_rate;
                let frozen = self.neurons.get(&nid).map_or(false, |n| n.frozen);
                let dropped = would_drop && !frozen;  // never drop frozen — Task A signal must flow

                self.dropout_mask.insert(nid, dropped);

                if dropped {
                    if let Some(n) = self.neurons.get_mut(&nid) {
                        n.activation = 0.0;
                    }
                    continue;
                }

                let mut sum = 0.0f32;
                for &src_id in &prev_layer {
                    if let Some(src) = self.neurons.get(&src_id) {
                        if self.dropout_mask.get(&src_id) == Some(&true) { continue; }
                        for syn in &src.synapses {
                            if syn.target == nid {
                                sum += src.activation * syn.effective_strength();
                            }
                        }
                    }
                }

                if let Some(n) = self.neurons.get_mut(&nid) {
                    n.activate(sum);
                    if n.activation > 0.6 { n.last_fired = self.time; }
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
        let loss = mse_loss(output, target);
        let n_layers = self.layers.len();
        let mut deltas: HashMap<NeuronId, f32> = HashMap::new();

        // Output layer — update bias immediately
        for (i, &nid) in self.layers[n_layers - 1].iter().enumerate() {
            let o = output.get(i).copied().unwrap_or(0.0);
            let t = target.get(i).copied().unwrap_or(0.0);
            let delta = (o - t) * o * (1.0 - o);
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

        // Hidden layers backwards
        for layer_idx in (1..n_layers).rev() {
            let curr_layer = self.layers[layer_idx].clone();
            let prev_layer = self.layers[layer_idx - 1].clone();
            let mut prev_deltas: HashMap<NeuronId, f32> = HashMap::new();

            for &src_id in &prev_layer {
                // Skip neurons that were dropped in the forward pass
                if self.dropout_mask.get(&src_id) == Some(&true) { continue; }

                let src_act = self.neurons[&src_id].activation;
                let is_input = self.layer_of.get(&src_id) == Some(&0usize);

                let syn_snap: Vec<(NeuronId, f32)> = self.neurons[&src_id]
                    .synapses.iter()
                    .filter(|s| curr_layer.contains(&s.target))
                    .map(|s| (s.target, s.strength))
                    .collect();

                let mut grad_sum = 0.0f32;

                for (tgt_id, pre_str) in &syn_snap {
                    if self.dropout_mask.get(tgt_id) == Some(&true) { continue; }
                    let tgt_delta = *deltas.get(tgt_id).unwrap_or(&0.0);
                    grad_sum += tgt_delta * pre_str;

                    let clipped = (tgt_delta * src_act).clamp(-1.0, 1.0);
                    if let Some(n) = self.neurons.get_mut(&src_id) {
                        if n.frozen { continue; }
                        let is_consolidated = n.group_id.map_or(false, |gid| self.consolidated_group_ids.contains(&gid));
                        let mut eff_lr = if is_consolidated { 0.0 } else { self.current_lr };
                        if eff_lr > 0.0 && self.config.mass_consolidation_k > 0.0 {
                            eff_lr /= 1.0 + self.config.mass_consolidation_k * n.mass;
                        }
                        for syn in n.synapses.iter_mut() {
                            if syn.frozen { continue; }
                            if syn.target == *tgt_id {
                                // Uniform weight clamp for all synapses.
                                // Input cap (0.5) was a failed experiment — starved gradient,
                                // caused neuron death at 54.4% accuracy. Removed.
                                let strength_cap = self.config.weight_clamp;
                                syn.strength = (syn.strength * (1.0 - self.config.weight_decay)
                                    - eff_lr * clipped)
                                    .clamp(-strength_cap, strength_cap);
                                // Mark as active if it carried a non-trivial gradient.
                                // Used by phase-3 pruning to distinguish dormant from active.
                                if clipped.abs() > 0.001 {
                                    syn.last_active = self.tick_count as u32;
                                }
                            }
                        }
                    }
                }

                if !is_input {
                    let delta = (grad_sum * src_act * (1.0 - src_act)).clamp(-1.0, 1.0);
                    prev_deltas.insert(src_id, delta);
                    if let Some(n) = self.neurons.get_mut(&src_id) {
                        if n.frozen { continue; }
                        let is_consolidated = n.group_id.map_or(false, |gid| self.consolidated_group_ids.contains(&gid));
                        let mut eff_lr = if is_consolidated { 0.0 } else { self.current_lr };
                        if eff_lr > 0.0 && self.config.mass_consolidation_k > 0.0 {
                            eff_lr /= 1.0 + self.config.mass_consolidation_k * n.mass;
                        }
                        n.weight = (n.weight * (1.0 - self.config.bias_decay)
                            - eff_lr * delta)
                            .clamp(-self.config.weight_clamp, self.config.weight_clamp);
                    }
                } else {
                    prev_deltas.insert(src_id, grad_sum.clamp(-1.0, 1.0));
                }
            }

            deltas = prev_deltas;
        }

        loss
    }

    // -------------------------------------------------------------------------
    // Training tick
    // -------------------------------------------------------------------------

    pub fn train_tick(
        &mut self, input: &[f32], target: &[f32], rng: &mut impl Rng,
    ) -> TickResult {
        // Forward with dropout enabled
        let output = self.forward_pass(input, true, rng);
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