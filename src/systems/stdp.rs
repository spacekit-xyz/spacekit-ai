use crate::types::*;
use crate::neuron::Neuron;
use std::collections::HashMap;

/// System 3: Spike-Timing-Dependent Plasticity (STDP)
///
/// The *when* of a signal matters as much as the *strength*.
/// Pre fires before post → strengthen (causal relationship)
/// Post fires before pre → weaken (coincidence, not cause)
///
/// This implements Hebbian learning with a temporal window,
/// replacing the purely gradient-based weight update.
pub fn update_stdp(
    neurons: &mut HashMap<NeuronId, Neuron>,
    pre_id: NeuronId,
    post_id: NeuronId,
    current_time: f64,
    config: &EnvironmentConfig,
) {
    let pre_fired = neurons.get(&pre_id).map(|n| n.last_fired).unwrap_or(0.0);
    let post_fired = neurons.get(&post_id).map(|n| n.last_fired).unwrap_or(0.0);

    let delta_t = (post_fired - pre_fired) as f32;

    // Outside the STDP window entirely — no effect
    if delta_t.abs() > config.stdp_window {
        return;
    }

    let delta_w = if delta_t > 0.0 {
        // Pre fired before post: causal → potentiate
        config.a_plus * (-delta_t / config.tau_plus).exp()
    } else if delta_t < 0.0 {
        // Post fired before pre: acausal → depress
        -config.a_minus * (delta_t / config.tau_minus).exp()
    } else {
        0.0 // simultaneous — no STDP effect
    };

    if let Some(pre_neuron) = neurons.get_mut(&pre_id) {
        for synapse in pre_neuron.synapses.iter_mut() {
            if synapse.target == post_id {
                // Clamp preserves sign: inhibitory synapses stay inhibitory
                synapse.strength = (synapse.strength + delta_w).clamp(-1.5, 1.5);

                if delta_w > 0.0 {
                    synapse.facilitation = (synapse.facilitation + 0.001).min(2.0);
                } else {
                    synapse.depression = (synapse.depression - 0.0005).max(0.1);
                }

                synapse.timing_offset = delta_t;
                break;
            }
        }
    }
}

/// Run STDP across all recently active neuron pairs in a layer
pub fn update_stdp_layer(
    neurons: &mut HashMap<NeuronId, Neuron>,
    active_pairs: &[(NeuronId, NeuronId)],
    current_time: f64,
    config: &EnvironmentConfig,
) {
    for &(pre, post) in active_pairs {
        update_stdp(neurons, pre, post, current_time, config);
    }
}

/// Update last_fired timestamp for neurons that crossed activation threshold
pub fn record_firing(
    neurons: &mut HashMap<NeuronId, Neuron>,
    fired_ids: &[NeuronId],
    current_time: f64,
) {
    for &id in fired_ids {
        if let Some(n) = neurons.get_mut(&id) {
            n.last_fired = current_time;
        }
    }
}

/// Returns which neurons fired (crossed threshold) in this tick
pub fn get_fired_neurons(
    neurons: &HashMap<NeuronId, Neuron>,
    threshold: f32,
) -> Vec<NeuronId> {
    neurons
        .values()
        .filter(|n| n.activation >= threshold)
        .map(|n| n.id)
        .collect()
}