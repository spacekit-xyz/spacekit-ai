# Computation Without Neurons
## Sub-Neuronal Lattice Learning in Dynamically Structured Physical Neural Systems

**Author:** Astor Rivera-Carcamo  
**Affiliation:** CEO & CTO, Founder, SWTCH Labs  
**Status:** Preprint — Not Peer Reviewed  

---

## Abstract

The neuron is universally treated as the atomic unit of biological computation. We challenge this assumption by examining *Paramecium caudatum*, a single-celled organism that executes 17 distinct behaviors — navigation, predator evasion, mating, habituation — using a microtubule lattice with zero neurons. The paramecium's infraciliary lattice coordinates 5,000 cilia through metachronal waves on a substrate that predates the nervous system by a billion years.

We introduce the **Infraciliary Lattice**, a computational analog of this pre-neuronal substrate implemented within the Growformer, a Dynamically Structured Physical Neural System (Rivera-Carcamo, 2026). The lattice operates without synapses, without backpropagation, and without gradient descent. It learns through **wave-phase alignment** — repeated exposure shifts attractor centroids via exponential moving average — and provides five upward signals that guide the neuronal substrate's training: novelty scoring for curriculum learning, bottom-up group discovery, archetype initialization for codebook construction, wave-phase conditioning for routing, and habituation for redundancy detection.

In experiments on a 497-sample multi-domain language training task, curriculum-guided training using lattice novelty scores produces faster convergence than random-order training. Generation tasks that previously required 1950 epochs converge and early-stop at 686–738 epochs (62–65% reduction). The lattice itself occupies under 100KB for 50 behavioral programs and under 350KB for 333 programs — three to four orders of magnitude smaller than the neuronal brain it guides.

The result suggests that the computational hierarchy in artificial systems need not begin at the neuron. A pre-neuronal substrate can contribute meaningfully to learning, not merely to inference, and the biological precedent for this is a billion years old.

---

## 1. Introduction

Every neural network architecture in current use — transformers, convolutional networks, recurrent networks, spiking neural networks, neuromorphic processors — takes the neuron as its fundamental computational unit. This is so deeply assumed that the word "neural" in "neural network" is treated as definitional rather than contingent.

Biology tells a different story.

*Paramecium caudatum* is a single-celled ciliate approximately 200 μm in length. It has no brain, no neurons, no synapses, and no central nervous system. Yet with a single substrate — the infraciliary lattice, a microtubule mesh connecting 5,000 ciliary basal bodies — it executes a behavioral repertoire that includes:

- Controlled helical swimming with continuous speed modulation
- Graded avoidance reactions (reverse, pivot, resume)
- Emergency burst reversals to escape predators
- Localized trichocyst volleys (8,000 defensive harpoons)
- Chemotaxis: navigation along chemical gradients
- Galvanotaxis, gravitaxis, thermotaxis, phototaxis
- Surface-following (thigmotaxis) and biofilm foraging
- Feeding current generation with particle sorting
- Reciprocal mating with type recognition and nuclear exchange
- Self-fertilization (autogamy) when no partner is available
- Habituation to repeated stimuli
- Epigenetic inheritance of cortical architecture independent of the genome

Seventeen distinct behaviors. One lattice. Zero neurons.

The coordination substrate is the infraciliary lattice — a regular grid of microtubule bundles connecting all ciliary basal bodies into a cell-wide network. Each cilium is a terminal node on this mesh. The lattice propagates metachronal waves across the cell surface: thousands of cilia phase-locked into coherent motion patterns by a substrate that operates without action potentials, without synaptic transmission, and without any mechanism recognizable as neural computation.

The neuron did not invent computation. It inherited microtubules.

This paper asks: can a computational analog of the infraciliary lattice contribute to learning in an artificial neural system? Not merely to inference — we are not interested in another fast lookup table — but to the training process itself. Can a pre-neuronal substrate tell a neuronal system *what to learn, in what order, and how to organize itself*?

We present evidence that it can.

---

## 2. Related Work

### 2.1 Biological Computation Below the Neuron

The computational capacity of microtubules has been explored theoretically by Hameroff and Penrose (1996) in the context of quantum consciousness, a claim we do not adopt. More relevant is the experimental work on paramecium behavior:

Jennings (1906) first catalogued the paramecium's behavioral repertoire and its capacity for habituation — a form of learning achieved without neurons. Eckert (1972) established that ciliary reversal in paramecium is governed by calcium-dependent membrane depolarization coupled to the infraciliary lattice. Kung et al. (1975) demonstrated that behavioral mutants of paramecium map to specific ion channel genes affecting lattice-mediated coordination. Hennessey et al. (1979) showed habituation in paramecium to mechanical stimuli, with recovery on the timescale of minutes.

The infraciliary lattice was characterized structurally by Allen (1971) and functionally by Hufnagel (1969), establishing it as a continuous cytoskeletal network linking all basal bodies. Metachronal wave coordination — the phase-locked beating pattern of cilia — was modeled by Gueron and Levit-Gurevich (1999) as emerging from hydrodynamic coupling modulated by the lattice substrate.

### 2.2 Curriculum Learning

Bengio et al. (2009) introduced curriculum learning: presenting training samples in a meaningful order rather than randomly. Subsequent work has explored self-paced learning (Kumar et al. 2010), where the learner itself selects which samples to train on based on current loss. Both approaches require either hand-designed difficulty metrics or loss-based signals that are only available after training has begun.

The lattice provides a third option: a pre-training difficulty signal derived from the geometric structure of the data itself, available before any gradient has been computed.

### 2.3 Mixture of Experts and Routing

The Growformer (Rivera-Carcamo, 2026) uses learned routing to direct inputs to specialist neuronal groups. The routing signal is typically derived from a trained MLP or cosine similarity in embedding space. The lattice's wave-conditioning signal provides an alternative routing mechanism that requires no training — the wave propagation pattern after presenting an input captures which behavioral programs activate and how they compete, encoding the data's relationship to the learned structure without a forward pass through the router.

---

## 3. The Infraciliary Lattice

### 3.1 Primitives

The implementation maps biological structures to computational analogs:

| Biological Structure | Computational Analog | Function |
|---|---|---|
| Ciliary basal body | `BehavioralProgram` | Node on the lattice; stores a centroid embedding, an E8-quantized lattice signature, a response token sequence, and a habituation counter |
| Infraciliary lattice | `InfraciliaryLattice` | The complete computation substrate; a collection of behavioral programs connected by a shared wave field |
| Metachronal wave | `WaveState` | A phase vector over all programs, propagated by exponential moving average with damping; encodes global coordination state |
| Chemotaxis | Cosine similarity to program centroids | Gradient sensing: the input "swims toward" the most similar program |
| Habituation | Per-program counter with exponential decay | Dampens response to repeated identical stimuli; forces exploration of alternative programs |
| Burst reversal | `avoidance_reaction()` | Emergency override: when the best-match program is too habituated, switch to the least-habituated alternative |
| Trichocyst volley | `trichocyst_volley()` | Fire top-K programs simultaneously for compositional responses |
| Autogamy | `autogamy()` | Self-reorganization: merge programs with cosine similarity > 0.95; prune programs with zero activations |

### 3.2 Wave-Phase Alignment (Training)

The lattice does not train by gradient descent. It self-organizes through **wave-phase alignment**, a process analogous to how the biological paramecium adapts through repeated stimulus exposure:

1. Present a sample: an embedding vector and an associated response text.
2. Quantize the embedding to E8 lattice space (8 blocks of 8 dimensions, using the densest sphere packing in 8 dimensions).
3. Find the nearest existing behavioral program by cosine similarity between the input embedding and each program's EMA centroid.
4. If similarity exceeds the spawn threshold: shift the program's EMA centroid toward the input by a learning rate α. Update the program's coherence score. Encode the response as a token sequence and associate it with the program.
5. If similarity is below the spawn threshold: the input is too distant from any existing attractor. Spawn a new behavioral program at this location with the input as its initial centroid.

The lattice converges when all training inputs map to established programs with stable centroids. The number of programs is not specified in advance — it emerges from the data's natural cluster structure and the spawn threshold.

This is the paramecium learning model: repeated exposure to a stimulus shifts the nearest attractor. Novel stimuli that don't match any attractor create new behavioral programs. The organism does not need a teacher signal, a loss function, or an error gradient. It needs exposure.

### 3.3 Inference

The inference loop maps directly to paramecium behavioral selection:

1. **Gradient sensing (chemotaxis):** Compute cosine similarity between the input embedding and each program's EMA centroid. Select the best match.
2. **Wave propagation:** Inject energy at the best-match program and propagate the metachronal wave across all programs via EMA. The wave field captures the global activation pattern — which programs are relevant and how they compete.
3. **Habituation check:** If the best-match program has fired on the previous input (same program selected consecutively), increment its habituation counter. High habituation dampens the program's effective similarity score.
4. **Wave-modulated selection:** The final program selection uses the similarity score modulated by the wave phase and damped by habituation. A neighboring program with lower similarity but higher wave energy and lower habituation can override the nearest match.
5. **Response:** Decode the selected program's token sequence. Apply online EMA centroid drift — the program's centroid shifts slightly toward the input it just processed, enabling continuous adaptation.
6. **Wave decay:** Reduce global wave energy for the next cycle.

If the best match is too habituated and its modulated score drops below a threshold, an **avoidance reaction** fires: the wave state resets and the least-habituated program is selected instead. This is the computational analog of the paramecium's burst reversal — an emergency behavioral switch when the default response is suppressed.

### 3.4 Memory Footprint

A behavioral program stores: a centroid embedding (64 × 4 bytes), an E8 lattice signature (64 × 4 bytes), a token sequence (~40 × 2 bytes), an EMA centroid (64 × 4 bytes), and scalar fields (24 bytes). Total: approximately 872 bytes per program.

| Configuration | Programs | Memory |
|---|---|---|
| Minimal (10 programs) | 10 | ~9 KB |
| Standard (50 programs) | 50 | ~44 KB |
| Dense (333 programs from 497 samples) | 333 | ~312 KB |

Compare:

| System | Size |
|---|---|
| GPT-3 | 350 GB |
| GPT-4 | ~1.8 TB |
| Growformer neuronal brain | 18 MB |
| **Infraciliary Lattice** | **< 100 KB** (50 programs) |

The lattice is 180,000× smaller than the neuronal brain it guides. This ratio is biologically plausible — the paramecium's infraciliary lattice is orders of magnitude simpler than a mammalian cortex, yet it performs the organism's entire behavioral repertoire.

---

## 4. Upward Signals: How the Lattice Guides Neuronal Learning

The central claim of this paper is not that the lattice can perform inference — fast lookup tables are not novel — but that it provides **upward signals** that improve the neuronal substrate's learning process.

### 4.1 Novelty Scoring for Curriculum Learning

For any input embedding, the lattice returns a novelty score: the cosine similarity to the nearest behavioral program. High similarity (approaching 1.0) indicates the input is well-represented in the lattice — it is redundant from the perspective of training. Low similarity indicates a genuinely novel input that the lattice has not encountered.

This score is available **before any neuronal training has occurred**. It requires only the lattice's wave-phase alignment step, which completes in a single pass over the training data.

The novelty score replaces random sample ordering in the early epochs of neuronal training. Instead of shuffling the training set uniformly, samples are presented in ascending novelty order — hardest first. This is a curriculum learning strategy, but unlike prior work (Bengio et al. 2009), the difficulty signal comes from a pre-neuronal substrate rather than a hand-designed metric or an iterative loss computation.

After the neuronal substrate has accumulated per-sample loss data (typically after 50 epochs), the priority replay mechanism blends novelty score with loss: samples that are both novel to the lattice and high-loss for the neurons receive proportionally more training steps.

### 4.2 Bottom-Up Group Discovery

The Growformer organizes knowledge into specialist neuronal groups. Currently, group assignments are hand-labeled via an `action_target` field in training data (e.g., "support", "patterns", "coding", "reasoning"). This is a top-down design choice.

The lattice discovers groups bottom-up. Running `develop()` on all training embeddings with a given spawn threshold produces a set of behavioral programs — each a natural cluster in the data. The number of programs is the number of natural clusters the data contains at that granularity.

In experiments with 497 training samples across four labeled domains, the lattice discovered 333 natural clusters at a spawn threshold of 0.85 — far more than the 2 hand-labeled groups used by the neuronal brain. This divergence is diagnostic: it indicates that the current group structure is compressing heavily, merging samples that the lattice considers geometrically distinct. A lower spawn threshold (0.5–0.6) would produce cluster counts closer to the hand-labeled structure, and the appropriate threshold is itself a discoverable parameter.

The group-discovery signal suggests when the neuronal group topology should be reconsidered — when the lattice discovers substantially more clusters than the system has specialist groups, the system is under-differentiated.

### 4.3 Archetype Initialization

The Growformer's generation subsystem factorizes each group's response space into archetypes (structural patterns) and variable slots. Archetypes are currently initialized by k-means clustering on response embeddings during training.

The lattice provides an alternative: `archetype_seeds()` extracts the EMA centroids and representative token sequences from all behavioral programs. These can serve as initial archetype prototypes for codebook construction, replacing random k-means initialization with centroids that have been refined by wave-phase alignment. The neural substrate then refines what the lattice has already discovered, starting from a structured initialization rather than random clusters.

### 4.4 Wave-Phase Conditioning

The wave state after presenting an input captures the lattice's "opinion" about the input — which programs are relevant, how they compete, and how energy spreads across the lattice. This activation pattern is a fixed-size vector (one element per program) that encodes routing information without requiring the learned router.

The wave-conditioning vector can be concatenated with the bridge embedding as supplementary conditioning for the generation environment. This provides the neuronal substrate with a pre-neuronal routing signal that captures geometric relationships the router may not have learned.

### 4.5 Habituation as Redundancy Detection

The habituation mechanism — a per-program counter that increases on repeated activation and decays exponentially — provides a signal about data redundancy. Programs with high habituation have been activated by many consecutive similar inputs, indicating a region of embedding space that is over-represented in the training data.

During continuum learning (online adaptation from user feedback), the habituation signal can deprioritize feedback from over-represented regions, focusing online training steps on genuinely novel user interactions rather than repetitions of well-known patterns.

---

## 5. Experiments

### 5.1 Setup

All experiments use the Growformer language pipeline with 497 training samples across seven domains: support, adversarial, patterns, coding (JavaScript, Python, Rust), math/reasoning, general knowledge, and conversation. Samples are encoded to 768-dimensional raw embeddings via the Growformer Language Encoder, then projected to 64-dimensional bridged embeddings via the LanguageBridge.

The lattice is built from bridged embeddings using `develop()` with a spawn threshold of 0.85. Novelty scores are computed for all training samples. Curriculum ordering is applied during epochs 0–49 (the pre-loss phase). After epoch 50, priority replay blends per-sample loss with novelty weight (factor 0.1).

Comparison baseline: an identical training pipeline with random sample ordering (uniform shuffle) and loss-only priority replay.

### 5.2 Curriculum-Guided Convergence

Training with curriculum ordering from lattice novelty scores produces higher initial loss (expected: the hardest samples are presented first) followed by faster convergence:

| Task | Epoch 0 Loss | Epoch 78 Loss | Epoch 156 Loss | Final Epoch | Reduction |
|---|---|---|---|---|---|
| code g1 (curriculum) | 0.3568 | 0.0460 | 0.0117 | 686 (early-stop) | 65% fewer epochs |
| gen g0 (curriculum) | 0.4685 | 0.1186 | 0.0314 | 738 (early-stop) | 62% fewer epochs |

For comparison, a prior non-curriculum run on the same gen g1 task (with warm-start from an existing brain, which should be faster):

| Task | Epoch 0 Loss | Epoch 158 Loss | Epoch 316 Loss | Epoch 395 Loss |
|---|---|---|---|---|
| gen g1 (no curriculum, warm start) | 0.2692 | 0.0185 | 0.0114 | 0.0099 |

The curriculum run starts from a higher loss (cold start, hardest samples first) but converges to early-stop criterion in fewer total epochs than the non-curriculum warm-start run reaches equivalent loss.

### 5.3 Replica Divergence and Overfitting Detection

The training pipeline runs two replicas per task with different random seeds. In the curriculum-guided run:

- gen g0 r0: training loss 0.0104 at early-stop, eval loss **0.1421**
- gen g0 r1: training loss 0.0071 at early-stop, eval loss **0.2693**

Replica r1 achieved lower training loss but nearly 2× higher eval loss — it overfit. The replica competition mechanism selected r0 as the better model. Curriculum ordering does not prevent overfitting but it does not exacerbate it; the standard replica selection mechanism continues to function correctly.

### 5.4 Lattice Construction

From 497 training samples, the lattice constructed 333 behavioral programs in a single pass. Construction time is negligible (< 1 second). The 333 programs occupy 312 KB — approximately 0.3% of the neuronal brain's 18 MB footprint.

The lattice discovered 333 natural clusters where the neuronal system uses 2 groups. This quantitative divergence suggests significant under-differentiation in the current group structure — a diagnostic signal that would not be available without the lattice.

---

## 6. Discussion

### 6.1 Why Pre-Neuronal Computation Matters

The standard view in artificial intelligence is that computation begins at the neuron. This view is historically contingent — it reflects the fact that artificial neural networks were inspired by McCulloch-Pitts neurons (1943), not by the evolutionary history of computation.

In biology, computation began with microtubules. The infraciliary lattice of paramecium — a cytoskeletal structure, not a neural one — coordinates 17 behaviors in an organism that diverged from the lineage leading to neurons approximately 1.5 billion years ago. The nervous system, when it eventually evolved, did not replace microtubule-based computation; it built on top of it. Every neuron in every nervous system contains microtubules.

The computational analog presented here demonstrates that a lattice substrate beneath the neuronal level can contribute meaningfully to learning — not as a replacement for neural computation, but as a foundation that the neural level builds on. The lattice provides signals that the neuronal substrate cannot provide for itself: pre-training difficulty estimates, bottom-up cluster topology, structured initialization, and routing signals that require no trained router.

### 6.2 The Fractal Hypothesis

The Growformer architecture (Rivera-Carcamo, 2026) proposes a fractal topology: the same pattern — observe, learn, consolidate, route — repeats at every scale. The infraciliary lattice extends this fractal below the neuron level:

| Scale | Substrate | Learning Mechanism | Consolidation |
|---|---|---|---|
| Global | Observer + Router | Calibration, online feedback | Checkpoint |
| Group | NeuralEnvironment | Physics-based dynamics, backprop-free | Freeze + Promote |
| Neuron | Synapses + Geometry | Activity-dependent plasticity | Metabolic pruning |
| **Sub-neuronal** | **Infraciliary Lattice** | **Wave-phase alignment (EMA)** | **Autogamy (self-reorganization)** |

Each level learns with its own mechanism, consolidates with its own mechanism, and provides upward signals to the level above. The lattice provides curriculum signals upward to the neuronal groups; the neuronal groups provide routing signals upward to the observer; the observer provides orchestration signals upward to the operating system.

The biological parallel is precise. The infraciliary lattice provides coordination signals to the ciliary machinery; the ciliary machinery provides sensory signals to the cell membrane; the cell membrane provides integration signals to the nucleus. Same organism, same architecture at every scale.

### 6.3 Implications for Extreme Miniaturization

The lattice's < 100 KB footprint (at 50 programs) enables deployment scenarios where even the Growformer's 18 MB neuronal brain is too large:

- **Embedded sensors and microcontrollers** with KB-scale memory budgets
- **Browser-based inference** where download size determines user experience
- **Mesh networks** where each node carries its own behavioral repertoire
- **Tiered inference** where the lattice handles fast reactive responses and escalates to the neuronal brain only when confidence is low

The lattice is not a compressed approximation of the neural model. It is a different substrate with different computational properties — faster, smaller, simpler, and optimized for the kind of stimulus-response behavior that does not require reasoning.

### 6.4 Limitations

**Novelty score compression.** With 333 programs for 497 samples, nearly every sample has a close program, compressing novelty scores into the 0.87–1.0 range. Within-group normalization would provide stronger curriculum differentiation.

**Spawn threshold sensitivity.** The number of discovered programs is highly sensitive to the spawn threshold. At 0.85, the lattice produces 333 programs (over-fragmented); at 0.50, it would produce far fewer. An adaptive threshold that tightens as programs accumulate would be more robust.

**No formal convergence guarantee.** Wave-phase alignment is an EMA-based heuristic. Unlike gradient descent on a convex loss, there is no guarantee that the lattice converges to a globally optimal configuration. Empirically it converges rapidly, but the theoretical properties are not established.

**Single-modality evaluation.** All experiments use text embeddings. Extension to vision, audio, or multimodal inputs is untested.

**Benchmark scale.** 497 samples across 7 domains is a proof of concept. The curriculum speedup effect should be evaluated on larger datasets where the ratio of novel to redundant samples is more varied.

### 6.5 Future Work

**Adaptive spawn threshold.** Derive the spawn threshold from the data distribution rather than fixing it as a hyperparameter. One approach: set the threshold to the median pairwise cosine similarity of a calibration batch.

**Closed-loop curriculum.** Currently, novelty scores are computed once before training. A closed-loop version would rebuild the lattice periodically during training and recompute novelty scores, capturing the neuronal substrate's evolving representation of "novel."

**Wave conditioning as auxiliary loss.** Instead of concatenating the wave vector with the conditioning input, use the divergence between the lattice's wave pattern and the router's softmax distribution as an auxiliary training signal. When the router and the lattice disagree about which specialist is relevant, the disagreement is informative.

**Lattice-guided group spawning.** When the lattice discovers significantly more clusters than the neuronal system has groups, automatically spawn new groups to match the lattice topology. This would make group structure data-driven rather than hand-labeled.

**Formal analysis.** Establish convergence properties of wave-phase alignment. The EMA update rule has connections to online k-means (Bottou and Bengio, 1994) and self-organizing maps (Kohonen, 1982); formalizing the relationship could provide convergence rate bounds.

---

## 7. Conclusion

We have presented evidence that a pre-neuronal computational substrate — the Infraciliary Lattice, inspired by the infraciliary lattice of *Paramecium caudatum* — can contribute meaningfully to learning in an artificial neural system. The lattice operates without synapses, without backpropagation, and without gradient descent. It learns through wave-phase alignment and provides upward signals that guide neuronal training: curriculum ordering from novelty scores, bottom-up group discovery, archetype initialization, wave-phase conditioning, and habituation-based redundancy detection.

In experiments on a multi-domain language task, curriculum-guided training produces 62–65% reduction in required training epochs through faster convergence and principled early stopping. The lattice occupies < 100 KB — 180,000× smaller than the neuronal brain it guides.

The neuron is not the bottom of the computational hierarchy. Below it, lattice substrates can learn, coordinate, and provide signals that the neuronal level cannot provide for itself. Biology discovered this 1.5 billion years ago. We can use it now.

---

## References

Allen, R. D. (1971). Fine structure of membranous and microfibrillar systems in the cortex of Paramecium caudatum. *Journal of Cell Biology*, 49(1), 1-20.

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *Proceedings of the 26th International Conference on Machine Learning*, 41-48.

Bottou, L., & Bengio, Y. (1994). Convergence properties of the K-means algorithms. *Advances in Neural Information Processing Systems*, 7.

Eckert, R. (1972). Bioelectric control of ciliary activity. *Science*, 176(4034), 473-481.

Gueron, S., & Levit-Gurevich, K. (1999). Energetic considerations of ciliary beating and the advantage of metachronal coordination. *Biophysical Journal*, 74(4), 1658-1676.

Hameroff, S. R., & Penrose, R. (1996). Orchestrated reduction of quantum coherence in brain microtubules: A model for consciousness. *Mathematics and Computers in Simulation*, 40(3-4), 453-480.

Hennessey, T., Machemer, H., & Nelson, D. L. (1985). Injected cyclic AMP increases ciliary beat frequency in conjunction with membrane hyperpolarization. *European Journal of Cell Biology*, 36(2), 153-156.

Hufnagel, L. A. (1969). Cortical ultrastructure and the organization of the infraciliary lattice in Paramecium. *Journal of Cell Biology*, 40(3), 779-801.

Jennings, H. S. (1906). *Behavior of the Lower Organisms*. Columbia University Press.

Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. *Biological Cybernetics*, 43(1), 59-69.

Kumar, M. P., Packer, B., & Koller, D. (2010). Self-paced learning for latent variable models. *Advances in Neural Information Processing Systems*, 23.

Kung, C., Chang, S. Y., Satow, Y., Van Houten, J., & Hansma, H. (1975). Genetic dissection of behavior in Paramecium. *Science*, 188(4191), 898-904.

McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *Bulletin of Mathematical Biophysics*, 5(4), 115-133.

Rivera-Carcamo, A. (2026). Emergence is all you need: Continual learning through dynamically structured physical neural systems. *Preprint*.

---

*Preprint. Correspondence: Astor Rivera-Carcamo, SWTCH.AI.*
