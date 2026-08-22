# IoT - Internet of Things

**Current memory footprint:**

The Phase 3 system holds two fully trained specialists simultaneously. What's actually in memory at inference time is tiny — Group 0 has roughly 120-160 frozen synapses across 16 neurons, Group 1 similar. The entire dual-task system is kilobytes, not megabytes. The checkpoint serializes to ~70KB including all metadata. That is a remarkable number for a system that holds two learned tasks with zero forgetting.

**Why the bio-design produces this:**

Pruning is not cosmetic. Dead synapses are gone — they consume no memory, no compute, no representation. A standard MLP trained on spiral keeps all 272 synapses whether they're useful or not. The Growformer prunes to 139-163 active synapses and stops there. The network is exactly as large as the task requires, not as large as the architecture permits.

KWTA compounds this at inference. Only 4 of 16 neurons activate per forward pass. You don't compute the other 12. On constrained hardware this matters enormously — inference cost is proportional to active neurons, not total neurons.

As tasks accumulate, each new group adds only its own pruned footprint. Not a growing dense matrix. A small frozen sparse graph.

**Browser viability:**

The core architecture is pure arithmetic — no special ops, no BLAS, no GPU dependency. The forward pass is dot products over sparse synapse lists. This compiles to WebAssembly trivially. The Rust codebase already has the right shape for wasm-pack — no OS dependencies, no filesystem requirements at inference time, deterministic computation.

A browser deployment would load the serialized DimensionManager checkpoint, run inference through the frozen groups, and route via the cosine similarity router. All of that is feasible in under 1MB of WASM binary plus the checkpoint. Running in a web worker keeps the main thread free.

The Three.js visualizer you already have is evidence the browser environment works for this project — the heavier rendering runs fine, and the inference is far lighter than the rendering.

**IoT viability:**

This is where the architecture has genuine advantages over anything transformer-based. A Raspberry Pi Zero has 512MB RAM and a single 1GHz ARM core. The Growformer's inference path — sparse forward pass over frozen neurons, cosine similarity routing, output head activation — is achievable on hardware that cannot run any meaningful transformer inference at all.

Microcontrollers are the more interesting question. An ESP32 has 520KB of SRAM. The current two-task system at ~70KB serialized is within range after accounting for runtime overhead. You would need to strip the training infrastructure entirely — inference-only build — but the frozen group representation is compact enough that embedded deployment is plausible.

The pruning threshold becomes a deployment parameter in this context. Tighter pruning before promotion reduces the frozen group size further. A group trained to 90% accuracy with aggressive pruning might consolidate to 80-100 synapses — potentially under 20KB per task.

**What needs to happen to make this real:**

The training and inference paths need to be separated. Right now they're coupled in NeuralEnvironment. An inference-only build strips out backprop, geometry simulation, pruning, mirror coupling, STDP — everything except the forward pass and KWTA. That build would be dramatically smaller and fast enough for real-time inference on constrained hardware.

The DimensionManager checkpoint loader is already most of what's needed — it deserializes frozen groups and runs predict(). The gap is building a no-std compatible inference crate that can be compiled for embedded targets.

**The honest constraint:**

Training cannot happen on IoT hardware at current scale. 4000 epochs over 400 samples is a desktop workload. The deployment model is: train on desktop, promote groups, serialize the frozen DimensionManager, deploy the inference-only checkpoint to the device. The device never trains — it only routes and infers. That constraint is actually a feature for IoT — the device behavior is fixed and auditable, not drifting from on-device learning.

Phase 3c compositional generalization strengthens this further. If a novel task can be solved by blending existing frozen groups with a handful of examples, that blending computation is cheap enough to run on a Raspberry Pi. You get adaptation on-device without full retraining.