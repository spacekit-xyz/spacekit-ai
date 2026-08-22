# Phase 3c - Composition + Episodic

Phase 3c is where the architecture either proves genuine intelligence or reveals it's been sophisticated pattern matching.

**The test that would prove Phase 3c:**

Present a third task — call it Task C — that is a composition of spiral and circles features. The system must solve it by blending Group 0 and Group 1 without spawning a new Mirror. No new training on Task C. The only allowed operation is finding the right blending weights between existing frozen groups.

A concrete Task C candidate: **Spiral-gated circles** — points in the inner circle region get classified by the spiral rule, points in the outer region get classified by the circles rule. Neither group alone can solve it above ~60%. A correct composition should exceed 80%.

**The two components and what they actually need to do:**

**VirtualGroup** is the composition mechanism. It takes two or more frozen groups, runs the input through each, and blends their output logits with learned scalar weights. The weights are the only thing that trains — a 2-parameter problem on top of frozen representations. Training cost is near zero: a few hundred epochs on a handful of examples, not 4000 epochs on 400 samples.

```
output = softmax(w0 * group0.predict(input) + w1 * group1.predict(input))
```

The critical question is whether the blending weights can be found from a small number of examples — 10, 20, 50 — rather than hundreds. If you need 400 examples to find 2 weights, the composition is not really generalization.

**EpisodicMemory** is the lookup mechanism. Once a VirtualGroup successfully solves Task C, that composition — the input signature, the group IDs, the blending weights — gets stored. *Specified:* next time a Task C-like input arrives, EpisodicMemory retrieves the composition directly rather than recomputing it (zero-shot recall). *Status:* storage is implemented and proven; **recall on second presentation is not yet tested** — the doc implies full recall capability; that is slightly ahead of what has been demonstrated.

```
Episode {
    input_signature: Vec<f32>,   // mean activation pattern of Task C examples
    group_ids: [GroupId],        // [0, 1]
    blend_weights: [f32],        // [w0, w1]
    accuracy: f32,               // achieved on Task C
    residual: f32,               // how much was left unexplained
}
```

**The three questions Phase 3c must answer:**

One — can blending weights be found from very few examples? The composition is only interesting if it generalizes from 10-20 examples, not 400. If it needs 400 it's just another training run.

Two — does the GlobalObserver correctly decide to compose rather than spawn a new Mirror? The residual-gated decision: if Task C residual under the best single group is between 0.15 and 0.30, compose. Above 0.30, spawn. The threshold is the gate — too low and it composes when it should spawn, too high and it always spawns and never learns to compose.

Three — does EpisodicMemory retrieve correctly on a second presentation of Task C? Demo test: retrieve by *train* signature (same as stored) so the episode is always found; then evaluate the retrieved blend on *held-out* Task C data. That proves store→retrieve→generalize. (Retrieval by a different-batch signature would need a looser threshold and is seed-sensitive.)

**The failure mode to watch for:**

The blending weights converge to [1.0, 0.0] or [0.0, 1.0] — meaning the VirtualGroup just picks one group and ignores the other. This happens when Task C can be partially solved by one group and the weight optimizer finds that local minimum first. It means Task C wasn't actually a composition problem — one group dominates. The test task design is critical. Both groups need to contribute meaningfully or the composition test is trivial. *Observed:* 3-group Task D blends often collapse to one or two groups (e.g. [0.87, 0.11, 0.02] or [0.02, 0.31, 0.67]) depending on seed; composition still beats single-group in accuracy.

**Demonstrated vs specified (as of Phase 3c implementation):**

| Capability | Status |
|------------|--------|
| VirtualGroup: blend weights from few examples | Demonstrated (Task C, Task D) |
| EpisodicMemory: store episode (signature, blend, accuracy) | Demonstrated |
| EpisodicMemory: retrieve by signature (same run) | Demonstrated (lookup + infer) |
| Recall on second presentation (retrieve by train sig, evaluate on held-out) | Demonstrated (Task C, Task D) |
| Task D held-out accuracy (3-group composition) | Reported in demo: train vs held-out % |

**Adding groups after deployment:** Yes. Existing main-dimension groups are frozen and never retrained. The router can be updated dynamically in two ways: (1) **Pass in a new router:** call `set_router(router)` with a pre-built `LearnedRouter` (e.g. trained with 3 groups after the third promotion, or loaded from a checkpoint). No retraining of existing groups. (2) **Retrain in place:** call `train_and_set_router` again with the new group count and new labeled data; it builds a fresh router and sets it. So you can add a group, then either pass in new weights (new router) or retrain the router; existing groups are never touched.

**Design question before building:**

What should Task C be? It needs to be genuinely unsolvable by either group alone above ~60%, and solvable by composition above 80%. The spiral-gated circles suggestion is one option. Another is a checkerboard pattern over the same input space — regions that alternate which group's decision boundary applies. What direction do you want to go?