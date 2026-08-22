# When Composite Accuracy Lies: Authenticating Routing Over Frozen Specialists

### A public whitepaper on promote–freeze learning and honest routing evaluation

**Author:** Astor Rivera-Carcamo  
**Affiliation:** SWTCH Labs, SWTCH.AI, SpaceKit.xyz  
**Status:** Public preprint; not peer reviewed  
**Evidence current through:** July 2026

---

## Abstract

Growformer is a neural learning substrate designed to add new specialists without rewriting specialists that have already been consolidated. A new capability learns in an isolated environment called a **Mirror**. Once it meets a promotion criterion, it moves into **Main**, the permanent store, where it is frozen. Later learning occurs elsewhere and has no gradient path back into the promoted specialist.

This design makes retention straightforward: frozen specialists do not forget because their parameters no longer change. That is useful, but it is not a solution to forgetting inside one shared neural network. Growformer belongs to the parameter-isolation family, alongside systems such as Progressive Neural Networks. Its harder scientific problem is what happens next: given many frozen specialists, how does the system select the right one for each input, and when can their knowledge be combined?

The main findings are:

1. **Retention works as designed.** Across five sequential Split-MNIST tasks, average accuracy was 97.7% both at promotion and after all later tasks, for measured forgetting of 0.0%.
2. **A fixed global blend is not a general composition method.** On a balanced task that requires one specialist in an inner region and another in an outer region, a fixed blend scored 69.9%, below either specialist alone.
3. **Accuracy can make a failed router look successful.** A learned router scored 81.3%, but selected the specialist associated with the true region only 55.4% of the time. Fourteen of twenty runs collapsed to choosing one specialist everywhere.
4. **Routing needs independent authentication.** Region agreement, boundary-localized errors, routing diversity, and confidence behavior reveal failure modes that composite accuracy hides.
5. **Boundary-aware routing produced a qualified positive.** With region supervision during training, a boundary-aware router reached 92.5% accuracy, 85.3% region agreement, and no collapsed runs. A later method removed region labels from the loss and reached 93.8% accuracy, 85.6% region agreement, and no collapsed runs, but pseudo-labeled from an r-proxy feature using an orientation prior derived from true-radius analysis.
6. **The transferable result is a certification protocol.** Gates are declared before evaluation; independent certifiers authenticate the mechanism; negative, invalid, and under-resolved outcomes remain distinct; and additional evidence increases resolution rather than relaxing a threshold.

The reusable contribution is not a claim that routing is solved in general. It is a disciplined way to distinguish genuine specialist selection from a deceptively high score produced by collapse.

---



## 1. The Problem

Continual-learning systems must acquire new capabilities without damaging old ones. There are two fundamentally different versions of that problem.


| Setting               | What changes during later learning                          | Main difficulty                               |
| --------------------- | ----------------------------------------------------------- | --------------------------------------------- |
| Shared representation | New and old tasks use some of the same trainable parameters | Preventing interference and forgetting        |
| Parameter isolation   | Each task has a separate specialist that becomes frozen     | Selecting and combining the right specialists |


Growformer studies the second setting. It does not claim to prevent forgetting while repeatedly modifying one shared representation. Instead, it prevents later tasks from modifying earlier specialists at all.

That changes the location of risk. Weight corruption becomes structurally unlikely after promotion, but a perfectly preserved specialist is still useless if the system routes an input to the wrong one. **Routing error, rather than weight corruption, is the central operational risk.**

This paper is scoped to the promote–freeze substrate and routing authentication. Language-grounding and encoder-certification results are reported separately in the project's grounding-loop specification, and frozen-encoder predictive and action-adapter results are reported in the world-model specification. They are not evidence cells in this paper.

### 1.1 What Growformer is

Growformer is a dynamic neural substrate in which connectivity can grow, strengthen, weaken, and be pruned during learning. Neurons have geometric and metabolic state in addition to ordinary activations and weights. The resulting specialist is a sparse learned subgraph rather than merely a dense layer of predetermined width.

These physical and biological analogies motivate the design, but they are not evidence by themselves. The claims in this paper rest on measured retention and routing behavior.

### 1.2 What Growformer is not

Growformer is not:

- a demonstrated solution to catastrophic forgetting in shared weights;
- a general-purpose routing solution for arbitrary domains;
- a claim that sparse growth is always optimal or follows a universal law;
- evidence that one global model should replace the library of frozen specialists.

---



## 2. The Promote–Freeze Lifecycle

Growformer separates plastic learning from stable storage.

1. **Create a Mirror.** A fresh, isolated environment is created for a new task or dynamics regime.
2. **Grow and train a specialist.** Its weights and structure can change while it learns.
3. **Evaluate promotion criteria.** The specialist must meet a task-specific performance threshold and any required safety or stability checks.
4. **Promote into Main.** The accepted specialist moves into the permanent store.
5. **Freeze it.** Later tasks receive no gradient path into that specialist.
6. **Route at inference.** A dispatcher selects a frozen specialist from the input. The substrate supports abstention, but abstention behavior is not evaluated in this paper.

The Mirror/Main separation resembles the fast-learning and stable-storage distinction in Complementary Learning Systems, but this is an architectural analogy, not a biological mechanism claim.

### 2.1 What zero forgetting means here

If a frozen specialist produces exactly the same output before and after another task is trained, its measured forgetting is zero. In Growformer this is an expected consequence of isolation. It should be audited, but it should not be presented as evidence that shared neural representations have become immune to interference.

### 2.2 The cost of isolation

Storage grows approximately linearly with the number of promoted specialists. Pruning can reduce the size of each specialist, but it does not make total memory growth sub-linear. Inference can remain roughly constant per input when only one routed specialist is activated, although the router itself must still search or index the available choices.

---



## 3. Why Routing Requires Its Own Evidence

Suppose two frozen specialists each solve a different part of a task. A combined system can appear accurate for at least three very different reasons:

- it learned the true condition that determines which specialist applies;
- it found a proxy that happens to correlate with that condition; or
- it simply chooses the globally stronger specialist almost all the time.

On an imbalanced test set, the third behavior can look excellent. Even on a balanced test set, always selecting one specialist can produce a respectable score if each specialist is partially correct outside its home region.

For that reason, Growformer uses a declared protocol and several independent measures.

### 3.1 Certification protocol

The protocol is intended to transfer beyond the particular radial task used here. Its rules, the failure each prevents, and their status in this paper are:


| Rule                                                                                                     | Failure prevented                                                                           | Status and instance                                                                                                                                                            |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Declare gates and the verdict table before evaluating the method.                                        | Moving a threshold after seeing the result.                                                 | Exercised in Sections 6.1 and 6.2.                                                                                                                                             |
| Keep `PASS`, `FAIL`, `BELOW_RESOLUTION`, and `INVALID` distinct.                                         | Reading an underpowered or contaminated evaluation as evidence for or against a method.     | Negative verdicts are exercised in Sections 5.4 and 5.7; under-resolution and invalid-evaluation verdicts are exercised in the companion encoder-certification work, not here. |
| Certify with held-out quantities that the training objective and method-selection process did not shape. | A model being rewarded for the same quantity later presented as independent authentication. | Exercised here when margin-to-latent correlation was removed from the verdict after an earlier loss directly shaped the margin.                                                |
| Check data and quantity provenance before interpreting a score.                                          | Training/evaluation leakage or a system certifying evidence it generated.                   | Exercised in companion encoder-certification work; stated but not otherwise exercised here.                                                                                    |
| Increase sample resolution rather than relaxing a declared threshold.                                    | Converting an unresolved result into a pass by weakening the gate.                          | Exercised directly in companion work. Section 5.7 provides a related in-paper warning: a favorable small-budget result deteriorated when the certification budget increased.   |
| Publish negative and indeterminate outcomes without converting them into positive narrative.             | Survivorship bias and hidden failure modes.                                                 | Exercised by the fixed-blend and competence-routing negatives in Sections 5.4 and 5.7.                                                                                         |


The status column is part of the claim boundary. A rule demonstrated only in companion work is not presented as an empirical result of this routing study.

### 3.2 Core routing certifiers


| Measure                      | Question it answers                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| Composite accuracy           | Does the final prediction match the task label?                                          |
| Region agreement             | Does the router choose the specialist associated with the true generating region?        |
| Routing diversity            | Does the router use both specialists, or collapse to one choice?                         |
| Boundary error profile       | Are mistakes concentrated near the genuine switch boundary, where ambiguity is expected? |
| Margin-to-latent correlation | Does routing confidence vary with distance from the true boundary?                       |
| Confident-wrong reliance     | When one specialist is confidently wrong, does the router down-weight it?                |


No single measure is sufficient. Accuracy evaluates the product outcome; the remaining measures authenticate the mechanism.

A certifier must be independent of the training objective and method-selection process. An earlier boundary-router loss directly regressed the routing margin toward distance from the true boundary. That made margin-to-latent correlation circular: it measured optimization of a training target rather than independently authenticating routing. The shaping term was removed, and the correlation was demoted to observational evidence and excluded from the verdict. More generally, if a loss shapes a candidate certification quantity, that quantity cannot certify the same run.

### 3.3 Why boundary localization matters

A real boundary learner should make more routing errors close to the boundary than far inside either region. A router that always selects the inner specialist has a different signature: almost no routing errors in the inner region and almost total failure in the outer region. Both patterns can produce similar average task accuracy, but only one represents learned routing.

---



## 4. Retention Audit: Five Sequential Digit Tasks

The retention audit used five binary digit tasks: 0 versus 1, 2 versus 3, 4 versus 5, 6 versus 7, and 8 versus 9. Images were mapped through one fixed 64-dimensional projection. Each task trained in its own Mirror, met its promotion criterion, moved into Main, and was frozen before the next task began.

Split MNIST is an easy benchmark and is used here only to verify the lifecycle.


| Task        | Accuracy at promotion | Accuracy after all five tasks | Forgetting |
| ----------- | --------------------- | ----------------------------- | ---------- |
| 0 versus 1  | 99.5%                 | 99.5%                         | 0.0%       |
| 2 versus 3  | 95.5%                 | 95.5%                         | 0.0%       |
| 4 versus 5  | 97.5%                 | 97.5%                         | 0.0%       |
| 6 versus 7  | 98.0%                 | 98.0%                         | 0.0%       |
| 8 versus 9  | 98.0%                 | 98.0%                         | 0.0%       |
| **Average** | **97.7%**             | **97.7%**                     | **0.0%**   |


The conclusion is narrow: later Mirror training did not alter promoted specialists. This does not establish superiority over shared-network continual-learning methods, nor does it make Split MNIST a hard benchmark.

---



## 5. The Decisive Routing Study



### 5.1 The balanced switched task

The central study used two frozen two-dimensional specialists:

- a **spiral specialist**, trained on a double-spiral classification rule; and
- a **circles specialist**, trained on a concentric-circles rule.

The composite task applied the spiral rule inside a circle of radius 0.4 and the circles rule outside it. Sampling was stratified so that half of the examples came from each region. Each run used 30 training examples and approximately 370 held-out examples. The study covered 20 fixed runs.

This design matters. Earlier pilot tasks placed most examples in one region, allowing one specialist to appear competitive without meaningful composition. The balanced task removes that shortcut.

### 5.2 Methods compared

The study compared:

- either frozen specialist used alone;
- an oracle that globally selects the better single specialist;
- a fixed weighted blend of both specialist outputs;
- selection by whichever specialist appears more confident;
- learned input-dependent routing based on the two specialist outputs;
- a boundary-aware router trained with region supervision;
- a boundary-aware router trained from specialist-output pseudo-labels without region labels in its loss;
- gates given the true radial variable, used only as diagnostic ceilings; and
- an oracle that always knows the true region.



### 5.3 Main results


| Method                                                                              | Held-out accuracy | Region agreement          | Collapsed runs            | Interpretation                                                          |
| ----------------------------------------------------------------------------------- | ----------------- | ------------------------- | ------------------------- | ----------------------------------------------------------------------- |
| Spiral specialist alone                                                             | 76.6% ± 4.1%      | Not applicable            | Not applicable            | Single-specialist reference                                             |
| Circles specialist alone                                                            | 73.5% ± 2.2%      | Not applicable            | Not applicable            | Single-specialist reference                                             |
| Oracle best single specialist                                                       | 77.1% ± 3.4%      | Not applicable            | Not applicable            | Best global choice                                                      |
| Fixed global blend                                                                  | 69.9% ± 5.7%      | Not applicable            | Not applicable            | Finished negative                                                       |
| Confidence-based selection                                                          | 69.5% ± 6.2%      | Not reported as certified | Not reported as certified | Input dependence alone is insufficient                                  |
| Learned router on specialist outputs                                                | 81.3% ± 7.8%      | 55.4%                     | 14 of 20                  | Accuracy mirage; rejected                                               |
| Boundary-aware router, region-supervised training                                   | 92.5% ± 5.9%      | 85.3% ± 8.3%              | 0 of 20                   | Qualified anti-collapse result                                          |
| Boundary-aware router, pseudo-labeled by an r-informed specialist-feature threshold | 93.8% ± 3.9%      | 85.6% ± 7.2%              | 0 of 20                   | Qualified no-region-label-loss result; not switching-variable discovery |
| Gate given true radius                                                              | 91.5% to 100.0%   | Tracks true region        | 0                         | Diagnostic ceiling, not deployable evidence                             |
| Oracle region switch                                                                | 100.0%            | 100.0%                    | 0                         | Upper bound                                                             |


The baseline values in the supervised and pseudo-label reruns differ slightly from the pooled study because they were recomputed inside their own fixed evaluation paths. The global blend was 68.4% in those reruns rather than 69.9%; the confidence baseline was 69.9% in the pseudo-label rerun rather than 69.5%. These are split and rerun differences, not new methods.

### 5.4 Finished negative: fixed global composition

A fixed blend assigns the same weight to each specialist for every input. The balanced task requires a piecewise decision: use one specialist inside the boundary and the other outside it. No constant pair of weights can express that switch.

The measured result matches the structural limitation. The blend reached 69.9%, below the spiral specialist at 76.6%, the circles specialist at 73.5%, and the best global single choice at 77.1%. This is a negative result, not a tuning failure to be hidden.

### 5.5 The 81.3% routing mirage

The learned router appears to improve substantially over the single specialists, but its mechanism does not survive authentication:

- It agreed with the true inner/outer region on only 55.4% of held-out points.
- Fourteen of twenty runs landed at exactly 50% region agreement.
- Those collapsed runs showed the signature of always choosing the spiral specialist: approximately 0% inner-region routing error and 100% outer-region routing error.
- Only five of twenty runs reached at least 85% composite accuracy.
- In a post-hoc exploratory analysis of those five high-scoring runs, routing confidence correlated with the true radial variable; no alternative high-accuracy mechanism was observed in that small subset.

The mean of 81.3% therefore combines a large collapsed cluster with a small favorable tail. Composite accuracy alone would have misclassified this method as successful.

### 5.6 The signal was present, but the router failed to use it

The circles specialist's output correlated 0.87 with distance from the center, while the spiral specialist's output had approximately zero correlation. The switch information was therefore strongly represented in the available specialist features.

This separates two questions:

1. **Is the switching information present?** Yes, strongly.
2. **Does the learned routing mechanism recover it reliably from sparse boundary data?** No.

The failure is best described as unstable routing under sparse boundary coverage, not absence of useful information.

### 5.7 A rejected competence-routing hypothesis

A separate approach trained each specialist to estimate whether it was likely to be correct. At the primary training budget it reached 78.9% ± 6.1% composite accuracy, 56.7% ± 7.8% region agreement, and collapsed in 10 of 20 runs. Its routing confidence remained high on confidently wrong examples, and its boundary correlation had the wrong sign.

This mechanism was rejected. Publishing the negative matters because a smaller training budget had looked substantially better; performance deteriorated as the certification budget increased.

---



## 6. Qualified Positive Results



### 6.1 Region-supervised boundary-aware routing

The boundary-aware router expands its effective decision region near uncertain boundaries. During training it received region supervision; at inference it saw only the frozen specialists' outputs and related features. The true radius was not an inference input.

At the canonical 30-example budget across 20 runs:


| Pre-declared criterion                       | Result                                      | Verdict |
| -------------------------------------------- | ------------------------------------------- | ------- |
| Accuracy above fixed global blend            | 92.5% versus 68.4%                          | Pass    |
| No constant-specialist collapse              | 0 of 20 runs                                | Pass    |
| Errors concentrated near boundary            | 26.2% near boundary versus 8.7% in interior | Pass    |
| Low reliance on confidently wrong specialist | 0.18, below the 0.50 limit                  | Pass    |
| Mean region agreement at least 80%           | 85.3%                                       | Pass    |
| Stretch target above 90% region agreement    | 85.3%                                       | Not met |


Across five training-set sizes from 20 to 200 examples, none of 100 runs collapsed and the boundary-aware router beat the fixed blend at every size.

The correct attribution is limited: region supervision explains much of the accuracy. The architecture earns the anti-collapse result and stable behavior across training sizes. It does not show that the relevant region was discovered without supervision.

### 6.2 Pseudo-label training without region labels in the loss

The next study removed the true region and radius from the training loss. Pseudo-labels were formed from the frozen specialists' scalar outputs, especially a threshold on the circles specialist. A fixed orientation prior from the earlier correlation analysis established which side of that threshold should route to which specialist. True regions were used only after training to evaluate the router.

Structurally, this is an r-proxy gate: it thresholds a feature previously shown, using true radius, to correlate 0.87 with r, and applies a direction derived from that same true-radius analysis. It does not demonstrate independent discovery of the switching variable.


| Criterion or comparison                                   | Result             | Verdict |
| --------------------------------------------------------- | ------------------ | ------- |
| No collapsed runs                                         | 0 of 20            | Pass    |
| Accuracy above fixed blend                                | 93.8% versus 68.4% | Pass    |
| Accuracy above confidence selection                       | 93.8% versus 69.9% | Pass    |
| Original pre-declared region-agreement floor at least 60% | 85.6% ± 7.2%       | Pass    |
| Stricter supervised-study benchmark at least 80%          | 85.6% ± 7.2%       | Pass    |
| Confident-wrong reliance below 0.50                       | 0.12               | Pass    |
| No agreement decay as data grows                          | 85.2% to 85.9%     | Pass    |


Two alternative pseudo-labeling strategies reached 92.1% and 93.6% accuracy, with no collapsed runs.

The 60% floor is the criterion actually registered for this study and has not been rewritten after evaluation. It was deliberately looser than the supervised-study floor because the no-region-label-loss method was allowed to be weaker while still being required to beat chance. The 80% row is a retrospective comparison to the stricter earlier benchmark, not part of the original verdict table.

“No-region-label-loss” has a precise and narrow meaning here: no region label or radius entered the loss. It does **not** mean assumption-free discovery. The method retained a fixed orientation prior derived from true-radius analysis, and the result has been established only on this two-dimensional construction.

---



## 7. Input-Only Routing Beyond the Central Study

Two additional experiments asked the router to select among five frozen specialists from the input alone at test time.


| Domain        | Protocol                                                                                                                    | Result                                                                                                  | Scope                                           |
| ------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Split MNIST   | Five binary digit tasks; task identity available for router training but absent at test; three fixed runs                   | 86.0% router agreement, 84.9% mean task accuracy, 3 of 3 non-collapsed                                  | Bounded multi-run result on an easy domain      |
| CIFAR-10 lite | Five binary class-pair tasks; frozen, fingerprinted 128-dimensional patch features; nearest-neighbor routing; one fixed run | 41.0% router agreement versus 20.5% embedding-similarity baseline, 68.2% task accuracy, zero forgetting | Single-run result, one point above its 40% gate |


The CIFAR result is a preliminary single-run observation that local structure in a frozen feature bank may support better-than-chance dispatch. It is not a supported routing claim: it is one point above its gate, has no confidence interval, and does not meet this paper's multi-run resolution standard. It is not a full Split-CIFAR-100 result or class-incremental learning.

---



## 8. Scope Extension: Frozen-Encoder Predictive Adapters

The promote–freeze contract also extends to predictive and action adapters bound to a frozen, fingerprinted sensory encoder. That work uses its own evidence cells for prediction, regime routing, action return, persistence, and encoder-pin stability. Because those studies are largely orthogonal to this paper's routing-authentication thesis and do not share one uniform multi-run protocol, their results are not summarized as evidence here. Architecture, run counts, diagnostics, and bounded results are reported separately in [WORLD_MODELS.md](WORLD_MODELS.md) and [JEPA_ADAPTER_PROMOTION.md](JEPA_ADAPTER_PROMOTION.md).

---



## 9. What the Evidence Supports Today



### Supported

- New parameter-isolated specialists can be promoted and frozen without measured changes to earlier specialists.
- Fixed global output blending fails on the balanced switched task studied here.
- Composite accuracy alone can conceal constant-specialist routing collapse.
- Region agreement, routing diversity, boundary error profiles, and confidence diagnostics expose that collapse.
- Boundary-aware routing avoids collapse on the studied task under both region-supervised and narrowly defined pseudo-label training.
- Input-only routing works above its baselines in the bounded multi-run MNIST evaluation.



### Not supported

- Prevention of forgetting inside one continually updated shared model.
- General routing across open-world tasks or arbitrary high-dimensional domains.
- A universal advantage over mixture-of-experts, adapter fusion, model merging, or Progressive Networks.
- Robust CIFAR routing across many runs or full Split-CIFAR-100.
- A supported CIFAR routing result from the current single run.
- Formal superiority of the substrate's geometric or algebraic engineering choices.

---



## 10. Limitations

1. **Parameter isolation moves rather than eliminates difficulty.** Retention is inherited from freezing; routing and memory growth remain.
2. **The central routing task is synthetic and two-dimensional.** It was designed to expose mechanism, not to represent deployment complexity.
3. **The strongest routing results use task-specific structure.** The supervised method receives region labels during training. The pseudo-label method retains a fixed orientation prior derived from true-radius analysis.
4. **The apparent information ceiling is task-local.** The supervised and pseudo-label methods both settle near 85% region agreement; this should not be treated as a universal limit.
5. **Split MNIST is weak.** Its role is lifecycle auditing, not competitive continual-learning evaluation.
6. **The CIFAR-10 lite result is preliminary.** It uses frozen patch features, binary class pairs, and one run.
7. **The proxy-gated result is task-informed.** Its threshold orientation was derived from true-radius analysis, so it does not establish discovery of an unknown switching variable.
8. **Storage grows with specialist count.** Pruning reduces per-specialist cost but does not remove linear growth.
9. **The work is not peer reviewed.** Results are internal preprint evidence and should be independently reproduced.

---



## 11. Research Roadmap

The next evidence should increase difficulty rather than add terminology.

1. Replicate frozen-feature CIFAR routing across many runs with confidence intervals.
2. Evaluate full Split-CIFAR-100 and CORe50 with matched isolated baselines.
3. Test boundary-aware routing on higher-dimensional, non-radial, and transferred composites.
4. Remove fixed orientation assumptions and evaluate open-world specialist discovery.
5. Compare directly with Progressive Networks, adapter fusion, task arithmetic, and mixture-of-experts dispatch under matched protocols.
6. Evaluate production-scale routing latency, memory growth, abstention, and failure recovery.
7. Publish a pinned independent-reproduction package and seek external replication.



---



## 12. Glossary

**Boundary-aware router:** A router whose effective decision behavior expands near uncertain switching boundaries.

**Collapsed run:** A run in which routing becomes nearly constant and selects one specialist for almost every input.

**Composite task:** A task whose correct rule changes across inputs, requiring different specialists in different regions or regimes.

**Frozen specialist:** A promoted neural subgraph whose trainable state no longer changes.

**Main:** The permanent store of promoted, frozen specialists.

**Mirror:** An isolated environment in which one new specialist can learn.

**Promotion:** The transition from a plastic Mirror specialist to a frozen Main specialist after evaluation criteria are met.

**Region agreement:** The fraction of examples for which the router's specialist choice matches the specialist assigned by the task's true generating region.

**Router:** The mechanism that selects which frozen specialist should process an input.

---



## References

Fedus, W., Zoph, B., and Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. Journal of Machine Learning Research, 23.

Ilharco, G., et al. (2023). *Editing Models with Task Arithmetic*. International Conference on Learning Representations.

Kirkpatrick, J., et al. (2017). *Overcoming Catastrophic Forgetting in Neural Networks*. Proceedings of the National Academy of Sciences, 114(13), 3521–3526.

Kumaran, D., Hassabis, D., and McClelland, J. (2016). *What Learning Systems Do Intelligent Agents Need? Complementary Learning Systems Theory Updated*. Trends in Cognitive Sciences, 20(7), 512–534.

LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.

Mallya, A., and Lazebnik, S. (2018). *PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning*. IEEE Conference on Computer Vision and Pattern Recognition.

McClelland, J. L., McNaughton, B. L., and O'Reilly, R. C. (1995). *Why There Are Complementary Learning Systems in the Hippocampus and Neocortex*. Psychological Review, 102(3), 419–457.

Pfeiffer, J., et al. (2021). *AdapterFusion: Non-Destructive Task Composition for Transfer Learning*. European Chapter of the Association for Computational Linguistics.

Rusu, A. A., et al. (2016). *Progressive Neural Networks*. arXiv:1606.04671.

Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. International Conference on Learning Representations.

Wortsman, M., et al. (2022). *Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time*. International Conference on Machine Learning.

---

*Public preprint. Correspondence: Astor Rivera-Carcamo, SWTCH.AI.*