# JEPA Adapter Promotion Contract

**Status:** Normative for Phase 3i world-model toys and any future JEPA / latent-dynamics adapters on Growformer.  
**Related:** [WORLD_MODELS.md](WORLD_MODELS.md) §3.2, [GROWFORMER_WHITEPAPER.md](GROWFORMER_WHITEPAPER.md) §3.3 / §4.3.1, implementation [`src/dimension/jepa_adapters.rs`](../src/dimension/jepa_adapters.rs).

---

## 1. Purpose

Continual learning in Growformer is **parameter isolation** (Mirror → promote → frozen Main). JEPA / world-model adapters must not reopen shared-substrate forgetting. This document states what may train, what must freeze, and what promotion verifies.

## 2. Roles

| Role | Mutable after init? | Promotes to Main? | Notes |
| --- | --- | --- | --- |
| **JEPA / sensory encoder** | **No** | N/A (shared, pinned) | Constructed once; fingerprint recorded; never receives gradient |
| **Predictor adapter** (next-latent / affinity) | Yes, in Mirror only | **Yes** | One adapter (or pair) per dynamics regime / domain |
| **Authenticated router** (cone) | Train-time only on composite | Optional / separate | Consumes affinity (or specialist) scalars; no `r` at inference |
| **AI OS planner** | Policy layer | N/A | Latent rollouts over *selected* predictors — not VirtualGroup blend |

## 3. Fingerprint pin

On encoder construction, compute a deterministic fingerprint over all encoder parameters (`FrozenJepaEncoder::fingerprint`).

**Promotion rules** (`JepaPromotionBundle::promote`):

1. Every predictor’s `encoder_pin` must equal the live encoder fingerprint.
2. The bundle stores `encoder_fingerprint` immutably.
3. At load / inference, `verify_encoder` must succeed; fingerprint drift is a hard error (not a warning).

Changing the encoder requires a **new pin** and re-training / re-promoting all predictors that depended on the old latent space. There is no “lightly fine-tune the backbone” path under this contract.

## 4. Mirror → Main lifecycle

```
1. Freeze encoder E; pin = fingerprint(E)
2. Mirror A: train PredictorAdapter_A on regime-A transitions only (encoder forward-only)
3. Mirror B: train PredictorAdapter_B on regime-B transitions only
4. Promote {A, B} into Main under pin  (no gradient into E or into other promoted adapters)
5. Train / update router on composite affinity features (region labels train-only if used)
6. Inference: z = E(obs); route among frozen predictors; plan via short latent rollouts
```

**Forbidden:**

- Gradient into `E` during Mirror or router training
- Updating a previously promoted predictor while training a new Mirror (isolation)
- One mega world-model group that absorbs all regimes without promote–freeze boundaries
- Claiming retention while the encoder continues to train

## 5. Routing honesty (WM Task E)

Composite next-step MSE alone cannot certify routing (same lesson as Task E accuracy). Use:

- **Regime agreement** (router choice vs generative regime)
- **Degeneracy / entropy** (constant-predictor collapse)
- **Cone MSE vs VirtualGroup / confidence-argmax floors**
- **n-sweep** (no regime-agree decay as train coverage grows)

Reproduce: `cargo run --release --bin growformer-demos -- --phase3i-jepa-wm`  
Artifact: [`phase3i_jepa_wm_results.txt`](../phase3i_jepa_wm_results.txt).

## 6. Relation to language `GroupAdapter`

[`GroupAdapter`](../src/dimension/language.rs) is a PEFT-style map for language bridge dims. The **same isolation idea** applies: backbone / encoder frozen; adapter params are the promotable delta. Phase 3i predictors are the world-model analogue of that pattern in latent dynamics space — not a replacement for chat fingerprint routing.

## 7. Non-goals

- Full LeCun AMI / JEPA-2 training at scale
- Luna / companion chat adapters as the first JEPA surface
- Replacing Main Dimension with a single learned world model
- Using VirtualGroup global blending as the planning mechanism

## 8. Energy-based adapters (Phase 3j)

Phase 3i predictors score next-latents with MSE. Phase 3j promotes **energy landscapes**
\(E_\theta(z_t, z_{t+1})\) (EB-JEPA / GeoWorld-adjacent):

| Piece | Role |
| --- | --- |
| Energy head | Softplus MLP on `[z; z'; Δz]`; true pairs low, contrasts high (margin) |
| Proposal head | Residual next-latent for planning / MSE eval |
| Affinity head | Oracle-free regime score for cone routing |

**Naming:** this is **not** metabolic synapse `energy_budget_per_neuron`. Docs and APIs use
`EnergyAdapter` / `E(z,z')` / “latent energy” for the EBM sense.

**Promotion:** same pin rules via `EnergyPromotionBundle` — encoder frozen; only energy /
proposal / affinity params promote ([`energy_jepa.rs`](../src/dimension/energy_jepa.rs)).

**Extra certifiers (beyond §5):** energy margin \(E_{\mathrm{away}} - E_{\mathrm{home}} > 0.01\)
(softplus scale on the toy); cone true-pair energy ≤ VG average energy.

Reproduce: `cargo run --release --bin growformer-demos -- --phase3j-energy-wm`  
Artifact: [`phase3j_energy_wm_results.txt`](../phase3j_energy_wm_results.txt).

**Roadmap after 3j (implemented):**

| Phase | Flag | Artifact |
| --- | --- | --- |
| **3k Geometric** | `--phase3k-geo-wm` | [`phase3k_geo_wm_results.txt`](../phase3k_geo_wm_results.txt) |
| **3ℓ Probabilistic** | `--phase3l-prob-wm` | [`phase3l_prob_wm_results.txt`](../phase3l_prob_wm_results.txt) |
| **3m Neuro-symbolic** | `--phase3m-sym-wm` | [`phase3m_sym_wm_results.txt`](../phase3m_sym_wm_results.txt) |
| **3n Action** | `--phase3n-action-wm` | [`phase3n_action_wm_results.txt`](../phase3n_action_wm_results.txt) |
| **3o Compose** | `--phase3o-compose-wm` | [`phase3o_compose_wm_results.txt`](../phase3o_compose_wm_results.txt) |
| **3p Hard transfer** | `--phase3p-hard-wm` | [`phase3p_hard_wm_results.txt`](../phase3p_hard_wm_results.txt) |
| **3q Deploy** | `--phase3q-deploy-wm` | [`phase3q_deploy_wm_results.txt`](../phase3q_deploy_wm_results.txt) |
| **3r Beyond-toy** | `--phase3r-beyond-toy` | [`phase3r_beyond_toy_results.txt`](../phase3r_beyond_toy_results.txt) |
| **3s Open ladder** | `--phase3s-open-ladder` | [`phase3s_open_ladder_results.txt`](../phase3s_open_ladder_results.txt) |
| **3t Act surface** | `--phase3t-act-wm` | [`phase3t_act_wm_results.txt`](../phase3t_act_wm_results.txt) |
| **3u V-JEPA bridge** | `--phase3u-vjepa-wm` | [`phase3u_vjepa_wm_results.txt`](../phase3u_vjepa_wm_results.txt) |
| **3v Scene-graph WM** | `--phase3v-scene-wm` | [`phase3v_scene_wm_results.txt`](../phase3v_scene_wm_results.txt) |
| **3w Scene host** | `--phase3w-scene-host` | [`phase3w_scene_host_results.txt`](../phase3w_scene_host_results.txt) |
| **V-JEPA smoke** | `scripts/smoke_vjepa_export.sh` | mock CI; `VJEPA_MODE=hf` optional |
| **Layer 0 graph** | `--layer0-concept-graph` | [`layer0_concept_graph_results.txt`](../layer0_concept_graph_results.txt) |
| **4a CF-MNIST scaffold** | `--phase4a-context-free-mnist` | [`phase4a_context_free_mnist_results.txt`](../phase4a_context_free_mnist_results.txt) |
| **4b CF LearnedRouter** | `--phase4b-cf-mnist-router` | [`phase4b_cf_mnist_router_results.txt`](../phase4b_cf_mnist_router_results.txt) |
| **4c Split-CIFAR scaffold** | `--phase4c-split-cifar-scaffold` | [`phase4c_split_cifar_scaffold_results.txt`](../phase4c_split_cifar_scaffold_results.txt) |
| **4d CF MNIST 5-task** | `--phase4d-cf-mnist-full` | [`phase4d_cf_mnist_full_results.txt`](../phase4d_cf_mnist_full_results.txt) |
| **4e Split-CIFAR-10 lite** | `--phase4e-split-cifar-lite` | [`phase4e_split_cifar_lite_results.txt`](../phase4e_split_cifar_lite_results.txt) (needs `python3 scripts/export_cifar10.py`) |
| **4f CIFAR frozen patches** | `--phase4f-split-cifar-frozen` | [`phase4f_split_cifar_frozen_results.txt`](../phase4f_split_cifar_frozen_results.txt) |
| **5a WM↔DM citizens** | `--phase5a-wm-dm` | [`phase5a_wm_dm_results.txt`](../phase5a_wm_dm_results.txt) |
| **5b Product act-loop** | `--phase5b-product-act` | [`phase5b_product_act_results.txt`](../phase5b_product_act_results.txt) |
| **5c External product** | `--phase5c-external-product` | [`phase5c_external_product_results.txt`](../phase5c_external_product_results.txt) |
| **5d V-JEPA vision D′** | `--phase5d-vjepa-vision` | [`phase5d_vjepa_vision_results.txt`](../phase5d_vjepa_vision_results.txt) |
| **5e WM brain.bin** | `--phase5e-wm-brain` | [`phase5e_wm_brain_results.txt`](../phase5e_wm_brain_results.txt) |
| **5f Live SpaceKit** | `--phase5f-live-spacekit` | [`phase5f_live_spacekit_results.txt`](../phase5f_live_spacekit_results.txt) |
| **5g V-JEPA real-log** | `--phase5g-vjepa-real-log` | [`phase5g_vjepa_real_log_results.txt`](../phase5g_vjepa_real_log_results.txt) |
| **SpaceKit stdio** | `--wm-host-stdio scene\|acting\|deploy` | [`WM_SPACEKIT_HOST.md`](WM_SPACEKIT_HOST.md) |

Code: [`wm_frontier.rs`](../src/dimension/wm_frontier.rs), [`wm_transfer.rs`](../src/dimension/wm_transfer.rs), [`wm_proof.rs`](../src/dimension/wm_proof.rs), [`wm_open.rs`](../src/dimension/wm_open.rs), [`wm_act.rs`](../src/dimension/wm_act.rs), [`wm_vjepa.rs`](../src/dimension/wm_vjepa.rs), [`wm_scene.rs`](../src/dimension/wm_scene.rs), [`wm_scene_host.rs`](../src/dimension/wm_scene_host.rs), [`context_free_mnist.rs`](../src/dimension/context_free_mnist.rs).  
Export: [`scripts/export_vjepa_features.py`](../scripts/export_vjepa_features.py).  
Beyond-toy proof ladder: [`WORLD_MODELS.md`](WORLD_MODELS.md) §8. SpaceKit host: [`WM_SPACEKIT_HOST.md`](WM_SPACEKIT_HOST.md).
