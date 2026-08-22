//! Fractal Topology — Main Dimension, Mirror Dimension, Promotion Gate, GlobalObserver.
//! Phase 3: isolated env per task; no shared substrate.

pub mod action;
pub mod action_classifier;
pub mod codegen;
pub mod composition;
pub mod cone_router;
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub mod context_free_mnist;
pub mod embedding;
pub mod energy_jepa;
pub mod generation;
pub mod generation_head;
pub mod group_gen;
pub mod jepa_adapters;
pub mod language;
pub mod main_dim;
pub mod mirror_dim;
pub mod observer;
pub mod policy;
pub mod promotion;
pub mod router;
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub mod split_cifar_scaffold;
pub mod wm_act;
pub mod wm_citizen;
pub mod wm_dm;
pub mod wm_frontier;
pub mod wm_open;
pub mod wm_product;
pub mod wm_proof;
pub mod wm_scene;
pub mod wm_scene_host;
pub mod wm_transfer;
pub mod wm_vjepa;
// manager after wm_dm (WmCitizenRecord field)
pub mod manager;
pub mod paramecium;
pub mod polarity_probe;
pub mod tool;

pub use crate::clifford::GroupRotor;
pub use action::{action_from_routing, ActionJson, ActionType};
pub use action_classifier::{
    action_target_to_type, action_type_one_hot, group_id_one_hot, ActionClassifier,
};
pub use codegen::{generate_code_from_action, CodeGeneration};
pub use composition::{
    routing_entropy_bits, routing_entropy_degenerate, Episode, EpisodicMemory, RoutingEntropyGuard,
    VirtualGroup,
};
pub use cone_router::{
    cone_features, AdjustableConeRouter, ConeConfig, ConeDecision, ConeSample, LabelFreeStrategy,
    CONE_FEATURE_DIM,
};
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub use context_free_mnist::{
    run_phase4a_context_free_mnist, run_phase4b_cf_mnist_router, run_phase4d_cf_mnist_full,
    run_phase4d_cf_mnist_multiseed, CfMnistMultiSeedResult, CfMnistRouterResult,
    ContextFreeMnistResult,
};
pub use embedding::{
    compute_group_embedding, cosine_similarity, retrieve_relevant_groups, GroupEmbedding,
};
pub use energy_jepa::{
    run_energy_wm_task_e_seed, EnergyAdapter, EnergyPromotionBundle, EnergyWmSeedResult,
};
pub use generation::{render_action_template, GeneratedResponse};
pub use generation_head::GenerationHead;
pub use group_gen::GroupGenEnv;
pub use group_gen::IndexedGenEnv;
pub use jepa_adapters::{
    generate_transitions, run_wm_task_e_seed, step_dynamics, stratified_wm_split,
    FrozenJepaEncoder, JepaPromotionBundle, PredictorAdapter, WmSeedResult, WmTransition,
    WM_INNER_RADIUS, WM_LATENT_DIM, WM_OBS_DIM,
};
pub use language::{
    append_language_samples_from_training_jsonl_dir, causal_index_token,
    causal_subtype_index_token, is_brain_training_jsonl_filename,
    is_inference_guardrails_jsonl_filename, load_language_samples_jsonl, route_language_embedding,
    sentiment_lattice_index_body_with_causal, CalibrationCoverage, CalibrationDataset,
    CalibrationReport, CalibrationRequirements, CausalAnnotation, EmaSmoother, EncoderPreset,
    GroupAdapter, HashingLanguageEncoder, LanguageBridge, LanguageConfig, LanguageEncoder,
    LanguageRoutingDecision, LanguageRuntime, LanguageSample, SENTIMENT_CAUSAL_INDEX_CORE,
};
pub use main_dim::{FrozenGroupEnv, MainDimension};
pub use manager::{
    CheckpointSizeSummary, DimensionManager, DimensionManagerConfig, EpisodicSummary, GroupSummary,
    MirrorSummary,
};
pub use mirror_dim::{EpochResult, MirrorDimension};
pub use observer::{GlobalObserver, RoutingConfig};
pub use policy::{ContinuousPolicy, Policy};
pub use promotion::{evaluate_promotion, promote, PromotionDecision, PromotionGateConfig};
pub use router::{attend_by_query, KnnRouter, LearnedRouter};
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
pub use split_cifar_scaffold::{
    run_phase4c_split_cifar_scaffold, run_phase4e_split_cifar_lite, run_phase4f_knn_router_probe,
    run_phase4f_split_cifar_frozen, SplitCifarFrozenResult, SplitCifarLiteResult,
    SplitCifarRouterProbeResult, SplitCifarScaffoldResult,
};
pub use tool::{ParamType, ToolCallInfo, ToolParam, ToolRegistry, ToolResult, ToolSchema};
pub use wm_act::{
    act_step_disk, run_phase3t_disk_act_seed, run_phase3t_host_act_seed,
    run_phase3t_visuomotor_act_seed, train_disk_acting_bundle, ActDecision, ActSeedResult,
    ActingHostRequest, ActingHostResponse, ActingHostSession, ActingWmBundle,
};
pub use wm_citizen::{WmCitizenKind, WmCitizenRecord};
pub use wm_dm::{install_wm_citizens, run_phase5a_wm_dm_spike, WmDmSpikeResult};
pub use wm_frontier::{
    geometric_energy, latent_interval, run_phase3k_geo_seed, run_phase3l_prob_seed,
    run_phase3m_sym_seed, symbolic_energy, FrozenGeometricEncoder, GeoWmSeedResult,
    ProbWmSeedResult, SymWmSeedResult, WorldRule,
};
pub use wm_open::{
    ensure_frozen_vision_encoder, render_visuomotor, run_phase3s_spacekit_host_seed,
    run_phase3s_visuomotor_seed, step_visuomotor, write_visuomotor_log, FrozenVisionEncoder,
    SpacekitHostSeedResult, VisuomotorSeedResult, WmHostRequest, WmHostResponse, WmHostSession,
    VISION_PIXELS, VISION_SIDE,
};
pub use wm_product::{
    run_phase5b_product_act_loop, run_phase5c_external_product_loop,
    run_phase5f_live_spacekit_episode, ExternalProductLoopResult, LiveSpacekitEpisodeResult,
    ProductActLoopResult,
};
pub use wm_proof::{
    ensure_frozen_encoder_file, run_phase3r_action_rank_seed, run_phase3r_foreign_seed,
    run_phase3r_sim_loop, ActionRankSeedResult, ForeignDomainResult, ForeignProofSeedResult,
    FrozenExternalEncoder, SimLoopResult, SimStepLog,
};
pub use wm_scene::{
    evaluate_scene_wm_bundle, goal_scene, pick_block, run_phase3v_scene_seed, sample_scene,
    scene_act_step, scene_act_step_routed, scene_deploy_step, step_scene, train_scene_wm_bundle,
    FrozenSceneEncoder, SceneActDecision, SceneGraph, SceneStepDecision, SceneWmBundle,
    SceneWmSeedResult,
};
pub use wm_scene_host::{
    run_phase3w_scene_host_seed, SceneHostRequest, SceneHostResponse, SceneHostSeedResult,
    SceneHostSession,
};
pub use wm_transfer::{
    deploy_step, load_composed_bundle, plan_action, run_phase3n_action_seed,
    run_phase3o_compose_seed, run_phase3p_hard_seed, run_phase3q_deploy_seed, save_composed_bundle,
    step_dynamics_action, train_composed_bundle, ActionEnergyAdapter, ActionWmSeedResult,
    ComposeWmSeedResult, ComposedWmBundle, DeployDecision, DeploySeedResult, FrozenHardEncoder,
    HardWmSeedResult, WmAction, ACTION_DIM, HARD_OBS_DIM,
};
pub use wm_vjepa::{
    build_export_from_frozen_vision, build_export_from_real_log, build_rust_mock_export,
    dump_visuomotor_log, ensure_vjepa_export, run_phase3u_vjepa_seed,
    run_phase5d_vjepa_vision_seed, run_phase5g_vjepa_real_log_seed, FrozenVjepaExport,
    VjepaWmSeedResult,
};
