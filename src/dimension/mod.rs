//! Fractal Topology — Main Dimension, Mirror Dimension, Promotion Gate, GlobalObserver.
//! Phase 3: isolated env per task; no shared substrate.

pub mod composition;
pub mod cone_router;
pub mod jepa_adapters;
pub mod energy_jepa;
pub mod wm_frontier;
pub mod wm_transfer;
pub mod wm_proof;
pub mod wm_open;
pub mod wm_act;
pub mod wm_vjepa;
pub mod wm_scene;
pub mod wm_scene_host;
pub mod context_free_mnist;
pub mod split_cifar_scaffold;
pub mod action;
pub mod action_classifier;
pub mod codegen;
pub mod generation;
pub mod generation_head;
pub mod group_gen;
pub mod embedding;
pub mod language;
pub mod main_dim;
pub mod mirror_dim;
pub mod policy;
pub mod promotion;
pub mod router;
pub mod observer;
pub mod manager;
pub mod tool;
pub mod paramecium;
pub mod polarity_probe;

pub use composition::{EpisodicMemory, Episode, RoutingEntropyGuard, VirtualGroup, routing_entropy_bits, routing_entropy_degenerate};
pub use cone_router::{
    cone_features, AdjustableConeRouter, ConeConfig, ConeDecision, ConeSample, LabelFreeStrategy,
    CONE_FEATURE_DIM,
};
pub use jepa_adapters::{
    generate_transitions, run_wm_task_e_seed, step_dynamics, stratified_wm_split, FrozenJepaEncoder,
    JepaPromotionBundle, PredictorAdapter, WmSeedResult, WmTransition, WM_INNER_RADIUS,
    WM_LATENT_DIM, WM_OBS_DIM,
};
pub use energy_jepa::{
    run_energy_wm_task_e_seed, EnergyAdapter, EnergyPromotionBundle, EnergyWmSeedResult,
};
pub use wm_frontier::{
    geometric_energy, latent_interval, run_phase3k_geo_seed, run_phase3l_prob_seed,
    run_phase3m_sym_seed, symbolic_energy, FrozenGeometricEncoder, GeoWmSeedResult,
    ProbWmSeedResult, SymWmSeedResult, WorldRule,
};
pub use wm_transfer::{
    deploy_step, load_composed_bundle, plan_action, run_phase3n_action_seed,
    run_phase3o_compose_seed, run_phase3p_hard_seed, run_phase3q_deploy_seed,
    save_composed_bundle, step_dynamics_action, train_composed_bundle, ActionEnergyAdapter,
    ActionWmSeedResult, ComposedWmBundle, ComposeWmSeedResult, DeployDecision, DeploySeedResult,
    FrozenHardEncoder, HardWmSeedResult, WmAction, ACTION_DIM, HARD_OBS_DIM,
};
pub use wm_proof::{
    ensure_frozen_encoder_file, run_phase3r_action_rank_seed, run_phase3r_foreign_seed,
    run_phase3r_sim_loop, ActionRankSeedResult, ForeignDomainResult, ForeignProofSeedResult,
    FrozenExternalEncoder, SimLoopResult, SimStepLog,
};
pub use wm_open::{
    ensure_frozen_vision_encoder, render_visuomotor, run_phase3s_spacekit_host_seed,
    run_phase3s_visuomotor_seed, step_visuomotor, write_visuomotor_log, FrozenVisionEncoder,
    SpacekitHostSeedResult, VisuomotorSeedResult, WmHostRequest, WmHostResponse, WmHostSession,
    VISION_PIXELS, VISION_SIDE,
};
pub use wm_act::{
    act_step_disk, run_phase3t_disk_act_seed, run_phase3t_host_act_seed,
    run_phase3t_visuomotor_act_seed, train_disk_acting_bundle, ActDecision, ActSeedResult,
    ActingHostRequest, ActingHostResponse, ActingHostSession, ActingWmBundle,
};
pub use wm_vjepa::{
    build_rust_mock_export, ensure_vjepa_export, run_phase3u_vjepa_seed, FrozenVjepaExport,
    VjepaWmSeedResult,
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
pub use context_free_mnist::{
    run_phase4a_context_free_mnist, run_phase4b_cf_mnist_router, CfMnistRouterResult,
    ContextFreeMnistResult,
};
pub use split_cifar_scaffold::{run_phase4c_split_cifar_scaffold, SplitCifarScaffoldResult};
pub use action::{ActionJson, ActionType, action_from_routing};
pub use codegen::{CodeGeneration, generate_code_from_action};
pub use generation::{GeneratedResponse, render_action_template};
pub use embedding::{GroupEmbedding, compute_group_embedding, cosine_similarity, retrieve_relevant_groups};
pub use language::{
    append_language_samples_from_training_jsonl_dir, causal_index_token, causal_subtype_index_token,
    CalibrationCoverage, CalibrationDataset, CalibrationRequirements, CalibrationReport,
    CausalAnnotation, EncoderPreset, EmaSmoother, GroupAdapter, HashingLanguageEncoder, LanguageBridge,
    LanguageConfig, LanguageEncoder, LanguageRoutingDecision, LanguageRuntime, LanguageSample,
    is_brain_training_jsonl_filename, is_inference_guardrails_jsonl_filename, load_language_samples_jsonl,
    route_language_embedding, sentiment_lattice_index_body_with_causal, SENTIMENT_CAUSAL_INDEX_CORE,
};
pub use crate::clifford::GroupRotor;
pub use main_dim::{MainDimension, FrozenGroupEnv};
pub use mirror_dim::{MirrorDimension, EpochResult};
pub use promotion::{PromotionGateConfig, PromotionDecision, evaluate_promotion, promote};
pub use policy::{ContinuousPolicy, Policy};
pub use router::{LearnedRouter, attend_by_query};
pub use observer::{GlobalObserver, RoutingConfig};
pub use action_classifier::{ActionClassifier, action_target_to_type, action_type_one_hot, group_id_one_hot};
pub use generation_head::GenerationHead;
pub use group_gen::GroupGenEnv;
pub use group_gen::IndexedGenEnv;
pub use manager::{DimensionManager, DimensionManagerConfig, GroupSummary, MirrorSummary, EpisodicSummary, CheckpointSizeSummary};
pub use tool::{ToolSchema, ToolParam, ParamType, ToolRegistry, ToolCallInfo, ToolResult};
