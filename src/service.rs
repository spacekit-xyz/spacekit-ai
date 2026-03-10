use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::dimension::{
    generate_code_from_action, render_action_template, ActionJson, CalibrationDataset, CalibrationReport,
    CalibrationRequirements, CodeGeneration, DimensionManager, DimensionManagerConfig, EncoderPreset,
    GeneratedResponse, LanguageConfig, LanguageSample,
};
use crate::types::{EnvironmentConfig, GroupId, Sample};

pub struct LanguageService {
    pub dm: DimensionManager,
    pub support_gid: GroupId,
    pub coding_gid: GroupId,
    pub calibration: CalibrationReport,
}

impl LanguageService {
    pub fn new_default() -> Result<Self, String> {
        let (dm, support_gid, coding_gid, report) = build_language_demo_manager(0.2)?;
        Ok(Self {
            dm,
            support_gid,
            coding_gid,
            calibration: report,
        })
    }

    pub fn action(&mut self, text: &str) -> Result<ActionJson, String> {
        self.dm.route_text_to_action(text)
    }

    pub fn generation(&mut self, text: &str) -> Result<(ActionJson, GeneratedResponse), String> {
        let action = self.action(text)?;
        let resp = render_action_template(&action);
        Ok((action, resp))
    }

    pub fn codegen(&mut self, text: &str) -> Result<(ActionJson, Option<CodeGeneration>), String> {
        let action = self.action(text)?;
        let code = generate_code_from_action(&action, text);
        Ok((action, code))
    }
}

pub fn build_language_demo_manager(
    ema_alpha: f32,
) -> Result<(DimensionManager, GroupId, GroupId, CalibrationReport), String> {
    let mut data_rng = StdRng::seed_from_u64(7);
    let config = DimensionManagerConfig {
        mirror_config: phase2_base_config(),
        mirror_layer_sizes: vec![2, 16, 16, 1],
        promotion_check_interval: 999_999,
        max_concurrent_mirrors: 2,
        calibration_samples: 50,
        reserve_pool_size: 0,
    };
    let mut dm = DimensionManager::new(config);

    dm.spawn_mirror("support", 100)
        .ok_or_else(|| "failed to spawn support mirror".to_string())?;
    dm.spawn_mirror("coding", 101)
        .ok_or_else(|| "failed to spawn coding mirror".to_string())?;
    let cal_support = generate_spiral_data(50, &mut data_rng);
    let cal_coding = generate_concentric_circles_data(50, &mut data_rng);
    let support_gid = dm
        .force_promote("support", &cal_support)
        .ok_or_else(|| "failed to promote support mirror".to_string())?;
    let coding_gid = dm
        .force_promote("coding", &cal_coding)
        .ok_or_else(|| "failed to promote coding mirror".to_string())?;

    dm.configure_language(LanguageConfig {
        encoder: EncoderPreset::BertClass,
        bridge_output_dim: 64,
        ema_alpha,
        ood_similarity_threshold: 0.15,
        gle_http_endpoint: std::env::var("GROWFORMER_GLE_HTTP_ENDPOINT").ok(),
        gle_checkpoint: std::env::var("GROWFORMER_GLE_CHECKPOINT").ok(),
    });

    let calibration = build_language_calibration_dataset();
    let requirements = CalibrationRequirements {
        multilingual_required: true,
        ..CalibrationRequirements::default()
    };
    let report = dm.calibrate_language_bridge(&calibration, &requirements)?;

    let mut support_prompts = Vec::new();
    let mut coding_prompts = Vec::new();
    for i in 0..200 {
        support_prompts.push(format!(
            "customer support account login password reset billing help ticket {}",
            i
        ));
        support_prompts.push(format!(
            "help desk cannot access account needs recovery and verification {}",
            i
        ));
        coding_prompts.push(format!(
            "write rust code function parser json serde implementation {}",
            i
        ));
        coding_prompts.push(format!(
            "debug c segmentation fault stack trace pointer module {}",
            i
        ));
    }
    dm.set_group_language_vector_from_texts(support_gid, &support_prompts)?;
    dm.set_group_language_vector_from_texts(coding_gid, &coding_prompts)?;

    Ok((dm, support_gid, coding_gid, report))
}

fn phase2_base_config() -> EnvironmentConfig {
    EnvironmentConfig {
        learning_rate: 0.15,
        weight_decay: 0.0000025,
        bias_decay: 0.0,
        dropout_rate: 0.0,
        geometry_noise: 0.0,
        competitive_k: 4,
        lateral_inhibition: 0.12,
        lr_decay: 0.00008,
        sigma_inhib: 2.0,
        debye_length: 1.5,
        thermal_noise: 0.02,
        k_repel: 0.2,
        gravity_g: 0.05,
        damping: 0.2,
        mass_win_threshold: 0.15,
        mass_decay: 0.00009,
        mass_growth: 0.0005,
        homeostasis_lr: 0.0,
        growth_radius: 2.0,
        prune_interval: 500,
        weight_clamp: 5.0,
        max_synapses_per_neuron: 64,
        energy_budget_per_neuron: 100.0,
        pruning_threshold: 0.001,
        mirror_coupling_strength: 0.001,
        geometry_interval: 500,
        stdp_enabled: false,
        mass_consolidation_k: 0.0,
        ..EnvironmentConfig::default()
    }
}

fn generate_concentric_circles_data(n_per_class: usize, rng: &mut impl rand::Rng) -> Vec<Sample> {
    use std::f32::consts::PI;
    let mut data = Vec::with_capacity(n_per_class * 2);
    let noise = 0.05_f32;
    for _ in 0..n_per_class {
        let theta = rng.gen::<f32>() * 2.0 * PI;
        let r = 0.5 + rng.gen_range(-noise..noise);
        data.push((vec![r * theta.cos(), r * theta.sin()], [0.0]));
    }
    for _ in 0..n_per_class {
        let theta = rng.gen::<f32>() * 2.0 * PI;
        let r = 1.0 + rng.gen_range(-noise..noise);
        data.push((vec![r * theta.cos(), r * theta.sin()], [1.0]));
    }
    data.shuffle(rng);
    data
}

fn generate_spiral_data(n_per_class: usize, rng: &mut impl rand::Rng) -> Vec<Sample> {
    use std::f32::consts::PI;
    let mut data = Vec::new();
    for class in 0..2 {
        for i in 0..n_per_class {
            let t = (i as f32 / n_per_class as f32) * PI;
            let offset = if class == 0 { 0.0 } else { PI };
            let r = t / (4.0 * PI);
            let x = r * (t + offset).cos() + rng.gen_range(-0.05..0.05_f32);
            let y = r * (t + offset).sin() + rng.gen_range(-0.05..0.05_f32);
            data.push((vec![x, y], [class as f32]));
        }
    }
    data.shuffle(rng);
    data
}

fn build_language_calibration_dataset() -> CalibrationDataset {
    let mut samples = Vec::new();
    let domains = vec![
        "customer_support",
        "coding_tool_use",
        "knowledge_qa",
        "safety_refusal",
        "procedural_instruction",
        "short_conversation",
        "multi_turn_followup",
        "adversarial_noisy",
    ];
    let languages = ["english", "english", "english", "spanish", "french"];
    for domain in domains {
        for i in 0..500 {
            let lang = languages[i % languages.len()];
            let text = format!("{} sample {} in {}", domain, i, lang);
            samples.push(LanguageSample {
                domain: domain.to_string(),
                text,
                semantic_intent: format!("{}_intent", domain),
                action_target: if domain == "coding_tool_use" {
                    Some("tool_runner".to_string())
                } else {
                    None
                },
                policy_regime: if domain == "safety_refusal" {
                    "strict".to_string()
                } else {
                    "default".to_string()
                },
                language_channel: lang.to_string(),
            });
        }
    }
    CalibrationDataset { samples }
}

