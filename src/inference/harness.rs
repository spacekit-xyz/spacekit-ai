//! Built-in inference plugins run through a small harness so [`crate::service::LanguageService`]
//! stays orchestration-only while domain logic lives under [`super::plugins`].

use std::sync::Arc;

use crate::brain::BrainPackageHeader;
use crate::dimension::DimensionManager;
use crate::growformer_lang::MetaConcept;
use crate::micro_brain::MetaResult;

use super::manifest::{BrainPluginsManifest, InferenceThresholds};

/// Short-circuit generation: plugin-produced text replaces lattice `generate_with_e8_for_topic`.
#[derive(Debug, Clone)]
pub struct GenerationPreemptOutcome {
    pub text: String,
    pub confidence: f32,
    pub template_id: &'static str,
}

/// Post-processing flags for a finished [`crate::service::GeneratedResponse::template_id`].
#[derive(Clone, Copy, Default)]
pub struct TemplatePostprocessFlags {
    pub skip_coherence_truncate: bool,
    pub skip_metacog: bool,
}

/// One loadable inference plugin (compile-time registered; brain TOML supplies config only).
pub trait BrainInferencePlugin: Send + Sync {
    fn skip_weak_gk_for_meta_conditioning(
        &self,
        dm: &DimensionManager,
        inference_profile: Option<&str>,
        thresholds_from_manifest: Option<&InferenceThresholds>,
        concept: MetaConcept,
        margin: f32,
        confidence: f32,
    ) -> bool {
        let _ = (
            dm,
            inference_profile,
            thresholds_from_manifest,
            concept,
            margin,
            confidence,
        );
        false
    }

    fn extend_subject_keywords(
        &self,
        dm: &DimensionManager,
        inference_profile: Option<&str>,
        intent_text: &str,
        subject_kw: &mut Vec<String>,
    ) {
        let _ = (dm, inference_profile, intent_text, subject_kw);
    }

    fn try_preempt_generation(
        &self,
        dm: &DimensionManager,
        inference_profile: Option<&str>,
        thresholds_from_manifest: Option<&InferenceThresholds>,
        intent_text: &str,
        meta_result: Option<&MetaResult>,
        topic_hint: Option<&str>,
    ) -> Option<GenerationPreemptOutcome> {
        let _ = (
            dm,
            inference_profile,
            thresholds_from_manifest,
            intent_text,
            meta_result,
            topic_hint,
        );
        None
    }

    fn template_postprocess_flags(&self, template_id: &str) -> TemplatePostprocessFlags {
        let _ = template_id;
        TemplatePostprocessFlags::default()
    }

    /// Export-time: set header / manifest defaults. Return `true` if this plugin set
    /// `header.inference_profile` (harness will not overwrite it from the previous header).
    fn export_brain_plugins(
        &self,
        dm: &DimensionManager,
        header: &mut BrainPackageHeader,
        manifest: &mut BrainPluginsManifest,
    ) -> bool {
        let _ = (dm, header, manifest);
        false
    }
}

/// Registered built-in plugins; construct with [`Self::new`] (see [`crate::inference::plugins::default_inference_harness`]).
///
/// Cheap [`Clone`] (shared plugin table) so callers can hold a copy across `active_dm_mut()` borrows.
#[derive(Clone)]
pub struct InferenceHarness {
    plugins: Arc<Vec<Box<dyn BrainInferencePlugin>>>,
}

impl InferenceHarness {
    pub fn new(plugins: Vec<Box<dyn BrainInferencePlugin>>) -> Self {
        Self {
            plugins: Arc::new(plugins),
        }
    }

    pub fn skip_weak_gk_for_meta_conditioning(
        &self,
        dm: &DimensionManager,
        inference_profile: Option<&str>,
        thresholds_from_manifest: Option<&InferenceThresholds>,
        concept: MetaConcept,
        margin: f32,
        confidence: f32,
    ) -> bool {
        self.plugins.iter().any(|p| {
            p.skip_weak_gk_for_meta_conditioning(
                dm,
                inference_profile,
                thresholds_from_manifest,
                concept,
                margin,
                confidence,
            )
        })
    }

    pub fn extend_subject_keywords(
        &self,
        dm: &DimensionManager,
        inference_profile: Option<&str>,
        intent_text: &str,
        subject_kw: &mut Vec<String>,
    ) {
        for p in self.plugins.iter() {
            p.extend_subject_keywords(dm, inference_profile, intent_text, subject_kw);
        }
    }

    pub fn try_preempt_generation(
        &self,
        dm: &DimensionManager,
        inference_profile: Option<&str>,
        thresholds_from_manifest: Option<&InferenceThresholds>,
        intent_text: &str,
        meta_result: Option<&MetaResult>,
        topic_hint: Option<&str>,
    ) -> Option<GenerationPreemptOutcome> {
        for p in self.plugins.iter() {
            if let Some(o) = p.try_preempt_generation(
                dm,
                inference_profile,
                thresholds_from_manifest,
                intent_text,
                meta_result,
                topic_hint,
            ) {
                return Some(o);
            }
        }
        None
    }

    pub fn template_postprocess_flags(&self, template_id: &str) -> TemplatePostprocessFlags {
        let mut f = TemplatePostprocessFlags::default();
        for p in self.plugins.iter() {
            let pf = p.template_postprocess_flags(template_id);
            f.skip_coherence_truncate |= pf.skip_coherence_truncate;
            f.skip_metacog |= pf.skip_metacog;
        }
        f
    }

    pub fn apply_export_brain_plugins(
        &self,
        dm: &DimensionManager,
        previous_inference_profile: Option<&str>,
        header: &mut BrainPackageHeader,
        manifest: &mut BrainPluginsManifest,
    ) {
        let mut profile_owned = false;
        for p in self.plugins.iter() {
            if p.export_brain_plugins(dm, header, manifest) {
                profile_owned = true;
                break;
            }
        }
        if !profile_owned {
            header.inference_profile = previous_inference_profile.map(|s| s.to_string());
        }
    }
}
