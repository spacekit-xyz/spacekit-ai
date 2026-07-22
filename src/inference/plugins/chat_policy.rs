//! Chat-policy plugin: locale-keyed greeting/identity shortcuts and compose-bleed recovery.
//!
//! Policy data lives in inference TOML `[chat_policy]`; this plugin only delegates so
//! [`crate::service::LanguageService`] stays orchestration-focused.

use crate::inference::chat_policy::BleedHit;
use crate::inference::harness::BrainInferencePlugin;
use crate::inference::inference_toml::inference_toml_loaded;

/// Built-in plugin registered in [`super::default_inference_harness`].
pub struct ChatPolicyPlugin;

impl BrainInferencePlugin for ChatPolicyPlugin {
    fn match_identity_query(&self, language_channel: Option<&str>, text: &str) -> bool {
        inference_toml_loaded()
            .chat_policy()
            .match_identity_query(language_channel, text)
    }

    fn match_greeting(&self, language_channel: Option<&str>, text: &str) -> bool {
        inference_toml_loaded()
            .chat_policy()
            .match_greeting(language_channel, text)
    }

    fn detect_compose_bleed(
        &self,
        language_channel: Option<&str>,
        prompt: &str,
        response: &str,
    ) -> Option<BleedHit> {
        inference_toml_loaded()
            .chat_policy()
            .detect_compose_bleed(language_channel, prompt, response)
    }

    fn bleed_fallback(
        &self,
        language_channel: Option<&str>,
        prompt: &str,
        bleed: &BleedHit,
    ) -> Option<(String, String)> {
        inference_toml_loaded()
            .chat_policy()
            .bleed_fallback(language_channel, prompt, bleed)
    }
}
