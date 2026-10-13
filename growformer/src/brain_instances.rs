//! Several brains in one process.
//!
//! A [`Runtime`] holds its own brain, conversation and agent state, but the
//! inference configuration it reads is process-wide: the inference rules
//! (`inference_toml`), the topic graph and the domain grounding graph. Loading
//! a second brain's rules therefore used to change the first brain's answers.
//!
//! [`BrainInstance`] keeps each brain's configuration with the brain and makes
//! it the active one before every call. Switching is cheap: the saved
//! configuration is put back by reference (rules) or by clone (graphs), never
//! re-parsed. Brains used one after another behave exactly as if each were the
//! only one loaded.
//!
//! The legacy single-brain API (`crate::wasm::growformer_*` functions) is the
//! "default" slot: [`activate_default`] restores its configuration, and
//! [`capture_default`] saves it after the legacy API changes it.
//!
//! Only one thread uses this at a time (WASM is single-threaded; native hosts
//! should keep a set of instances on one thread or behind one lock).

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use crate::growformer_lang;
use crate::inference::inference_toml::{self, LoadedInferenceToml};
use crate::inference::world_grounding::{self, GroundingSnapshot};
use crate::runtime::Runtime;
use crate::topic_graph::TopicGraph;

/// The process-wide inference configuration one brain runs with.
#[derive(Clone, Debug, Default)]
pub struct BrainConfig {
    inference: Option<Arc<LoadedInferenceToml>>,
    topic_graph: Option<TopicGraph>,
    grounding: GroundingSnapshot,
}

impl BrainConfig {
    /// Save what is active right now.
    pub fn capture() -> Self {
        Self {
            inference: inference_toml::inference_toml_snapshot(),
            topic_graph: growformer_lang::topic_graph(),
            grounding: world_grounding::grounding_snapshot(),
        }
    }

    /// Make this configuration the active one.
    pub fn install(&self) {
        inference_toml::install_inference_toml_snapshot(self.inference.clone());
        match &self.topic_graph {
            Some(g) => {
                let _ = growformer_lang::init_topic_graph_direct(g.clone());
            }
            None => growformer_lang::clear_topic_graph(),
        }
        world_grounding::restore_grounding(&self.grounding);
    }

    pub fn has_inference_rules(&self) -> bool {
        self.inference.is_some()
    }
    pub fn has_topic_graph(&self) -> bool {
        self.topic_graph.is_some()
    }
    pub fn has_grounding(&self) -> bool {
        self.grounding.is_loaded()
    }
}

/// Slot id 0 is the legacy single-brain API.
const DEFAULT_SLOT: u64 = 0;

thread_local! {
    static ACTIVE: Cell<u64> = const { Cell::new(DEFAULT_SLOT) };
    static NEXT_ID: Cell<u64> = const { Cell::new(1) };
    static DEFAULT_CONFIG: RefCell<BrainConfig> = RefCell::new(BrainConfig::default());
}

fn active() -> u64 {
    ACTIVE.with(|a| a.get())
}

fn set_active(id: u64) {
    ACTIVE.with(|a| a.set(id));
}

/// Before a legacy single-brain call: put back its configuration if another
/// brain was active.
pub fn activate_default() {
    if active() != DEFAULT_SLOT {
        DEFAULT_CONFIG.with(|c| c.borrow().install());
        set_active(DEFAULT_SLOT);
    }
}

/// After the legacy API changes the configuration: remember it as the default slot's.
pub fn capture_default() {
    DEFAULT_CONFIG.with(|c| *c.borrow_mut() = BrainConfig::capture());
    set_active(DEFAULT_SLOT);
}

/// One loaded brain with its own inference configuration.
pub struct BrainInstance {
    id: u64,
    runtime: Runtime,
    config: BrainConfig,
}

impl BrainInstance {
    /// Load a brain (`.bin` bytes). The brain may carry its own inference rules;
    /// they apply to this instance only.
    pub fn from_brain_bytes(data: &[u8]) -> Result<Self, String> {
        Self::create(|| Runtime::from_brain_bytes(data))
    }

    /// An instance with no brain (tests, or a brain loaded later with [`Self::load_brain`]).
    pub fn empty() -> Result<Self, String> {
        Self::create(Runtime::empty)
    }

    fn create(make: impl FnOnce() -> Result<Runtime, String>) -> Result<Self, String> {
        // Whatever was active stays intact: save it, build this brain from the
        // built-in defaults, then capture what the brain installed.
        let previous_slot = active();
        let previous = BrainConfig::capture();
        BrainConfig::default().install();
        let built = make();
        let config = BrainConfig::capture();
        previous.install();
        if previous_slot == DEFAULT_SLOT {
            DEFAULT_CONFIG.with(|c| *c.borrow_mut() = previous);
        }
        set_active(previous_slot);
        let runtime = built?;
        let id = NEXT_ID.with(|n| {
            let id = n.get();
            n.set(id + 1);
            id
        });
        Ok(Self {
            id,
            runtime,
            config,
        })
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn config(&self) -> &BrainConfig {
        &self.config
    }

    /// Make this brain's configuration active and run `f` on its runtime.
    pub fn with<R>(&mut self, f: impl FnOnce(&mut Runtime) -> R) -> R {
        if active() != self.id {
            if active() == DEFAULT_SLOT {
                // Keep the legacy slot's latest state before replacing it.
                DEFAULT_CONFIG.with(|c| *c.borrow_mut() = BrainConfig::capture());
            }
            self.config.install();
            set_active(self.id);
        }
        f(&mut self.runtime)
    }

    /// Run `f`, then save any configuration change it made as this brain's.
    fn reconfigure<R>(
        &mut self,
        f: impl FnOnce(&mut Runtime) -> Result<R, String>,
    ) -> Result<R, String> {
        let out = self.with(f);
        self.config = BrainConfig::capture();
        out
    }

    /// Replace the brain. Rules embedded in the new brain replace this instance's.
    pub fn load_brain(&mut self, data: &[u8]) -> Result<(), String> {
        self.reconfigure(|rt| rt.load_brain(data))
    }

    /// Domain inference rules (e.g. `inference_crypto.toml`), merged with the built-in core.
    pub fn load_inference_toml(&mut self, toml_str: &str) -> Result<(), String> {
        self.reconfigure(|rt| {
            inference_toml::reload_inference_toml_from_str(toml_str)?;
            rt.apply_loaded_generation_config();
            Ok(())
        })
    }

    /// Topic routing graph, with an optional overlay merged on top.
    pub fn load_topic_graph(
        &mut self,
        base_toml: &str,
        overlay_toml: Option<&str>,
    ) -> Result<(), String> {
        let base =
            TopicGraph::from_toml(base_toml).map_err(|e| format!("topic graph base: {}", e))?;
        let graph = match overlay_toml {
            Some(o) => base.merge_overlay(
                TopicGraph::from_toml_quiet(o)
                    .map_err(|e| format!("topic graph overlay: {}", e))?,
            ),
            None => base,
        };
        self.reconfigure(|_| growformer_lang::init_topic_graph_direct(graph))
    }

    pub fn clear_topic_graph(&mut self) {
        let _ = self.reconfigure(|_| {
            growformer_lang::clear_topic_graph();
            Ok(())
        });
    }

    /// Domain grounding graph (concept nodes and edges).
    pub fn load_grounding_graph(&mut self, toml_str: &str) -> Result<(), String> {
        self.reconfigure(|_| world_grounding::load_grounding_graph_from_str(toml_str))
    }

    pub fn clear_grounding_graph(&mut self) {
        let _ = self.reconfigure(|_| {
            world_grounding::clear_grounding_graph();
            Ok(())
        });
    }
}

impl Drop for BrainInstance {
    fn drop(&mut self) {
        // The process-wide configuration is this brain's; hand it back to the
        // legacy slot so the single-brain API keeps working as before.
        if active() == self.id {
            DEFAULT_CONFIG.with(|c| c.borrow().install());
            set_active(DEFAULT_SLOT);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOPICS_A: &str = r#"
[[nodes]]
topic = "alpha_topic"
concept = "greeting"

[[nodes.rules]]
any = ["alpha"]
"#;

    fn topic_ids() -> Vec<String> {
        growformer_lang::topic_graph()
            .map(|g| format!("{:?}", g))
            .into_iter()
            .collect()
    }

    #[test]
    fn instances_keep_their_own_configuration() {
        let mut a = BrainInstance::empty().expect("a");
        let mut b = BrainInstance::empty().expect("b");
        assert_ne!(a.id(), b.id());

        a.load_inference_toml("").expect("rules a");
        let rules_a = a.config().inference.clone().expect("a has rules");
        b.load_inference_toml("").expect("rules b");
        let rules_b = b.config().inference.clone().expect("b has rules");
        assert!(!Arc::ptr_eq(&rules_a, &rules_b));

        // Topic graph only on a, grounding only on b.
        a.load_topic_graph(TOPICS_A, None).expect("topics a");
        assert!(a.config().has_topic_graph());
        assert!(!b.config().has_topic_graph());
        b.load_grounding_graph("").expect("grounding b");
        assert!(b.config().has_grounding());
        assert!(!a.config().has_grounding());

        // Switching restores each brain's configuration exactly.
        a.with(|_| {
            let active = inference_toml::inference_toml_snapshot().expect("active rules");
            assert!(Arc::ptr_eq(&active, &rules_a));
            assert!(!world_grounding::grounding_snapshot().is_loaded());
        });
        let a_topics = topic_ids();
        b.with(|_| {
            let active = inference_toml::inference_toml_snapshot().expect("active rules");
            assert!(Arc::ptr_eq(&active, &rules_b));
            assert!(world_grounding::grounding_snapshot().is_loaded());
            assert!(growformer_lang::topic_graph().is_none());
        });
        a.with(|_| assert_eq!(topic_ids(), a_topics));

        // Dropping the active brain hands the configuration back to the legacy slot.
        drop(a);
        activate_default();
        b.with(|_| assert!(world_grounding::grounding_snapshot().is_loaded()));
        drop(b);
        assert!(!world_grounding::grounding_snapshot().is_loaded());

        creating_an_instance_leaves_the_active_one_alone();
    }

    // Part of the test above: the configuration is process-wide on native, so the
    // checks must not run in parallel with each other.
    fn creating_an_instance_leaves_the_active_one_alone() {
        let mut a = BrainInstance::empty().expect("a");
        a.load_grounding_graph("").expect("grounding a");
        a.with(|_| ());
        let before = inference_toml::inference_toml_snapshot();
        let _b = BrainInstance::empty().expect("b");
        let after = inference_toml::inference_toml_snapshot();
        match (before, after) {
            (Some(x), Some(y)) => assert!(Arc::ptr_eq(&x, &y)),
            (None, None) => {}
            _ => panic!("creating b changed a's rules"),
        }
        assert!(world_grounding::grounding_snapshot().is_loaded());
    }
}
