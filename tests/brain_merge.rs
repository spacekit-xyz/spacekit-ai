// tests/brain_merge.rs — integration tests for Brain A + Brain B overlay merge

use growformer::dimension::manager::DimensionManager;
use growformer::dimension::manager::DimensionManagerConfig;

#[test]
fn merge_overlay_adds_gen_envs() {
    let base = DimensionManager::new(DimensionManagerConfig::default());
    let overlay = DimensionManager::new(DimensionManagerConfig::default());
    let mut merged = base;
    let summary = merged.merge_overlay_brain(overlay);
    assert_eq!(summary.overlay_groups, 0);
    assert_eq!(summary.gen_envs_added, 0);
    assert_eq!(summary.code_envs_added, 0);
}

#[test]
fn merge_overlay_renumbers_groups() {
    let mut base = DimensionManager::new(DimensionManagerConfig::default());
    let mut overlay = DimensionManager::new(DimensionManagerConfig::default());

    let env = growformer::dimension::group_gen::IndexedGenEnv::empty_with_output_dim(192);
    base.group_gen_envs.insert(0, env.clone());
    overlay.group_gen_envs.insert(0, env.clone());
    overlay.group_gen_envs.insert(1, env);

    let summary = base.merge_overlay_brain(overlay);
    assert_eq!(summary.gen_envs_added, 2);
    assert!(
        base.group_gen_envs.contains_key(&0),
        "base group 0 preserved"
    );
    let new_0 = summary.group_id_map[&0];
    let new_1 = summary.group_id_map[&1];
    assert!(
        base.group_gen_envs.contains_key(&new_0),
        "overlay group 0 remapped"
    );
    assert!(
        base.group_gen_envs.contains_key(&new_1),
        "overlay group 1 remapped"
    );
    assert_ne!(new_0, 0, "overlay group 0 should be renumbered");
    assert_eq!(base.group_gen_envs.len(), 3, "total should be 3 gen envs");
}

#[test]
fn merge_summary_has_correct_counts() {
    let mut base = DimensionManager::new(DimensionManagerConfig::default());
    let mut overlay = DimensionManager::new(DimensionManagerConfig::default());

    let env = growformer::dimension::group_gen::IndexedGenEnv::empty_with_output_dim(192);
    base.group_gen_envs.insert(0, env.clone());
    base.group_gen_envs.insert(1, env.clone());
    overlay.group_gen_envs.insert(0, env);

    let summary = base.merge_overlay_brain(overlay);
    assert_eq!(
        summary.base_groups_before, 2,
        "offset should be max(existing)+1"
    );
    assert_eq!(summary.gen_envs_added, 1);
    assert_eq!(base.group_gen_envs.len(), 3);
}
