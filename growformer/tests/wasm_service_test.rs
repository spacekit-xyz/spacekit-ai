use growformer::dimension::action::ActionPayload;
use growformer::dimension::{ActionType, LanguageConfig};
use growformer::service::LanguageService;

#[test]
fn test_wasm_compatible_service_lifecycle() {
    let config = LanguageConfig::default();
    let mut svc = LanguageService::new_with_config(config).expect("init service");

    let action = svc.action("implement a rust web server").expect("action");
    assert_eq!(action.action_type, ActionType::CodingAssist);
    if let Some(ActionPayload::CodingAssist { language_hint, .. }) = &action.payload {
        assert_eq!(language_hint, "rust");
    } else {
        panic!("expected CodingAssist payload, got {:?}", action.payload);
    }

    let (_action, response) = svc
        .generation("help me reset my password")
        .expect("generation");
    assert!(!response.text.is_empty(), "generation should produce text");

    let (code_action, code) = svc.codegen("implement a rust web server").expect("codegen");
    assert_eq!(code_action.action_type, ActionType::CodingAssist);
    assert!(
        code.is_none(),
        "an empty service must abstain instead of fabricating an untrained code stub"
    );
}
