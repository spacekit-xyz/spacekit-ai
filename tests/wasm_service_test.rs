use growformer::dimension::{ActionType, LanguageConfig};
use growformer::dimension::action::ActionPayload;
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

    let (_action, response) = svc.generation("help me reset my password").expect("generation");
    assert!(!response.text.is_empty(), "generation should produce text");

    let (_action, code) = svc.codegen("implement a rust web server").expect("codegen");
    let code = code.expect("should produce code for coding prompt");
    assert_eq!(code.language, "rust");
    assert!(!code.code.is_empty(), "code output should be non-empty");
}
