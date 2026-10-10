//! Write a tiny random GGUF + tokenizer.json for smoke tests:
//! `cargo run --example tiny_model --features candle -- <out_dir> [llama|qwen2|qwen3]`
//! Answers from this model are random — it only exercises the plumbing.
fn main() {
    let mut args = std::env::args().skip(1);
    let out = std::path::PathBuf::from(args.next().unwrap_or_else(|| "tiny-model".into()));
    let arch = args.next().unwrap_or_else(|| "qwen3".into());
    std::fs::create_dir_all(&out).expect("create out dir");
    let m = out.join(format!("tiny-{arch}.gguf"));
    std::fs::write(&m, growformer_system1::lm::testkit::gguf(&arch)).expect("write gguf");
    let t = out.join("tokenizer.json");
    std::fs::write(&t, growformer_system1::lm::testkit::tokenizer_json()).expect("write tokenizer");
    println!("{}\n{}", m.display(), t.display());
}
