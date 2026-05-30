//! Internal dev binary (`growformer-cli`) — same surface as `growformer`, no entitlement gate.

fn main() {
    if let Err(e) = growformer::run_cli(std::env::args()) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
