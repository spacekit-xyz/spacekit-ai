//! Standalone growformer CLI (growformer team development only).
//! End users invoke growformer via `spacekit agent` with entitlement enforcement.

fn main() {
    if let Err(e) = growformer::run_cli(std::env::args()) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
