use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let build_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    println!("cargo:rustc-env=GROWFORMER_BUILD_UNIX={}", build_unix);

    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=GROWFORMER_TARGET={}", target);

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=GROWFORMER_PROFILE={}", profile);

    // Best-effort git SHA; workspace might not be a git repo (that's ok).
    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "nogit".to_string());
    println!("cargo:rustc-env=GROWFORMER_GIT_SHA={}", git_sha);
}
