#!/usr/bin/env bash
set -euo pipefail

echo "== Growformer Code Tasks Demo =="

echo
echo "[1/5] Rust code sample"
cargo run -- --language-code-text "implement a web server in rust"

echo
echo "[2/5] JavaScript code sample"
cargo run -- --language-code-text "implement debounce helper in javascript"

echo
echo "[3/5] Python code sample"
cargo run -- --language-code-text "remove duplicate list entries in python"

echo
echo "[4/5] Batch codegen eval (full holdouts)"
cargo run -- --language-code-eval --code-eval-report reports/m5_codegen_eval_holdouts.json

echo
echo "[5/5] Codegen validation gate (full holdouts)"
cargo run -- --validate-codegen --code-eval-report reports/m5_codegen_eval_holdouts.json

echo
echo "Demo complete."

