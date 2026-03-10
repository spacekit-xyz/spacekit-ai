# Growformer Coding Curriculum v1 Template

Use this template to build a systematic coding curriculum for M5 retention experiments.

## 1) Stack Profile

- Project name:
- Primary languages: `python`, `rust`, `javascript/typescript`
- Runtime targets: (CLI, backend API, workers, frontend)
- Key non-functional constraints: (latency, memory, safety, reliability)

## 2) Library Matrix (Per Language)

Fill this with the exact libraries/frameworks Growformer should learn.

| Language | Tier | Libraries / Frameworks |
|---|---|---|
| Python | Core | `stdlib`, `typing`, `dataclasses`, `pytest` |
| Python | App | `<fastapi>`, `<pydantic>`, `<sqlalchemy>`, `<pandas>` |
| Rust | Core | `std`, `serde`, `thiserror`, `tokio` |
| Rust | App | `<axum/actix>`, `<sqlx/diesel>`, `<tracing>` |
| JS/TS | Core | `node`, `promises`, `jest/vitest` |
| JS/TS | App | `<express/fastify>`, `<react/next>`, `<zod>` |

## 3) Task Taxonomy

For each language, target a balanced distribution:

- Implementation (`coding_implementation`)
- Debug (`coding_debug`)
- Refactor (`coding_refactor`)
- Optimize (`coding_optimize`)
- Testing (`coding_testing`)

Recommended starting ratio:

- 35% implementation
- 20% debugging
- 15% refactor
- 15% optimize
- 15% testing

## 4) Difficulty Bands

Label each sample with one of:

- `L1`: small function / utility
- `L2`: module-level logic
- `L3`: multi-file integration
- `L4`: production constraints / edge-heavy

Target mix:

- `L1` 30%
- `L2` 35%
- `L3` 25%
- `L4` 10%

## 5) JSONL Row Schema

Use this schema for train/eval examples:

```json
{
  "task_id": "rs_train_001",
  "text": "Implement Rust interval merge function over sorted ranges.",
  "semantic_intent": "coding_implementation",
  "domain": "coding_rust",
  "action_target": "coding",
  "policy_regime": "default",
  "language_channel": "english",
  "code_language": "rust",
  "libraries": ["std", "serde"],
  "difficulty": "L2",
  "split": "train"
}
```

## 6) Eval Rubric

Track at least:

- Routing/action quality:
  - `coding_action_rate`
  - fallback false positives
- Codegen quality:
  - `language_match_rate`
  - `specialized_stub_rate`
  - syntax validity (compile/lint where feasible)
- Retention quality:
  - per-domain post-sequence retention ratio
  - target: `>= 0.97`

## 7) Retention Protocol

1. Train domain A (Python), eval A baseline.
2. Train domain B (Rust), eval A+B.
3. Train domain C (JS/TS), eval A+B+C.
4. Compute final retention ratio per domain vs baseline.

Use `retention_eval_splits.json` as the execution plan and keep it versioned.

## 8) Demo Checklist (For Stakeholders)

- Show single-prompt code outputs (Python/Rust/JS).
- Show batch eval report with per-language breakdown.
- Show validation gate pass/fail.
- Show retention chart after sequential training.

