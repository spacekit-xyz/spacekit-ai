#!/usr/bin/env python3
"""Pre-registered embedding RAG baseline vs growformer raw lattice retrieval.

Encoder: sentence-transformers/all-MiniLM-L6-v2 (cosine k-NN on training JSONL `text` field).
Decision rule (pre-registered): if RAG passes >=3 scorable cases and brain passes <=1,
switch retrieval core to embedding k-NN for grounding.

Fair frozen run (see docs/BET_D_FROZEN_PROTOCOL.md):
  python3 scripts/rag_baseline_battery.py --round frozen --live-brain --corpus spacekit
  python3 scripts/rag_baseline_battery.py --heldout-v2 --protocol-frozen \\
    --live-brain --corpus spacekit --round frozen_v2

Legacy audit (hand-frozen brain dict, asymmetric growformer corpus):
  python3 scripts/rag_baseline_battery.py --round 4 --no-live-brain --corpus growformer
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

NEGATIVE_LABELS = {
    "negative_mild",
    "negative_strong",
    "sarcastic",
    "cautiously_negative",
    "capitulation",
    "disappointed",
    "formal_complaint",
    "polite_negative",
    "resigned_annoyance",
    "etf_delay_bearish",
    "mortgage_rate_complaint",
    "fee_complaint",
}
POSITIVE_LABELS = {
    "positive_mild",
    "positive_strong",
    "euphoric",
    "cautiously_positive",
    "grateful_but_cautious",
    "hopium",
}

CASES = [
    {
        "id": "case1_sentiment_bitcoin",
        "corpus": "sentiment",
        "query": "Bitcoin crashed after the ETF delay",
        "scorable": True,
    },
    {
        "id": "case2_crypto_bitcoin",
        "corpus": "crypto",
        "query": "Bitcoin crashed after the ETF delay",
        "scorable": True,
    },
    {
        "id": "case3_fintech_chase",
        "corpus": "fintech",
        "query": "Chase raised my mortgage rate without notice",
        "scorable": True,
    },
    {
        "id": "case4_sentiment_chase_wrong_brain",
        "corpus": "sentiment",
        "query": "Chase raised my mortgage rate without notice",
        "scorable": True,
    },
]

# Frozen raw brain top-1 from brain-raw-diag --battery --top-k 3 (2026-07-04 round 4).
# SpaceKit crypto/fintech .bin + scenario topics + service.rs lexical-hint fix.
BRAIN_RAW_FROZEN = {
    "case1_sentiment_bitcoin": {
        "topic_hint": "positive_strong",
        "score": 1.359,
        "text_preview": "Earnings beat expectations by 15%. Revenue guidance raised for next quarter.",
        "pass_brain": False,
    },
    "case2_crypto_bitcoin": {
        "topic_hint": "etf_delay_bearish",
        "score": 1.415,
        "text_preview": "Spot Bitcoin sold off overnight after regulators pushed the ETF decision back again.",
        "pass_brain": True,
    },
    "case3_fintech_chase": {
        "topic_hint": "mortgage_rate_complaint",
        "score": 1.569,
        "text_preview": "My bank bumped my home loan rate mid-term without sending any notice letter.",
        "pass_brain": True,
    },
    "case4_sentiment_chase_wrong_brain": {
        "topic_hint": "positive_strong",
        "score": 1.029,
        "text_preview": "Earnings beat expectations by 15%. Revenue guidance raised for next quarter.",
        "pass_brain": False,
    },
}

# Product-scoped scored battery (default `--battery` in growformer-llm; cases 2–3 only).
SCORED_BATTERY_CASE_IDS = frozenset({"case2_crypto_bitcoin", "case3_fintech_chase"})
BRAIN_ELIGIBLE_CORPORA = frozenset({"crypto", "fintech"})
BRAIN_SKIP_REASON = (
    "general sentiment corpus has no SpaceKit brain (sentiment-brain-v3.bin deprecated)"
)

HELDOUT_PROMPTS_PATH = ROOT / "data/sentiment/eval_battery_heldout_prompts.jsonl"
HELDOUT_V2_PROMPTS_PATH = ROOT / "data/sentiment/eval_battery_heldout_prompts_v2.jsonl"
LLM_ROOT = ROOT.parent / "growformer-llm"


def spacekit_sentiment_root() -> Path:
    env = os.environ.get("SPACEKIT_SENTIMENT_ROOT")
    if env:
        return Path(env)
    return ROOT.parent.parent / "spacekit" / "spacekit-projects" / "sentiment"


def brain_case_eligible(corpus: str) -> bool:
    return corpus in BRAIN_ELIGIBLE_CORPORA


def brain_paths(corpus: str) -> tuple[Path, Path]:
    """SpaceKit crypto/fintech brains only (no deprecated neurokit sentiment brain)."""
    sk = spacekit_sentiment_root()
    if corpus == "crypto":
        return (
            sk / "crypto/agent/crypto-brain.bin",
            sk / "crypto/crypto-sentiment-analysis.gf.toml",
        )
    if corpus == "fintech":
        return (
            sk / "fintech/agent/fintech-brain.bin",
            sk / "fintech/fintech-sentiment-analysis.gf.toml",
        )
    raise ValueError(f"no brain configured for corpus {corpus!r}")

# Frozen raw brain top-1 from brain-raw-diag (2026-07-04, after held-out inference TOML paraphrase rules).
BRAIN_RAW_HELDOUT = {
    "heldout_crypto_etf_delay": {
        "topic_hint": "etf_delay_bearish",
        "score": 1.408,
        "text_preview": "Spot Bitcoin sold off overnight after regulators pushed the ETF decision back again.",
        "pass_brain": True,
    },
    "heldout_fintech_mortgage_hike": {
        "topic_hint": "mortgage_rate_complaint",
        "score": 1.526,
        "text_preview": "My bank bumped my home loan rate mid-term without sending any notice letter.",
        "pass_brain": True,
    },
}


def load_heldout_cases(path: Path = HELDOUT_PROMPTS_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(
                {
                    "id": obj["case_id"],
                    "corpus": obj["corpus"],
                    "query": obj["prompt"],
                    "scorable": bool(obj.get("scored", True)),
                    "expected_topic": obj.get("expected_topic"),
                }
            )
    return cases


def rag_score_case_id(case_id: str) -> str:
    """Map held-out ids to battery scoring rubric."""
    if case_id in ("heldout_crypto_etf_delay", "heldout_v2_crypto_etf_kick"):
        return "case2_crypto_bitcoin"
    if case_id in ("heldout_fintech_mortgage_hike", "heldout_v2_fintech_repriced"):
        return "case3_fintech_chase"
    return case_id


# Gap-paraphrase rows registered before v2 lock (may exist in train shards; prompt must not).
V2_EXPECTED_STORE_TEXT = {
    "heldout_v2_crypto_etf_kick": (
        "Spot Bitcoin sold off overnight after regulators pushed the ETF decision back again."
    ),
    "heldout_v2_fintech_repriced": (
        "My bank bumped my home loan rate mid-term without sending any notice letter."
    ),
}


def resolve_commit_hash(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("BET_D_PATH_A_V3_COMMIT") or os.environ.get("BET_D_FROZEN_COMMIT")
    if env:
        return env.strip()
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return "unknown"


def validate_protocol_frozen(args: argparse.Namespace, eval_cases: list[dict[str, Any]]) -> None:
    if not args.heldout_v2:
        raise SystemExit("--protocol-frozen requires --heldout-v2 (v1 held-out is retired)")
    if args.corpus != "spacekit":
        raise SystemExit("--protocol-frozen requires --corpus spacekit")
    if not args.live_brain:
        raise SystemExit("--protocol-frozen requires --live-brain (no frozen brain dict)")
    for case in eval_cases:
        rows = load_corpus(corpus_globs(case["corpus"], args.corpus))
        q = case["query"].strip().lower()
        if any(r.text.strip().lower() == q for r in rows):
            raise SystemExit(
                f"protocol-frozen guard: prompt exact match in corpus for {case['id']!r}"
            )


def result_output_path(args: argparse.Namespace) -> Path:
    out_dir = ROOT / "agent-data/brain-rag-baseline"
    if args.round in ("1", "") and not args.heldout and not args.heldout_v2:
        suffix = ""
    elif args.heldout_v2 and args.round not in ("1", ""):
        suffix = f"_{args.round}"
    elif args.heldout_v2:
        suffix = "_heldout_v2"
    elif args.heldout:
        suffix = "_heldout"
    elif args.round.isdigit():
        suffix = "" if args.round == "1" else f"_round{args.round}"
    else:
        suffix = f"_{args.round}"
    return out_dir / f"rag_baseline_results{suffix}.json"


@dataclass
class MemoryRow:
    task_id: str
    text: str
    label: str
    source: str


@dataclass
class Hit:
    rank: int
    score: float
    task_id: str
    label: str
    text: str


def corpus_globs(name: str, corpus_mode: str = "spacekit") -> list[Path]:
    if corpus_mode == "growformer":
        if name == "sentiment":
            return sorted((ROOT / "data/sentiment").glob("train_sentiment_*.jsonl"))
        if name == "crypto":
            return sorted((ROOT / "data/crypto").glob("train_sentiment_*.jsonl"))
        if name == "fintech":
            base = sorted((ROOT / "data/fintech").glob("train_sentiment_*.jsonl"))
            base += sorted((ROOT / "data/fintech").glob("train_identity_*.jsonl"))
            return base
        raise ValueError(name)
    if corpus_mode == "spacekit":
        sk = spacekit_sentiment_root()
        if name == "sentiment":
            return sorted((ROOT / "data/sentiment").glob("train_sentiment_*.jsonl"))
        if name == "crypto":
            return sorted((sk / "crypto/data").glob("train_sentiment_*.jsonl"))
        if name == "fintech":
            base = sorted((sk / "fintech/data").glob("train_sentiment_*.jsonl"))
            base += sorted((sk / "fintech/data").glob("train_identity_*.jsonl"))
            return base
        raise ValueError(name)
    raise ValueError(f"unknown corpus_mode: {corpus_mode}")


def answer_row_in_corpus(rows: list[MemoryRow], text: str) -> bool:
    t = text.strip().lower()
    return any(r.text.strip().lower() == t for r in rows)


def parse_brain_raw_diag_stdout(stdout: str) -> dict[str, Any]:
    """Parse JSON report from mixed stdout (growformer logs topic-graph lines first)."""
    text = stdout.strip()
    if not text:
        raise RuntimeError("brain-raw-diag produced empty stdout")
    start = text.find("{")
    if start < 0:
        raise RuntimeError(
            f"brain-raw-diag stdout has no JSON object (first 500 chars): {text[:500]!r}"
        )
    try:
        obj, _end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"brain-raw-diag JSON parse failed: {e}; json_start={text[start:start + 500]!r}"
        ) from e
    if not isinstance(obj, dict):
        raise RuntimeError("brain-raw-diag JSON root is not an object")
    return obj


def tinystories_brain_raw_diag_cmd(
    brain: Path,
    project: Path,
    prompt: str,
    top_k: int,
) -> list[str]:
    release_bin = LLM_ROOT / "target/release/tinystories"
    args = [
        "brain-raw-diag",
        "--brain",
        str(brain.resolve()),
        "--project",
        str(project.resolve()),
        "--prompt",
        prompt,
        "--top-k",
        str(top_k),
        "--json",
    ]
    if release_bin.is_file():
        return [str(release_bin), *args]
    return [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--bin",
        "tinystories",
        "--",
        *args,
    ]


def run_brain_raw_diag(
    brain: Path,
    project: Path,
    prompt: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """Live brain raw top-K via growformer-llm (fair arm — no frozen dict)."""
    if not LLM_ROOT.is_dir():
        raise RuntimeError(f"growformer-llm not found at {LLM_ROOT}")
    cmd = tinystories_brain_raw_diag_cmd(brain, project, prompt, top_k)
    proc = subprocess.run(
        cmd,
        cwd=str(LLM_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"brain-raw-diag failed ({proc.returncode}): {proc.stderr[-2000:]}"
        )
    return parse_brain_raw_diag_stdout(proc.stdout)


def brain_top1_from_report(report: dict[str, Any]) -> dict[str, Any]:
    cands = report.get("candidates") or []
    if not cands:
        return {
            "topic_hint": report.get("topic_hint") or report.get("forced_topic"),
            "forced_topic": report.get("forced_topic"),
            "retrieval_path": report.get("retrieval_path"),
            "score": 0.0,
            "text_preview": "",
            "witness_ok": False,
            "hard_reject": True,
            "topic": "",
        }
    c0 = cands[0]
    return {
        "topic_hint": report.get("topic_hint") or report.get("forced_topic"),
        "forced_topic": report.get("forced_topic"),
        "retrieval_path": report.get("retrieval_path"),
        "score": c0.get("score", 0.0),
        "text_preview": c0.get("text_preview", ""),
        "witness_ok": c0.get("witness_ok", False),
        "hard_reject": c0.get("hard_reject", False),
        "topic": c0.get("topic", ""),
    }


def score_brain_case(
    case_id: str,
    query: str,
    brain_top: dict[str, Any],
    store_status: str,
) -> tuple[bool, str]:
    text = brain_top.get("text_preview") or ""
    if not text.strip():
        return False, "no raw candidates"
    label = (brain_top.get("topic") or brain_top.get("forced_topic") or "").lower()
    hit = Hit(rank=1, score=float(brain_top.get("score") or 0), task_id="", label=label, text=text)
    return score_rag(rag_score_case_id(case_id), query, hit, store_status)


def load_corpus(paths: list[Path]) -> list[MemoryRow]:
    rows: list[MemoryRow] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                if not text:
                    continue
                rows.append(
                    MemoryRow(
                        task_id=str(obj.get("task_id", "")),
                        text=text,
                        label=str(obj.get("semantic_intent", "")),
                        source=path.name,
                    )
                )
    return rows


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return v / n


def top_k(query_emb: np.ndarray, matrix: np.ndarray, rows: list[MemoryRow], k: int) -> list[Hit]:
    sims = matrix @ query_emb
    order = np.argsort(-sims)[:k]
    out: list[Hit] = []
    for rank, idx in enumerate(order, start=1):
        r = rows[int(idx)]
        out.append(
            Hit(
                rank=rank,
                score=float(sims[int(idx)]),
                task_id=r.task_id,
                label=r.label,
                text=r.text,
            )
        )
    return out


def text_has_any(text: str, terms: list[str]) -> bool:
    t = text.lower()
    return any(term in t for term in terms)


def opposite_mortgage_polarity(text: str) -> bool:
    t = text.lower()
    positive_markers = ["great mortgage rate", "happy with the process", "really happy"]
    return any(m in t for m in positive_markers)


def corpus_store_status(corpus: str, query: str, corpus_mode: str = "spacekit") -> str:
    """Pre-register store coverage before retrieval."""
    rows = load_corpus(corpus_globs(corpus, corpus_mode))
    q = query.lower().strip()
    if any(r.text.lower().strip() == q for r in rows):
        return "STORE_OK_EXACT"
    if "mortgage" in q and "chase" in q:
        qualifying = [
            r
            for r in rows
            if "mortgage" in r.text.lower()
            or ("rate" in r.text.lower() and "raised" in r.text.lower())
            or ("interest rate" in r.text.lower() and "hike" in r.text.lower())
        ]
        if not qualifying:
            return "STORE_EMPTY"
        exactish = [r for r in rows if "mortgage" in r.text.lower() and "rate" in r.text.lower()]
        if not exactish:
            return "STORE_PARTIAL"
        return "STORE_OK"
    if "bitcoin" in q and "crash" in q:
        qualifying = [
            r
            for r in rows
            if "bitcoin" in r.text.lower()
            and any(
                w in r.text.lower()
                for w in ("crash", "crashed", "plunged", "dump", "sell-off", "selloff", "support")
            )
        ]
        if not qualifying:
            return "STORE_PARTIAL"
        if any(r.text.lower().strip() == q for r in rows):
            return "STORE_OK_EXACT"
        return "STORE_OK"
    return "STORE_OK"


def score_rag(case_id: str, query: str, top1: Hit, store_status: str) -> tuple[bool | None, str]:
    label = top1.label.lower()
    text = top1.text

    if case_id in ("case1_sentiment_bitcoin", "case2_crypto_bitcoin"):
        decline_terms = [
            "bitcoin",
            "btc",
            "crypto",
            "etf",
            "crash",
            "crashed",
            "slip",
            "support",
            "liquidation",
            "dump",
            "sell-off",
            "selloff",
            "decline",
            "slid",
            "shelved",
        ]
        polar_ok = label in NEGATIVE_LABELS
        topical_ok = text_has_any(text, decline_terms)
        if polar_ok and topical_ok:
            return True, "negative label + crypto/decline terms in retrieved headline"
        if label in POSITIVE_LABELS:
            return False, f"positive label `{label}` on crash query"
        if not topical_ok:
            return False, "no crypto/decline overlap in retrieved headline"
        return False, f"label `{label}` not in negative family"

    if case_id == "case3_fintech_chase":
        if store_status == "STORE_EMPTY":
            return None, "no mortgage/rate rows in fintech corpus"
        fintech_terms = [
            "mortgage",
            "rate",
            "interest",
            "bank",
            "credit",
            "chase",
            "wells fargo",
            "loan",
            "apr",
        ]
        topical_ok = text_has_any(text, fintech_terms)
        polar_ok = label in NEGATIVE_LABELS or label in {"neutral", "mixed"}
        if store_status == "STORE_PARTIAL" and not topical_ok:
            return None, "STORE_PARTIAL — no mortgage-rate row; closest miss"
        if topical_ok and polar_ok and not opposite_mortgage_polarity(text):
            return True, "fintech terms + non-positive complaint polarity"
        if opposite_mortgage_polarity(text):
            return False, "retrieved opposite-sentiment mortgage praise"
        return False, f"topical/polarity miss (label={label})"

    if case_id == "case4_sentiment_chase_wrong_brain":
        if opposite_mortgage_polarity(text) or label in POSITIVE_LABELS:
            return False, "opposite polarity (praise / positive label) at rank 1"
        if label in NEGATIVE_LABELS or label in {"neutral", "mixed"}:
            return True, "non-positive top-1 on rate-hike complaint"
        return False, f"label `{label}`"

    return False, "unhandled case"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="JSON output only")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--round",
        type=str,
        default="1",
        help="Result tag (1, 2, frozen, frozen_v2, …)",
    )
    parser.add_argument(
        "--heldout",
        action="store_true",
        help="Run v1 held-out paraphrase prompts (archival only)",
    )
    parser.add_argument(
        "--heldout-v2",
        action="store_true",
        help="Run v2 held-out prompts (eval_battery_heldout_prompts_v2.jsonl)",
    )
    parser.add_argument(
        "--protocol-frozen",
        action="store_true",
        help="One-shot frozen protocol: v2 held-out + live brain + spacekit corpus",
    )
    parser.add_argument(
        "--corpus",
        choices=("spacekit", "growformer"),
        default="spacekit",
        help="RAG index source (default spacekit — unified with SpaceKit brain train)",
    )
    parser.add_argument(
        "--live-brain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run brain-raw-diag live per case (required for --protocol-frozen)",
    )
    parser.add_argument(
        "--commit-hash",
        default=None,
        help="Git commit for results JSON (default: BET_D_FROZEN_COMMIT or git rev-parse HEAD)",
    )
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.heldout and args.heldout_v2:
        print("Use only one of --heldout or --heldout-v2", file=sys.stderr)
        return 1
    if args.protocol_frozen:
        args.live_brain = True
        args.corpus = "spacekit"

    if args.heldout_v2:
        eval_cases = load_heldout_cases(HELDOUT_V2_PROMPTS_PATH)
        brain_frozen = None
    elif args.heldout:
        eval_cases = load_heldout_cases(HELDOUT_PROMPTS_PATH)
        brain_frozen = BRAIN_RAW_HELDOUT
    else:
        eval_cases = CASES
        brain_frozen = BRAIN_RAW_FROZEN

    if args.protocol_frozen:
        validate_protocol_frozen(args, eval_cases)

    commit_hash = resolve_commit_hash(args.commit_hash)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install: pip install sentence-transformers", file=sys.stderr)
        return 1

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    corpora: dict[str, list[MemoryRow]] = {}
    embeddings: dict[str, np.ndarray] = {}
    for name in ("sentiment", "crypto", "fintech"):
        paths = corpus_globs(name, args.corpus)
        rows = load_corpus(paths)
        corpora[name] = rows
        texts = [r.text for r in rows]
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings[name] = np.asarray(emb, dtype=np.float32)

    results: list[dict[str, Any]] = []
    rag_pass = 0
    rag_pass_at_2_only = 0
    rag_scorable = 0
    brain_pass = 0

    for case in eval_cases:
        cid = case["id"]
        corpus = case["corpus"]
        query = case["query"]
        store = corpus_store_status(corpus, query, args.corpus)
        rows = corpora[corpus]
        q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        hits = top_k(q_emb, embeddings[corpus], rows, args.top_k)
        top1 = hits[0]
        score_id = rag_score_case_id(cid)
        pass_rag, reason = score_rag(score_id, query, top1, store)
        top2 = hits[1] if len(hits) > 1 else None
        pass_rag2: bool | None = None
        pass_rag2_reason = ""
        if top2 is not None:
            pass_rag2, pass_rag2_reason = score_rag(score_id, query, top2, store)

        if pass_rag is not None:
            rag_scorable += 1
            if pass_rag:
                rag_pass += 1
            elif pass_rag2:
                rag_pass_at_2_only += 1

        prompt_in_store = any(r.text.strip().lower() == query.strip().lower() for r in rows)
        expected_store = case.get("expected_store_text") or V2_EXPECTED_STORE_TEXT.get(cid)
        answer_in_store = (
            answer_row_in_corpus(rows, expected_store) if expected_store else None
        )

        if brain_case_eligible(corpus):
            if args.live_brain:
                brain_path, project_path = brain_paths(corpus)
                report = run_brain_raw_diag(brain_path, project_path, query, top_k=3)
                brain_top = brain_top1_from_report(report)
                pass_brain, pass_brain_reason = score_brain_case(cid, query, brain_top, store)
                brain_top["pass_brain"] = pass_brain
                brain_top["pass_brain_reason"] = pass_brain_reason
            else:
                if brain_frozen is None or cid not in brain_frozen:
                    print(f"No frozen brain entry for {cid}; use --live-brain", file=sys.stderr)
                    return 1
                brain_top = dict(brain_frozen[cid])
                pass_brain = bool(brain_top.get("pass_brain", False))
                pass_brain_reason = "frozen dict"
            if pass_brain:
                brain_pass += 1
        else:
            brain_top = {"skipped": True, "reason": BRAIN_SKIP_REASON}
            pass_brain = None
            pass_brain_reason = BRAIN_SKIP_REASON

        entry: dict[str, Any] = {
            "case_id": cid,
            "query": query,
            "corpus": corpus,
            "corpus_mode": args.corpus,
            "corpus_rows": len(rows),
            "store_status": store,
            "prompt_in_store": prompt_in_store,
            "answer_in_store": answer_in_store,
            "encoder": model_name,
            "rag_top1": asdict(top1),
            "rag_topk": [asdict(h) for h in hits],
            "pass_rag": pass_rag,
            "pass_rag_reason": reason,
            "pass_rag_at_2": pass_rag2,
            "pass_rag_at_2_reason": pass_rag2_reason,
            "brain_raw_top1": brain_top,
            "pass_brain": pass_brain,
            "pass_brain_reason": pass_brain_reason,
            "brain_eligible": brain_case_eligible(corpus),
            "live_brain": args.live_brain,
        }
        if top2 is not None:
            entry["rag_top2"] = asdict(top2)
        results.append(entry)

        if not args.json:
            print(f"========== {cid} ==========")
            print(f"query: {query}")
            print(
                f"corpus: {corpus} ({len(rows)} rows, mode={args.corpus}) "
                f"store_status={store}"
            )
            print(
                f"prompt_in_store={prompt_in_store} answer_in_store={answer_in_store}"
            )
            print(
                f"rag #1: score={top1.score:.3f} label={top1.label} id={top1.task_id}\n"
                f"        {top1.text[:140]}"
            )
            print(f"pass_rag@1: {pass_rag} — {reason}")
            if top2 is not None:
                print(
                    f"rag #2: score={top2.score:.3f} label={top2.label} id={top2.task_id}\n"
                    f"        {top2.text[:140]}"
                )
                print(f"pass_rag@2: {pass_rag2} — {pass_rag2_reason}")
            if brain_case_eligible(corpus):
                print(
                    f"brain raw #1: topic={brain_top.get('topic_hint')} "
                    f"score={brain_top.get('score', 0):.3f}\n"
                    f"              {(brain_top.get('text_preview') or '')[:140]}"
                )
                print(f"pass_brain: {pass_brain} — {pass_brain_reason}")
            else:
                print(f"brain: N/A — {pass_brain_reason}")
            print()

    scored_ids = {c["id"] for c in eval_cases if c.get("scorable", True)}
    subset_ids = (
        scored_ids
        if args.heldout_v2 or args.heldout
        else SCORED_BATTERY_CASE_IDS
    )
    brain_eligible_count = sum(
        1 for c in eval_cases if brain_case_eligible(c["corpus"])
    )
    brain_pass_scored = sum(
        1 for e in results if e["case_id"] in subset_ids and e.get("pass_brain") is True
    )
    rag_pass_scored = sum(
        1 for e in results if e["case_id"] in subset_ids and e.get("pass_rag") is True
    )
    rag_pass_scored_at_2 = sum(
        1
        for e in results
        if e["case_id"] in subset_ids
        and (e.get("pass_rag") is True or e.get("pass_rag_at_2") is True)
    )
    rag_scorable_scored = sum(
        1 for e in results if e["case_id"] in subset_ids and e.get("pass_rag") is not None
    )

    if args.heldout_v2:
        decision = {
            "heldout_prompts": str(HELDOUT_V2_PROMPTS_PATH.relative_to(ROOT)),
            "protocol": "BET_D_FROZEN_v2",
            "rag_pass_count": rag_pass,
            "rag_pass_at_2_only_count": rag_pass_at_2_only,
            "rag_scorable_count": rag_scorable,
            "brain_pass_count": brain_pass,
            "rag_pass_scored": rag_pass_scored,
            "rag_pass_scored_at_2": rag_pass_scored_at_2,
            "brain_pass_scored": brain_pass_scored,
            "scored_case_ids": sorted(scored_ids),
            "rule": "Held-out v2: both arms pass all scored prompts => HELDOUT_BOTH_PASS",
            "outcome": None,
        }
        n = len(scored_ids)
        if brain_pass_scored >= n and rag_pass_scored >= n:
            decision["outcome"] = "HELDOUT_BOTH_PASS"
        elif brain_pass_scored >= n:
            decision["outcome"] = "BRAIN_HELDOUT_ONLY"
        elif rag_pass_scored >= n:
            decision["outcome"] = "RAG_HELDOUT_ONLY"
        else:
            decision["outcome"] = "HELDOUT_GAP"
    elif args.heldout:
        decision = {
            "heldout_prompts": str(HELDOUT_PROMPTS_PATH.relative_to(ROOT)),
            "protocol": "heldout_v1_archival",
            "rag_pass_count": rag_pass,
            "rag_pass_at_2_only_count": rag_pass_at_2_only,
            "rag_scorable_count": rag_scorable,
            "brain_pass_count": brain_pass,
            "rule": "v1 held-out archival — do not use for product decisions",
            "outcome": None,
        }
        n = len(scored_ids)
        if brain_pass_scored >= n and rag_pass_scored >= n:
            decision["outcome"] = "HELDOUT_BOTH_PASS"
        elif brain_pass_scored >= n:
            decision["outcome"] = "BRAIN_HELDOUT_ONLY"
        elif rag_pass_scored >= n:
            decision["outcome"] = "RAG_HELDOUT_ONLY"
        else:
            decision["outcome"] = "HELDOUT_GAP"
    else:
        decision = {
            "rag_pass_count": rag_pass,
            "rag_pass_at_2_only_count": rag_pass_at_2_only,
            "rag_scorable_count": rag_scorable,
            "brain_pass_count": brain_pass,
            "brain_eligible_count": brain_eligible_count,
            "brain_eligible_case_ids": sorted(SCORED_BATTERY_CASE_IDS),
            "brain_skip_reason": BRAIN_SKIP_REASON,
            "scored_battery_case_ids": sorted(SCORED_BATTERY_CASE_IDS),
            "rag_pass_scored": rag_pass_scored,
            "rag_pass_scored_at_2": rag_pass_scored_at_2,
            "rag_scorable_scored": rag_scorable_scored,
            "brain_pass_scored": brain_pass_scored,
            "rule_full_battery": "RAG >=3 scorable passes AND brain <=1 => SWITCH_TO_EMBEDDING_RAG",
            "rule_scored_subset": "SpaceKit cases 2–3: both arms pass => HYBRID (exploratory)",
            "outcome_full_battery": None,
            "outcome_scored_subset": None,
        }
        if rag_pass >= 3 and brain_pass <= 1:
            decision["outcome_full_battery"] = "SWITCH_TO_EMBEDDING_RAG"
        elif rag_pass <= 2:
            decision["outcome_full_battery"] = "POPULATE_STORE_FIRST"
        else:
            decision["outcome_full_battery"] = "HYBRID_OR_INCONCLUSIVE"

        if brain_pass_scored >= 2 and rag_pass_scored >= 2:
            decision["outcome_scored_subset"] = "HYBRID_DOMAIN_BRAIN"
        elif brain_pass_scored >= 2:
            decision["outcome_scored_subset"] = "BRAIN_DOMAIN_RETRIEVAL"
        else:
            decision["outcome_scored_subset"] = "RETRIEVAL_GAP"

        decision["outcome"] = decision["outcome_full_battery"]
        decision["rule"] = decision["rule_full_battery"]

    report = {
        "round": args.round,
        "commit_hash": commit_hash,
        "corpus_mode": args.corpus,
        "live_brain": args.live_brain,
        "protocol_frozen": args.protocol_frozen,
        "encoder": model_name,
        "cases": results,
        "decision": decision,
    }

    out_dir = ROOT / "agent-data/brain-rag-baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_output_path(args)
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("========== DECISION ==========")
        if args.heldout_v2 or args.heldout:
            print(
                f"Held-out: RAG@1 {rag_pass}/{rag_scorable} RAG@2-only {rag_pass_at_2_only} | "
                f"Brain {brain_pass}/{len(eval_cases)}"
            )
            print(f"Outcome: {decision['outcome']}")
        else:
            print(
                f"RAG@1 pass: {rag_pass}/{rag_scorable} scorable | "
                f"Brain pass: {brain_pass}/{brain_eligible_count} "
                f"(SpaceKit crypto/fintech only; sentiment cases N/A)"
            )
            print(
                f"Scored subset (cases 2–3): RAG@1 {rag_pass_scored}/{rag_scorable_scored} "
                f"RAG@1|2 {rag_pass_scored_at_2}/{rag_scorable_scored} | "
                f"Brain {brain_pass_scored}/2"
            )
            print(f"Full-battery outcome: {decision['outcome_full_battery']}")
            print(f"Product-scoped outcome: {decision['outcome_scored_subset']}")
        print(f"commit_hash={commit_hash} corpus={args.corpus} live_brain={args.live_brain}")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
