#!/usr/bin/env python3
"""Drain browser RealTraffic captures from a spacekit storage node into traffic JSONL.

§18.2 last hop: the Agent Hub WASM path POSTs capture batches as DID-authed documents to the
storage node's `growformer_capture` collection (one document per flush, under a single shared
capture-service DID so one listing returns everything). This tool lists that collection, flattens
each document's `records`, groups by agent, and writes `capture_artifacts/traffic_web_<agent>.jsonl`
in the exact schema `growformer-demos --audit-capture` reads.

It writes `traffic_web_<agent>.jsonl` (a full snapshot, overwritten each run) so it never clobbers
CLI captures in `traffic_<agent>.jsonl`; `--audit-capture` reads every `traffic_*.jsonl`.

Usage:
  python scripts/drain_capture.py --storage-url https://node.example \
      [--did did:spacekit:growformer-capture] [--collection growformer_capture] \
      [--out-dir capture_artifacts]
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path


def fetch_documents(storage_url, collection, did):
    url = f"{storage_url.rstrip('/')}/api/documents/{collection}"
    req = urllib.request.Request(url, headers={"Authorization": f"DID {did}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    docs = payload.get("documents", [])
    if not isinstance(docs, list):
        raise ValueError(f"unexpected list response shape: {type(docs)}")
    return docs


def safe_agent(name):
    s = re.sub(r"[^A-Za-z0-9_-]", "_", name or "")
    return s or "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storage-url", required=True, help="storage node base URL")
    ap.add_argument("--did", default="did:spacekit:growformer-capture", help="shared capture-service DID")
    ap.add_argument("--collection", default="growformer_capture")
    ap.add_argument("--out-dir", default="capture_artifacts")
    args = ap.parse_args()

    try:
        docs = fetch_documents(args.storage_url, args.collection, args.did)
    except Exception as e:  # noqa: BLE001 - surface any transport/parse failure plainly
        print(f"Failed to list {args.collection} from {args.storage_url}: {e}", file=sys.stderr)
        sys.exit(1)

    # Group records by agent, dedup by (phrase, timestamp, session) to absorb re-flushed batches.
    by_agent = {}
    seen = set()
    n_docs = 0
    n_records = 0
    for doc in docs:
        n_docs += 1
        data = doc.get("data", {}) if isinstance(doc, dict) else {}
        records = data.get("records", []) if isinstance(data, dict) else []
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            phrase = (rec.get("phrase") or "").strip()
            if not phrase:
                continue
            agent = rec.get("agent") or data.get("agent") or "unknown"
            key = (agent, phrase, rec.get("timestamp_unix"), rec.get("session_id"))
            if key in seen:
                continue
            seen.add(key)
            by_agent.setdefault(agent, []).append(rec)
            n_records += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for agent, recs in sorted(by_agent.items()):
        path = out_dir / f"traffic_web_{safe_agent(agent)}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  wrote {len(recs):>6} records -> {path}")

    print(
        f"\nDrained {n_docs} documents -> {n_records} unique records across {len(by_agent)} agents."
    )
    if n_records:
        print(f"Next: growformer-demos --audit-capture {args.out_dir} <companion_dir>  # triage / bucketing")
    else:
        print("No records yet — capture accumulates as users chat through Agent Hub.")


if __name__ == "__main__":
    main()
