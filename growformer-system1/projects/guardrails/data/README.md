Put labelled JSONL here, one check per line, e.g.

{"state": {"stage": "output", "prompt": "...", "content": "..."}, "labels": {"policy_violation": "none", "sensitive_data": false, "response_quality": 3}}

Label only the questions in that stage's group. The best source is your own traffic: run
`gf-system1 serve --log decisions.jsonl --log-state`, review the logged checks, and save
the corrected ones here. Then `eval` the open model on them, and `train` / `train-probe`
once you have a few hundred.
