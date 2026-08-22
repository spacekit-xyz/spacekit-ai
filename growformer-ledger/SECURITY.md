# Security policy

Report suspected integrity, canonicalization, or chain-verification
vulnerabilities privately through the repository's security-reporting channel.
Do not publish exploit details before a fix and disclosure plan are agreed.

Include the affected revision, a minimal deterministic reproduction, impact,
and mitigation.

The ledger detects modifications under its documented assumptions; it is not a
consensus system, signature scheme, remote timestamp authority, or substitute
for independent artifact custody.

Never commit credentials, private datasets, production evaluation ledgers, or
model artifacts. Rotate credentials that enter Git history.
