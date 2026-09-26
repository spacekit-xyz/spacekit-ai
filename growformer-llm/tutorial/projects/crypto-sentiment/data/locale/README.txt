Locale-specific sentiment training data (Phase 2).

Place JSONL shards here, e.g.:
  es/train_sentiment_crypto.jsonl
  fr/train_sentiment_crypto.jsonl

Train a brain per locale (or one multilingual encoder + single brain) and load the matching
blob at inference. Declarative rescore tables can mirror the English file as
  growformer/data/inference/sentiment_crypto_rescore.es.toml
See growformer/docs/RETRIEVAL_EXTENSIBILITY.md.

Retrieval stopwords / intent tokens for non-English UI: add matching [locales.<lang>] in
growformer/data/inference/retrieval_lexicon.toml (same keys as [locales.en]).
