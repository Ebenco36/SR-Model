# Dataset scaffold generated from corpus_txt.zip

This folder contains **robust scaffolding datasets** to bootstrap your pipeline.
They are intentionally **silver/weakly-labeled** and meant for:
- rapid prototyping of NER + QA training code
- pre-labeling for human correction in Doccano/Label Studio
- data generation experiments later

## Files

- `docs_raw.jsonl`: one record per paper with `doc_id` and `text`.
- `splits.json`: paper-level train/dev/test split (by `doc_id`).
- `ner_silver_v0.jsonl`: span-based NER prelabels (character offsets).
  - format: `{doc_id, text, spans:[{start,end,label}]}`
- `qa_silver_v0.jsonl`: SQuAD-like extractive QA examples.
  - format: `{id, doc_id, question, context, answers:{text:[...], answer_start:[...]}}`

## Notes
- Prelabels are heuristic and may contain false positives (especially COUNTRY).
- Use these to validate the training loop first; then replace with gold.
