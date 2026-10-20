# Changelog

## v5.1 — Learning and reasoning improvements
- Added an evidence-aware, capped learning ledger in existing IndexedDB metadata.
- Added explicit positive/negative feedback capture and relevance-based lesson retrieval.
- Updated system and team prompts to favour structured problem-solving, evidence checks and uncertainty over exposed `<think>` tags.
- Added a blank-generation fallback.
- Made Llama 3.2 1B the default model and the fallback for auto-restoring a previously selected 3B model.
- Kept existing UI, controls, search actions, document retrieval, memory, voice, goals, and agent team features.

## v5.2 — Training pipeline and portable learning ledger
- Full memory export/import now includes Bob's capped explicit-feedback learning ledger.
- Added separate TRAINING pipeline: dataset preparation, LoRA/QLoRA fine-tuning, baseline/candidate generation, side-by-side comparison, and adapter merge.
- Added held-out starter evaluation tasks and privacy/licensing/deployment guidance.
- Existing dashboard and browser model selection remain unchanged; tuned weights are not automatically deployed.
