# Crazy Bob v5.1 — 1B Learning Upgrade

Crazy Bob is a GitHub Pages-compatible, installable iPhone web app for local AI inference using WebLLM/WebGPU. This update preserves the existing dashboard and existing feature set while improving the reasoning instructions and adding an evidence-aware learning ledger.

## Preserved

- Existing dashboard, styling, layout, controls and install icons
- WebLLM local inference and model selector
- Llama 3.2 1B remains available and is now the default selection
- Smaller models remain available for devices that need them; 1.5B and 3B options remain in the selector
- Persistent chat history, user profile, working memory and episodic memory
- Uploaded-document retrieval, chunking, keyword/semantic search and RRF fusion
- Planner → Writer → Reviewer team mode
- Voice input/output and voice settings
- Existing search actions: news, research, web search, weather, currency, crypto, define, translate and URL summarisation
- API key settings and test buttons
- Proactive goals, skills schemas, backup, model restore and sleep/idle cycle
- GitHub Pages relative asset paths and PWA icons/manifest

## New in v5.1

### Evidence-aware learning ledger
- Stores up to 80 compact lessons in the existing IndexedDB metadata store; no destructive database migration.
- Learns from clear user feedback such as “that worked”, “that helped”, “still broken” or “that is wrong”.
- Stores the feedback outcome, a short topic, evidence excerpt, timestamp and confidence.
- Retrieves lessons relevant to a future query and provides them to Bob as prior experience, not unquestionable facts.
- Does not automatically assume that an answer was correct merely because it was generated.

### Better problem-solving instructions
- Replaces requests to output hidden `<think>` tags with a concise problem-solving standard: understand the goal, use evidence, check assumptions, verify where possible and communicate uncertainty.
- Simple questions can still be answered directly; difficult tasks are prompted to use a more structured approach.
- Adds an empty-response fallback so a blank model result is not silently saved as a normal answer.

### Default model
- Llama 3.2 1B q4f16 is the default selection.
- If a remembered 3B model is detected on startup, automatic restore falls back to the 1B model to reduce iPhone crash risk.
- Users can still manually choose another listed model.

## Important technical boundary

Bob remains a **1B-parameter language model when Llama 3.2 1B is selected**. Memory, retrieval, structured prompting and feedback can improve usefulness over time, but they do not update the model's neural-network weights. Actual weight improvement requires a separate fine-tuning/training pipeline and a compatible exported model. The learning ledger is intentionally small, local and reversible by clearing site data; it is not claimed to be AGI or autonomous model training.

## Deploy to GitHub Pages

1. Extract the ZIP.
2. Upload the contents of the `crazy-bob-v5` folder to the root of your GitHub Pages repository (so `index.html`, `bob-worker.js`, `manifest.json` and the icon files are at the published root).
3. In GitHub, open **Settings → Pages** and select the correct branch and root folder.
4. Open the published HTTPS site in Safari on iPhone. Add to Home Screen if desired.
5. Load Llama 3.2 1B. Initial model download needs a reliable connection and browser/device WebGPU support; after download it may be cached by the browser.

## Device notes

- Local model inference requires a compatible browser and WebGPU. Browser cache can be cleared by iOS storage management, and background work may pause when Safari is suspended.
- The embedding model is optional; keyword retrieval remains the fallback if embeddings cannot load.
- API keys entered for search providers are stored in local browser storage for this app. Treat the device as private and do not publish a repository containing personal keys.

## Verification status

The included JavaScript was syntax-checked with Node.js and static checks were run for duplicate HTML IDs, referenced local assets and PWA manifest validity. This is not a claim of full live-device/browser QA: the final GitHub Pages deployment and actual WebGPU model download must still be checked on the target iPhone.


## v5.2: training and evaluation
The `TRAINING/` directory contains an explicit, offline fine-tuning workflow for a compatible 1B checkpoint, dataset preparation from Bob's private memory export, baseline/tuned comparison and merge scripts. See `TRAINING/README.md`. This does not alter the dashboard or silently replace the WebLLM model. A fine-tuned Hugging Face model requires a separate MLC/WebLLM conversion before it can run in the browser.
