# Crazy Bob 1B — measured improvement and fine-tuning pipeline

This pipeline is separate from the iPhone app. It keeps Bob's web dashboard and local WebLLM runtime unchanged while allowing you to prepare privacy-controlled examples, fine-tune a compatible 1B Hugging Face checkpoint with LoRA, compare baseline and tuned responses, and export a merged model for a later WebLLM/MLC conversion.

## Important reality check

- Memory and prompts can improve Bob's *system behaviour* but do not change neural weights.
- This pipeline can change weights only after you run training on compatible hardware with the model weights available and any required model licence accepted.
- The official Llama 3.2 1B checkpoint may require accepting Meta's licence and authenticating with Hugging Face. This folder does not contain model weights or credentials.
- Fine-tuning is not guaranteed to improve general intelligence. The held-out evaluation must be reviewed; do not ship a worse checkpoint.
- A merged Hugging Face model is **not automatically a WebLLM browser model**. MLC conversion/compilation is a separate compatibility step. The current GitHub Pages app continues to use its existing WebLLM model IDs until a converted model is compiled, hosted and explicitly integrated.
- Never upload personal chat exports or API keys to a public GitHub repository. Prepare data locally and remove identifying/private material.

## End-to-end workflow

Use Python 3.10–3.11 recommended, a CUDA-capable NVIDIA GPU with enough VRAM, and a clean private machine. NVIDIA CUDA uses 4-bit QLoRA; Apple Silicon uses standard fp16 LoRA on MPS because bitsandbytes 4-bit quantization is CUDA-only. Speed and memory depend on sequence length and batch settings. CPU-only training is intentionally not enabled by default.

```bash
cd TRAINING
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Export data from Bob

On the iPhone app, open Settings → **Export full memory (JSON)**. Transfer the JSON to your private training computer by a method you trust. It includes chat, profile, rules, events, documents metadata and the explicit-feedback learning ledger; treat it as private data.

### 2. Prepare a reviewed dataset

```bash
python prepare_dataset.py --input /path/to/crazy-bob-memory-YYYY-MM-DD.json --out data --min-answer-chars 40
```

The script uses only assistant turns followed by clear positive user feedback for supervised examples. It excludes turns followed by negative feedback, removes system messages, strips obvious secrets, deduplicates and writes a train/validation split. **Inspect `data/review.jsonl` and `data/train.jsonl` before training.** Automated secret filtering is imperfect; manually remove sensitive data. The tool does not train directly on every private conversation.

If there are fewer than 20 usable examples, collect more explicit feedback or create a carefully reviewed domain-specific dataset. More examples are not automatically better.

### 3. Baseline evaluation

```bash
python evaluate.py --model meta-llama/Llama-3.2-1B-Instruct --tasks eval/tasks.jsonl --out results/baseline.jsonl
```

You may need `huggingface-cli login` (or `hf auth login`) after accepting the model licence. `eval/tasks.jsonl` is a small starter smoke test, not a scientifically sufficient benchmark. Add held-out tasks relevant to Bob, and never use evaluation examples for training.

### 4. Fine-tune with LoRA/QLoRA

```bash
python train_lora.py --model meta-llama/Llama-3.2-1B-Instruct --data data/train.jsonl --validation data/validation.jsonl --out outputs/bob-1b-lora
```

The default is conservative. Adjust `--epochs`, `--lr`, `--max-length`, and `--batch-size` only after checking GPU memory and validation results. Stop if validation loss worsens consistently. Keep the original checkpoint untouched.

### 5. Compare baseline against tuned adapter

```bash
python evaluate.py --model meta-llama/Llama-3.2-1B-Instruct --adapter outputs/bob-1b-lora --tasks eval/tasks.jsonl --out results/tuned.jsonl
python compare_results.py --baseline results/baseline.jsonl --candidate results/tuned.jsonl
```

This compares outputs side by side; it does not pretend a language model can objectively score every open-ended answer. Review correctness, evidence, safety, instruction following and regressions manually. Keep the adapter only if it improves the target tasks without unacceptable regressions.

### 6. Merge weights for conversion experiments

```bash
python merge_adapter.py --base meta-llama/Llama-3.2-1B-Instruct --adapter outputs/bob-1b-lora --out outputs/bob-1b-merged
```

Keep the original and adapter as rollback copies. Do not overwrite the baseline.

### 7. Make a browser model (separate deployment step)

The app's current `modelSelect` IDs refer to prebuilt WebLLM/MLC models. To use tuned weights in the browser, follow the current MLC LLM documentation to convert/compile the merged checkpoint to a supported model library and quantization, host its model artifacts on a suitable static/CDN host, and integrate the exact model library URL/config into WebLLM. Confirm browser memory use, WebGPU compatibility, cache size and licence terms. GitHub Pages may not be suitable for very large model artifacts. Do not change the production model ID until the compiled model loads and passes on-device tests. Until then, the trained checkpoint is a research artifact and the app remains fully usable with its existing 1B model.

## Data format

Each JSONL line uses `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`. Training code accepts only user/assistant messages. Do not include system prompts, credentials, hidden reasoning, or private data.

## What this pipeline does not do

It does not run continuous autonomous training on an iPhone, does not guarantee AGI, and does not silently replace the live model. Training is explicit, versioned and evaluated on separate hardware. This is safer, reproducible, and easier to roll back.
