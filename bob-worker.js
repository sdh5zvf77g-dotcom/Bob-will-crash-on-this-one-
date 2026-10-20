/**
 * Bob v4 Web Worker — background engine
 * Handles: text chunking helpers, rule consolidation, profile cleanup,
 * and (when available) embedding calculations off the main thread.
 */

let embeddingPipeline = null;
let embeddingReady = false;
let embeddingError = null;

// Lightweight cosine similarity
function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom ? dot / denom : 0;
}

async function ensureEmbedder() {
  if (embeddingReady) return true;
  if (embeddingError) return false;
  try {
    // Dynamic import inside worker
    const { pipeline, env } = await import("https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2");
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    // Prefer WASM in worker for broader compatibility; WebGPU if available
    embeddingPipeline = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
      quantized: true,
      progress_callback: (p) => {
        self.postMessage({ type: "embed_progress", data: p });
      }
    });
    embeddingReady = true;
    self.postMessage({ type: "embed_ready" });
    return true;
  } catch (err) {
    embeddingError = String(err.message || err);
    self.postMessage({ type: "embed_error", error: embeddingError });
    return false;
  }
}

async function embedTexts(texts) {
  const ok = await ensureEmbedder();
  if (!ok || !embeddingPipeline) return null;
  const vectors = [];
  for (const t of texts) {
    const out = await embeddingPipeline(t, { pooling: "mean", normalize: true });
    // out.data is Float32Array-like
    vectors.push(Array.from(out.data));
  }
  return vectors;
}

function consolidateRules(rules) {
  // Deduplicate near-identical rules and drop very short / empty ones
  const cleaned = [];
  const seen = new Set();
  for (const r of rules) {
    const text = (typeof r === "string" ? r : r?.text || "").trim();
    if (text.length < 12) continue;
    const key = text.toLowerCase().replace(/\s+/g, " ").slice(0, 120);
    if (seen.has(key)) continue;
    seen.add(key);
    cleaned.push(text);
  }
  return cleaned.slice(0, 40);
}

function scoreKeyword(query, text) {
  const q = query.toLowerCase();
  const terms = q.split(/[^a-z0-9\u0e00-\u0e7f]+/).filter(t => t.length > 2);
  if (!terms.length) return 0;
  const t = (text || "").toLowerCase();
  let score = 0;
  for (const term of terms) {
    if (t.includes(term)) score += 2;
    const m = t.match(new RegExp(term, "gi"));
    if (m) score += Math.min(m.length, 3);
  }
  return score;
}

self.onmessage = async (e) => {
  const { id, type, payload } = e.data || {};
  try {
    if (type === "ping") {
      self.postMessage({ id, type: "pong" });
      return;
    }

    if (type === "init_embedder") {
      const ok = await ensureEmbedder();
      self.postMessage({ id, type: "init_embedder_done", ok });
      return;
    }

    if (type === "embed") {
      const vectors = await embedTexts(payload.texts || []);
      self.postMessage({ id, type: "embed_result", vectors, ok: !!vectors });
      return;
    }

    if (type === "semantic_search") {
      const { query, chunks, queryVector } = payload;
      // chunks: [{ id, text, vector? }]
      let results = [];
      if (queryVector && chunks.some(c => c.vector)) {
        results = chunks
          .filter(c => c.vector)
          .map(c => ({ ...c, score: cosineSim(queryVector, c.vector) }))
          .filter(c => c.score > 0.25)
          .sort((a, b) => b.score - a.score)
          .slice(0, payload.max || 5);
      }
      // Fallback / hybrid with keyword
      if (!results.length) {
        results = chunks
          .map(c => ({ ...c, score: scoreKeyword(query, c.text) }))
          .filter(c => c.score > 0)
          .sort((a, b) => b.score - a.score)
          .slice(0, payload.max || 5);
      }
      self.postMessage({ id, type: "semantic_search_result", results });
      return;
    }

    if (type === "consolidate_rules") {
      const cleaned = consolidateRules(payload.rules || []);
      self.postMessage({ id, type: "consolidate_rules_result", rules: cleaned });
      return;
    }

    if (type === "compress_profile_hint") {
      // Worker only prepares a compact transcript; main thread still runs the LLM
      const msgs = payload.messages || [];
      const transcript = msgs.map(m => {
        const who = m.role === "user" ? "User" : (m.agent || "Bob");
        return `${who}: ${String(m.content || "").slice(0, 350)}`;
      }).join("\n").slice(0, 5500);
      self.postMessage({ id, type: "compress_profile_hint_result", transcript });
      return;
    }

    self.postMessage({ id, type: "error", error: "Unknown task: " + type });
  } catch (err) {
    self.postMessage({ id, type: "error", error: String(err.message || err) });
  }
};

self.postMessage({ type: "worker_ready" });
