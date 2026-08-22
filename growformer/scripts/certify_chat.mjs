// Chat-output certifier — applies the project's certifier-first discipline to the
// generative chat path. It loads the SAME wasm engine + companion artifacts the
// browser bundles, runs a HELD-OUT eval set (data/chat_certify/*.jsonl) through
// `growformer_converse`, and scores each line against the companion's own
// `[response_shaping]` / `[validation]` contract:
//
//   - garble / MASK-leak / tokenization collapse  -> hard fail
//   - forbidden_phrases                            -> hard fail
//   - voice_violation_patterns                     -> hard fail
//   - required_signal_present (broad pet voice)    -> fail when expect_signal
//   - length_bounds                                -> reported (soft; see note)
//
// It prints per-category pass rates and an overall PASS/FAIL verdict. This is the
// "measure before you serve" gate for chat, analogous to the encoder certifier.
//
// Usage:
//   node scripts/certify_chat.mjs
//   N=6 EVAL=data/chat_certify/luna_chat_eval.jsonl node scripts/certify_chat.mjs
//   SKIP=fragments node scripts/certify_chat.mjs   # ablate a layer
//   EXTRA_EVAL=path/to/prompts.txt node scripts/certify_chat.mjs
//     — append plain-text prompts (one per line) or JSONL with {prompt|text|phrase}
//       as category=real_traffic (capture holdout). Soft-reported; hard gate still
//       requires authored EVAL + real_traffic garble/forbidden/voice = 0.
import { readFileSync, existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SK = process.env.SK || "/Users/astor/Projects/2026/spacekit";
const LUNA = process.env.COMPANION || `${SK}/spacekit-projects/companions/luna`;
const JS = process.env.JS || `${SK}/spacekit-js`;

const WASM = process.env.WASM || `${JS}/growformer-pkg/growformer_bg.wasm`;
const GLUE = process.env.GLUE || `${JS}/growformer-pkg/growformer.js`;
const BRAIN = process.env.BRAIN || `${LUNA}/agent/luna-v3-3d.bin`;
const EVAL = resolve(process.env.EVAL || `${HERE}/../data/chat_certify/luna_chat_eval.jsonl`);

// [inference] artifact mapping from luna.gf.toml (browser load order).
const INFER_TOML = `${LUNA}/data/inference_pets.toml`;
const GUARDRAILS = `${LUNA}/data/inference_guardrails.jsonl`;
const TOPIC_GRAPH = `${LUNA}/data/knowledge_graph_pet_overlay.toml`;
const FRAGMENTS = `${LUNA}/data/luna_fragments_v2.jsonl`;
const GROUNDING = `${LUNA}/data/pet_world_grounding.toml`;

// ---- Validation contract (mirrors inference_pets.toml) ----
// min is a soft empty/truncation catch only (informational column); the in-voice
// floor is required-signal presence, not raw length. See GROUNDING_LOOP_SPEC §19.4.
const MIN_CHARS = 16;
const MAX_CHARS = 320;
// Garble: training MASK leak, known decode-collapse signatures, and generic
// tokenization artifacts (bracket-period, run-on ellipses with dashes, comma pileups).
const GARBLE = /MASK|\bschedule both\b|puddle brush|\bvibrates\b|\]\.|opinions\.\.\.\.|,,|\.\.\.\.-|\bsleeping minutes\b/i;
const FORBIDDEN = [
  "I'm doing well, thank you for asking", "I'm Pet Companion", "ready to help",
  "micro-brain chatbot", "for dogs, cats, birds, fish", "as an AI", "I cannot",
  "I'm an AI", "overlay merge", "groups added",
].map((s) => s.toLowerCase());
const VOICE_VIOLATION = [
  /\*[^*]+\*/,
  /\bLuna\s+(walks|sits|comes|looks|runs|trots|stretches)\b/,
  /\bshe\s+(walks|sits|purrs|stretches|comes|looks)\b/,
  /^The cat\s+/,
];
// required_signal_patterns (TOML) broadened with [fragment_compose].vocalizations,
// so valid short pet lines ("Mew.", "Nya.") are not falsely flagged.
const VOX = ["mrrp","mrr","chirp","trill","prrp","kek","purr","meow","mew","nya","mrrow","mrrt","brrt","brup","hrrr","huff","reee","yowl","ack","chirrup","prrt","grrk"];
const SIGNAL = [
  new RegExp(`\\b(${VOX.join("|")})\\b`, "i"),
  /\bmy (tail|ear|paw|nose|eyes|whiskers|face|forehead|head|belly|chin)\b/i,
  /\b(slow blink|head[- ]bump|knead|stretch|crouch|pounce|trot|curl|flatten|flick)\b/i,
  /\b(the |your )(bowl|lap|hand|voice|knee|leg|shoulder|floor|pillow)\b/i,
];

function parse(raw) {
  const o = typeof raw === "string" ? JSON.parse(raw) : raw;
  return { text: o?.text ?? "", confidence: o?.confidence ?? 0, template_id: o?.template_id ?? o?.action_type ?? "" };
}
const hasSignal = (t) => SIGNAL.some((re) => re.test(t));
const isGarble = (t) => GARBLE.test(t) || t.trim().length === 0;
const forbiddenHit = (t) => { const l = t.toLowerCase(); return FORBIDDEN.find((p) => l.includes(p)) || null; };
const voiceHit = (t) => VOICE_VIOLATION.find((re) => re.test(t)) || null;

// ---- Engine bootstrap ----
const gf = await import(pathToFileURL(GLUE).href);
gf.initSync({ module: readFileSync(WASM) });
gf.growformer_init();
gf.growformer_load_brain(new Uint8Array(readFileSync(BRAIN)));
if (!gf.growformer_ready()) throw new Error("brain not ready");
const info = typeof gf.growformer_brain_info() === "string" ? JSON.parse(gf.growformer_brain_info()) : gf.growformer_brain_info();

const skip = new Set((process.env.SKIP || "").split(",").map((s) => s.trim()).filter(Boolean));
function load(name, path, fn) {
  if (skip.has(name)) { console.log(`SKIP ${name}`); return; }
  if (!existsSync(path)) { console.log(`MISSING ${name}: ${path}`); return; }
  if (typeof fn !== "function") { console.log(`NO GLUE FN for ${name}`); return; }
  try { fn(readFileSync(path, "utf-8")); } catch (e) { console.log(`${name} load FAILED:`, String(e)); }
}
load("inference_toml", INFER_TOML, gf.growformer_load_inference_toml);
load("guardrails", GUARDRAILS, gf.growformer_load_inference_guardrails_jsonl);
load("topic_graph", TOPIC_GRAPH, (t) => gf.growformer_load_topic_graph(t));
load("fragments", FRAGMENTS, gf.growformer_load_fragments_jsonl);
load("grounding", GROUNDING, gf.growformer_load_grounding_graph);

// ---- Run eval ----
const N = Number(process.env.N || 6);
const rows = readFileSync(EVAL, "utf-8").split("\n").map((l) => l.trim()).filter(Boolean).map((l) => JSON.parse(l));

function loadExtraEval(path) {
  if (!path || !existsSync(path)) return [];
  const out = [];
  for (const line of readFileSync(path, "utf-8").split("\n").map((l) => l.trim()).filter(Boolean)) {
    if (line.startsWith("#")) continue;
    if (line.startsWith("{")) {
      try {
        const o = JSON.parse(line);
        const prompt = (o.prompt || o.text || o.phrase || "").trim();
        // Capture holdout: hard-gate garble/forbidden/voice; signal is optional unless set.
        if (prompt) out.push({ prompt, category: o.category || "real_traffic", expect_signal: !!o.expect_signal });
      } catch {
        /* skip bad json line */
      }
    } else {
      out.push({ prompt: line, category: "real_traffic", expect_signal: false });
    }
  }
  return out;
}
const extraPath = process.env.EXTRA_EVAL || "";
const extraRows = loadExtraEval(extraPath);
if (extraRows.length) rows.push(...extraRows);

console.log(`brain=${BRAIN.split("/").pop()} agent=${info.agent_name} wasm=${WASM.split("/").pop()} eval=${EVAL.split("/").pop()} prompts=${rows.length} N=${N} skip=[${[...skip].join(",")}]${extraRows.length ? ` extra=${extraRows.length}` : ""}\n`);

// Canned fallback lines route through `[[rules.lattice_misfire_fallback]]`; a spike
// after an enforcement change signals the gate is over-triggering on good output.
const FALLBACK_TEMPLATES = new Set(["training_fallback", "identity_fallback", "grounding_fallback", "bedtime_fallback", "general_comfort_fallback"]);

const cats = {};
const fails = [];
let total = 0;
for (const row of rows) {
  const cat = row.category || "uncategorized";
  cats[cat] ??= { n: 0, garble: 0, forbidden: 0, voice: 0, noSignal: 0, len: 0, fallback: 0, pass: 0 };
  for (let i = 0; i < N; i++) {
    gf.growformer_reset_conversation();
    const r = parse(gf.growformer_converse(row.prompt));
    total++; cats[cat].n++;
    const g = isGarble(r.text);
    const f = forbiddenHit(r.text);
    const v = voiceHit(r.text);
    const sig = hasSignal(r.text);
    const lenBad = r.text.length < MIN_CHARS || r.text.length > MAX_CHARS;
    if (g) cats[cat].garble++;
    if (f) cats[cat].forbidden++;
    if (v) cats[cat].voice++;
    if (row.expect_signal && !sig) cats[cat].noSignal++;
    if (lenBad) cats[cat].len++;
    if (FALLBACK_TEMPLATES.has(r.template_id)) cats[cat].fallback++;
    // Hard gate: garble/forbidden/voice/missing-signal. Length is reported, not gated.
    const hardPass = !g && !f && !v && !(row.expect_signal && !sig);
    if (hardPass) cats[cat].pass++;
    else fails.push({ prompt: row.prompt, cat, conf: r.confidence, reason: g ? "GARBLE" : f ? `FORBIDDEN(${f})` : v ? "VOICE" : "NO_SIGNAL", text: r.text });
  }
}

// ---- Report ----
const pct = (a, b) => (b ? ((100 * a) / b).toFixed(1) : "  -  ").padStart(5);
let agg = { n: 0, garble: 0, forbidden: 0, voice: 0, noSignal: 0, len: 0, fallback: 0, pass: 0 };
console.log("category        n   pass%  garble forbid voice noSig  len>bnd  fallbck");
console.log("-------------- ---  -----  ------ ------ ----- -----  -------  -------");
for (const [cat, c] of Object.entries(cats)) {
  for (const k of Object.keys(agg)) agg[k] += c[k];
  console.log(`${cat.padEnd(14)} ${String(c.n).padStart(3)}  ${pct(c.pass, c.n)}  ${String(c.garble).padStart(6)} ${String(c.forbidden).padStart(6)} ${String(c.voice).padStart(5)} ${String(c.noSignal).padStart(5)}  ${String(c.len).padStart(7)}  ${String(c.fallback).padStart(7)}`);
}
console.log("-------------- ---  -----  ------ ------ ----- -----  -------  -------");
console.log(`${"OVERALL".padEnd(14)} ${String(agg.n).padStart(3)}  ${pct(agg.pass, agg.n)}  ${String(agg.garble).padStart(6)} ${String(agg.forbidden).padStart(6)} ${String(agg.voice).padStart(5)} ${String(agg.noSignal).padStart(5)}  ${String(agg.len).padStart(7)}  ${String(agg.fallback).padStart(7)}`);

if (fails.length) {
  console.log(`\n${fails.length} hard failures (showing up to 20):`);
  for (const f of fails.slice(0, 20)) console.log(`  [${f.reason}] (${f.cat}) conf=${f.conf.toFixed(3)} "${f.prompt}" -> "${f.text}"`);
}

const passRate = (100 * agg.pass) / agg.n;
const THRESHOLD = Number(process.env.THRESHOLD || 100);
const verdict = agg.garble === 0 && agg.forbidden === 0 && agg.voice === 0 && passRate >= THRESHOLD ? "PASS" : "FAIL";
console.log(`\n==> CHAT CERTIFY: ${verdict}  pass=${passRate.toFixed(1)}%  (gate: garble=0 forbidden=0 voice=0 pass>=${THRESHOLD}%)`);
process.exit(verdict === "PASS" ? 0 : 1);
