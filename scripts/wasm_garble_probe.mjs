// Probe: does the REBUILT growformer_bg.wasm (the artifact the browser bundles) garble
// "bad cat" on the multi-turn converse path? Isolates engine from deployed brain/data.
import { readFileSync, existsSync } from "node:fs";
import { pathToFileURL } from "node:url";

const SK = "/Users/astor/Projects/2026/spacekit";
const LUNA = `${SK}/spacekit-projects/companions/luna`;
const JS = `${SK}/spacekit-js`;

const WASM = `${JS}/growformer-pkg/growformer_bg.wasm`;
const GLUE = `${JS}/growformer-pkg/growformer.js`;
const BRAIN = process.env.BRAIN || `${LUNA}/agent/luna-v3-3d.bin`;
// Canonical [inference] artifact mapping from luna.gf.toml (browser load order).
const INFER_TOML = `${LUNA}/data/inference_pets.toml`;
const GUARDRAILS = `${LUNA}/data/inference_guardrails.jsonl`;
const TOPIC_GRAPH = `${LUNA}/data/knowledge_graph_pet_overlay.toml`;
const FRAGMENTS = `${LUNA}/data/luna_fragments_v2.jsonl`;
const GROUNDING = `${LUNA}/data/pet_world_grounding.toml`;

const GARBLE = /MASK|\bschedule both\b|puddle brush|\bvibrates\b|\]\.|opinions/i;

function parse(raw) {
  const o = typeof raw === "string" ? JSON.parse(raw) : raw;
  return { text: o?.text ?? "", confidence: o?.confidence ?? 0, action_type: o?.action_type ?? "" };
}

const gf = await import(pathToFileURL(GLUE).href);
gf.initSync({ module: readFileSync(WASM) });
gf.growformer_init();
gf.growformer_load_brain(new Uint8Array(readFileSync(BRAIN)));
if (!gf.growformer_ready()) throw new Error("brain not ready");

const info = typeof gf.growformer_brain_info() === "string" ? JSON.parse(gf.growformer_brain_info()) : gf.growformer_brain_info();
console.log(`brain=${BRAIN.split("/").pop()} agent=${info.agent_name} groups=${info.num_groups} profile=${info.inference_profile ?? "(none)"}`);

// Load full [inference] artifact set in the same order AgentHub uses.
// Set SKIP="guardrails,fragments" etc. to ablate individual layers.
const skip = new Set((process.env.SKIP || "").split(",").map((s) => s.trim()).filter(Boolean));
function load(name, path, fn) {
  if (skip.has(name)) { console.log(`SKIP ${name}`); return; }
  if (!existsSync(path)) { console.log(`MISSING ${name}: ${path}`); return; }
  if (typeof fn !== "function") { console.log(`NO GLUE FN for ${name}`); return; }
  try { fn(readFileSync(path, "utf-8")); console.log(`loaded ${name}`); }
  catch (e) { console.log(`${name} load FAILED:`, String(e)); }
}
load("inference_toml", INFER_TOML, gf.growformer_load_inference_toml);
load("guardrails", GUARDRAILS, gf.growformer_load_inference_guardrails_jsonl);
load("topic_graph", TOPIC_GRAPH, (t) => gf.growformer_load_topic_graph(t));
load("fragments", FRAGMENTS, gf.growformer_load_fragments_jsonl);
load("grounding", GROUNDING, gf.growformer_load_grounding_graph);

const N = Number(process.env.N || 6);
const PROMPTS = (process.env.PROMPTS || "bad cat|who are you|i feel sad|good kitty|what are you doing").split("|");
let totalGarble = 0;
for (const prompt of PROMPTS) {
  gf.growformer_reset_conversation();
  console.log(`\n--- growformer_converse("${prompt}") x${N} ---`);
  for (let i = 1; i <= N; i++) {
    const r = parse(gf.growformer_converse(prompt));
    const bad = GARBLE.test(r.text);
    if (bad) totalGarble++;
    console.log(`[${i}]${bad ? " GARBLE" : "      "} conf=${r.confidence.toFixed(3)} "${r.text}"`);
  }
}
console.log(`\n==> TOTAL GARBLE: ${totalGarble}  (wasm=${WASM.split("/").pop()}, skip=[${[...skip].join(",")}])`);
