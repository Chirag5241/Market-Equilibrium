// Compact the full-information LLM runs into website/data/llm_full_info.json for the site's results panel.
// Run after llm_full_info.mjs:  node agents/export_for_site.mjs
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const R = path.join(here, "results");
const src = fs.readFileSync(path.join(here, "llm_full_info.mjs"), "utf8");
const objective = (src.match(/const OBJECTIVE_TEXT = `([\s\S]*?)`;/) || [])[1] || "";

// cost per case: ledger lines written after the smoke test, tagged "<case>:..."
const smokeT = Math.max(0, ...["minimal", "objective"].map((c) => { const f = path.join(R, `full_info_${c}_smoke.json`); return fs.existsSync(f) ? fs.statSync(f).mtimeMs : 0; }));
const ledger = fs.existsSync(path.join(R, "llm_ledger.jsonl")) ? fs.readFileSync(path.join(R, "llm_ledger.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)) : [];
const spend = (c) => ledger.filter((q) => Date.parse(q.t) > smokeT && q.tag.startsWith(c + ":")).reduce((a, q) => ({ calls: a.calls + 1, inTok: a.inTok + q.inTok, outTok: a.outTok + q.outTok, reasoning: a.reasoning + q.reasoning, cost: a.cost + q.cost }), { calls: 0, inTok: 0, outTok: 0, reasoning: 0, cost: 0 });

const out = { generated: new Date().toISOString(), goalMinimal: "Trade favorably and increase your utility.", objectiveText: objective, cases: {} };
for (const c of ["minimal", "objective"]) {
  const f = path.join(R, `full_info_${c}.json`);
  if (!fs.existsSync(f)) continue;
  const res = JSON.parse(fs.readFileSync(f, "utf8"));
  out.cfg = { N: res.cfg.N, seed: res.cfg.seed, shock: res.cfg.shock, share: res.cfg.share, rounds: res.cfg.rounds, maxTurns: res.cfg.maxTurns, priceIn: res.cfg.priceIn, priceOut: res.cfg.priceOut, cap: res.cfg.cap, model: res.model };
  const meets = res.rounds.flatMap((r) => r.meetings || []);
  const ex = meets.filter((q) => q.executed);
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  out.cases[c] = {
    stopped: res.stopped,
    F: res.rounds.map((r) => r.F), Fref: res.rounds.map((r) => r.Fref), groupCash: res.rounds.map((r) => r.groupCash),
    stats: {
      meetings: meets.length, withSurplus: meets.filter((q) => q.nashS > 2e-12).length, executed: ex.length,
      blocked: meets.filter((q) => q.d && !q.executed).length, walked: meets.filter((q) => q.ended === "walked").length,
      unreadable: meets.filter((q) => q.ended === "unreadable").length, noAgreement: meets.filter((q) => !q.d && q.ended !== "walked").length,
      surplusVsNash: ex.length ? sum(ex.map((q) => (q.nashS > 0 ? Math.min(1.5, q.S / q.nashS) : 1))) / ex.length : null,
      firstSpeakerShare: ex.length ? sum(ex.map((q) => q.gi / (q.gi + q.gj))) / ex.length : null,
      messages: sum(meets.map((q) => q.turns.length)),
    },
    spend: spend(c),
    rounds: res.rounds.slice(1).map((r) => ({ t: r.t, meetings: r.meetings.map((q) => ({ i: q.i, j: q.j, ended: q.ended, executed: q.executed, nashS: q.nashS, S: q.S, gi: q.gi, gj: q.gj, d: q.d, z: q.z, turns: q.turns.map((t) => ({ s: t.speaker, a: t.action, d: t.d, z: t.z, m: (t.message || "").slice(0, 280) })) })) })),
  };
}
fs.mkdirSync(path.join(here, "../website/data"), { recursive: true });
const dest = path.join(here, "../website/data/llm_full_info.json");
fs.writeFileSync(dest, JSON.stringify(out));
console.log(`wrote ${dest} (${(fs.statSync(dest).size / 1024).toFixed(0)} KB); cases: ${Object.keys(out.cases).join(", ")}`);
for (const [c, v] of Object.entries(out.cases)) console.log(c, JSON.stringify(v.stats), "spend", JSON.stringify(v.spend));
