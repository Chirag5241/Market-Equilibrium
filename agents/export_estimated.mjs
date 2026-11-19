// Compact the estimated-utility runs into website/data/llm_estimated.json for the site's results panel.
// Run after llm_estimated.mjs:  node agents/export_estimated.mjs
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const R = path.join(here, "results");
const read = (f) => (fs.existsSync(path.join(R, f)) ? JSON.parse(fs.readFileSync(path.join(R, f), "utf8")) : null);
const sum = (a) => a.reduce((s, v) => s + v, 0), mean = (a) => (a.length ? sum(a) / a.length : null);

const out = { generated: new Date().toISOString() };
const P = read("est_particle.json");
if (P) {
  const meets = P.meetings.flat();
  out.particle = {
    F: P.rounds.map((r) => r.F), Fref: P.rounds.map((r) => r.Fref), belief: P.rounds.map((r) => r.belief), groupCash: P.rounds.map((r) => r.groupCash),
    stats: { meetings: meets.length, executed: meets.filter((q) => q.executed).length, proposals: P.rounds[P.rounds.length - 1].proposals, rejects: P.rounds[P.rounds.length - 1].rejects, silent: meets.filter((q) => !q.props).length },
  };
  out.cfg = { N: P.cfg.N, seed: P.cfg.seed, shock: P.cfg.shock, share: P.cfg.share, rounds: P.cfg.rounds, maxProposals: P.cfg.maxProposals, particles: P.cfg.particles, cap: P.cfg.cap, priceIn: P.cfg.priceIn, priceOut: P.cfg.priceOut };
}
const L = read("est_llm.json");
if (L) {
  const smokeT = fs.existsSync(path.join(R, "est_llm_smoke.json")) ? fs.statSync(path.join(R, "est_llm_smoke.json")).mtimeMs : 0;
  const ledger = fs.existsSync(path.join(R, "llm_ledger_estimated.jsonl")) ? fs.readFileSync(path.join(R, "llm_ledger_estimated.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)).filter((q) => Date.parse(q.t) > smokeT) : [];
  const meets = L.rounds.flatMap((r) => r.meetings || []), ex = meets.filter((q) => q.executed);
  const lastEst = (q) => [...q.props].reverse().find((p) => p.est && isFinite(p.est.lam) && p.est.lam > 0);
  const errs = meets.map((q) => { const p = lastEst(q); return p ? { lam: Math.abs(Math.log(p.est.lam / q.truth.lam)), d: Math.abs(p.est.d - q.truth.d), tau: Math.abs(p.est.tau - q.truth.tau) } : null; }).filter(Boolean);
  out.llmEstimate = {
    stopped: L.stopped, model: L.model,
    F: L.rounds.map((r) => r.F), groupCash: L.rounds.map((r) => r.groupCash),
    belief: L.rounds.map((r) => { const es = (r.meetings || []).map((q) => { const p = lastEst(q); return p ? Math.abs(Math.log(p.est.lam / q.truth.lam)) : null; }).filter((v) => v !== null); return es.length ? mean(es) : null; }),
    stats: {
      meetings: meets.length, withSurplus: meets.filter((q) => q.nashS > 2e-12).length, executed: ex.length,
      proposals: sum(meets.map((q) => q.props.filter((p) => !p.silent).length)), estimates: sum(meets.map((q) => q.props.length)),
      firstAccepted: meets.filter((q) => q.props[0] && q.props[0].ok).length,
      surplusVsNash: ex.length ? mean(ex.map((q) => (q.nashS > 0 ? Math.min(1.5, q.S / q.nashS) : 1))) : null,
      proposerShare: ex.length ? mean(ex.map((q) => q.gi / (q.gi + q.gj))) : null,
      errLam: mean(errs.map((v) => v.lam)), errD: mean(errs.map((v) => v.d)), errTau: mean(errs.map((v) => v.tau)),
    },
    spend: ledger.reduce((a, q) => ({ calls: a.calls + 1, inTok: a.inTok + q.inTok, outTok: a.outTok + q.outTok, cost: a.cost + q.cost }), { calls: 0, inTok: 0, outTok: 0, cost: 0 }),
    rounds: L.rounds.slice(1).map((r) => ({ t: r.t, meetings: r.meetings.map((q) => ({ i: q.i, j: q.j, executed: q.executed, nashS: q.nashS, S: q.S, gi: q.gi, gj: q.gj, d: q.d, z: q.z, truth: q.truth, m1: q.msg1, m2: q.msg2, props: q.props.map((p) => ({ est: p.est, ok: !!p.ok, silent: !!p.silent, d: p.d, z: p.z })) })) })),
  };
}
fs.mkdirSync(path.join(here, "../website/data"), { recursive: true });
const dest = path.join(here, "../website/data/llm_estimated.json");
fs.writeFileSync(dest, JSON.stringify(out));
console.log(`wrote ${dest} (${(fs.statSync(dest).size / 1024).toFixed(0)} KB)`);
if (out.particle) console.log("particle", JSON.stringify(out.particle.stats));
if (out.llmEstimate) console.log("llmEstimate", JSON.stringify(out.llmEstimate.stats), "spend", JSON.stringify(out.llmEstimate.spend));
