// Estimated-utility solvers on the deck's cash market, with private types.
//
// Both methods use the deck's estimated-utility rule: the proposer plugs an ESTIMATE of the counterpart's
// coefficients (c, b) into the cash-market Nash formula, and the market executes the trade only if both TRUE
// gains are positive. They differ only in where the estimate comes from:
//
//   particle     — the older deq run_B / ParticleBeliefs design, adapted to cash: each proposer keeps 3,000
//                  guesses of the counterpart's (d*, tau, lambda) drawn from the deck's ranges and reweights them
//                  after every accept or reject. No LLM.
//   llmEstimate  — as in the older llm_regimes "D-infer" set-up: the two investors (independent gpt-6-luna agents,
//                  each knowing only its own type) exchange one message each, then the proposer's LLM outputs its
//                  estimate of (d*, tau, lambda) and the formula computes the trade. After a rejection the proposer
//                  is told and may revise its estimate, up to 3 proposals in a meeting (same budget as particle).
//
// Same market and the same meeting matching as the true-utility solver on the website.
// Information: types private; holdings and bond data public; the deck's prior ranges and the shock's description
// public, but not who was hit. The pre-shock price P0 is NOT given (with it the type could be backed out exactly).
//
// Run:   node agents/llm_estimated.mjs                 (both methods; N 90, seed 7, 16 rounds, $10 cap)
//        node agents/llm_estimated.mjs --smoke         (particle in full, LLM for one meeting)
//        add --resume to continue an interrupted LLM run.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const S = require(path.join(here, "../website/engine.js"));

const args = Object.fromEntries(process.argv.slice(2).map((a, n, all) => (a.startsWith("--") ? [a.slice(2), all[n + 1] && !all[n + 1].startsWith("--") ? all[n + 1] : true] : null)).filter(Boolean));
const CFG = {
  N: +(args.N || 90), seed: +(args.seed || 7), variant: "baseline", shock: args.shock || "riskUp", share: +(args.share || 0.1),
  rounds: +(args.rounds || 16), maxProposals: 3, particles: 3000, concurrency: +(args.concurrency || 24),
  cap: +(args.cap || 10), priceIn: 0.11, priceOut: 0.55, maxOut: 6000, smoke: !!args.smoke, resume: !!args.resume,
};
const env = Object.fromEntries(fs.readFileSync(path.join(here, ".env"), "utf8").split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#") && l.includes("=")).map((l) => { const q = l.indexOf("="); return [l.slice(0, q).trim(), l.slice(q + 1).trim().replace(/^["']|["']$/g, "")]; }));
const ENDPOINT = env.AZURE_OPENAI_ENDPOINT.replace(/\/$/, ""), KEY = env.AZURE_OPENAI_KEY;
const DEPLOYMENTS = [env.DEPLOYMENT_NAME_6_LUNA, env.DEPLOYMENT_NAME_6_LUNA_2].filter(Boolean);
const OUT = path.join(here, "results");
fs.mkdirSync(OUT, { recursive: true });

// ---------- spend ledger with a hard cap (separate from the full-information experiment) ----------
const ledgerPath = path.join(OUT, "llm_ledger_estimated.jsonl");
const ledger = { spent: 0, reserved: 0, calls: 0, inTok: 0, outTok: 0, reasoning: 0 };
if (fs.existsSync(ledgerPath)) for (const l of fs.readFileSync(ledgerPath, "utf8").split("\n").filter(Boolean)) ledger.spent += JSON.parse(l).cost;
class CapReached extends Error {}
const costOf = (i, o) => (i * CFG.priceIn + o * CFG.priceOut) / 1e6;
async function chat(deployment, messages, tag) {
  const reserve = costOf(Math.ceil(messages.reduce((s, m) => s + m.content.length, 0) / 3.5), CFG.maxOut);
  if (ledger.spent + ledger.reserved + reserve > CFG.cap) throw new CapReached(`cap $${CFG.cap} reached (spent $${ledger.spent.toFixed(4)})`);
  ledger.reserved += reserve;
  try {
    for (let attempt = 0; ; attempt++) {
      let r;
      try {
        r = await fetch(`${ENDPOINT}/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${KEY}` }, body: JSON.stringify({ model: deployment, messages, max_completion_tokens: CFG.maxOut, response_format: { type: "json_object" } }) });
      } catch (err) { if (attempt >= 6) throw err; await new Promise((res) => setTimeout(res, 2 ** attempt * 1000)); continue; }
      if (r.status === 429 || r.status >= 500) { if (attempt >= 6) throw new Error(`HTTP ${r.status} after retries`); await new Promise((res) => setTimeout(res, (+r.headers.get("retry-after") || 2 ** attempt) * 1000)); continue; }
      const j = await r.json();
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${JSON.stringify(j).slice(0, 300)}`);
      const u = j.usage || {}, inTok = u.prompt_tokens || 0, outTok = u.completion_tokens || 0, reasoning = (u.completion_tokens_details && u.completion_tokens_details.reasoning_tokens) || 0, cost = costOf(inTok, outTok);
      ledger.spent += cost; ledger.calls++; ledger.inTok += inTok; ledger.outTok += outTok; ledger.reasoning += reasoning;
      fs.appendFileSync(ledgerPath, JSON.stringify({ t: new Date().toISOString(), tag, deployment, inTok, outTok, reasoning, cost }) + "\n");
      return (j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) || "";
    }
  } finally { ledger.reserved -= reserve; }
}
const parseJSON = (raw) => { try { return JSON.parse(raw); } catch { const mm = raw.match(/\{[\s\S]*\}/); if (mm) try { return JSON.parse(mm[0]); } catch {} } return null; };
async function pool(items, n, fn) { const out = new Array(items.length); let next = 0; await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => { while (next < items.length) { const q = next++; out[q] = await fn(items[q], q); } })); return out; }

// ---------- market (same as the website) ----------
const m = S.buildMarket({ N: CFG.N, seed: CFG.seed, variant: CFG.variant, shock: CFG.shock, share: CFG.share });
const e = m.post, K = S.K, BN = S.T.map((t) => (t % 1 ? t.toFixed(1) : String(t)) + "y");
const matching = (t) => S.makeRng(S.hashSeed("match", m.cfg.seed, t)).perm(m.N); // identical to engine.step
const ref = S.createRun(m, "true", {});
for (let t = 0; t < CFG.rounds; t++) S.step(m, ref);
const f6 = (v) => (+v).toPrecision(6), id2 = (i) => "#" + String(i).padStart(2, "0");
const groupCash = (cash) => [0, 1, 2].map((q) => m.group.reduce((s, g, i) => s + (g === q ? cash[i] - m.m0[i] : 0), 0));

// coefficients implied by a type estimate, using only public data (post-shock mu, sigma^2, alpha)
function coefFromType(dstar, tau, lam, xj) {
  const c = S.T.map((t, k) => S.D[k] * e.mu[k] + m.alpha * S.vPref(t, dstar, tau)), b = S.T.map((t, k) => lam * S.D[k] ** 2 * e.s2[k]);
  return { a: c.map((v, k) => v - 2 * b[k] * xj[k]), b };
}

// ---------- method 1: particle beliefs (3,000 particles, deck prior), via the website engine ----------
function runParticle() {
  const run = S.createRun(m, "est", { particles: CFG.particles, prior: "deck", maxProposals: CFG.maxProposals });
  for (let t = 0; t < CFG.rounds; t++) S.step(m, run);
  const res = { method: "particle", cfg: CFG, rounds: run.history.map((h, t) => ({ t, F: h.F, Fref: ref.history[t].F, belief: h.belief, proposals: h.proposals, rejects: h.rejects, trades: h.trades, groupCash: t < run.rounds.length ? groupCash(run.rounds[t].cash) : groupCash(run.cash) })) };
  res.meetings = run.rounds.map((R) => R.meetings.map((q) => ({ i: q.i, j: q.j, executed: !!q.d, props: q.props, d: q.d, z: q.z })));
  fs.writeFileSync(path.join(OUT, "est_particle.json"), JSON.stringify(res));
  const h = run.history;
  console.log(`[particle] F after rounds 1/4/16: ${[1, 4, CFG.rounds].map((t) => (100 * h[t].F).toFixed(2) + "%").join(" / ")}; proposals ${run.proposals}, rejected ${(100 * run.rejects / run.proposals).toFixed(0)}%, belief error |log lam| ${h[CFG.rounds].belief.toFixed(3)}`);
  return res;
}

// ---------- method 2: LLM estimate ----------
const PUBLIC_TEXT = `MARKET (public)
- Over-the-counter corporate bond market. No exchange and no posted price: investors meet in pairs.
- Six zero-coupon corporate bonds, maturities ${BN.join(", ")}. Holdings are in units and cannot go below zero. Cash counts one-for-one; borrowing is unlimited.
- Utility of any investor: U = cash + sum_k (c_k x_k - b_k x_k^2), with
    c_k = D_k * mu_k + alpha * v_k,   v_k = exp(-(T_k - d*)^2 / (2 tau^2)),   b_k = lambda * D_k^2 * sigma2_k.
- Public bond data: T = [${S.T.join(", ")}], D = [${S.D.map(f6).join(", ")}], mu = [${e.mu.map(f6).join(", ")}], sigma2 = [${e.s2.map(f6).join(", ")}], alpha = ${m.alpha}.
- Each investor's type (d*, tau, lambda) is PRIVATE. Public prior: d* is in one of three groups, [1,4], [4,9] or [9,25] years (group not public); tau is in [1.5, 4]; lambda = ${m.lamBase} x Uniform(0.5, 1.5) before the shock. A recent shock ${m.shock.kind === "risk" ? `multiplied lambda by ${m.shock.mult} for ${Math.round(100 * m.shock.share)}% of investors; who was hit is private` : `changed the default hazard for every bond (already reflected in mu and sigma2)`}.
- Holdings of everyone are public.
- THE TRADE RULE: in each meeting the proposer's trade is computed by a fixed formula, the Nash bargain with cash, using the proposer's TRUE utility and the proposer's ESTIMATE of the counterpart's type. The market executes it only if it strictly raises BOTH investors' TRUE utilities. Otherwise it is rejected, and the proposer may revise its estimate (at most ${CFG.maxProposals} proposals per meeting).`;

class Agent {
  constructor(id) { this.id = id; this.deployment = DEPLOYMENTS[id % DEPLOYMENTS.length]; this.memory = []; }
  system() {
    const i = this.id;
    return `You are investor ${id2(i)}, an independent agent acting only for yourself. Trade favorably and increase your utility.\n\n${PUBLIC_TEXT}\n\nYOUR PRIVATE TYPE: d* = ${f6(m.dstar[i])}, tau = ${f6(m.tau[i])}, lambda = ${f6(e.lam[i])}${m.hit[i] ? " (you were hit by the shock)" : ""}.\nYOUR COEFFICIENTS: c = [${e.c[i].map(f6).join(", ")}], b = [${e.b[i].map(f6).join(", ")}].\n\nAlways reply with ONE JSON object and nothing else.`;
  }
  mem() { return this.memory.slice(-5).map((q) => `round ${q.round} with ${id2(q.partner)} (you were the ${q.role}): ${q.text}`).join("\n") || "(no earlier meetings)"; }
  async call(user, tag) { return parseJSON(await chat(this.deployment, [{ role: "system", content: this.system() }, { role: "user", content: user }], tag)) || {}; }
}
const holdLine = (X, C, i) => `${id2(i)}: ${BN.map((b, k) => `${b}=${X[i][k].toFixed(4)}`).join(" ")} cash=${C[i].toFixed(4)}`;

async function llmMeeting(agents, round, i, j, X, C) {
  const A = agents[i], B = agents[j], ctx = (me, other) => `ROUND ${round}. You are meeting ${id2(other)}. ${id2(i)} is the proposer.\nYOUR HOLDINGS: ${holdLine(X, C, me)}\nCOUNTERPART HOLDINGS: ${holdLine(X, C, other)}\nYOUR EARLIER MEETINGS:\n${agents[me].mem()}`;
  const tag = (s) => `llmEstimate:r${round}:${id2(i)}-${id2(j)}:${s}`;
  // 1-2: one message each
  const m1 = await A.call(`${ctx(i, j)}\n\nStep 1 of the meeting: send one message to ${id2(j)} (for example, ask about its preferences or say what you are looking for). It will reply once, then you will estimate its type.\nReply: {"message": "..."}`, tag("msg1"));
  const msg1 = String(m1.message || "").slice(0, 600);
  const m2 = await B.call(`${ctx(j, i)}\n\n${id2(i)} says: "${msg1}"\n\nReply once. ${id2(i)} will then estimate your type and the market will compute a trade from that estimate; it executes only if it strictly benefits you too. You may say whatever you like about your preferences.\nReply: {"message": "..."}`, tag("msg2"));
  const msg2 = String(m2.message || "").slice(0, 600);
  // 3+: estimates, up to maxProposals
  const props = [];
  let deal = null;
  for (let a = 0; a < CFG.maxProposals && !deal; a++) {
    const fb = props.map((p, n) => `Proposal ${n + 1}: your estimate d*=${f6(p.est.d)}, tau=${f6(p.est.tau)}, lambda=${f6(p.est.lam)} -> ${p.silent ? "the formula found no trade worth proposing" : `trade: you receive ${tradeText(p.d)} and pay cash ${p.z.toFixed(6)} -> REJECTED by the market (not strictly better for both by true utilities)`}.`).join("\n");
    const r = await A.call(`${ctx(i, j)}\n\nConversation:\n${id2(i)} (you): "${msg1}"\n${id2(j)}: "${msg2}"\n${fb ? `\nEarlier in this meeting:\n${fb}\n` : ""}\nEstimate ${id2(j)}'s private type. The formula will plug it into the Nash bargain with your own true utility.\nReply: {"d_star": number, "tau": number, "lambda": number, "note": "one sentence on your reasoning"}`, tag(`est${a + 1}`));
    const est = { d: Number(r.d_star), tau: Number(r.tau), lam: Number(r.lambda), note: String(r.note || "").slice(0, 300) };
    if (![est.d, est.tau, est.lam].every(isFinite) || est.tau <= 0 || est.lam <= 0) { props.push({ est, bad: true, silent: true }); continue; }
    const { a: aj, b: bj } = coefFromType(est.d, est.tau, est.lam, X[j]);
    const s = S.nashMoney(S.margVec(e, i, X[i]), e.b[i], X[i], aj, bj, X[j]);
    if (!(s.S > 2e-12)) { props.push({ est, silent: true }); continue; }
    const [Ai, Aj] = S.bondGains(m, X, i, j, s.d), gi = Ai - s.z, gj = Aj + s.z;
    const ok = gi > 0 && gj > 0 && s.d.every((v, k) => X[i][k] + v >= -1e-12 && X[j][k] - v >= -1e-12);
    props.push({ est, d: s.d, z: s.z, gi, gj, S: Ai + Aj, ok });
    if (ok) deal = props[props.length - 1];
  }
  const nash = S.trueNash(m, X, i, j);
  const rec = { i, j, msg1, msg2, props, executed: !!deal, d: deal && deal.d, z: deal && deal.z, gi: deal && deal.gi, gj: deal && deal.gj, S: deal && deal.S, nashS: Math.max(0, nash.S), truth: { d: m.dstar[j], tau: m.tau[j], lam: e.lam[j] } };
  remember(agents, round, rec);
  return rec;
}
function tradeText(d) { const p = d.map((v, k) => (Math.abs(v) < 1e-9 ? null : `${v > 0 ? "+" : ""}${v.toFixed(5)} ${BN[k]}`)).filter(Boolean); return p.length ? p.join(", ") : "no bonds"; }
function remember(agents, round, rec) {
  const last = rec.props[rec.props.length - 1];
  agents[rec.i].memory.push({ round, partner: rec.j, role: "proposer", text: rec.executed ? `traded after ${rec.props.length} proposal(s); your gain ${rec.gi.toExponential(3)}; your final estimate of its type d*=${f6(last.est.d)}, tau=${f6(last.est.tau)}, lambda=${f6(last.est.lam)}` : `no trade after ${rec.props.length} proposal(s)` });
  agents[rec.j].memory.push({ round, partner: rec.i, role: "counterpart", text: rec.executed ? `traded; your gain ${rec.gj.toExponential(3)}` : "no trade" });
}

async function runLLM() {
  const agents = m.dstar.map((d, i) => new Agent(i));
  const x = m.x0.map((r) => r.slice()), cash = m.m0.slice();
  const file = path.join(OUT, `est_llm${CFG.smoke ? "_smoke" : ""}.json`);
  let res = { method: "llmEstimate", cfg: CFG, model: DEPLOYMENTS, rounds: [{ t: 0, F: 0, Fref: 0, groupCash: groupCash(cash), meetings: [] }], stopped: null };
  let start = 0;
  if (CFG.resume && !CFG.smoke && fs.existsSync(file)) {
    res = Object.assign(JSON.parse(fs.readFileSync(file, "utf8")), { cfg: CFG, stopped: null });
    const last = res.rounds[res.rounds.length - 1];
    if (last.x) { last.x.forEach((r, i) => (x[i] = r.slice())); last.cash.forEach((v, i) => (cash[i] = v)); }
    for (const r of res.rounds.slice(1)) for (const rec of r.meetings) remember(agents, r.t, rec);
    start = res.rounds.length - 1;
    console.log(`[llmEstimate] resuming after round ${start}`);
  }
  for (let t = start; t < (CFG.smoke ? 1 : CFG.rounds); t++) {
    const perm = matching(t), X = x.map((r) => r.slice()), C = cash.slice();
    let pairs = []; for (let n = 0; n + 1 < m.N; n += 2) pairs.push([perm[n], perm[n + 1]]);
    if (CFG.smoke) pairs = pairs.filter(([i, j]) => S.trueNash(m, X, i, j).S > 2e-12).slice(0, 1);
    let recs;
    try { recs = await pool(pairs, CFG.concurrency, ([i, j]) => llmMeeting(agents, t + 1, i, j, X, C)); }
    catch (err) { if (err instanceof CapReached) { res.stopped = `${err.message} during round ${t + 1}`; break; } throw err; }
    for (const r of recs) if (r.executed) { for (let k = 0; k < K; k++) { x[r.i][k] = Math.max(0, x[r.i][k] + r.d[k]); x[r.j][k] = Math.max(0, x[r.j][k] - r.d[k]); } cash[r.i] -= r.z; cash[r.j] += r.z; }
    const W = S.welfare(e, x);
    res.rounds.push({ t: t + 1, W, F: (W - m.W0) / (m.Wstar - m.W0), Fref: ref.history[t + 1].F, groupCash: groupCash(cash), meetings: recs, x: x.map((r) => r.slice()), cash: cash.slice() });
    const nP = recs.reduce((s, r) => s + r.props.length, 0), nOk = recs.filter((r) => r.executed).length;
    console.log(`[llmEstimate] round ${t + 1}: F = ${(100 * res.rounds.at(-1).F).toFixed(2)}% (true-utility ${(100 * ref.history[t + 1].F).toFixed(2)}%), ${nOk} trades from ${nP} proposals, spent $${ledger.spent.toFixed(4)}`);
    res.ledger = Object.assign({}, ledger);
    fs.writeFileSync(file, JSON.stringify(res));
  }
  res.ledger = Object.assign({}, ledger);
  fs.writeFileSync(file, JSON.stringify(res));
  return res;
}

console.log(`Market: N ${m.N}, seed ${CFG.seed}, ${CFG.shock} (${Math.round(100 * CFG.share)}% hit), ${CFG.rounds} rounds. Deployments: ${DEPLOYMENTS.join(", ")}. Cap $${CFG.cap} (already spent $${ledger.spent.toFixed(4)} on this experiment).`);
if (!args.llmOnly) runParticle();
if (!args.particleOnly) {
  const r = await runLLM().catch((err) => ({ stopped: `error: ${err.message}` }));
  if (r.stopped) console.log(`[llmEstimate] stopped: ${r.stopped}`);
}
console.log(`LLM calls ${ledger.calls}, input tokens ${ledger.inTok}, output tokens ${ledger.outTok} (reasoning ${ledger.reasoning}). Experiment total $${ledger.spent.toFixed(4)} at $${CFG.priceIn}/$${CFG.priceOut} per 1M.`);
