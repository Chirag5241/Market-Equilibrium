// Full-information LLM negotiation on the deck's corporate-bond market.
//
// Each investor is an independent agent (its own system prompt and its own memory of the meetings it took
// part in) backed by an Azure gpt-6-luna deployment. Meetings use exactly the same random matching as the
// true-utility solver in website/engine.js, so the two can be compared round by round.
//
// Two cases, identical information, different instructions:
//   minimal   — "trade favorably and increase your utility"; no hint about how to optimise.
//   objective — adds the objective the market is judged by: the symmetric Nash product of the two gains,
//               with the joint-surplus / equal-split structure and marginal values spelled out.
//
// Run:   node agents/llm_full_info.mjs --case both            (defaults: N 90, seed 7, 16 rounds, $10 cap)
//        node agents/llm_full_info.mjs --smoke                 (one meeting per case, to check cost)
// Reads AZURE_OPENAI_ENDPOINT (…/openai/v1), AZURE_OPENAI_KEY and DEPLOYMENT_NAME_6_LUNA[_2] from agents/.env.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const S = require(path.join(here, "../website/engine.js"));

// ---------- configuration ----------
const args = Object.fromEntries(process.argv.slice(2).map((a, n, all) => (a.startsWith("--") ? [a.slice(2), all[n + 1] && !all[n + 1].startsWith("--") ? all[n + 1] : true] : null)).filter(Boolean));
const CFG = {
  N: +(args.N || 90), seed: +(args.seed || 7), variant: "baseline", shock: args.shock || "riskUp", share: +(args.share || 0.1),
  rounds: +(args.rounds || 16), maxTurns: 4, concurrency: +(args.concurrency || 24),
  cap: +(args.cap || 10), priceIn: +(args.priceIn || 0.11), priceOut: +(args.priceOut || 0.55), // $ per 1M tokens (conservative Azure rate)
  maxOut: +(args.maxOut || 6000), smoke: !!args.smoke,
  cases: args.case === "minimal" || args.case === "objective" ? [args.case] : ["minimal", "objective"],
};
const env = Object.fromEntries(fs.readFileSync(path.join(here, ".env"), "utf8").split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#") && l.includes("=")).map((l) => { const q = l.indexOf("="); return [l.slice(0, q).trim(), l.slice(q + 1).trim().replace(/^["']|["']$/g, "")]; }));
const ENDPOINT = env.AZURE_OPENAI_ENDPOINT.replace(/\/$/, ""), KEY = env.AZURE_OPENAI_KEY;
const DEPLOYMENTS = [env.DEPLOYMENT_NAME_6_LUNA, env.DEPLOYMENT_NAME_6_LUNA_2].filter(Boolean);
const OUT = path.join(here, "results");
fs.mkdirSync(OUT, { recursive: true });

// ---------- spend ledger with a hard cap ----------
const ledgerPath = path.join(OUT, "llm_ledger.jsonl");
const ledger = { spent: 0, reserved: 0, calls: 0, inTok: 0, outTok: 0, reasoning: 0 };
if (fs.existsSync(ledgerPath)) for (const l of fs.readFileSync(ledgerPath, "utf8").split("\n").filter(Boolean)) ledger.spent += JSON.parse(l).cost;
class CapReached extends Error {}
const costOf = (inTok, outTok) => (inTok * CFG.priceIn + outTok * CFG.priceOut) / 1e6;

async function chat(deployment, messages, tag) {
  const estIn = Math.ceil(messages.reduce((s, m) => s + m.content.length, 0) / 3.5);
  const reserve = costOf(estIn, CFG.maxOut);
  if (ledger.spent + ledger.reserved + reserve > CFG.cap) throw new CapReached(`cap $${CFG.cap} reached (spent $${ledger.spent.toFixed(4)})`);
  ledger.reserved += reserve;
  try {
    for (let attempt = 0; ; attempt++) {
      let r;
      try {
        r = await fetch(`${ENDPOINT}/chat/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${KEY}` },
          body: JSON.stringify({ model: deployment, messages, max_completion_tokens: CFG.maxOut, response_format: { type: "json_object" } }),
        });
      } catch (err) { // dropped connection, DNS hiccup, etc.
        if (attempt >= 6) throw err;
        await new Promise((res) => setTimeout(res, 2 ** attempt * 1000));
        continue;
      }
      if (r.status === 429 || r.status >= 500) {
        if (attempt >= 6) throw new Error(`HTTP ${r.status} after retries`);
        const wait = (+r.headers.get("retry-after") || 2 ** attempt) * 1000;
        await new Promise((res) => setTimeout(res, wait));
        continue;
      }
      const j = await r.json();
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${JSON.stringify(j).slice(0, 300)}`);
      const u = j.usage || {}, inTok = u.prompt_tokens || 0, outTok = u.completion_tokens || 0;
      const reasoning = (u.completion_tokens_details && u.completion_tokens_details.reasoning_tokens) || 0;
      const cost = costOf(inTok, outTok);
      ledger.spent += cost; ledger.calls++; ledger.inTok += inTok; ledger.outTok += outTok; ledger.reasoning += reasoning;
      fs.appendFileSync(ledgerPath, JSON.stringify({ t: new Date().toISOString(), tag, deployment, inTok, outTok, reasoning, cost }) + "\n");
      return (j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) || "";
    }
  } finally { ledger.reserved -= reserve; }
}

// ---------- market (same as the website) ----------
const m = S.buildMarket({ N: CFG.N, seed: CFG.seed, variant: CFG.variant, shock: CFG.shock, share: CFG.share });
const e = m.post, K = S.K, BN = S.T.map((t) => (t % 1 ? t.toFixed(1) : String(t)) + "y");
const GROUP = ["short", "medium", "long"];
const f6 = (v) => (+v).toPrecision(6);
const matching = (t) => S.makeRng(S.hashSeed("match", m.cfg.seed, t)).perm(m.N); // identical to engine.step

// true-utility reference on the same matching
const ref = S.createRun(m, "true", {});
for (let t = 0; t < CFG.rounds; t++) S.step(m, ref);

// ---------- prompts ----------
const MARKET_TEXT = `MARKET
- Over-the-counter corporate bond market. No exchange and no posted price: investors meet in pairs and agree on a trade.
- Six zero-coupon corporate bonds, maturities ${BN.join(", ")}. Bond holdings are in units and can never go below zero.
- Cash counts one-for-one in utility, and borrowing is unlimited, so cash may go negative.
- Each round every investor is paired with one other investor. A meeting allows at most ${CFG.maxTurns} messages in total.
- A trade = a bundle of bonds moving one way plus one cash payment. The market executes an agreed trade only if it is feasible and strictly raises BOTH investors' true utilities.
- Utility of any investor: U = cash + sum_k (c_k * x_k - b_k * x_k^2), where x_k is units of bond k held, c_k is that investor's value per unit (discounted expected payoff plus a small maturity-preference bonus) and b_k penalises concentration risk.`;

const OBJECTIVE_TEXT = `HOW TRADES ARE JUDGED (the objective to maximise)
- The market's benchmark for every meeting is symmetric Nash bargaining: among feasible trades, maximise the PRODUCT of the two utility gains, g_you * g_them, with both gains positive.
- For a bond trade D (units you receive, negative = you give) and cash z you pay: g_you = A_you(D) - z and g_them = A_them(D) + z, where A_you(D) = V_you(x_you + D) - V_you(x_you) and A_them(D) = V_them(x_them - D) - V_them(x_them), and V = sum_k (c_k x_k - b_k x_k^2).
- Because cash is linear, the product is maximised in two steps: (1) choose the bonds D to maximise the JOINT SURPLUS S(D) = A_you(D) + A_them(D); (2) set the cash so the two gains are equal: z = (A_you - A_them) / 2, giving each side S/2.
- Marginal value of bond k: a_k = c_k - 2 b_k x_k (the cash value of one more small unit at current holdings). S(D) = sum_k [ (a_you,k - a_them,k) D_k - (b_you,k + b_them,k) D_k^2 ]. Units should flow toward whoever has the higher marginal value, until the two marginal values meet or a holding runs out.
- Asking for more than half of the surplus makes it a worse bargain by this objective, even if the counterpart accepts.`;

const REPLY_TEXT = `REPLY FORMAT
Reply with ONE JSON object and nothing else:
{"action": "propose" | "accept" | "end", "message": "one or two sentences to the counterpart", "bonds_you_receive": {${BN.map((b) => `"${b}": number`).join(", ")}}, "cash_you_pay": number}
- "propose": a new offer; "bonds_you_receive" (negative = you give) and "cash_you_pay" (negative = you receive cash) are required.
- "accept": accept the counterpart's latest offer exactly as stated.
- "end": walk away without a trade.`;

const goalText = (caseId) => (caseId === "minimal" ? `YOUR GOAL\nTrade favorably and increase your utility.` : `YOUR GOAL\nTrade favorably and increase your utility, using the objective below.\n\n${OBJECTIVE_TEXT}`);

function coeffs(i) { return `c = {${BN.map((b, k) => `${b}: ${f6(e.c[i][k])}`).join(", ")}}\n  b = {${BN.map((b, k) => `${b}: ${f6(e.b[i][k])}`).join(", ")}}`; }
function holdingsLine(X, C, i) { return `#${String(i).padStart(2, "0")} ${GROUP[m.group[i]].padEnd(6)} ${BN.map((b, k) => `${b}=${X[i][k].toFixed(4)}`).join(" ")} cash=${C[i].toFixed(4)}`; }

// ---------- independent agents ----------
class Agent {
  constructor(id) {
    this.id = id;
    this.deployment = DEPLOYMENTS[id % DEPLOYMENTS.length];
    this.memory = []; // only the meetings this agent took part in
  }
  system(caseId) {
    return `You are investor #${String(this.id).padStart(2, "0")}, an independent trading agent acting only for yourself.\n\n${MARKET_TEXT}\n\nYOUR UTILITY COEFFICIENTS\n  ${coeffs(this.id)}\n\n${goalText(caseId)}\n\n${REPLY_TEXT}`;
  }
  async act(caseId, ctx) {
    const { round, other, X, C, turns, k } = ctx, me = this.id;
    const hist = turns.length ? turns.map((t) => `#${String(t.speaker).padStart(2, "0")}: ${t.action.toUpperCase()}${t.action === "propose" ? ` (#${String(t.speaker).padStart(2, "0")} receives ${tradeText(t.d)}; pays cash ${t.z.toFixed(6)})` : ""}${t.message ? ` — "${t.message}"` : ""}`).join("\n") : "(none yet; you speak first)";
    const last = [...turns].reverse().find((t) => t.action === "propose");
    const mem = this.memory.slice(-6).map((q) => `round ${q.round} with #${String(q.partner).padStart(2, "0")}: ${q.outcome}${q.d ? `; you received ${tradeText(q.d)}, paid cash ${q.z.toFixed(6)}, your utility gain ${q.gain.toExponential(3)}` : ""}`).join("\n") || "(no earlier meetings)";
    const user = `ROUND ${round}. You are meeting #${String(other).padStart(2, "0")}. This is message ${k + 1} of at most ${CFG.maxTurns}.

YOUR POSITION: ${holdingsLine(X, C, me)}

COUNTERPART #${String(other).padStart(2, "0")} (full information: you know its utility exactly)
  ${holdingsLine(X, C, other)}
  ${coeffs(other)}

EVERYONE'S HOLDINGS AND CASH THIS ROUND
${X.map((r, i) => holdingsLine(X, C, i)).join("\n")}

YOUR EARLIER MEETINGS
${mem}

THIS MEETING SO FAR
${hist}
${last && last.speaker === other ? `\nThe counterpart's latest offer means YOU would receive ${tradeText(last.d.map((v) => -v))} and RECEIVE cash ${last.z.toFixed(6)}.` : ""}`;
    const raw = await chat(this.deployment, [{ role: "system", content: this.system(caseId) }, { role: "user", content: user }], `${caseId}:r${round}:#${me}`);
    return parseReply(raw);
  }
}
function tradeText(d) { const p = d.map((v, k) => (Math.abs(v) < 1e-9 ? null : `${v > 0 ? "+" : ""}${v.toFixed(5)} ${BN[k]}`)).filter(Boolean); return p.length ? p.join(", ") : "no bonds"; }
function parseReply(raw) {
  let o = null;
  try { o = JSON.parse(raw); } catch { const mm = raw.match(/\{[\s\S]*\}/); if (mm) try { o = JSON.parse(mm[0]); } catch {} }
  if (!o) return { action: "end", message: "(unreadable reply)", bad: true };
  const action = ["propose", "accept", "end"].includes(o.action) ? o.action : "end";
  if (action !== "propose") return { action, message: String(o.message || "").slice(0, 400) };
  const b = o.bonds_you_receive || {};
  const d = BN.map((n) => { const v = Number(b[n]); return isFinite(v) ? v : 0; }), z = Number(o.cash_you_pay);
  if (!isFinite(z) || d.every((v) => v === 0)) return { action: "end", message: String(o.message || "").slice(0, 400), bad: true };
  return { action, d, z, message: String(o.message || "").slice(0, 400) };
}

// ---------- one meeting ----------
async function meeting(caseId, agents, round, i, j, X, C) {
  const turns = [];
  let deal = null, ended = "limit";
  for (let k = 0; k < CFG.maxTurns; k++) {
    const me = k % 2 === 0 ? i : j, other = me === i ? j : i;
    const r = await agents[me].act(caseId, { round, other, X, C, turns, k });
    turns.push(Object.assign({ speaker: me }, r));
    if (r.action === "accept") {
      const prev = turns.slice(0, -1).reverse().find((t) => t.action === "propose");
      if (prev && prev.speaker === other) { deal = prev.speaker === i ? { d: prev.d, z: prev.z } : { d: prev.d.map((v) => -v), z: -prev.z }; ended = "agreed"; }
      else ended = "accepted nothing";
      break;
    }
    if (r.action === "end") { ended = r.bad ? "unreadable" : "walked"; break; }
  }
  const nash = S.nashMoney(S.margVec(e, i, X[i]), e.b[i], X[i], S.margVec(e, j, X[j]), e.b[j], X[j]);
  const rec = { i, j, turns: turns.map((t) => ({ speaker: t.speaker, action: t.action, d: t.d, z: t.z, message: t.message })), ended, nashS: Math.max(0, nash.S), executed: false };
  if (deal) {
    const feas = deal.d.every((v, k) => X[i][k] + v >= -1e-12 && X[j][k] - v >= -1e-12);
    const [Ai, Aj] = S.bondGains(m, X, i, j, deal.d), gi = Ai - deal.z, gj = Aj + deal.z;
    Object.assign(rec, { d: deal.d, z: deal.z, gi, gj, S: Ai + Aj, feasible: feas, executed: feas && gi > 0 && gj > 0 });
  }
  remember(agents, round, rec);
  return rec;
}
// each agent remembers only its own meeting
function remember(agents, round, rec) {
  const { i, j } = rec;
  for (const [me, other, sign] of [[i, j, 1], [j, i, -1]]) agents[me].memory.push({ round, partner: other, outcome: rec.executed ? "traded" : rec.d ? "agreed but blocked by the market (not strictly better for both)" : rec.ended === "walked" ? "no trade (someone walked away)" : "no trade", d: rec.executed ? rec.d.map((v) => sign * v) : null, z: rec.executed ? sign * rec.z : 0, gain: rec.executed ? (sign > 0 ? rec.gi : rec.gj) : 0 });
}

async function pool(items, n, fn) { const out = new Array(items.length); let next = 0; await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => { while (next < items.length) { const q = next++; out[q] = await fn(items[q], q); } })); return out; }

// ---------- one case ----------
async function runCase(caseId) {
  const agents = m.dstar.map((d, i) => new Agent(i));
  const x = m.x0.map((r) => r.slice()), cash = m.m0.slice();
  let res = { caseId, cfg: CFG, rounds: [], stopped: null, model: DEPLOYMENTS };
  const file = path.join(OUT, `full_info_${caseId}${CFG.smoke ? "_smoke" : ""}.json`);
  const groupCash = () => [0, 1, 2].map((q) => m.group.reduce((s, g, i) => s + (g === q ? cash[i] - m.m0[i] : 0), 0));
  let start = 0;
  if (args.resume && !CFG.smoke && fs.existsSync(file)) {
    const old = JSON.parse(fs.readFileSync(file, "utf8"));
    res = Object.assign(old, { cfg: CFG, stopped: null });
    const last = res.rounds[res.rounds.length - 1];
    if (last.x) { last.x.forEach((r, i) => (x[i] = r.slice())); last.cash.forEach((v, i) => (cash[i] = v)); }
    for (const r of res.rounds.slice(1)) for (const rec of r.meetings) remember(agents, r.t, rec);
    start = res.rounds.length - 1;
    console.log(`[${caseId}] resuming after round ${start}`);
  } else res.rounds.push({ t: 0, F: 0, W: S.welfare(e, x), Fref: ref.history[0].F, groupCash: groupCash(), meetings: [] });
  for (let t = start; t < (CFG.smoke ? 1 : CFG.rounds); t++) {
    const perm = matching(t), X = x.map((r) => r.slice()), C = cash.slice();
    let pairs = []; for (let n = 0; n + 1 < m.N; n += 2) pairs.push([perm[n], perm[n + 1]]);
    if (CFG.smoke) pairs = pairs.filter(([i, j]) => S.trueNash(m, X, i, j).S > 2e-12).slice(0, 1);
    let recs;
    try { recs = await pool(pairs, CFG.concurrency, ([i, j]) => meeting(caseId, agents, t + 1, i, j, X, C)); }
    catch (err) { if (err instanceof CapReached) { res.stopped = `${err.message} during round ${t + 1}`; break; } throw err; }
    for (const r of recs) if (r.executed) { for (let k = 0; k < K; k++) { x[r.i][k] = Math.max(0, x[r.i][k] + r.d[k]); x[r.j][k] = Math.max(0, x[r.j][k] - r.d[k]); } cash[r.i] -= r.z; cash[r.j] += r.z; }
    const W = S.welfare(e, x);
    res.rounds.push({ t: t + 1, W, F: (W - m.W0) / (m.Wstar - m.W0), Fref: ref.history[t + 1].F, groupCash: groupCash(), meetings: recs, x: x.map((r) => r.slice()), cash: cash.slice() });
    const done = recs.filter((r) => r.executed).length, blocked = recs.filter((r) => r.d && !r.executed).length;
    console.log(`[${caseId}] round ${t + 1}: F = ${(100 * res.rounds.at(-1).F).toFixed(2)}% (true-utility ${(100 * ref.history[t + 1].F).toFixed(2)}%), ${done} trades, ${blocked} blocked, spent $${ledger.spent.toFixed(4)}`);
    res.ledger = Object.assign({}, ledger);
    fs.writeFileSync(file, JSON.stringify(res));
  }
  res.ledger = Object.assign({}, ledger);
  fs.writeFileSync(file, JSON.stringify(res));
  return res;
}

console.log(`Market: N ${m.N}, seed ${CFG.seed}, ${CFG.shock} (${Math.round(100 * CFG.share)}% hit), ${CFG.rounds} rounds. Deployments: ${DEPLOYMENTS.join(", ")}. Cap $${CFG.cap} (already spent $${ledger.spent.toFixed(4)}).`);
const results = await Promise.all(CFG.cases.map((c) => runCase(c).catch((err) => ({ caseId: c, stopped: `error: ${err.message}` }))));
console.log(`Done. Calls ${ledger.calls}, input tokens ${ledger.inTok}, output tokens ${ledger.outTok} (reasoning ${ledger.reasoning}). Spent this run $${(ledger.spent).toFixed(4)} at $${CFG.priceIn}/$${CFG.priceOut} per 1M.`);
for (const r of results) if (r.stopped) console.log(`[${r.caseId}] stopped early: ${r.stopped}`);
