/*
 * Simulation engine for "Decentralized Equilibrium Discovery After Shocks" (CS 580 proposal, Nov 2026).
 *
 * Part 1, duration market (proposal sections 0.1 to 0.4)
 *   N agents, K divisible instruments with duration d_k and idiosyncratic variance s2_k (independent).
 *   Agent i: preferred duration dStar, tolerance tau, risk aversion lam.
 *     v_ik = exp(-(d_k - dStar_i)^2 / (2 tau_i^2))
 *     U_i(x) = sum_k x_ik v_ik - lam_i sum_k x_ik^2 s2_k
 *   Bilateral barter trade: x_i += D, x_j -= D, feasible iff -x_i <= D <= x_j.
 *   Nash bargain: D(r)_k = clip((a_ik - r a_jk) / (2 s2_k (lam_i + r lam_j)), -x_ik, x_jk),
 *   with a_lk = v_lk - 2 lam_l s2_k x_lk, and bisection on F(r) = g_i(D(r)) - r g_j(D(r)).
 *   Regimes A (solver, full info), B (solver, estimated counterpart, halfspace learning),
 *   C (stylised LLM, full info), D (stylised LLM, private info via noisy disclosure).
 *
 * Part 2, endogenous term structure (proposal section 1)
 *   Money plus K zero-coupon bonds. U_i = m_i + sum_k x_ik [D_i(T_k) mu_k + alpha_i v_ik] - lam_i sum_k x_ik^2 D_i(T_k)^2 s_k^2.
 *   Money makes utility transferable, so the Nash bargain maximises joint surplus bond by bond and
 *   splits it equally with a money transfer.
 *
 * Works in the browser (window.Engine) and in Node (module.exports).
 */
(function (root) {
  "use strict";

  // ---------- random numbers ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeRng(seed) {
    const u = mulberry32(seed);
    let spare = null;
    u.normal = function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let a, b, r;
      do { a = 2 * u() - 1; b = 2 * u() - 1; r = a * a + b * b; } while (r >= 1 || r === 0);
      const f = Math.sqrt(-2 * Math.log(r) / r);
      spare = b * f; return a * f;
    };
    u.range = (lo, hi) => lo + (hi - lo) * u();
    u.int = (n) => Math.floor(u() * n);
    return u;
  }
  function hashSeed() {
    let h = 2166136261 >>> 0;
    for (let a = 0; a < arguments.length; a++) {
      const s = String(arguments[a]);
      for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
      h ^= 0x9e3779b9; h = Math.imul(h, 16777619) >>> 0;
    }
    return h;
  }
  function randomMatching(N, rng) {
    const p = []; for (let i = 0; i < N; i++) p.push(i);
    for (let i = N - 1; i > 0; i--) { const j = rng.int(i + 1); const t = p[i]; p[i] = p[j]; p[j] = t; }
    const pairs = []; for (let i = 0; i + 1 < N; i += 2) pairs.push([p[i], p[i + 1]]);
    return pairs;
  }
  const clip = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  // =====================================================================================
  // Part 1: duration market
  // =====================================================================================
  const DURATION_SETS = {
    4: [1, 3, 7, 20],
    6: [1, 2, 5, 7, 10, 20],
    8: [0.5, 1, 2, 3, 5, 7, 10, 20],
  };
  // Clienteles: who typically wants which part of the curve.
  const TYPES = [
    { key: "bank", name: "Bank", d: [0.6, 3], tau: [1.2, 3], lam: [0.9, 2.0] },
    { key: "fund", name: "Bond fund", d: [3, 8], tau: [2, 5], lam: [0.5, 1.2] },
    { key: "insurer", name: "Insurer", d: [6, 12], tau: [2.5, 5], lam: [0.6, 1.4] },
    { key: "pension", name: "Pension fund", d: [12, 22], tau: [3, 7], lam: [0.4, 1.0] },
  ];
  const SHOCKS = {
    ldi: {
      label: "Insurers must match longer liabilities",
      detail: "A solvency rule change pushes every insurer's preferred duration out by 2.5 years per unit of shock size.",
    },
    riskoff: {
      label: "Banks turn risk-averse",
      detail: "Bank risk aversion λ rises by 100% per unit of shock size, so banks want smaller, more diversified positions.",
    },
    idio: {
      label: "Idiosyncratic liability shocks",
      detail: "A random 30% of agents see their preferred duration scaled up or down by a factor of e^(±0.5 × size).",
    },
    longvol: {
      label: "Long-end volatility shock",
      detail: "Idiosyncratic variance σ²ₖ rises for long instruments (fully for d ≥ 10, partially below). Everyone's utility changes; agents with d* ≥ 10 are marked as hit.",
    },
  };
  const DEFAULT_PARAMS = {
    maxProposals: 3, // B, C, D: proposals per meeting
    priorSpread: 1, // B: noise on the stale prior
    llmAssets: 2, // C, D: instruments per offer
    llmLot: 0.1, // C, D: lot size
    llmNoise: 0.3, // C, D: size noise
    llmBias: 0.65, // C, D: proposer's surplus ask
    discloseNoise: 0.3, // D: cheap-talk noise
  };

  function valueVec(d, dStar, tau) { return d.map((dk) => Math.exp(-((dk - dStar) * (dk - dStar)) / (2 * tau * tau))); }

  function buildMarket(cfg) {
    const N = cfg.N, d = DURATION_SETS[cfg.K] || DURATION_SETS[6], K = d.length;
    const rng = makeRng(hashSeed("market", cfg.seed, N, K));
    const s2 = d.map(() => cfg.sigma2);
    const agents = [];
    for (let i = 0; i < N; i++) {
      const ty = TYPES[i % TYPES.length];
      agents.push({ type: ty.key, typeIdx: i % TYPES.length, dStar: rng.range(ty.d[0], ty.d[1]), tau: rng.range(ty.tau[0], ty.tau[1]), lam: rng.range(ty.lam[0], ty.lam[1]) });
    }
    const v = agents.map((a) => valueVec(d, a.dStar, a.tau));
    const supply = d.map(() => 3 * N);
    // Dirichlet-like random endowments
    const x = agents.map(() => new Array(K).fill(0));
    for (let k = 0; k < K; k++) {
      const w = agents.map(() => -Math.log(1 - rng()) + 0.05);
      const tot = w.reduce((s, z) => s + z, 0);
      for (let i = 0; i < N; i++) x[i][k] = (supply[k] * w[i]) / tot;
    }
    return { N, K, d, s2, agents, v, supply, xEndow: x, cfg: Object.assign({}, cfg) };
  }

  function utility(m, i, xi) {
    const v = m.v[i], lam = m.agents[i].lam;
    let u = 0; for (let k = 0; k < m.K; k++) u += xi[k] * v[k] - lam * xi[k] * xi[k] * m.s2[k];
    return u;
  }
  function marg(m, v, lam, xi, out) { for (let k = 0; k < m.K; k++) out[k] = v[k] - 2 * lam * m.s2[k] * xi[k]; return out; }

  // Section 0.4: the optimal bilateral solver. a_i, a_j marginal utilities; returns {d, gi, gj, iters, r}.
  // `target` (optional, 0..1) solves for the split g_i : g_j = target : (1 - target) instead of the Nash condition.
  function solvePair(K, s2, ai, li, aj, lj, xi, xj, target, trace) {
    const D = new Array(K);
    const at = (t) => {
      const r = Math.exp(t);
      let gi = 0, gj = 0;
      for (let k = 0; k < K; k++) {
        const dk = clip((ai[k] - r * aj[k]) / (2 * s2[k] * (li + r * lj)), -xi[k], xj[k]);
        D[k] = dk;
        gi += ai[k] * dk - li * s2[k] * dk * dk;
        gj += -aj[k] * dk - lj * s2[k] * dk * dk;
      }
      const F = target === undefined || target === null ? gi - r * gj : (1 - target) * gi - target * gj;
      return { gi, gj, F, r };
    };
    let lo = -18, hi = 18, it = 0, res;
    const flo = at(lo);
    if (flo.F <= 0) return { d: new Array(K).fill(0), gi: 0, gj: 0, iters: 0, r: 0 };
    const fhi = at(hi);
    if (fhi.F >= 0) return { d: new Array(K).fill(0), gi: 0, gj: 0, iters: 0, r: 0 };
    while (hi - lo > 1e-9 && it < 64) {
      const mid = 0.5 * (lo + hi);
      res = at(mid);
      if (trace) trace.push({ t: mid, F: res.F, gi: res.gi, gj: res.gj });
      if (res.F > 0) lo = mid; else hi = mid;
      it++;
    }
    res = at(0.5 * (lo + hi));
    const tol = 1e-12;
    if (!(res.gi > tol && res.gj > tol)) return { d: new Array(K).fill(0), gi: 0, gj: 0, iters: it, r: res.r };
    return { d: D.slice(), gi: res.gi, gj: res.gj, iters: it, r: res.r };
  }
  // Trace of the weighted problem over a grid of r (for the frontier figure)
  function frontier(K, s2, ai, li, aj, lj, xi, xj, n) {
    const out = [];
    for (let s = 0; s <= n; s++) {
      const t = -8 + (16 * s) / n, r = Math.exp(t);
      let gi = 0, gj = 0;
      for (let k = 0; k < K; k++) {
        const dk = clip((ai[k] - r * aj[k]) / (2 * s2[k] * (li + r * lj)), -xi[k], xj[k]);
        gi += ai[k] * dk - li * s2[k] * dk * dk;
        gj += -aj[k] * dk - lj * s2[k] * dk * dk;
      }
      out.push({ t, gi, gj, F: gi - r * gj });
    }
    return out;
  }

  const bufA = [], bufB = [];
  function trueNash(m, x, i, j) {
    const ai = marg(m, m.v[i], m.agents[i].lam, x[i], bufA[0] || (bufA[0] = new Array(16)));
    const aj = marg(m, m.v[j], m.agents[j].lam, x[j], bufB[0] || (bufB[0] = new Array(16)));
    return solvePair(m.K, m.s2, ai, m.agents[i].lam, aj, m.agents[j].lam, x[i], x[j]);
  }
  function trueGains(m, x, i, j, D) {
    let gi = 0, gj = 0;
    const li = m.agents[i].lam, lj = m.agents[j].lam;
    for (let k = 0; k < m.K; k++) {
      const ai = m.v[i][k] - 2 * li * m.s2[k] * x[i][k], aj = m.v[j][k] - 2 * lj * m.s2[k] * x[j][k];
      gi += ai * D[k] - li * m.s2[k] * D[k] * D[k];
      gj += -aj * D[k] - lj * m.s2[k] * D[k] * D[k];
    }
    return [gi, gj];
  }
  function feasible(m, x, i, j, D) {
    for (let k = 0; k < m.K; k++) if (x[i][k] + D[k] < -1e-12 || x[j][k] - D[k] < -1e-12) return false;
    return true;
  }
  // E_pair and the residual surplus R_pair (mean of g_i + g_j at the Nash trade)
  function pairResidual(m, x) {
    let E = 0, R = 0, n = 0;
    for (let i = 0; i < m.N; i++) for (let j = i + 1; j < m.N; j++) {
      const s = trueNash(m, x, i, j);
      let nn = 0; for (let k = 0; k < m.K; k++) nn += s.d[k] * s.d[k];
      E += Math.sqrt(nn); R += s.gi + s.gj; n++;
    }
    return { E: E / n, R: R / n };
  }

  // Run regime A offline until pairwise stable: the pre-shock allocation.
  function settle(m, x0, maxRounds) {
    const x = x0.map((r) => r.slice());
    const rng = makeRng(hashSeed("settle", m.cfg.seed));
    for (let t = 0; t < (maxRounds || 1500); t++) {
      let moved = 0;
      for (const [i, j] of randomMatching(m.N, rng)) {
        const s = trueNash(m, x, i, j);
        if (s.gi > 0) { for (let k = 0; k < m.K; k++) { x[i][k] += s.d[k]; x[j][k] -= s.d[k]; if (x[i][k] < 0) x[i][k] = 0; if (x[j][k] < 0) x[j][k] = 0; } moved++; }
      }
      if (t % 25 === 24 && pairResidual(m, x).E < 1e-6) break;
      if (!moved && t > 10) break;
    }
    return x;
  }

  function applyShock(m, kind, size) {
    const rng = makeRng(hashSeed("shock", m.cfg.seed, kind));
    m.pre = { v: m.v.map((r) => r.slice()), lam: m.agents.map((a) => a.lam), dStar: m.agents.map((a) => a.dStar), s2: m.s2.slice() };
    m.hit = new Array(m.N).fill(false);
    if (kind === "ldi") {
      m.agents.forEach((a, i) => { if (a.type === "insurer") { a.dStar += 2.5 * size; m.hit[i] = true; } });
    } else if (kind === "riskoff") {
      m.agents.forEach((a, i) => { if (a.type === "bank") { a.lam *= 1 + size; m.hit[i] = true; } });
    } else if (kind === "idio") {
      m.agents.forEach((a, i) => { if (rng() < 0.3) { a.dStar = clip(a.dStar * Math.exp((rng() < 0.5 ? -1 : 1) * 0.5 * size), 0.25, 30); m.hit[i] = true; } });
    } else if (kind === "longvol") {
      m.s2 = m.s2.map((s, k) => s * (1 + size * Math.min(1, m.d[k] / 10)));
      m.agents.forEach((a, i) => { if (a.dStar >= 10) m.hit[i] = true; });
    }
    m.v = m.agents.map((a) => valueVec(m.d, a.dStar, a.tau));
    m.shock = { kind, size };
  }

  // Full setup: market, pre-shock settle, shock. Returns m with m.xOld.
  function setup(cfg, shockKind, shockSize) {
    const m = buildMarket(cfg);
    m.xOld = settle(m, m.xEndow, 1500);
    m.Upre = m.xOld.map((xi, i) => utility(m, i, xi));
    applyShock(m, shockKind, shockSize);
    m.Upost0 = m.xOld.map((xi, i) => utility(m, i, xi));
    // Fixed anchors for welfare and fairness, all under post-shock preferences.
    const eq = m.supply.map((s) => s / m.N);
    m.fair = m.agents.map((a, i) => utility(m, i, eq)); // equal-division share = maximin share (U_i concave)
    m.bliss = m.agents.map((a, i) => utility(m, i, m.d.map((_, k) => clip(m.v[i][k] / (2 * a.lam * m.s2[k]), 0, m.supply[k]))));
    m.lot = 1; // one lot = S_k / (3N), a third of an equal share
    m.ref = fairness(m, m.xOld);
    const res = pairResidual(m, m.xOld);
    m.E0 = res.E; m.R0 = res.R;
    return m;
  }

  // ---------- runs ----------
  function createRun(m, regime, params) {
    const run = { regime, params: Object.assign({}, DEFAULT_PARAMS, params || {}), t: 0, x: m.xOld.map((r) => r.slice()), messages: 0, proposals: 0, rejects: 0, trades: 0, meet: { n: 0, share: 0, eff: 0 }, history: [], log: [] };
    if (regime === "B") initBeliefs(m, run);
    record(m, run);
    return run;
  }
  // Beliefs: agent i's estimate of agent j's theta = (v_j (K values), lam_j).
  // Prior: j's pre-shock theta (what a stable market reveals) plus noise.
  function initBeliefs(m, run) {
    const rng = makeRng(hashSeed("belief", m.cfg.seed));
    const sp = run.params.priorSpread;
    run.bel = [];
    for (let i = 0; i < m.N; i++) {
      const row = [];
      for (let j = 0; j < m.N; j++) {
        if (i === j) { row.push(null); continue; }
        const vv = m.pre.v[j].map((z) => clip(z + 0.12 * sp * rng.normal(), 0, 1.2));
        const lam = m.pre.lam[j] * Math.exp(0.25 * sp * rng.normal());
        row.push({ v: vv, lam });
      }
      run.bel.push(row);
    }
  }
  function beliefError(m, run) {
    if (!run.bel) return null;
    let s = 0, n = 0;
    for (let i = 0; i < m.N; i++) for (let j = 0; j < m.N; j++) {
      if (i === j) continue;
      const b = run.bel[i][j];
      let num = 0, den = 0;
      for (let k = 0; k < m.K; k++) { num += (b.v[k] - m.v[j][k]) ** 2; den += m.v[j][k] ** 2; }
      s += Math.sqrt(num / Math.max(den, 1e-9)); n++;
    }
    return s / n;
  }
  // Halfspace update: g_j(D) = -v.D + lam * b, with b = sum_k s2_k (2 x_jk D_k - D_k^2).
  // accepted => g_j > 0; rejected => g_j <= 0. Project (v, lam) if the belief violates it.
  function halfspace(m, belief, xj, D, accepted) {
    let b = 0, nn = 0;
    for (let k = 0; k < m.K; k++) { b += m.s2[k] * (2 * xj[k] * D[k] - D[k] * D[k]); nn += D[k] * D[k]; }
    const w = D.map((z) => -z); // coefficient on v
    const wl = b; // coefficient on lam
    let g = wl * belief.lam; for (let k = 0; k < m.K; k++) g += w[k] * belief.v[k];
    const norm2 = nn + wl * wl;
    if (norm2 < 1e-18) return;
    const margin = 1e-6 * Math.sqrt(norm2);
    let shift = 0;
    if (accepted && g <= margin) shift = (margin - g) / norm2; // move toward g > 0
    if (!accepted && g >= -margin) shift = (-margin - g) / norm2; // move toward g < 0
    if (!shift) return;
    for (let k = 0; k < m.K; k++) belief.v[k] = clip(belief.v[k] + shift * w[k], 0, 1.5);
    belief.lam = Math.max(0.02, belief.lam + shift * wl);
  }

  // Per-meeting outcome, scored before the trade executes: the proposer's share of the realised
  // surplus, and the realised Nash product as a fraction of the full-information Nash product.
  function scoreMeeting(m, run, i, j, D) {
    const [gi, gj] = trueGains(m, run.x, i, j, D);
    const s = trueNash(m, run.x, i, j), np = s.gi * s.gj;
    run.meet.n++;
    run.meet.share += gi / (gi + gj);
    run.meet.eff += np > 0 ? Math.min(1, (gi * gj) / np) : 1;
  }
  function execute(run, i, j, D) {
    for (let k = 0; k < D.length; k++) { run.x[i][k] = Math.max(0, run.x[i][k] + D[k]); run.x[j][k] = Math.max(0, run.x[j][k] - D[k]); }
    run.trades++;
  }

  // Stylised LLM offer: the bargain it believes in, simplified the way chat offers look.
  function llmOffer(m, run, i, j, aj, lj, attempt, rng) {
    const p = run.params;
    const ai = marg(m, m.v[i], m.agents[i].lam, run.x[i], new Array(m.K));
    const target = 0.5 + (p.llmBias - 0.5) * Math.pow(0.5, attempt);
    const base = solvePair(m.K, m.s2, ai, m.agents[i].lam, aj, lj, run.x[i], run.x[j], target);
    const D = base.d.slice();
    if (D.every((z) => z === 0)) return null;
    // keep the largest few instruments
    const order = D.map((z, k) => [Math.abs(z), k]).sort((a, b) => b[0] - a[0]);
    for (let r = p.llmAssets; r < order.length; r++) D[order[r][1]] = 0;
    for (let k = 0; k < m.K; k++) {
      if (!D[k]) continue;
      D[k] *= Math.exp(p.llmNoise * rng.normal() - 0.5 * p.llmNoise * p.llmNoise);
      if (p.llmLot > 0) D[k] = p.llmLot * Math.round(D[k] / p.llmLot);
      D[k] = clip(D[k], -run.x[i][k], run.x[j][k]);
    }
    if (D.every((z) => z === 0)) return null;
    return D;
  }

  function step(m, run) {
    const rng = makeRng(hashSeed("round", m.cfg.seed, run.t));
    const rngR = makeRng(hashSeed("regime", m.cfg.seed, run.t, run.regime));
    const pairs = randomMatching(m.N, rng);
    const p = run.params;
    for (const pr of pairs) {
      const flip = rng() < 0.5;
      const i = flip ? pr[1] : pr[0], j = flip ? pr[0] : pr[1]; // i proposes
      if (run.regime === "A") {
        const s = trueNash(m, run.x, i, j);
        run.messages += 1;
        if (s.gi > 0) { run.messages += 1; run.proposals++; scoreMeeting(m, run, i, j, s.d); execute(run, i, j, s.d); }
        continue;
      }
      if (run.regime === "B") {
        const ai = marg(m, m.v[i], m.agents[i].lam, run.x[i], new Array(m.K));
        for (let a = 0; a < p.maxProposals; a++) {
          const bel = run.bel[i][j];
          const aj = marg(m, bel.v, bel.lam, run.x[j], new Array(m.K));
          const s = solvePair(m.K, m.s2, ai, m.agents[i].lam, aj, bel.lam, run.x[i], run.x[j]);
          if (s.gi <= 0) break; // believes no trade exists
          run.proposals++; run.messages += 2;
          const [gi, gj] = trueGains(m, run.x, i, j, s.d);
          const ok = gi > 1e-12 && gj > 1e-12 && feasible(m, run.x, i, j, s.d);
          halfspace(m, bel, run.x[j], s.d, ok);
          if (ok) { scoreMeeting(m, run, i, j, s.d); execute(run, i, j, s.d); break; }
          run.rejects++;
        }
        continue;
      }
      // C and D
      let aj, lj = m.agents[j].lam;
      const ajTrue = marg(m, m.v[j], lj, run.x[j], new Array(m.K));
      if (run.regime === "C") aj = ajTrue;
      else {
        run.messages += 2; // disclosure exchange
        aj = ajTrue.map((z) => z + p.discloseNoise * 0.25 * rngR.normal());
        lj = lj * Math.exp(p.discloseNoise * 0.4 * rngR.normal());
      }
      for (let a = 0; a < p.maxProposals; a++) {
        const D = llmOffer(m, run, i, j, aj, lj, a, rngR);
        if (!D) break;
        run.proposals++; run.messages += 2;
        const [gi, gj] = trueGains(m, run.x, i, j, D);
        if (gi > 1e-12 && gj > 1e-12 && feasible(m, run.x, i, j, D)) { scoreMeeting(m, run, i, j, D); execute(run, i, j, D); break; }
        run.rejects++;
      }
    }
    run.t++;
    record(m, run);
  }

  function record(m, run) {
    const r = pairResidual(m, run.x);
    const h = { t: run.t, E: r.E, R: r.R, messages: run.messages, trades: run.trades, proposals: run.proposals, rejects: run.rejects, belief: run.regime === "B" ? beliefError(m, run) : null };
    h.share = run.meet.n ? run.meet.share / run.meet.n : null;
    h.eff = run.meet.n ? run.meet.eff / run.meet.n : null;
    run.history.push(Object.assign(h, fairness(m, run.x)));
  }

  // ---------- limited information: one proposal under a perturbed belief ----------
  // For every ordered pair with a strictly improving trade at x, replace theta_j = (v_j, lam_j) by
  // v_j + e z, lam_j e^{e z'} and let i propose the predicted Nash trade. Score acceptance, the
  // Nash-product loss 1 - g_i g_j / NP* and the shift in i's share of the surplus. Pairs are split by
  // full-information surplus so the local (large-surplus) and aggregate behaviour can be compared.
  function beliefSweep(m, x, es, reps) {
    const rng = makeRng(hashSeed("sweep", m.cfg.seed));
    const pairs = [], ai = new Array(m.K), aj = new Array(m.K);
    for (let i = 0; i < m.N; i++) for (let j = 0; j < m.N; j++) {
      if (i === j) continue;
      const s = trueNash(m, x, i, j);
      if (!(s.gi > 1e-9 && s.gj > 1e-9)) continue;
      const np = s.gi * s.gj, sh = s.gi / (s.gi + s.gj);
      marg(m, m.v[i], m.agents[i].lam, x[i], ai);
      const res = es.map(() => ({ n: 0, acc: 0, loss: 0, shift: 0 }));
      for (let r = 0; r < reps; r++) {
        const z = m.v[j].map(() => rng.normal()), zl = rng.normal();
        es.forEach((e, q) => {
          const lh = m.agents[j].lam * Math.exp(e * zl);
          marg(m, m.v[j].map((v, k) => v + e * z[k]), lh, x[j], aj);
          const p = solvePair(m.K, m.s2, ai, m.agents[i].lam, aj, lh, x[i], x[j]);
          const R = res[q]; R.n++;
          if (!(p.gi > 0)) return;
          const [gi, gj] = trueGains(m, x, i, j, p.d);
          if (gi > 1e-12 && gj > 1e-12) { R.acc++; R.loss += 1 - (gi * gj) / np; R.shift += Math.abs(gi / (gi + gj) - sh); }
        });
      }
      pairs.push({ s: s.gi + s.gj, res });
    }
    pairs.sort((a, b) => a.s - b.s);
    const agg = (G) => es.map((e, q) => {
      let n = 0, acc = 0, loss = 0, shift = 0;
      for (const p of G) { const R = p.res[q]; n += R.n; acc += R.acc; loss += R.loss; shift += R.shift; }
      return { e, acc: n ? acc / n : null, loss: acc ? loss / acc : null, shift: acc ? shift / acc : null };
    });
    return { pairs: pairs.length, top: agg(pairs.slice(Math.floor((3 * pairs.length) / 4))), all: agg(pairs) };
  }

  // ---------- welfare and fairness of an allocation ----------
  // Utilities are concave and not monotone (holding too much of a bond hurts), and there is no free
  // disposal, so U_i can be negative. Normalised utility u_i = U_i(x_i) / U_i(bliss) uses fixed anchors
  // U_i(0) = 0 and the bliss bundle, which makes it invariant to positive affine rescaling of U_i.
  // EF1 and PROP1 treat one lot of an instrument as an item and use the goods-and-chores versions
  // (Aziz et al.): the item may be removed from either bundle, since a lot can be a good or a chore.
  function fairness(m, x) {
    const N = m.N, K = m.K, q = m.lot, tol = 1e-9;
    const U = x.map((xi, i) => utility(m, i, xi));
    let W = 0, sumU = 0, minU = Infinity, ef = 0, ef1 = 0, prop = 0, prop1 = 0, maxEnvy = 0;
    const g = [];
    for (let i = 0; i < N; i++) {
      const v = m.v[i], lam = m.agents[i].lam, xi = x[i];
      const dU = (y, k, t) => t * (v[k] - lam * m.s2[k] * (2 * y[k] + t)); // U_i(y + t e_k) - U_i(y)
      W += U[i] - m.Upost0[i]; g.push(Math.max(0, U[i] - m.Upost0[i]));
      const u = U[i] / m.bliss[i]; sumU += u; if (u < minU) minU = u;
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const Uij = utility(m, i, x[j]), envy = Uij - U[i];
        if (envy / m.bliss[i] > maxEnvy) maxEnvy = envy / m.bliss[i];
        if (envy <= tol) { ef++; ef1++; continue; }
        for (let k = 0; k < K; k++) {
          const tj = Math.min(q, x[j][k]), ti = Math.min(q, xi[k]);
          if ((tj > 0 && U[i] >= Uij + dU(x[j], k, -tj) - tol) || (ti > 0 && U[i] + dU(xi, k, -ti) >= Uij - tol)) { ef1++; break; }
        }
      }
      if (U[i] >= m.fair[i] - tol) { prop++; prop1++; continue; }
      for (let k = 0; k < K; k++) {
        const ta = Math.min(q, m.supply[k] - xi[k]), tr = Math.min(q, xi[k]);
        if ((ta > 0 && U[i] + dU(xi, k, ta) >= m.fair[i] - tol) || (tr > 0 && U[i] + dU(xi, k, -tr) >= m.fair[i] - tol)) { prop1++; break; }
      }
    }
    let gs = 0, gd = 0;
    for (let i = 0; i < N; i++) { gs += g[i]; for (let j = 0; j < N; j++) gd += Math.abs(g[i] - g[j]); }
    const P = N * (N - 1);
    return { W, meanU: sumU / N, minU, ef: ef / P, ef1: ef1 / P, prop: prop / N, prop1: prop1 / N, maxEnvy, gini: gs > 1e-12 ? gd / (2 * N * gs) : null };
  }
  function epsTime(run, eps) { const h = run.history.find((r) => r.E <= eps); return h ? h.t : null; }
  function gainsSince(m, run) { return run.x.map((xi, i) => utility(m, i, xi) - m.Upost0[i]); }

  // =====================================================================================
  // Part 2: endogenous term structure
  // =====================================================================================
  const MATURITIES = [0.5, 1, 2, 3, 5, 7, 10, 20, 30];
  // Illustrative Treasury curve (Nelson-Siegel), continuously compounded.
  function treasuryYield(T) {
    const b0 = 0.0445, b1 = -0.006, b2 = 0.008, L = 2.0;
    const z = T / L, f = (1 - Math.exp(-z)) / z;
    return b0 + b1 * f + b2 * (f - Math.exp(-z));
  }
  const BOND_SHOCKS = {
    vol: { label: "Default scare at the long end", detail: "Payoff uncertainty σₖ rises with maturity: by 60% × size at 10 years and beyond, proportionally less below." },
    loss: { label: "Expected-loss shock", detail: "Expected annual loss rises by 30 bp × size at 10 years and beyond, proportionally less below, so μₖ falls." },
    short: { label: "Short-end funding stress", detail: "Payoff uncertainty of bonds up to 2 years rises by 60% × size; longer bonds see 15% of that." },
  };
  function buildBondMarket(cfg) {
    const N = cfg.N, T = MATURITIES, K = T.length;
    const rng = makeRng(hashSeed("bonds", cfg.seed, N));
    const loss0 = 0.004;
    const mu = T.map((t) => Math.exp(-loss0 * t));
    const sd = T.map((t) => 0.03 + 0.012 * Math.pow(t, 0.8));
    const agents = [];
    for (let i = 0; i < N; i++) {
      const ty = TYPES[i % TYPES.length];
      agents.push({
        type: ty.key, typeIdx: i % TYPES.length,
        dStar: rng.range(ty.d[0], ty.d[1]) * 1.2, tau: rng.range(ty.tau[0], ty.tau[1]) * 1.3,
        lam: rng.range(1.0, 4.0), alpha: rng.range(0.004, 0.02), rho: treasuryYield(5) + rng.range(-0.008, 0.008),
      });
    }
    const v = agents.map((a) => valueVec(T, a.dStar, a.tau));
    const supply = T.map(() => N);
    const x = agents.map(() => new Array(K).fill(0));
    for (let k = 0; k < K; k++) {
      const w = agents.map(() => -Math.log(1 - rng()) + 0.05);
      const tot = w.reduce((s, z) => s + z, 0);
      for (let i = 0; i < N; i++) x[i][k] = (supply[k] * w[i]) / tot;
    }
    const money = agents.map(() => 60);
    const bm = { N, K, T, mu, sd, loss0, agents, v, supply, discount: cfg.discount, cfg: Object.assign({}, cfg), xEndow: x, mEndow: money };
    bm.D = discountMatrix(bm);
    return bm;
  }
  function discountMatrix(bm) {
    return bm.agents.map((a) => bm.T.map((t) => (bm.discount === "individual" ? Math.exp(-a.rho * t) : Math.exp(-treasuryYield(t) * t))));
  }
  function bondMarginal(bm, i, k, xik) {
    const a = bm.agents[i], D = bm.D[i][k];
    return D * bm.mu[k] + a.alpha * bm.v[i][k] - 2 * a.lam * D * D * bm.sd[k] * bm.sd[k] * xik;
  }
  function bondCurv(bm, i, k) { const D = bm.D[i][k]; return bm.agents[i].lam * D * D * bm.sd[k] * bm.sd[k]; }
  // TU Nash bargain: per bond, maximise joint surplus; money splits it equally.
  function bondTrade(bm, x, m, i, j) {
    const D = new Array(bm.K).fill(0);
    let gi = 0, gj = 0;
    for (let k = 0; k < bm.K; k++) {
      const pi = bondMarginal(bm, i, k, x[i][k]), pj = bondMarginal(bm, j, k, x[j][k]);
      const ci = bondCurv(bm, i, k), cj = bondCurv(bm, j, k);
      const d = clip((pi - pj) / (2 * (ci + cj)), -x[i][k], x[j][k]);
      D[k] = d; gi += pi * d - ci * d * d; gj += -pj * d - cj * d * d;
    }
    const S = gi + gj;
    if (!(S > 1e-14)) return null;
    let dm = (gj - gi) / 2; // money to i (negative: i pays)
    if (m[i] + dm < 0 || m[j] - dm < 0) return null;
    return { d: D, dm, surplus: S };
  }
  function bondState(bm, x) {
    // implied prices: holdings-weighted mean of holders' marginal valuations; band: min/max yields among holders
    const out = [];
    let disp = 0;
    for (let k = 0; k < bm.K; k++) {
      let wsum = 0, psum = 0, ylo = Infinity, yhi = -Infinity, ys = [];
      const thresh = (1e-3 * bm.supply[k]) / bm.N;
      for (let i = 0; i < bm.N; i++) {
        if (x[i][k] <= thresh) continue;
        const p = bondMarginal(bm, i, k, x[i][k]);
        const y = -Math.log(Math.max(p, 1e-6)) / bm.T[k];
        wsum += x[i][k]; psum += x[i][k] * p; ys.push(y);
        if (y < ylo) ylo = y; if (y > yhi) yhi = y;
      }
      const P = psum / wsum, y = -Math.log(P) / bm.T[k];
      const mean = ys.reduce((s, z) => s + z, 0) / ys.length;
      const sdv = Math.sqrt(ys.reduce((s, z) => s + (z - mean) * (z - mean), 0) / ys.length);
      disp += sdv;
      const tr = treasuryYield(bm.T[k]);
      out.push({ T: bm.T[k], P, y, ylo, yhi, sd: sdv, holders: ys.length, treasury: tr, spread: bm.discount === "individual" ? null : y - tr });
    }
    return { curve: out, dispersion: disp / bm.K };
  }
  function bondPairResidual(bm, x, m) {
    let E = 0, n = 0;
    for (let i = 0; i < bm.N; i++) for (let j = i + 1; j < bm.N; j++) {
      const s = bondTrade(bm, x, m, i, j); n++;
      if (s) { let q = 0; for (const z of s.d) q += z * z; E += Math.sqrt(q); }
    }
    return E / n;
  }
  function settleBonds(bm) {
    const x = bm.xEndow.map((r) => r.slice()), m = bm.mEndow.slice();
    const rng = makeRng(hashSeed("bsettle", bm.cfg.seed));
    for (let t = 0; t < 600; t++) {
      for (const [i, j] of randomMatching(bm.N, rng)) { const s = bondTrade(bm, x, m, i, j); if (s) applyBond(x, m, i, j, s); }
      if (t % 20 === 19 && bondPairResidual(bm, x, m) < 1e-7) break;
    }
    return { x, m };
  }
  function applyBond(x, m, i, j, s) {
    for (let k = 0; k < s.d.length; k++) { x[i][k] = Math.max(0, x[i][k] + s.d[k]); x[j][k] = Math.max(0, x[j][k] - s.d[k]); }
    m[i] += s.dm; m[j] -= s.dm;
  }
  function shockBonds(bm, kind, size) {
    bm.pre = { mu: bm.mu.slice(), sd: bm.sd.slice() };
    if (kind === "vol") bm.sd = bm.sd.map((s, k) => s * (1 + 0.6 * size * Math.min(1, bm.T[k] / 10)));
    else if (kind === "loss") bm.mu = bm.T.map((t) => Math.exp(-(bm.loss0 + 0.003 * size * Math.min(1, t / 10)) * t));
    else if (kind === "short") bm.sd = bm.sd.map((s, k) => s * (1 + 0.6 * size * (bm.T[k] <= 2 ? 1 : 0.15)));
    bm.shock = { kind, size };
  }
  function setupBonds(cfg, kind, size) {
    const bm = buildBondMarket(cfg);
    const s = settleBonds(bm);
    bm.preState = bondState(bm, s.x);
    shockBonds(bm, kind, size);
    const run = { t: 0, x: s.x.map((r) => r.slice()), m: s.m.slice(), trades: 0, history: [] };
    bm.shockState = bondState(bm, run.x);
    recordBond(bm, run);
    return { bm, run };
  }
  function stepBonds(bm, run) {
    const rng = makeRng(hashSeed("bround", bm.cfg.seed, run.t));
    for (const [i, j] of randomMatching(bm.N, rng)) { const s = bondTrade(bm, run.x, run.m, i, j); if (s) { applyBond(run.x, run.m, i, j, s); run.trades++; } }
    run.t++;
    recordBond(bm, run);
  }
  function recordBond(bm, run) {
    const st = bondState(bm, run.x);
    run.state = st;
    run.history.push({ t: run.t, disp: st.dispersion, E: bondPairResidual(bm, run.x, run.m) });
  }

  const api = {
    makeRng, hashSeed, randomMatching, DURATION_SETS, TYPES, SHOCKS, DEFAULT_PARAMS,
    valueVec, buildMarket, utility, solvePair, frontier, trueNash, trueGains, pairResidual, settle, applyShock, setup,
    createRun, step, epsTime, gainsSince, beliefError, marg, fairness, beliefSweep,
    MATURITIES, treasuryYield, BOND_SHOCKS, setupBonds, stepBonds, bondState, bondMarginal,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Engine = api;
})(typeof window !== "undefined" ? window : this);
