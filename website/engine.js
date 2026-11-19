/* Decentralized Equilibrium Discovery After Shocks: OTC corporate-bond market with cash.
   Model, notation, data and conventions follow the interim deck (corporate_bond_with_notation.pdf)
   and its handoff note. Quasilinear cash with unrestricted borrowing; symmetric Nash bargaining.
   The random draws use this file's own generator, so seeds do not reproduce numpy's draws; a market
   can instead be built from explicit types (cfg.spec), which is how the deck's seed-0 reference
   values are checked. */
(function (root) {
  "use strict";

  // ---------- random numbers ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeRng(seed) {
    const u = mulberry32(seed);
    let spare = null;
    const f = () => u();
    f.range = (lo, hi) => lo + (hi - lo) * u();
    f.normal = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let a, b, r;
      do { a = 2 * u() - 1; b = 2 * u() - 1; r = a * a + b * b; } while (r === 0 || r >= 1);
      const k = Math.sqrt(-2 * Math.log(r) / r);
      spare = b * k; return a * k;
    };
    f.perm = (n) => { const p = []; for (let i = 0; i < n; i++) p.push(i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(u() * (i + 1)); const t = p[i]; p[i] = p[j]; p[j] = t; } return p; };
    return f;
  }
  function hashSeed() {
    let h = 2166136261 >>> 0;
    for (let a = 0; a < arguments.length; a++) {
      const s = String(arguments[a]);
      for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
      h ^= 0x9e3779b9; h = Math.imul(h, 16777619) >>> 0;
    }
    return h >>> 0;
  }
  const clip = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  // ---------- data (deck slide 5; handoff section 2.1) ----------
  const T = [2, 4, 6, 8.5, 12.5, 20];
  const BUCKET = ["1–3y", "3–5y", "5–7y", "7–10y", "10–15y", "15y+"];
  const TSY = [3.96, 4.025, 4.13, 4.25, 4.4, 4.67].map((v) => v / 100); // Treasury yields 2025-03-25, interpolated
  const OAS = [57.6, 78.8, 91.8, 110.6, 113.2, 112.8].map((v) => v / 1e4); // ICE BofA OAS, mean of 2025-03-24..28
  const K = T.length;
  const D = T.map((t, k) => Math.exp(-TSY[k] * t));
  const P0 = T.map((t, k) => D[k] * Math.exp(-OAS[k] * t));
  const spreadOf = (P, k) => -Math.log(P / D[k]) / T[k];

  const VARIANTS = {
    baseline: { label: "Baseline", detail: "λᵢ = 20uᵢ with uᵢ ~ U(0.5, 1.5); α = 0.03; h = 0.003.", alpha: 0.03, h: 0.003, wide: false },
    wide: { label: "100-fold risk-aversion spread", detail: "λᵢ = 20·exp(U(ln 0.1, ln 10)), so λ ranges from 2 to 200.", alpha: 0.03, h: 0.003, wide: true },
    zeros: { label: "Many zero holdings", detail: "α = 0.3 and h = 0.04, so many investors hold none of some bonds.", alpha: 0.3, h: 0.04, wide: false },
  };
  const SHOCKS = {
    riskUp: { label: "Risk-aversion shock, λ × 3", kind: "risk", mult: 3 },
    riskDown: { label: "Risk-aversion shock, λ ÷ 3", kind: "risk", mult: 1 / 3 },
    hazUp: { label: "Default-hazard shock, h × 1.25", kind: "hazard", mult: 1.25 },
    hazDown: { label: "Default-hazard shock, h × 0.8", kind: "hazard", mult: 0.8 },
  };
  const GROUPS = [[1, 4], [4, 9], [9, 25]]; // preferred maturity d* by group (short, medium, long)
  const LGD = 0.6;
  const GAIN_TOL = 2e-12;

  function credit(h, L) {
    if (L === undefined) L = LGD;
    const PD = T.map((t) => 1 - Math.exp(-h * t));
    return { mu: PD.map((p) => 1 - L * p), s2: PD.map((p) => L * L * p * (1 - p)) };
  }
  const vPref = (t, dstar, tau) => Math.exp(-((t - dstar) * (t - dstar)) / (2 * tau * tau));
  // c_ik = D_k mu_k + alpha v_ik ; b_ik = lam_i D_k^2 s2_k
  function economy(m, lam, h) {
    const cr = credit(h, m.L);
    const c = [], b = [];
    for (let i = 0; i < m.N; i++) {
      c.push(T.map((t, k) => D[k] * cr.mu[k] + m.alpha * vPref(t, m.dstar[i], m.tau[i])));
      b.push(T.map((t, k) => lam[i] * D[k] * D[k] * cr.s2[k]));
    }
    return { c, b, mu: cr.mu, s2: cr.s2, lam: lam.slice(), h };
  }
  const bondU = (e, i, xi) => { let u = 0; for (let k = 0; k < K; k++) u += e.c[i][k] * xi[k] - e.b[i][k] * xi[k] * xi[k]; return u; };
  const welfare = (e, x) => { let w = 0; for (let i = 0; i < x.length; i++) w += bondU(e, i, x[i]); return w; };
  const marginal = (e, i, xi, k) => e.c[i][k] - 2 * e.b[i][k] * xi[k];

  // Terminal allocation: for each bond, the common marginal value P*_k that clears supply (handoff 2.4).
  function terminal(e, X, N) {
    const xs = []; for (let i = 0; i < N; i++) xs.push(new Array(K).fill(0));
    const Ps = [];
    for (let k = 0; k < K; k++) {
      let lo = -5, hi = 5;
      for (let it = 0; it < 200; it++) {
        const mid = 0.5 * (lo + hi);
        let d = 0; for (let i = 0; i < N; i++) d += Math.max(0, (e.c[i][k] - mid) / (2 * e.b[i][k]));
        if (d > X[k]) lo = mid; else hi = mid;
      }
      const P = 0.5 * (lo + hi); Ps.push(P);
      for (let i = 0; i < N; i++) xs[i][k] = Math.max(0, (e.c[i][k] - P) / (2 * e.b[i][k]));
    }
    return { xs, Ps, W: welfare(e, xs) };
  }

  // Build the deck's market: draw types, calibrate holdings to P0, apply the shock, solve the terminal state.
  function buildMarket(cfg) {
    const N = cfg.N, V = VARIANTS[cfg.variant || "baseline"], sh = SHOCKS[cfg.shock || "riskUp"];
    const rng = makeRng(hashSeed("market", cfg.seed, N, cfg.variant || "baseline"));
    // hyperparameters: explicit settings override the variant's defaults
    const pick = (v, d) => (v === undefined || v === null ? d : v);
    const lamBase = pick(cfg.lamBase, 20), wide = pick(cfg.wide, V.wide), mult = pick(cfg.shockMult, sh.mult);
    const m = { N, K, alpha: pick(cfg.alpha, V.alpha), h0: pick(cfg.h, V.h), L: pick(cfg.L, LGD), lamBase, wide, cfg: Object.assign({}, cfg) };
    if (cfg.spec) {
      m.dstar = cfg.spec.dstar.slice(); m.tau = cfg.spec.tau.slice(); m.lam0 = cfg.spec.lam.slice();
    } else {
      m.dstar = []; const g = Math.floor(N / 3);
      for (let q = 0; q < 3; q++) for (let n = 0; n < (q < 2 ? g : N - 2 * g); n++) m.dstar.push(rng.range(GROUPS[q][0], GROUPS[q][1]));
      m.tau = []; for (let i = 0; i < N; i++) m.tau.push(rng.range(1.5, 4));
      m.lam0 = []; for (let i = 0; i < N; i++) m.lam0.push(wide ? lamBase * Math.exp(rng.range(Math.log(0.1), Math.log(10))) : lamBase * rng.range(0.5, 1.5));
    }
    m.group = m.dstar.map((d) => (d < 4 ? 0 : d < 9 ? 1 : 2));
    m.pre = economy(m, m.lam0, m.h0);
    m.x0 = []; for (let i = 0; i < N; i++) m.x0.push(T.map((t, k) => Math.max(0, (m.pre.c[i][k] - P0[k]) / (2 * m.pre.b[i][k]))));
    m.X = T.map((t, k) => m.x0.reduce((s, xi) => s + xi[k], 0));
    // shock
    m.hit = new Array(N).fill(false);
    const lam1 = m.lam0.slice(); let h1 = m.h0;
    if (sh.kind === "risk") {
      let idx;
      if (cfg.spec && cfg.spec.hit) idx = cfg.spec.hit;
      else { const p = makeRng(hashSeed("hit", cfg.seed, N)).perm(N); idx = p.slice(0, Math.round((cfg.share || 0.1) * N)); }
      for (const i of idx) { lam1[i] *= mult; m.hit[i] = true; }
    } else { h1 = m.h0 * mult; m.hit.fill(true); }
    m.post = economy(m, lam1, h1);
    m.shock = Object.assign({}, sh, { key: cfg.shock || "riskUp", share: sh.kind === "risk" ? cfg.share || 0.1 : 1, mult });
    const tm = terminal(m.post, m.X, N);
    m.xs = tm.xs; m.Ps = tm.Ps; m.Wstar = tm.W;
    m.W0 = welfare(m.post, m.x0);
    m.m0 = new Array(N).fill(50);
    m.value0 = m.x0.map((xi) => xi.reduce((s, v, k) => s + v * P0[k], 0)); // pre-shock portfolio value, for bp
    m.U0 = m.x0.map((xi, i) => bondU(m.post, i, xi)); // post-shock bond utility at the post-shock no-trade state
    m.lot = m.X.map((Xk) => Xk / (3 * N));
    m.fair0 = fairness(m, m.x0, m.m0);
    return m;
  }

  // ---------- one meeting ----------
  // Nash bargain with cash (deck slides 3-4): per-bond clipped transfer, then cash splits the surplus equally.
  // ai, bi: proposer-side marginal values and curvatures; aj, bj: counterpart's (true or estimated).
  function nashMoney(ai, bi, xi, aj, bj, xj) {
    const d = new Array(K); let Ai = 0, Aj = 0;
    for (let k = 0; k < K; k++) {
      const dk = clip((ai[k] - aj[k]) / (2 * (bi[k] + bj[k])), -xi[k], xj[k]);
      d[k] = dk; Ai += ai[k] * dk - bi[k] * dk * dk; Aj += -aj[k] * dk - bj[k] * dk * dk;
    }
    return { d, Ai, Aj, S: Ai + Aj, z: 0.5 * (Ai - Aj) };
  }
  const margVec = (e, i, xi) => T.map((t, k) => e.c[i][k] - 2 * e.b[i][k] * xi[k]);
  function trueNash(m, x, i, j) {
    const e = m.post;
    return nashMoney(margVec(e, i, x[i]), e.b[i], x[i], margVec(e, j, x[j]), e.b[j], x[j]);
  }
  function bondGains(m, x, i, j, d) {
    const e = m.post; let Ai = 0, Aj = 0;
    for (let k = 0; k < K; k++) {
      const ai = marginal(e, i, x[i], k), aj = marginal(e, j, x[j], k);
      Ai += ai * d[k] - e.b[i][k] * d[k] * d[k]; Aj += -aj * d[k] - e.b[j][k] * d[k] * d[k];
    }
    return [Ai, Aj];
  }
  const feasible = (x, i, j, d) => d.every((v, k) => x[i][k] + v >= -1e-12 && x[j][k] - v >= -1e-12);

  // ---------- negotiation methods (handoff section 5 names) ----------
  const METHODS = {
    true: { name: "True-utility solver", short: "True-utility", info: "full", by: "solver" },
    est: { name: "Estimated-utility solver", short: "Estimated-utility", info: "private", by: "solver" },
    llmFull: { name: "Full-information LLM negotiation", short: "LLM, full info", info: "full", by: "llm" },
    llmPriv: { name: "Private-information LLM negotiation", short: "LLM, private", info: "private", by: "llm" },
  };
  const DEFAULT_PARAMS = {
    maxProposals: 3, // proposals per meeting (estimated-utility solver and both LLM methods)
    prior: "diffuse", // estimated-utility solver: "diffuse" or "holdings" (pre-shock type inferred from public x0)
    tape: false, // estimated-utility solver: use last round's executed prices as a public signal
    particles: 40,
    llmAssets: 3, // stylised LLM: bonds per offer
    llmLot: 0.005, // stylised LLM: lot size (units of bond)
    llmNoise: 0.3, // stylised LLM: size noise
    llmAsk: 0.65, // stylised LLM: proposer's ask, share of predicted surplus
    discloseNoise: 0.5, // private LLM: disclosure noise, in units of 20 bp of price
  };

  function createRun(m, method, params) {
    const run = {
      method, params: Object.assign({}, DEFAULT_PARAMS, params || {}), t: 0,
      x: m.x0.map((r) => r.slice()), cash: m.m0.slice(), messages: 0, proposals: 0, rejects: 0, trades: 0, silent: 0,
      meet: { n: 0, eff: 0, share: 0 }, tape: null, history: [], rounds: [], bel: null,
    };
    if (method === "est") run.bel = new Map();
    record(m, run);
    return run;
  }

  // Particle beliefs about a counterpart's private type theta_j = (d*, tau, lam), as in the older
  // ParticleBeliefs code. Prior "diffuse": public ranges only (group unknown, lam log-uniform 3..100).
  // Prior "holdings": pre-shock type recovered from public holdings x0 and P0 (it is identified),
  // so only the shock to lam is unknown.
  function belief(m, run, i, j) {
    const key = i * m.N + j;
    let b = run.bel.get(key);
    if (b) return b;
    const P = run.params.particles, rng = makeRng(hashSeed("prior", m.cfg.seed, i, j));
    b = { d: new Float64Array(P), tau: new Float64Array(P), lam: new Float64Array(P), w: new Float64Array(P).fill(1 / P) };
    for (let p = 0; p < P; p++) {
      if (run.params.prior === "holdings") {
        b.d[p] = m.dstar[j]; b.tau[p] = m.tau[j];
        const r = rng(), sh = m.shock;
        const mult = sh.kind === "risk" ? (r < 0.5 ? 1 : r < 0.75 ? 3 : 1 / 3) : 1;
        b.lam[p] = m.lam0[j] * mult;
      } else if (run.params.prior === "deck") {
        // the deck's own parameter ranges: group unknown, the shock's description public but not who was hit
        const g = GROUPS[Math.floor(rng() * 3)], sh = m.shock;
        b.d[p] = rng.range(g[0], g[1]); b.tau[p] = rng.range(1.5, 4);
        let lam = m.lamBase * (m.wide ? Math.exp(rng.range(Math.log(0.1), Math.log(10))) : rng.range(0.5, 1.5));
        if (sh.kind === "risk" && rng() < sh.share) lam *= sh.mult;
        b.lam[p] = lam;
      } else {
        const g = GROUPS[Math.floor(rng() * 3)];
        b.d[p] = rng.range(g[0], g[1]); b.tau[p] = rng.range(1.5, 4);
        b.lam[p] = Math.exp(rng.range(Math.log(3), Math.log(100)));
      }
    }
    run.bel.set(key, b);
    return b;
  }
  function particleCB(m, b, p, k) {
    const e = m.post;
    return [D[k] * e.mu[k] + m.alpha * vPref(T[k], b.d[p], b.tau[p]), b.lam[p] * D[k] * D[k] * e.s2[k]];
  }
  function estimate(m, b, xj) {
    const a = new Array(K).fill(0), bb = new Array(K).fill(0);
    for (let p = 0; p < b.w.length; p++) {
      const w = b.w[p]; if (!w) continue;
      for (let k = 0; k < K; k++) { const [c, bk] = particleCB(m, b, p, k); a[k] += w * (c - 2 * bk * xj[k]); bb[k] += w * bk; }
    }
    return { a, b: bb };
  }
  function updateBelief(m, b, xj, d, z, accepted, rng) {
    let tot = 0;
    for (let p = 0; p < b.w.length; p++) {
      let g = z;
      for (let k = 0; k < K; k++) { const [c, bk] = particleCB(m, b, p, k); g += -(c - 2 * bk * xj[k]) * d[k] - bk * d[k] * d[k]; }
      if ((g > 0) !== accepted) b.w[p] *= 0.02;
      tot += b.w[p];
    }
    let ess = 0;
    for (let p = 0; p < b.w.length; p++) { b.w[p] /= tot; ess += b.w[p] * b.w[p]; }
    if (1 / ess < b.w.length / 3) { // resample with jitter
      const P = b.w.length, nd = new Float64Array(P), nt = new Float64Array(P), nl = new Float64Array(P);
      let c = b.w[0], q = 0; const u0 = rng() / P;
      for (let p = 0; p < P; p++) {
        const u = u0 + p / P; while (u > c && q < P - 1) { q++; c += b.w[q]; }
        nd[p] = clip(b.d[q] + 0.25 * rng.normal(), 1, 25); nt[p] = clip(b.tau[q] + 0.08 * rng.normal(), 1.5, 4); nl[p] = clip(b.lam[q] * Math.exp(0.06 * rng.normal()), 0.5, 600);
      }
      b.d = nd; b.tau = nt; b.lam = nl; b.w.fill(1 / P);
    }
  }
  function beliefError(m, run) {
    if (!run.bel || !run.bel.size) return null;
    let s = 0, n = 0;
    for (const [key, b] of run.bel) {
      const j = key % m.N; let l = 0;
      for (let p = 0; p < b.w.length; p++) l += b.w[p] * Math.log(b.lam[p]);
      s += Math.abs(l - Math.log(m.post.lam[j])); n++;
    }
    return s / n; // mean |log(lam_hat / lam)|
  }

  // Stylised LLM offer: start from the bargain it believes in, keep a few bonds, round to lots, add size
  // noise, and ask for a share of the predicted surplus that softens after each rejection.
  function llmOffer(m, run, i, j, aj, bj, attempt, rng) {
    const p = run.params, e = m.post, x = run.x;
    const ai = margVec(e, i, x[i]);
    const base = nashMoney(ai, e.b[i], x[i], aj, bj, x[j]);
    if (!(base.S > GAIN_TOL)) return null;
    const d = base.d.slice();
    const order = d.map((v, k) => [Math.abs(v), k]).sort((a, b) => b[0] - a[0]);
    for (let r = p.llmAssets; r < K; r++) d[order[r][1]] = 0;
    for (let k = 0; k < K; k++) {
      if (!d[k]) continue;
      d[k] *= Math.exp(p.llmNoise * rng.normal() - 0.5 * p.llmNoise * p.llmNoise);
      if (p.llmLot > 0) d[k] = p.llmLot * Math.round(d[k] / p.llmLot);
      d[k] = clip(d[k], -x[i][k], x[j][k]);
    }
    if (d.every((v) => v === 0)) return null;
    let Ai = 0, Aj = 0;
    for (let k = 0; k < K; k++) { Ai += ai[k] * d[k] - e.b[i][k] * d[k] * d[k]; Aj += -aj[k] * d[k] - bj[k] * d[k] * d[k]; }
    const S = Ai + Aj; if (!(S > GAIN_TOL)) return null;
    const ask = 0.5 + (p.llmAsk - 0.5) * Math.pow(0.5, attempt);
    return { d, z: Ai - ask * S, ai, aj, bj };
  }

  // per-bond executed prices: proposer's equal-split quote per bond, shifted pro rata so cash adds up to z
  function tapeAdd(tape, d, z, ai, bi, aj, bj) {
    let base = 0, abs = 0; const pk = new Array(K);
    for (let k = 0; k < K; k++) { pk[k] = 0.5 * (ai[k] + aj[k]) - 0.5 * (bi[k] - bj[k]) * d[k]; base += pk[k] * d[k]; abs += Math.abs(d[k]); }
    const adj = abs > 0 ? (z - base) / abs : 0;
    for (let k = 0; k < K; k++) {
      const q = Math.abs(d[k]); if (q < 1e-12) continue;
      const price = pk[k] + adj * Math.sign(d[k]);
      tape.q[k] += q; tape.pq[k] += q * price;
    }
  }

  function execute(m, run, i, j, d, z) {
    const [Ai, Aj] = bondGains(m, run.x, i, j, d);
    const best = trueNash(m, run.x, i, j);
    run.meet.n++; run.meet.eff += best.S > 0 ? Math.min(1, (Ai + Aj) / best.S) : 1; run.meet.share += (Ai - z) / (Ai + Aj);
    for (let k = 0; k < K; k++) { run.x[i][k] = Math.max(0, run.x[i][k] + d[k]); run.x[j][k] = Math.max(0, run.x[j][k] - d[k]); }
    run.cash[i] -= z; run.cash[j] += z; run.trades++;
    run._last = { d: d.slice(), z };
  }

  function step(m, run) {
    const N = m.N, e = m.post, p = run.params;
    const sp = m.cfg.spec && m.cfg.spec.perms && m.cfg.spec.perms[run.t];
    const perm = sp || makeRng(hashSeed("match", m.cfg.seed, run.t)).perm(N);
    const rng = makeRng(hashSeed("method", m.cfg.seed, run.t, run.method));
    const tape = { q: new Array(K).fill(0), pq: new Array(K).fill(0) };
    const x0 = run.x.map((r) => r.slice()); // all meetings in a round use start-of-round holdings
    const prevTape = run.tape;
    const rlog = { t: run.t + 1, x: x0, cash: run.cash.slice(), meetings: [] }; // round t+1, start-of-round state
    for (let n = 0; n + 1 < N; n += 2) {
      const i = perm[n], j = perm[n + 1]; // i proposes and receives the bond vector
      const rec = { i, j, d: null, z: 0, props: 0 }, p0 = run.proposals, tr0 = run.trades;
      meeting: {
      const ai = margVec(e, i, x0[i]);
      if (run.method === "true") {
        const s = nashMoney(ai, e.b[i], x0[i], margVec(e, j, x0[j]), e.b[j], x0[j]);
        run.messages += 1;
        if (s.S > GAIN_TOL) { run.messages += 1; run.proposals++; tapeAdd(tape, s.d, s.z, ai, e.b[i], margVec(e, j, x0[j]), e.b[j]); execute(m, run, i, j, s.d, s.z); }
        break meeting;
      }
      if (run.method === "est") {
        const b = belief(m, run, i, j);
        for (let a = 0; a < p.maxProposals; a++) {
          const est = estimate(m, b, x0[j]);
          if (p.tape && prevTape) for (let k = 0; k < K; k++) if (prevTape.P[k] !== null && x0[j][k] > 0) est.a[k] = prevTape.P[k];
          const s = nashMoney(ai, e.b[i], x0[i], est.a, est.b, x0[j]);
          if (!(s.S > GAIN_TOL)) { if (a === 0) run.silent++; break; }
          run.proposals++; run.messages += 2;
          const [Ai, Aj] = bondGains(m, x0, i, j, s.d);
          const ok = Ai - s.z > 0 && Aj + s.z > 0 && feasible(x0, i, j, s.d);
          updateBelief(m, b, x0[j], s.d, s.z, ok, rng);
          if (ok) { tapeAdd(tape, s.d, s.z, ai, e.b[i], est.a, est.b); execute(m, run, i, j, s.d, s.z); break; }
          run.rejects++;
        }
        break meeting;
      }
      // stylised LLM negotiation
      let aj = margVec(e, j, x0[j]), bj = e.b[j].slice();
      if (run.method === "llmPriv") {
        run.messages += 2; // disclosure exchange
        aj = aj.map((v) => v + p.discloseNoise * 0.002 * rng.normal());
        bj = bj.map((v) => v * Math.exp(p.discloseNoise * 0.3 * rng.normal()));
      }
      for (let a = 0; a < p.maxProposals; a++) {
        const o = llmOffer(m, Object.assign({}, run, { x: x0 }), i, j, aj, bj, a, rng);
        if (!o) break;
        run.proposals++; run.messages += 2;
        const [Ai, Aj] = bondGains(m, x0, i, j, o.d);
        if (Ai - o.z > 0 && Aj + o.z > 0 && feasible(x0, i, j, o.d)) { tapeAdd(tape, o.d, o.z, o.ai, e.b[i], o.aj, o.bj); execute(m, run, i, j, o.d, o.z); break; }
        run.rejects++;
      }
      }
      rec.props = run.proposals - p0;
      if (run.trades > tr0) { rec.d = run._last.d; rec.z = run._last.z; }
      rlog.meetings.push(rec);
    }
    run.rounds.push(rlog);
    run.tape = { q: tape.q, P: tape.q.map((q, k) => (q > 1e-12 ? tape.pq[k] / q : null)) };
    run.t++;
    record(m, run);
  }

  // Surplus still available across all pairs, and dispersion of holders' implied spreads (bp).
  function residual(m, x) {
    let R = 0, n = 0;
    for (let i = 0; i < m.N; i++) for (let j = i + 1; j < m.N; j++) { const s = trueNash(m, x, i, j); R += Math.max(0, s.S); n++; }
    let disp = 0;
    for (let k = 0; k < K; k++) {
      const ys = [];
      for (let i = 0; i < m.N; i++) if (x[i][k] > 1e-6) ys.push(spreadOf(Math.max(1e-6, marginal(m.post, i, x[i], k)), k));
      if (ys.length > 1) { const mu = ys.reduce((s, v) => s + v, 0) / ys.length; disp += Math.sqrt(ys.reduce((s, v) => s + (v - mu) * (v - mu), 0) / ys.length); }
    }
    return { R: R / n, disp: (1e4 * disp) / K };
  }
  function record(m, run) {
    const W = welfare(m.post, run.x), F = (W - m.W0) / (m.Wstar - m.W0), r = residual(m, run.x);
    const h = {
      t: run.t, M: (run.t * Math.floor(m.N / 2)), W, F, gap: Math.max(1 - F, 0), R: r.R, disp: r.disp,
      messages: run.messages, trades: run.trades, proposals: run.proposals, rejects: run.rejects,
      belief: run.method === "est" ? beliefError(m, run) : null,
      tape: run.tape ? run.tape.P.map((P, k) => (P !== null && P > 0 ? spreadOf(P, k) : null)) : null,
      eff: run.meet.n ? run.meet.eff / run.meet.n : null, share: run.meet.n ? run.meet.share / run.meet.n : null,
    };
    run.history.push(Object.assign(h, fairness(m, run.x, run.cash)));
  }
  const metersNeeded = (run, N, eps) => { const h = run.history.find((r) => r.F >= 1 - eps); return h ? h.M / N : null; };
  // utility gain since the shock, in bp of the investor's pre-shock portfolio value
  const gainsBp = (m, run) => run.x.map((xi, i) => (1e4 * (run.cash[i] - m.m0[i] + bondU(m.post, i, xi) - m.U0[i])) / m.value0[i]);

  // ---------- welfare and fairness (money is the common unit) ----------
  // U_i = m_i + V_i(x_i). EF compares whole holdings including cash. EF1/PROP1 treat one lot,
  // X_k/(3N), of a bond as an item and use the goods-and-chores versions (a lot may be removed from
  // either bundle), since V_i is not monotone. PROP's share is U_i(M/N, X/N).
  function fairness(m, x, cash) {
    const N = m.N, e = m.post, tol = 1e-12;
    const U = x.map((xi, i) => cash[i] + bondU(e, i, xi));
    const M = cash.reduce((s, v) => s + v, 0), share = m.X.map((Xk) => Xk / N);
    const g = x.map((xi, i) => (1e4 * (cash[i] - m.m0[i] + bondU(e, i, xi) - m.U0[i])) / m.value0[i]);
    let ef = 0, ef1 = 0, prop = 0, prop1 = 0, maxEnvy = 0, gh = 0, gt = 0;
    for (let i = 0; i < N; i++) {
      const dU = (y, k, t) => t * (e.c[i][k] - e.b[i][k] * (2 * y[k] + t));
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const Uij = cash[j] + bondU(e, i, x[j]), envy = Uij - U[i];
        if ((1e4 * envy) / m.value0[i] > maxEnvy) maxEnvy = (1e4 * envy) / m.value0[i];
        if (envy <= tol) { ef++; ef1++; continue; }
        for (let k = 0; k < K; k++) {
          const tj = Math.min(m.lot[k], x[j][k]), ti = Math.min(m.lot[k], x[i][k]);
          if ((tj > 0 && U[i] >= Uij + dU(x[j], k, -tj) - tol) || (ti > 0 && U[i] + dU(x[i], k, -ti) >= Uij - tol)) { ef1++; break; }
        }
      }
      const fair = M / N + bondU(e, i, share);
      if (U[i] >= fair - tol) { prop++; prop1++; }
      else for (let k = 0; k < K; k++) {
        const ta = Math.min(m.lot[k], m.X[k] - x[i][k]), tr = Math.min(m.lot[k], x[i][k]);
        if ((ta > 0 && U[i] + dU(x[i], k, ta) >= fair - tol) || (tr > 0 && U[i] + dU(x[i], k, -tr) >= fair - tol)) { prop1++; break; }
      }
      const gp = Math.max(0, g[i]); gt += gp; if (m.hit[i]) gh += gp;
    }
    let gd = 0, gs = 0; const gp = g.map((v) => Math.max(0, v));
    for (let i = 0; i < N; i++) { gs += gp[i]; for (let j = 0; j < N; j++) gd += Math.abs(gp[i] - gp[j]); }
    const P = N * (N - 1);
    return { ef: ef / P, ef1: ef1 / P, prop: prop / N, prop1: prop1 / N, maxEnvy, minGain: Math.min(...g), gini: gs > 1e-12 ? gd / (2 * N * gs) : null, hitShare: gt > 0 ? gh / gt : null };
  }

  // ---------- theory check: one proposal under a perturbed estimate ----------
  // For each ordered pair with surplus, replace the counterpart's marginal values a_j by a_j + e*z and its
  // curvature b_j by b_j*exp(e*z'/scale) and propose once. Report acceptance, relative surplus loss
  // 1 - S(Dhat)/S(D*), and the shift in the counterpart's share of the realised surplus.
  function beliefSweep(m, x, es, reps) {
    const rng = makeRng(hashSeed("sweep", m.cfg.seed)), e = m.post;
    const pairs = [];
    for (let i = 0; i < m.N; i++) for (let j = 0; j < m.N; j++) {
      if (i === j) continue;
      const ai = margVec(e, i, x[i]), aj = margVec(e, j, x[j]);
      const s = nashMoney(ai, e.b[i], x[i], aj, e.b[j], x[j]);
      if (!(s.S > 1e-9)) continue;
      const res = es.map(() => ({ n: 0, acc: 0, loss: 0, shift: 0 }));
      for (let r = 0; r < reps; r++) {
        const z = aj.map(() => rng.normal());
        es.forEach((eps, q) => {
          const ah = aj.map((v, k) => v + eps * z[k]);
          const sh = nashMoney(ai, e.b[i], x[i], ah, e.b[j], x[j]);
          const R = res[q]; R.n++;
          if (!(sh.S > 0)) return;
          const [Ai, Aj] = bondGains(m, x, i, j, sh.d);
          const gj = Aj + sh.z;
          if (Ai - sh.z > 0 && gj > 0) { R.acc++; R.loss += 1 - (Ai + Aj) / s.S; R.shift += Math.abs(gj / (Ai + Aj) - 0.5); }
        });
      }
      pairs.push({ s: s.S, res });
    }
    pairs.sort((a, b) => a.s - b.s);
    const agg = (G) => es.map((eps, q) => {
      let n = 0, acc = 0, loss = 0, shift = 0;
      for (const p of G) { const R = p.res[q]; n += R.n; acc += R.acc; loss += R.loss; shift += R.shift; }
      return { e: eps, acc: n ? acc / n : null, loss: acc ? loss / acc : null, shift: acc ? shift / acc : null };
    });
    return { pairs: pairs.length, top: agg(pairs.slice(Math.floor((3 * pairs.length) / 4))), all: agg(pairs) };
  }

  const api = {
    makeRng, hashSeed, T, BUCKET, TSY, OAS, D, P0, K, spreadOf, VARIANTS, SHOCKS, GROUPS, METHODS, DEFAULT_PARAMS,
    credit, vPref, economy, bondU, welfare, marginal, margVec, terminal, buildMarket, nashMoney, trueNash, bondGains,
    createRun, step, metersNeeded, gainsBp, fairness, beliefSweep, residual, beliefError,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Engine = api;
})(typeof window !== "undefined" ? window : this);
