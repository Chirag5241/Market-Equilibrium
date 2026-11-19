# Handoff: shared setup for the incomplete-information experiments

**Project:** CS 580, "Decentralized Equilibrium Discovery After Shocks"

**Written:** 2026-11-18, to accompany `corporate_bond_interim_v2.pdf` (the 7-slide interim deck).

**Who this is for:** whoever builds the slide-7 experiments, in which investors have private preferences and negotiate through LLMs.

**What it covers:**
- The model, notation and baseline market behind slides 1–6.
- The exact parameter values and the external data values.
- The random-seed conventions.
- The existing code.
- A check of the solver formulas on slide 2.

The goal is that new results can be shown as a direct continuation of slides 3–6. This document does not propose an experimental design for slide 7. Side analyses that are not in the deck are left out on purpose.

**How each value is labeled.** Every value carries one of five labels:
- **[data]**: taken from public data.
- **[chosen]**: set by us.
- **[random]**: drawn at random (seeded).
- **[calibrated]**: chosen so that the model reproduces data.
- **[computed]**: derived from the above.

---

## 1. Model (slides 1–2)

### 1.1 Objects

| Symbol (deck) | Meaning | Code name | Label |
|---|---|---|---|
| i = 1..N | investor | row index | – |
| k = 1..K | bond (zero-coupon corporate bond with maturity T_k) | column index | – |
| x_ik ≥ 0 | units of bond k held by investor i; no short positions | `x` (N,K) | – |
| m_i ∈ ℝ | cash of investor i; may be negative (unrestricted borrowing) | `m` (N,) | – |
| X_k = Σ_i x_ik | total supply of bond k; fixed, because trades only move bonds between investors | `X` (K,) | [calibrated] |
| T_k | maturity in years | `T`, `BUCKET_T` | [chosen] |
| D_k = exp(−y_k T_k) | Treasury discount factor, with y_k the Treasury yield at maturity T_k | `D` | [data] |
| h | default intensity per year, the same for all bonds | `hazard` | [chosen] |
| L | loss given default (share of face value lost) | `lgd` | [chosen] |
| PD_k = 1 − e^{−h T_k} | probability that bond k defaults before maturity | `credit()` | [computed] |
| Y_k | payoff at maturity: 1 if no default, 1−L if default | – | – |
| μ_k = E[Y_k] = 1 − L·PD_k | expected payoff | `mu` | [computed] |
| σ_k² = L²·PD_k(1−PD_k) | payoff variance | `sig2` | [computed] |
| d*_i | investor i's preferred maturity (years) | `dstar` | [random] |
| τ_i | width of investor i's maturity preference (years) | `tau` | [random] |
| v_ik = exp(−(T_k − d*_i)² / (2τ_i²)) | preference of investor i for maturity T_k; between 0 and 1, and equal to 1 at T_k = d*_i | `duration_value()` | [computed] |
| α | weight of the maturity preference, the same for all investors | `alpha` | [chosen] |
| λ_i | risk aversion of investor i | `lam` | [random] |

The deck uses v_ik but never writes out its formula. The formula above is the one the code uses.

### 1.2 Utility

U_i(m_i, x_i) = m_i + V_i(x_i), where:

V_i(x_i) = Σ_k [ c_ik x_ik − b_ik x_ik² ]
- c_ik = D_k μ_k + α v_ik
- b_ik = λ_i D_k² σ_k²

In code: `Economy.c`, `Economy.b`; `TSMarket.economy()` builds them.

Two consequences:
- Cash enters linearly, so the marginal utility of cash is 1 (the "quasilinear" assumption).
- There are no cross-bond terms. Default risk of different bonds is not correlated in V.

**Marginal value of bond k to investor i:** a_ik = ∂V_i/∂x_ik = c_ik − 2 b_ik x_ik (code: `econ.marginal(x)`).

This is the most cash investor i would give for one more small unit of bond k. It is the only "price-like" object inside the model. There is no market price.

### 1.3 One meeting (slide 2)

**Notation for a meeting:**
- Investors i and j meet.
- The trade is a bond vector Δ ∈ ℝ^K plus a cash amount z:
  - x_i' = x_i + Δ, x_j' = x_j − Δ
  - m_i' = m_i − z, m_j' = m_j + z
- Δ_k > 0 means i buys bond k from j. z > 0 means i pays j.
- Feasibility: −x_ik ≤ Δ_k ≤ x_jk, so no one ends with negative holdings.

**Gains in bond utility:**
- A_i(Δ) = V_i(x_i+Δ) − V_i(x_i)
- A_j(Δ) = V_j(x_j−Δ) − V_j(x_j)

**Total gains, including cash:** g_i = A_i − z and g_j = A_j + z.

**Symmetric Nash bargaining:** choose (Δ, z) to maximise g_i·g_j subject to g_i, g_j ≥ 0.

**Solution, as implemented in `deq/core.py: nash_money`:**
- Δ*_k = clip( (a_ik − a_jk) / (2(b_ik + b_jk)), −x_ik, x_jk ) for every k separately (`_delta_r` with r = 1).
- z* = ½(A_i(Δ*) − A_j(Δ*)), paid by i to j. This gives g_i = g_j = ½(A_i + A_j).
- If the joint surplus A_i + A_j ≤ 2·10⁻¹² (`GAIN_TOL`), the pair does not trade.

### 1.4 Separability check: the slide-2 solver is correct

The question was whether the "separability across bonds" on slide 2 is right. It is, for this model. Derivation:

```
A_i(Δ) = Σ_k [ a_ik Δ_k − b_ik Δ_k² ]          (expand V_i(x_i+Δ) − V_i(x_i))
A_j(Δ) = Σ_k [ −a_jk Δ_k − b_jk Δ_k² ]
A_i + A_j = Σ_k f_k(Δ_k),   f_k(d) = (a_ik − a_jk) d − (b_ik + b_jk) d²
```

**Step 1: the cash amount does not change the best bond trade.**
- For a fixed Δ, g_i + g_j = A_i + A_j does not depend on z.
- So maximising g_i·g_j over z splits that sum equally, which gives z* = ½(A_i − A_j).
- The product at that z is ¼(A_i + A_j)², so the best Δ is the one that maximises A_i + A_j.
- This holds for any bargaining weight, not only the equal split. Only z changes with the weight.

**Step 2: the best bond trade splits by bond.**
- A_i + A_j is a sum of one-variable functions f_k.
- The feasibility box is a product of intervals, one per bond.
- So the maximisation separates bond by bond.
- Each f_k is a concave parabola, since b > 0. Its maximiser restricted to [−x_ik, x_jk] is the unconstrained maximiser clipped to the interval. That is the slide-2 formula.

**Other slide-2 statements, checked:**
- **Interior trade equalises marginal values.** If Δ_k is not clipped, the marginal values after the trade are a'_ik = a'_jk = (b_jk a_ik + b_ik a_jk)/(b_ik + b_jk). The same formula appears on slide 6.
- **Pairwise stability.** No pair can gain if and only if, for each bond k, every holder has the same marginal value P*_k and every non-holder has a marginal value ≤ P*_k.
  - These are exactly the optimality conditions for maximising Σ_i V_i subject to Σ_i x_ik = X_k and x ≥ 0.
  - That problem is strictly concave because all b_ik > 0, since λ_i > 0 and 0 < PD_k < 1.
  - So the terminal bond allocation is unique. Cash holdings at the end depend on the order of meetings.
  - Code: `deq/metrics.py: welfare_max_money` solves the per-bond conditions directly by bisection on P*_k.
- **Marginal exchange rate.** ∂z*/∂Δ_k = ½(a_ik + a_jk) − (b_ik − b_jk)Δ_k = (a'_ik + a'_jk)/2.

**Conditions the separability depends on.** Keep them, or change the solver:
1. **No cross-bond terms in V.** For example, there is no default correlation between bonds. With a covariance term, f_k would couple across bonds and the clip formula would no longer be the solution.
2. **Cash enters linearly and borrowing is unlimited.** With a cash constraint, or utility that is concave in cash, the best Δ would depend on how the gain is split. The repo's no-cash solver `nash_barter` handles that case by bisection, but it is not the slide-2 solver.
3. **No constraint that couples bonds within one meeting.** Examples would be a cap on total trade size, a fixed cost per meeting, or integer lots.
4. **Separability is a property of the true problem.** An LLM or estimated-utility negotiator may propose bundles that are not bond-by-bond optimal. Those are still evaluated with the true A_i and A_j.

### 1.5 Per-bond executed price (slide 4)

A meeting settles one cash amount z for the whole bundle. To get a price per bond, the code splits z by bond. Each bond's own surplus f_k(Δ_k) is split equally, and z_k is the cash attributed to bond k. These z_k sum to z*.

The executed price of bond k in that meeting is p_e = z_k / Δ_k = ½(a_ik + a_jk) − ½(b_ik − b_jk)Δ_k:
- It uses the marginal values before the trade.
- Code: `deq/core.py: per_bond_prices`.
- It is the average price per unit traded. It is not the marginal exchange rate from slide 2.

The deck's tape quantities, computed per bond k and per round t:
- **Quantity:** q_e = |Δ_k|.
- **Volume-weighted executed price:** P̂_k(t) = Σ q_e p_e / Σ q_e.
- **Execution-implied spread:** ŝ_k(t) = −(1/T_k)·ln(P̂_k(t)/D_k).
- A bond with no trade in a round has no executed price in that round.
- Code: `deq/termstructure.py: run_tape`, `spread_from_price`.

**If a new negotiation protocol produces one bundle with one cash amount,** the per-bond price is not determined by the trade itself. Either apply the split above to the true utilities, or have the negotiators quote per-bond prices. Then say which was used. Otherwise new tapes are not comparable to slide 4.

---

## 2. Baseline market (slide 3): exact values

### 2.1 Bonds and external data

| | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ICE BofA US Corporate maturity bucket | 1–3y | 3–5y | 5–7y | 7–10y | 10–15y | 15y+ |
| FRED series | BAMLC1A0C13Y | BAMLC2A0C35Y | BAMLC3A0C57Y | BAMLC4A0C710Y | BAMLC7A0C1015Y | BAMLC8A0C15PY |
| T_k (years), representative maturity of the bucket [chosen] | 2 | 4 | 6 | 8.5 | 12.5 | 20 |
| Treasury yield y_k on 2025-03-25 [data, linear interpolation] | 3.960% | 4.025% | 4.130% | 4.250% | 4.400% | 4.670% |
| D_k = exp(−y_k T_k) [computed] | 0.923855 | 0.851292 | 0.780516 | 0.696805 | 0.576950 | 0.392979 |
| Pre-shock spread s_k, mean of the 5 daily OAS values 2025-03-24..28 [data] | 57.6 bp | 78.8 bp | 91.8 bp | 110.6 bp | 113.2 bp | 112.8 bp |
| P⁰_k = D_k·exp(−s_k T_k), pre-shock reference price [computed] | 0.913273 | 0.824878 | 0.738688 | 0.634283 | 0.500824 | 0.313612 |
| μ_k (h = 0.003, L = 0.6) [computed] | 0.996411 | 0.992843 | 0.989297 | 0.984893 | 0.977917 | 0.965059 |
| σ_k² [computed] | 0.002141 | 0.004243 | 0.006307 | 0.008836 | 0.012762 | 0.019744 |

**Notes on the data:**
- **Treasury yields.** These are H.15 constant-maturity yields from FRED, as published for 2025-03-25: DGS1 4.09, DGS2 3.96, DGS3 3.98, DGS5 4.07, DGS7 4.19, DGS10 4.31, DGS20 4.67, DGS30 4.65 (%).
  - y_k is linear interpolation in maturity (`treasury_discount`).
  - The yields are used as continuously compounded zero rates. That is a simplification we chose.
- **OAS.** OAS (option-adjusted spread) is ICE's evaluated spread over Treasuries. Daily values in %:

  | Date | 1–3y | 3–5y | 5–7y | 7–10y | 10–15y | 15y+ |
  |---|---|---|---|---|---|---|
  | 03-24 | 0.58 | 0.78 | 0.91 | 1.09 | 1.12 | 1.11 |
  | 03-25 | 0.57 | 0.77 | 0.90 | 1.09 | 1.12 | 1.11 |
  | 03-26 | 0.57 | 0.78 | 0.91 | 1.10 | 1.13 | 1.13 |
  | 03-27 | 0.57 | 0.79 | 0.92 | 1.11 | 1.13 | 1.13 |
  | 03-28 | 0.59 | 0.82 | 0.95 | 1.14 | 1.16 | 1.16 |

- The window 2025-03-24..28 is a calm calibration week, not a shock date.
- FRED serves ICE series for only the most recent 3 years. Use the numbers above rather than re-downloading.
- **What P⁰_k means in the model:** in the pre-shock state, every investor who holds bond k has marginal value exactly P⁰_k (Section 2.3).

### 2.2 Investors (N = 90)

| Item | Value | Label |
|---|---|---|
| N | 90 | [chosen] |
| Preferred-maturity groups ("30 per group" on slide 3) | investors 0–29: d*_i ~ U(1, 4); 30–59: d*_i ~ U(4, 9); 60–89: d*_i ~ U(9, 25). Indices are ordered by group. | group sizes and ranges [chosen], d*_i [random] |
| τ_i | U(1.5, 4) | [random] |
| λ_i | 20·u_i, u_i ~ U(0.5, 1.5), so λ_i ∈ [10, 30] | 20 [chosen], u_i [random] |
| α | 0.03 for everyone | [chosen] |
| h, L | 0.003 per year, 0.6 | [chosen] |
| Starting cash m_i | 50 for everyone | [chosen] |

Starting cash has no effect on bond trades or on bond welfare W, because cash is linear and unrestricted. Some scripts (`exp_invariant.py`) do not track cash at all.

### 2.3 Pre-shock state (initial condition) [calibrated]

- Each investor holds x⁰_ik = max{0, (c_ik − P⁰_k)/(2 b_ik)}. That is the amount at which its marginal value equals P⁰_k, or zero if even the first unit is worth less than P⁰_k.
- Supply is X_k = Σ_i x⁰_ik (`calibrate_supply`), so X differs by seed.
- By construction this state is pairwise stable for the pre-shock utilities: no pair can trade. In the baseline every investor holds every bond.
- Code: `x0 = welfare_max_money(e0, X)[0]`. This returns the same allocation.

### 2.4 Shocks (slide 3)

At time 0 the utilities switch from pre-shock to post-shock. Holdings x⁰ and cash stay the same. Trading starts in round 1.

| Name (use this) | Slide label | Definition | Who is affected |
|---|---|---|---|
| Risk-aversion shock | "Shock A" | λ_i ← 3λ_i or λ_i/3 for a randomly chosen share of investors: 10%, 50% or 90% (9, 45 or 81 of 90) | Chosen investors only; changes their b_ik |
| Default-hazard shock | "Shock B" | h ← 1.25h or 0.8h, one common multiplier for all six bonds (not a separate draw per bond) | Everyone: μ_k and σ_k² change, so c and b change |

**After the shock:**
- The terminal allocation x* is the unique pairwise-stable allocation for the post-shock utilities with the same X.
- P*_k is the common marginal value of the holders of bond k at x*. The deck calls it the "terminal marginal value".
- Code: `x_star, W_star, P_star = welfare_max_money(e1, X)`.
- Code comments call this "Walrasian". That name refers to a central exchange, which is a concept outside the model. In the model it is simply the terminal pairwise-stable allocation.

### 2.5 Random-number conventions

These matter for paired comparisons: same investors, same shocked set, same meeting order across methods.

| Draw | How | Used in |
|---|---|---|
| Market for seed s | `rng = np.random.default_rng(s)`. Inside `TSMarket`: d* group by group (30, 30, 30), then τ (90), then λ (90) | all slides |
| Shocked investors | `rng.choice(N, round(share·N), replace=False)` from the same `rng`, right after the market is built | slides 4–6 |
| Matching sequence | `np.random.default_rng(s + 100)`; one `permutation(N)` per round; pairs are (p[0],p[1]), (p[2],p[3]), … with the first of each pair as i | slides 4–5 |
| | `exp_invariant.py` used `s + 12345` instead | slide 6 |

### 2.6 Trading protocol and measures (slides 3, 6)

**Round:**
- One random perfect matching of all N investors, so N/2 meetings. All meetings in a round use the holdings at the start of the round; the pairs are disjoint.
- **Meeting count** M = rounds × N/2. M includes meetings with no trade.

**Welfare and convergence:**
- **Bond welfare:** W(M) = Σ_{i,k} [c_ik x_ik(M) − b_ik x_ik(M)²], using post-shock (true) c and b. Cash cancels in the sum.
- **Share of the gain realised:** F(M) = (W(M) − W(0)) / (W* − W(0)), where W* is the welfare at x*.
- **Meetings needed:** M_ε = min{M : F(M) ≥ 1−ε}. The deck reports M₀.₁ (90%) and M₀.₀₁ (99%) as multiples of N.

---

## 3. Scripts behind each slide

| Slide | Script(s) | Setting |
|---|---|---|
| 4 | `exp_ts.py`, `exp_ts_premium.py` | baseline market; risk-aversion shock; executed prices and spreads per round from `run_tape` |
| 5 | `exp_entry.py` (400 seeds), `exp_entry_random.py` (30,000 random scenarios), `analyze_entry.py` | baseline market; one investor kept out of trading for R ∈ {1, 2, 4, 8, 16, 32} rounds |
| 6 | `exp_invariant.py`, `analyze_invariant.py` | N ∈ {10, 20, 50, 100, 200, 500, 1000, 2000, 5000}, 30 seeds, 6 shocks, 6 market variants (9,720 runs) |

The slide-6 market variants differ from the baseline in only these values:
- **Risk aversion spread 100-fold:** λ_i = 20·exp(U(ln 0.1, ln 10)), so λ ranges from 2 to 200.
- **Many zero holdings:** α = 0.3 and h = 0.04.

---

## 4. Code to reuse

**Model code files:**
- `deq/__init__.py`, `deq/core.py`, `deq/termstructure.py`, `deq/metrics.py`.
- `metrics.py` imports `cvxpy` at the top, so install it, or copy `welfare_max_money` alone (pure numpy).
- `termstructure.py` imports pandas.
- No data files are needed if you use the constants below.
- The repo's Python environment is 3.11 (`.venv`).

**Key functions:**

| Function | What it does |
|---|---|
| `core.nash_money(ai, bi, xi, aj, bj, xj)` | Full-information solver for many pairs at once (inputs (P,K)). Returns Δ, z and each side's gain. This is the deck's "true-utility solver". |
| `core.per_bond_prices` | executed price per bond (Section 1.5) |
| `core.Economy` | holds c, b; `.marginal(x)`, `.bond_utility(x)`, `.utility(x, m)` |
| `termstructure.TSMarket` | draws investors; `.economy()` → Economy; `.copy()`; fields `lam`, `hazard`, `dstar`, `tau`, `group` |
| `termstructure.calibrate_supply`, `metrics.welfare_max_money` | supply X from P⁰; terminal allocation, W* and P* |
| `termstructure.run_tape` | full-information rounds with tape statistics (volume-weighted price, volume, holder-weighted mean marginal value) |

**Minimal construction of the slide-3 market.** This was run and matches the reference values below:

```python
import numpy as np
from deq.termstructure import TSMarket, calibrate_supply
from deq.metrics import welfare_max_money
from deq.core import nash_money

T  = np.array([2.0, 4.0, 6.0, 8.5, 12.5, 20.0])                    # bond maturities (years)
y  = np.array([3.96, 4.025, 4.13, 4.25, 4.40, 4.67]) / 100         # Treasury yields, 2025-03-25
s0 = np.array([57.6, 78.8, 91.8, 110.6, 113.2, 112.8]) / 1e4       # corporate OAS, mean of 2025-03-24..28
D  = np.exp(-y * T)                                                # Treasury discount factors
P0 = D * np.exp(-s0 * T)                                           # pre-shock reference price

def make_market(seed, N=90, shock=('risk_aversion', 0.10, 3.0)):
    """shock = ('risk_aversion', share_hit, multiplier) or ('default_hazard', None, multiplier)."""
    rng = np.random.default_rng(seed)
    mk = TSMarket(N, T, D, rng, hazard=0.003, lam0=20, alpha=0.03, tau_range=(1.5, 4))  # lgd=0.6 default
    e0 = mk.economy()                         # pre-shock utilities: c (N,K), b (N,K)
    X = calibrate_supply(e0, P0)              # total supply per bond
    x0 = welfare_max_money(e0, X)[0]          # pre-shock holdings (pairwise stable)
    mk1, hit = mk.copy(), np.zeros(N, bool)
    kind, share, mult = shock
    if kind == 'risk_aversion':
        idx = rng.choice(N, int(round(share * N)), replace=False)   # same rng, drawn after the market
        mk1.lam[idx] *= mult; hit[idx] = True
    else:
        mk1.hazard = mk1.hazard * mult; hit[:] = True
    e1 = mk1.economy()                        # post-shock (true) utilities
    x_star, W_star, P_star = welfare_max_money(e1, X)   # terminal allocation, its welfare, terminal marginal values
    return dict(pre=mk, post=mk1, e0=e0, e1=e1, X=X, x0=x0, m0=np.full(N, 50.0),
                x_star=x_star, W_star=W_star, P_star=P_star, hit=hit)

def one_round(e1, x, m, rng_match):
    """Full-information baseline: random perfect matching, joint Nash formula, equal split. In place."""
    p = rng_match.permutation(len(x)); I, J = p[0::2], p[1::2]
    a = e1.marginal(x)                                         # a_ik = c_ik - 2 b_ik x_ik
    dX, z, gain = nash_money(a[I], e1.b[I], x[I], a[J], e1.b[J], x[J])
    x[I] += dX; x[J] -= dX; np.maximum(x, 0, out=x)            # i receives dX, j gives dX
    m[I] -= z; m[J] += z                                       # i pays z to j
    return I, J, dX, z

mkt = make_market(seed=0)
x, m = mkt['x0'].copy(), mkt['m0'].copy()
rng_match = np.random.default_rng(0 + 100)
W0 = mkt['e1'].bond_utility(mkt['x0']).sum()
for t in range(1, 9):
    one_round(mkt['e1'], x, m, rng_match)
    print(t, (mkt['e1'].bond_utility(x).sum() - W0) / (mkt['W_star'] - W0))   # F after t rounds
```

**Reference values for seed 0** (10% risk-aversion ×3). Use them to check a re-implementation:
- First three investors:
  - d* = 2.910885, 1.809360, 1.122921
  - τ = 3.818560, 3.919815, 1.536766
  - λ = 27.214028, 14.942935, 12.824931
- X = [26.366702, 27.973772, 30.308710, 35.243081, 38.918741, 56.601652].
- x⁰ of investor 0 = [0.366282, 0.293536, 0.263466, 0.266696, 0.279672, 0.395510]. No zero holdings.
- Shocked investors: 12, 17, 24, 26, 34, 35, 61, 62, 72.
- P* = [0.912057, 0.822586, 0.735530, 0.630421, 0.496681, 0.309612]. That is +6.4 to +7.2 bp of spread versus pre-shock.
- W* − W(0) = 0.690857.
- F after rounds 1–8: 0.7898, 0.8873, 0.9384, 0.9630, 0.9767, 0.9895, 0.9942, 0.9971. So 90% is reached in 3 rounds and 99% in 7.

### 4.1 Existing incomplete-information and LLM code (older market)

This code was written for the proposal's earlier **no-cash** market, not the deck's bond market:
- Market: `deq/sim.py: Market`, N = 20.
- Bonds have durations 1–30 and a common variance of 0.1.
- d* ~ U(1, 30), τ ~ U(2, 8), λ ~ U(0.5, 2).
- Trades are bond-for-bond swaps solved by `nash_barter`.

Reusable pieces:
- `deq/llm.py`: Azure OpenAI client.
  - Logs every call's cost to `results/llm_ledger.jsonl`.
  - Stops at a spend cap (`LLM_SPEND_CAP_USD`, default 40).
  - Credentials come from `.env` and are not part of this handoff.
  - The project's LLM budget is $100 in total; $7.91 has been spent so far, all on the earlier market.
- `deq/llm_regimes.py`: prompt and JSON-parsing patterns for three set-ups:
  - an LLM proposer that sees both utilities;
  - LLMs that see only their own utility and talk for up to 4 turns;
  - an LLM that estimates the counterpart's (d*, τ, λ), after which the formula computes the trade.
  - Rule worth keeping: **the environment executes a proposed trade only if it is feasible and strictly raises both investors' true utilities.**
- `deq/sim.py: run_B` + `ParticleBeliefs`: an estimated-utility solver.
  - Each investor keeps a probability distribution over the counterpart's (d*, τ, λ), represented by 3,000 sampled candidates.
  - It updates that distribution from accept/reject outcomes and proposes the formula trade for the estimated counterpart.
  - It uses the no-cash solver. Using it with the bond market needs `nash_money` and the bond-market parameter ranges.
- Code names there are letters ("regime A/B/C/D", "B-prior", "D-infer"). Use the descriptive names in Section 5.

---

## 5. Terms to share

Slide 7 names four negotiation methods. Use these names in code, figures and text:

| Name | Meaning |
|---|---|
| true-utility solver | `nash_money` with both investors' true c, b. This is the baseline of slides 3–6. |
| estimated-utility solver | The same formula, with the counterpart's c, b replaced by an estimate. Executed only if both true gains are > 0. |
| full-information LLM negotiation | LLMs see both investors' utilities. |
| private-information LLM negotiation | Each LLM sees only its own utility. |

Other terms:
- **Private and public information, as fixed on slide 7:**
  - Private: θ_i = (d*_i, τ_i, λ_i).
  - Public: holdings x, and the bond data (T, D, μ, σ², h, L, α).
  - The slides do not say whether the prior ranges or group membership are public. Decide this and write it down.
- **investor** = the market participant. **LLM agent** = the program negotiating for an investor, if a separate word is needed.
- **round** = N/2 simultaneous meetings. **meeting** = one matched pair in one round. **proposal** = one offer within a meeting. One meeting can contain several proposals; slide 7 asks for meeting and proposal counts separately.
- **pre-shock reference price P⁰**, **terminal allocation x***, **terminal marginal value P***, **executed price**, **volume-weighted executed price P̂_k(t)**, **execution-implied spread ŝ_k(t)**, **share of the gain realised F(M)**, **meetings needed M_ε**, as defined above.
- Shock names: risk-aversion shock (×3 or ÷3; share hit 10/50/90%) and default-hazard shock (×1.25 or ×0.8, all bonds).
- Avoid "Walrasian price / equilibrium" in text (it appears in code comments; see Section 2.4). Avoid the old letter names.

## 6. Keep these identical so results continue the deck

1. Same market (Section 2), same seeds, and the same rules for the shocked set and the matching sequence.
   - If LLM cost forces a smaller N or fewer seeds, change only that.
   - Run the true-utility solver on exactly the same seeds as the reference.
2. Evaluate every method with the true post-shock utilities. Use the same W, F and M_ε, and count meetings with no trade.
3. Use the same trade execution rule:
   - feasible (−x_i ≤ Δ ≤ x_j);
   - both true gains > 0;
   - cash quasilinear, with borrowing allowed.
4. If any of the separability conditions in Section 1.4 is changed, say so. The slide-2 solver is then no longer the right benchmark.
5. State how per-bond executed prices are obtained when one cash amount covers a bundle (Section 1.5).
