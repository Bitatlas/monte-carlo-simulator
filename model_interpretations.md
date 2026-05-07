# ℹ️ About Models — Multi-Asset Monte Carlo Simulator

This page explains the mathematical assumptions, strengths, limitations, and best-use scenarios for every model and analytical engine in the platform.

---

## Simulation Models (Dashboard & Portfolio Simulator)

Five stochastic models are available. Each makes different assumptions about how returns behave over time. Choosing the right model depends on your asset, time horizon, and analytical goal.

---

### 🎲 Standard Monte Carlo

**What it does:**
Simulates returns by drawing independently each period from a Normal distribution parameterised by the historical mean (μ) and standard deviation (σ) of daily returns:

```
r_t ~ N(μ, σ²)
S_t = S_{t-1} × (1 + r_t)
```

**Mathematical assumptions:**
- Returns are i.i.d. (independent and identically distributed)
- Returns follow a Normal (Gaussian) distribution
- μ and σ are constant through time — no volatility clustering

**Strengths:**
- Extremely fast — scales to thousands of paths with no performance penalty
- Easy to interpret: parameters (mean, std) are familiar to most practitioners
- Provides a useful baseline/benchmark for comparing other models
- Appropriate when returns are genuinely close to normal (some bond instruments)

**Limitations:**
- Underestimates tail risk: real equity returns have **excess kurtosis** (fat tails) and **negative skew**
- Ignores volatility clustering: real markets have calm and turbulent periods that persist
- No autocorrelation: real returns show slight mean-reversion at short horizons
- Can produce unrealistically smooth simulation paths

**Best for:**
- Quick baseline projections
- Fixed-income assets with near-normal return distributions
- Educational comparisons against more sophisticated models
- Long time horizons where tail behaviour averages out

---

### 📉 Geometric Brownian Motion (GBM)

**What it does:**
Models asset prices as a continuous-time stochastic process. The price follows the stochastic differential equation:

```
dS = μ · S · dt + σ · S · dW_t
```

where W_t is a standard Wiener process (Brownian motion). The discrete-time exact solution is:

```
S_{t+Δt} = S_t · exp[(μ − σ²/2) · Δt + σ · √Δt · ε],    ε ~ N(0,1)
```

The term **−σ²/2** (Itô correction) ensures the expected price grows at rate μ, accounting for Jensen's inequality in the log-transform.

**Mathematical assumptions:**
- Log-returns are normally distributed: ln(S_{t+1}/S_t) ~ N((μ − σ²/2)Δt, σ²Δt)
- Prices are log-normally distributed (cannot go negative)
- Constant drift μ and volatility σ
- No jumps, no autocorrelation

**Strengths:**
- Theoretically grounded: foundation of Black-Scholes options pricing
- Log-normal prices prevent negative values (unlike simple Monte Carlo)
- Multiplicative nature of returns is correctly handled
- Widely used and accepted in quantitative finance

**Limitations:**
- Still assumes constant volatility (no GARCH-like clustering)
- Still assumes normal log-returns (underestimates crash probability)
- No jumps — cannot model sudden price discontinuities (earnings shocks, crashes)
- Historical μ and σ estimates are noisy and non-stationary

**Best for:**
- Equity and equity index simulations
- Options pricing context
- Longer time horizons where the log-normal approximation improves
- When you want a more rigorous version of simple Monte Carlo

---

### 📊 GARCH(1,1)

**What it does:**
Generalised Autoregressive Conditional Heteroskedasticity — models the fact that volatility in financial markets **clusters**: periods of high volatility tend to follow each other, as do calm periods.

The model has two equations:

```
Return equation:    r_t = μ + ε_t,           ε_t = σ_t · z_t,    z_t ~ N(0,1)
Variance equation:  σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
```

Parameters ω, α, β are estimated by **maximum likelihood** on the historical return series.
- **ω** (omega): baseline variance (long-run unconditional variance = ω / (1 − α − β))
- **α** (alpha): weight on the most recent shock (ARCH effect — reaction to news)
- **β** (beta): weight on the previous conditional variance (GARCH effect — persistence)
- **α + β**: total volatility persistence; if close to 1, volatility decays very slowly (long memory)

**Strengths:**
- Captures volatility clustering — the most important empirical feature of financial returns
- Produces **fat-tailed** simulated return distributions (closer to reality)
- Parameters are interpretable and estimated from data (not assumed)
- More realistic risk estimates in turbulent market periods
- Better Value-at-Risk (VaR) and Conditional VaR (CVaR) estimates

**Limitations:**
- Requires the `arch` Python library (optional install); falls back to GBM if unavailable
- Conditional normality still underestimates extreme tail events
- Parameters estimated from historical data may not represent future regimes
- Slower to fit than simpler models
- Overfitting risk with short historical series (< 2 years of daily data)

**Best for:**
- Risk management and stress testing
- High-volatility assets (individual tech stocks, Emerging Markets, sector ETFs)
- Periods following known volatility shocks (post-COVID, 2022 rate cycle)
- VaR/CVaR estimation for leveraged positions
- When you want the most realistic single-asset risk model

**GARCH model parameters (adjustable in UI):**
- **p (GARCH lag)**: order of lagged variance terms (default 1; try 2 for assets with longer volatility memory)
- **q (ARCH lag)**: order of lagged squared-return terms (default 1; standard for most assets)

---

### ⛓️ Markov Chain

**What it does:**
Discretises the historical return distribution into **K distinct states** (regimes) and estimates the probability of transitioning between them. Future simulations draw from the current state's empirical return distribution, then transition to the next state probabilistically.

**Model construction:**
1. Daily historical returns are clustered into K states using **K-means clustering** on return magnitude
2. A **K × K transition matrix P** is estimated: P_{ij} = Prob(state j at t+1 | state i at t)
3. Simulation proceeds step-by-step: at each period, draw a return from the current state's empirical distribution, then transition to a new state using P

**Mathematical assumptions:**
- Market regimes are discrete (K states)
- Transition probabilities are stationary (do not change over time)
- The Markov property holds: next state depends only on the current state, not history beyond t

**Strengths:**
- Captures **regime switching**: extended bull markets, bear markets, crisis periods
- Produces **non-Gaussian, multi-modal** final-value distributions
- More realistic representation of "fat tails" from regime persistence
- Regime transitions are interpretable: you can inspect the transition matrix
- Flexible: increasing K adds resolution to the regime model

**Limitations:**
- Results depend heavily on K (number of states) and historical period used
- Transition probabilities may be non-stationary (e.g. crisis regimes were rarer pre-2000)
- K-means clustering is sensitive to initialisation; results may vary slightly across runs
- Discretisation loses information within each state
- With too many states (K > 8), transition probabilities become unreliable (sparse data)

**Best for:**
- Assets with well-documented regime behaviour (equity indices, rates)
- Analysing persistence of bull/bear market phases
- When return distributions are clearly non-normal or bimodal
- Comparing against GBM to quantify regime risk
- Longer historical lookbacks (10+ years) to capture multiple cycles

**Markov parameter (adjustable in UI):**
- **States (K)**: number of discrete return regimes. 
  - 2 states: bull vs. bear
  - 3–5 states: crash / bear / neutral / bull / melt-up (recommended default: 5)
  - 7–10 states: fine-grained regime analysis (use only with 20+ years of data)

---

### 🔄 Feynman Path Integral

**What it does:**
Adapted from quantum mechanics, the Feynman Path Integral treats asset price evolution as a **sum over all possible price paths**, each weighted by an "action" that measures how probable that path is given the underlying dynamics.

**Mathematical concept:**
In quantum mechanics, the propagator (probability amplitude of going from state A to state B) is:

```
K(S_f, T | S_0, 0) = ∫ D[S(t)] · exp(i · S[path] / ℏ)
```

In the financial implementation:
- The "action" S[path] penalises paths that deviate far from the expected drift
- The integral is evaluated by **Monte Carlo importance sampling** over a large set of candidate paths
- Each candidate path is weighted by its action-score; paths with unrealistic dynamics receive low weight
- The resulting ensemble captures **path-dependent** correlations and rare events

**Strengths:**
- Natively models **path-dependent dynamics**: the history of the price affects future volatility
- Better representation of rare events and non-Gaussian tails
- Captures complex interactions not present in single-step models
- Computationally flexible: more paths/time steps = greater accuracy

**Limitations:**
- Computationally expensive (most demanding model)
- Harder to interpret than classical financial models
- Parameter choices (number of paths, time steps) significantly affect results
- Research-grade tool: not yet a standard in mainstream finance
- Requires careful tuning for each asset type

**Best for:**
- Research and academic exploration of complex market dynamics
- Stress testing and black-swan event analysis
- Comparing against classical models to identify exotic path dependencies
- Advanced users who want a quantum-finance perspective

**Feynman parameters (adjustable in UI):**
- **Paths**: number of candidate paths evaluated (100–2000; more = smoother, slower)
- **Time Steps**: granularity of the time discretisation (10–100; more = finer paths, slower)

---

## Portfolio Simulation Model (Portfolio Simulator tab)

**What it does:**
The Portfolio Simulator runs a **correlated Monte Carlo simulation** across 2–3 assets simultaneously, using the historical **joint covariance matrix** to preserve realistic correlations between assets.

**Method:**
1. Fetch historical daily returns for all selected assets
2. Estimate the annualised mean return vector **μ** and covariance matrix **Σ** from historical data
3. Use **Cholesky decomposition** of Σ to generate correlated return vectors:
   ```
   R_t = μ · Δt + L · ε_t,    where Σ = L · Lᵀ,   ε_t ~ N(0, I)
   ```
4. Apply portfolio weights to get the blended portfolio return each period
5. Optionally apply **rebalancing** (reset weights periodically) and/or **DCA** (add contributions)

**Rebalancing mechanics:**
At each rebalancing date, the portfolio drifted weights are reset to target weights. The trades required to rebalance incur a transaction cost applied to the absolute value of each trade:
```
cost = txn_cost_rate × Σ |ΔW_i| × Portfolio_Value
```
Rebalancing adds a **rebalancing bonus** in volatile, mean-reverting markets (assets oscillate around their means, so selling high and buying low captures excess returns). It subtracts value in strongly trending markets.

**DCA mechanics:**
A fixed contribution is added at each DCA interval. New contributions are allocated according to the target weights:
```
Portfolio_Value += contribution
Each_asset_i += contribution × weight_i
```

---

## Portfolio Optimization Model (Portfolio Optimizer tab)

**What it does:**
Exhaustively evaluates all N-choose-K combinations from the asset universe and ranks them using Kelly Criterion and Modern Portfolio Theory metrics.

### Individual Kelly Formula

For each asset i, the long-only analytical Kelly fraction is:

```
f*_i = max(0, (μ_i − r_f) / σ²_i)
```

where μ_i = annualised expected return, r_f = risk-free rate, σ²_i = annualised variance.

### Nekrasov Portfolio Kelly (K*)

For a multi-asset portfolio, the full-Kelly portfolio weight vector is:

```
w* = Σ⁻¹ · (μ − r_f · 1)
```

The Portfolio Kelly growth rate is:

```
K* = (μ − r_f)ᵀ · Σ⁻¹ · (μ − r_f)
```

This is the maximum achievable log-growth rate when optimal weights are used. It accounts for the full cross-asset covariance structure.

### Diversification Bonus

```
Diversification Bonus = K* − max(f*_i)
```

A positive diversification bonus means the combination of assets achieves **higher growth than the best single asset** thanks to their correlations. This is the core mathematical justification for diversification: it's not just risk reduction but a **free growth enhancement** for uncorrelated assets.

### Sharpe Ratio (equal-weight)

```
Sharpe = (μ_portfolio − r_f) / σ_portfolio,   (annualised, equal-weight)
```

Computed alongside K* to capture risk-adjusted return from a traditional MPT perspective.

### Best & Worst Diversifiers

For each ranked combination, the platform identifies:
- **Best Diversifier**: the asset in the combination that contributes the most to the Diversification Bonus (most negatively correlated to the others)
- **Worst Diversifier**: the asset that adds least diversification value (most correlated with peers)

These columns help you understand *which specific asset* is driving the combination's ranking.

---

## Kelly Criterion Engine (Kelly Analysis tab)

### Theoretical Full Kelly

Derived analytically from the historical return distribution:

```
f* = (μ − r_f) / σ²
```

### Half Kelly

```
f_half = f* / 2
```

Half Kelly is the institutional standard: it delivers ~75% of the Full Kelly growth rate with ~50% of the variance. The asymmetry arises because the Kelly growth function is concave, so moving half-way to the peak captures most of the growth but avoids the steep left side of the curve.

### Numerically Optimal Leverage

Found by simulation: run the model at leverage levels from 0 to 4× in small increments, compute the median log-growth across all simulated paths, and identify the leverage that maximises it. This accounts for model-specific dynamics (e.g. GARCH or Markov Chain paths may produce a different optimal than the analytical formula).

### Kelly Growth Curve

The growth curve plots G(f) = E[log(1 + f·r)] across leverage levels:
- **Peak** = Full Kelly
- **Left of peak** = under-leveraged (safe, but sub-optimal growth)
- **Right of peak** = over-leveraged (growth declines, eventually ruin)
- The **slope is steeper to the right of the peak** — overbetting is more harmful than underbetting by the same amount

---

## Kelly Game Engine (Kelly Game tab)

**What it does:**
An interactive simulation game using **real historical weekly returns** from the selected asset (shuffled randomly to prevent order-exploitation).

**Game mechanics:**
1. The optimal Kelly fraction is calculated from the asset's historical returns
2. Each round, the player chooses a bet fraction (0–2× Kelly)
3. A return is drawn from the shuffled historical weekly returns
4. Portfolio and a "perfect Kelly" benchmark are both updated
5. Score is tracked across rounds; drawdown and distance-from-Kelly are displayed

**Educational purpose:**
The game demonstrates two key Kelly properties:
- **Under-betting**: safe but wealth accumulates more slowly than the benchmark
- **Over-betting**: positive expected value per bet, but variance destroys compound wealth over enough rounds; "the house always wins against an overbetter"

---

## Choosing the Right Model

| Scenario | Recommended Model |
|----------|------------------|
| Quick estimate, any asset | 🎲 Standard Monte Carlo |
| Equity or equity index, baseline | 📉 GBM |
| High-volatility stock, risk management | 📊 GARCH(1,1) |
| Market cycle / regime analysis | ⛓️ Markov Chain |
| Tail risk / exotic dynamics research | 🔄 Feynman Path Integral |
| Multi-asset correlated portfolio | Portfolio Simulator (GBM-based) |
| Optimal multi-asset weights | Portfolio Optimizer (Kelly + MPT) |

**General rule:** use **GBM or GARCH** for planning and risk management; use **Markov Chain and Feynman** for research and stress-testing. Always compare at least two models — if results diverge significantly, **model uncertainty is itself a risk factor** worth accounting for.

---

## A Note on Parameter Estimation

All models are estimated from **historical data** downloaded via yfinance. Parameter quality depends on:

- **Lookback period**: longer periods average out regime noise but may include irrelevant history; shorter periods are reactive but noisy. For most assets, **10 years** captures a full market cycle.
- **Data frequency**: the platform uses **daily returns** for estimation (more data points = more stable parameters than monthly or weekly).
- **Non-stationarity**: financial parameters (especially μ) are unstable over time. A 10-year mean return is not guaranteed to persist.

> **Important:** All models project the *conditional distribution* of future outcomes given past data. They do not predict the future. Wide percentile bands are not a flaw — they are an honest representation of uncertainty.
