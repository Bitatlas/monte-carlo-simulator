# 📈 Multi-Asset Monte Carlo Simulator

A professional-grade financial simulator for projecting, stress-testing, and optimising investment portfolios across any asset class — powered by five stochastic models, Kelly Criterion mathematics, and an interactive learning game.

**Live app:** [henri-monte-carlo-simulator.streamlit.app](https://henri-monte-carlo-simulator.streamlit.app/)

![Monte Carlo Simulation](https://img.shields.io/badge/Monte%20Carlo-Simulation-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)

---

## 🗂️ Platform Overview

The app is organised into **8 tabs**, each covering a distinct analytical function:

| Tab | Name | Purpose |
|-----|------|---------|
| 1 | 📊 Dashboard | Configure parameters, run a simulation, view headline results |
| 2 | 🔬 Simulation Details | Full statistical breakdown + historical price chart |
| 3 | 📈 Kelly Analysis | Optimal leverage calculations and the Kelly growth curve |
| 4 | 🗂️ Portfolio Simulator | Multi-asset correlated simulation with rebalancing & DCA |
| 5 | 🔍 Portfolio Optimizer | Exhaustive Kelly + MPT ranking across asset universes |
| 6 | 🛠️ Use Cases | Guided scenarios and practical applications |
| 7 | ℹ️ About Models | Model assumptions, formulas, and guidance |
| 8 | 🎮 Kelly Game | Interactive game — learn Kelly Criterion by playing |

---

## 🚀 How to Use

### Step 1 — Dashboard (Single Asset Simulation)

Open the **📊 Dashboard** tab. Expand the **⚙️ Simulation Parameters** panel:

**Column 1 — Asset**
- Choose an *Asset Type*: Equity Index, Individual Stock, Sector ETF, or Bond
- Select the specific asset from the dropdown (or type a ticker for stocks)
- Set *Historical Data Years* — the lookback period used to estimate mean return and volatility
  - Tip: shorter windows (3–5 yr) reflect recent regime; longer (20+ yr) gives calmer estimates

**Column 2 — Investment**
- *Initial Investment ($)* — starting capital in USD
- *Time Horizon (Years)* — how far ahead to project (1–30 years)
- *Risk-Free Rate (%)* — used for Sharpe ratios and Kelly calculations; default 2%

**Column 3 — Model**
- *Simulation Model* — choose from five mathematical models (see About Models tab for details)
- *Simulations* — number of independent paths (10–3000); more = smoother distributions but slower
- Optional model-specific parameters appear when GARCH, Markov, or Feynman is selected

**Column 4 — Leverage**
- *Leverage Method*:
  - **Manual** — set leverage directly (0× = all cash, 1× = unlevered, 2× = 2:1 leverage)
  - **Kelly Criterion** — use the analytically-optimal leverage derived from historical returns
  - **Fractional Kelly** — Kelly × your chosen fraction (e.g. 0.5 = Half Kelly, safer)
  - **Numerical Optimization** — find the leverage that maximises expected log-growth via simulation

Click **▶ Run Simulation** — results appear below immediately.

**Dashboard Results:**
- 4 headline metrics: Initial Investment, Final Median Value (with 95% CI), Median CAGR, Leverage used
- Path Analysis panel: counts of paths below/above initial investment and above benchmark
- Simulation Paths chart (50 sampled paths plotted)
- Final Value Distribution histogram

---

### Step 2 — Simulation Details

Switch to the **🔬 Simulation Details** tab (results persist from the last run):

- **Historical Data Analysis** — annual return, volatility, Sharpe ratio, max drawdown from real data
- **Historical Price Performance** — actual price chart extended with the simulation's median, mean, and 90% confidence band
- **Simulation Statistics** — complete percentile table (5th, 25th, 50th, 75th, 95th), CAGR range, drawdown distributions, ruin probability
- **Sharpe Ratio Comparison** — bar chart comparing historical Sharpe vs. simulated Sharpe (median and mean) with uncertainty bar
- **Model Parameters** — if available, the fitted parameters (GARCH ω/α/β, Markov transition matrix, etc.)

---

### Step 3 — Kelly Analysis

The **📈 Kelly Analysis** tab shows:

- **Full Kelly Leverage** — the mathematically optimal leverage f* = (μ − r) / σ²
- **Half Kelly** — a common practical recommendation (f*/2)
- **Numerically Optimal Leverage** — found by maximising simulated log-growth (often close to Full Kelly)
- **Kelly Growth Curve** — interactive Plotly chart of expected log-growth vs. leverage; the peak marks Full Kelly
- Educational explanation of the Kelly Criterion with hover-over tooltips on financial terms

> **Tip:** The Kelly Criterion maximises *long-run* wealth but can involve large short-term swings. Use Fractional Kelly (50–75%) for a smoother ride.

---

### Step 4 — Portfolio Simulator

The **🗂️ Portfolio Simulator** tab lets you build a **2- or 3-asset portfolio** and run a correlated Monte Carlo simulation:

1. Select 2 or 3 assets (any mix of equity indices, stocks, ETFs, bonds)
2. Assign portfolio weights (auto-normalised to 100%)
3. Set investment parameters: initial amount, time horizon, number of simulations, historical data years
4. **Rebalancing** (optional): choose monthly / quarterly / annually + transaction cost (% per trade)
   - Rebalancing resets weights to target by selling over-weight and buying under-weight assets; can add a *rebalancing bonus* in volatile/mean-reverting markets
5. **Dollar-Cost Averaging (DCA)** (optional): add a fixed contribution each month or quarter
   - Reduces market-timing risk; particularly powerful over long horizons

Results include:
- Correlated portfolio paths (using the historical correlation matrix)
- Portfolio-level statistics (median, CAGR, max drawdown, Sharpe)
- Comparison of rebalanced vs. buy-and-hold outcomes
- Visualisation of DCA contributions vs. final wealth

---

### Step 5 — Portfolio Optimizer

The **🔍 Portfolio Optimizer** searches **all N-asset combinations** within a chosen universe and ranks them by Kelly + Modern Portfolio Theory metrics:

1. Select a preset universe (US Large Cap, US Equity ETFs, Global ETFs, Bonds, Commodities, Factor ETFs, High Growth, or Balanced Mix) or enter custom tickers
2. Choose how many assets per portfolio (2, 3, or 4)
3. Set the historical lookback period and risk-free rate
4. Click **Run Optimizer**

The engine computes for every combination:
- **Individual Kelly** — f*ᵢ = max(0, (μᵢ − r) / σᵢ²) per asset
- **Nekrasov Portfolio Kelly** — K\* = μᵀ Σ⁻¹ μ (full-Kelly with cross-asset covariance)
- **Diversification Bonus** — K\* − max(f*ᵢ) — extra growth from combining assets
- **Sharpe Ratio** — (μ_p − r) / σ_p (annualised, equal-weight portfolio)
- **Best & Worst Diversifiers** — assets at the two extremes of the diversification spectrum

Results are sorted by Diversification Bonus (highest first) so you can identify portfolios that genuinely benefit from combining assets rather than simply picking individually high-returning ones.

---

### Step 6 — Kelly Game 🎮

The **🎮 Kelly Game** is an interactive simulation game:

1. Pick a real asset (pulled live from yfinance)
2. Each "week" you choose what fraction of your portfolio to bet (0–2× Kelly)
3. The game draws from the asset's *actual historical weekly returns* (shuffled randomly)
4. Track your portfolio vs. the Kelly benchmark over multiple rounds
5. See your score, drawdown, and whether you over- or under-bet Kelly

Use the game to develop intuition about why betting more than Kelly is eventually ruinous even with positive expected returns.

---

## 📐 Mathematical Background

### Simulation Models

#### 🎲 Standard Monte Carlo
Each period's return is drawn i.i.d. from N(μ, σ²) where μ and σ are estimated from historical data:

```
r_t ~ N(μ, σ²)
S_t = S_{t-1} · (1 + r_t)
```

Simple, fast, but underestimates tail risk because real returns have excess kurtosis (fat tails).

---

#### 📉 Geometric Brownian Motion (GBM)
Continuous-time model underlying Black-Scholes. The stochastic differential equation:

```
dS = μ·S·dt + σ·S·dW
```

Exact discrete solution (Euler–Maruyama):

```
S_{t+1} = S_t · exp[(μ - σ²/2)·Δt + σ·√Δt·ε],   ε ~ N(0,1)
```

The drift correction term −σ²/2 ensures the **expected** price grows at μ (Itô's lemma). Log-returns are normally distributed; prices are log-normally distributed. More theoretically sound than simple MC.

---

#### 📊 GARCH(1,1)
Generalised Autoregressive Conditional Heteroskedasticity — models **volatility clustering** (turbulent periods tend to follow turbulent periods):

```
r_t = μ + ε_t,           ε_t = σ_t · z_t,    z_t ~ N(0,1)
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

Parameters ω, α, β are fitted by maximum likelihood on historical returns. The persistence of volatility shocks is α + β; when close to 1 the model exhibits long memory. GARCH produces **fat-tailed** return distributions, making it more realistic for risk management and VaR estimation.

---

#### ⛓️ Markov Chain
Discretises the return space into K states (e.g. crash, bear, neutral, bull, melt-up). A transition matrix P is estimated from historical returns using K-means clustering:

```
P_{ij} = Prob(state_j at t+1 | state_i at t)
```

Each simulated step draws a new state from the current state's row of P, then samples a return from that state's empirical distribution. Captures **regime switching** — extended bull or bear periods — and produces multi-modal final-value distributions.

---

#### 🔄 Feynman Path Integral
Quantum-mechanics-inspired approach. Rather than propagating a single path, it assigns a probability amplitude (action weight) to every possible price path:

```
K(S_f, T | S_0, 0) = ∫ DS(t) · exp(i·S[S(t)] / ℏ)
```

In the financial implementation, the "action" penalises paths that deviate from the drift; the integral is evaluated via Monte Carlo importance sampling over many candidate paths. This approach natively models **path-dependent** dynamics and is better at generating rare-event scenarios not captured by Gaussian models.

---

### Kelly Criterion

The Kelly Criterion finds the leverage f\* that maximises the **expected geometric growth rate** G:

```
G(f) = E[log(1 + f·r)]  ≈  f·μ − ½·f²·σ²
```

Setting dG/df = 0 gives:

```
f* = (μ − r_f) / σ²
```

where μ = expected return, r_f = risk-free rate, σ² = variance of returns.

**Key properties:**
- f\* > 1 → use leverage (expected return exceeds risk-adjusted cost)
- f\* < 0 → short the asset or hold cash
- Betting more than f\* *always* underperforms long-run (overbetting causes ruin)
- Half Kelly (f\*/2) halves variance while keeping ~75% of the growth rate — a common practical choice

**Nekrasov Portfolio Kelly** (used in the Portfolio Optimizer):

```
w* = Σ⁻¹ · (μ − r_f)
K* = (μ − r_f)ᵀ · Σ⁻¹ · (μ − r_f)
```

where Σ is the covariance matrix. K\* is the portfolio Kelly growth rate; the **Diversification Bonus** is K\* − max(f*ᵢ).

---

### Risk Metrics

| Metric | Formula |
|--------|---------|
| CAGR | (S_T / S_0)^(1/T) − 1 |
| Max Drawdown | max_{t∈[0,T]} (peak_t − S_t) / peak_t |
| Sharpe Ratio | (μ_p − r_f) / σ_p (annualised) |
| Ruin Probability | Fraction of paths ending below 1% of initial value |
| VaR (5%) | 5th percentile of final portfolio values |
| CVaR (5%) | Mean of paths below the VaR threshold |

---

## 🗂️ Asset Universe

| Type | Coverage |
|------|----------|
| **Equity Indices** | S&P 500, Nasdaq 100, Dow Jones, Russell 2000, EURO STOXX 50, STOXX 600, FTSE 100, DAX, CAC 40, SMI, Nikkei 225, Hang Seng, ASX 200, KOSPI, STI, TSX, Ibovespa, IPC Mexico, MSCI World, Emerging Markets, MSCI ACWI |
| **Individual Stocks** | Any ticker supported by yfinance (global equities) |
| **Sector ETFs** | XLK (Tech), XLF (Financials), XLE (Energy), XLV (Health), XLY (Cons. Disc.), XLP (Staples), XLI (Industrials), XLU (Utilities), XLB (Materials), XLC (Comms), CLRE (Return-Stacked Bonds & Futures), TLT (20+ Yr Treasury) |
| **Bonds** | US 10-Yr Treasury, US 30-Yr Treasury, US 3-Mo T-Bill, TLT, IEF, SHY |

---

## 🛠️ Installation & Local Run

```bash
# Clone
git clone https://github.com/Bitatlas/monte-carlo-simulator.git
cd monte-carlo-simulator

# Install dependencies
pip install -r requirements.txt

# Optional: GARCH support
pip install arch

# Run
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## ⚙️ Requirements

- Python 3.9+
- streamlit, pandas, numpy, matplotlib, plotly
- yfinance (live data)
- scikit-learn (Markov Chain state estimation)
- arch *(optional — enables GARCH model)*

---

## 📄 License

MIT License

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It does not constitute investment advice. Past performance is not indicative of future results. Leveraged investing involves significant risk of loss, including loss of the entire principal. All simulations are based on historical data and mathematical models with inherent limitations.


---

## Academic References

The models, metrics, and strategies implemented in this platform are grounded in the following peer-reviewed literature and authoritative texts.

### Monte Carlo Simulation in Finance
1. **Boyle, P. P.** (1977). "Options: A Monte Carlo approach." *Journal of Financial Economics*, 4(3), 323-338. — First application of Monte Carlo methods to option pricing.
2. **Glasserman, P.** (2003). *Monte Carlo Methods in Financial Engineering*. Springer, New York. — The definitive reference textbook for Monte Carlo simulation in quantitative finance.
3. **Metropolis, N., & Ulam, S.** (1949). "The Monte Carlo method." *Journal of the American Statistical Association*, 44(247), 335-341. — Original paper introducing the Monte Carlo method.

### Geometric Brownian Motion & Option Pricing
4. **Black, F., & Scholes, M.** (1973). "The pricing of options and corporate liabilities." *Journal of Political Economy*, 81(3), 637-654. — Foundational GBM-based option pricing model.
5. **Merton, R. C.** (1973). "Theory of rational option pricing." *Bell Journal of Economics and Management Science*, 4(1), 141-183. — Extended Black-Scholes to continuous-time GBM.
6. **Samuelson, P. A.** (1965). "Proof that properly anticipated prices fluctuate randomly." *Industrial Management Review*, 6(2), 41-49. — Mathematical basis for the random-walk / GBM price model.

### GARCH & Volatility Modelling
7. **Engle, R. F.** (1982). "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation." *Econometrica*, 50(4), 987-1007. — Introduced the ARCH model (Nobel Prize 2003).
8. **Bollerslev, T.** (1986). "Generalized autoregressive conditional heteroscedasticity." *Journal of Econometrics*, 31(3), 307-327. — Extended ARCH to GARCH(p,q).
9. **Engle, R. F., & Patton, A. J.** (2001). "What good is a volatility model?" *Quantitative Finance*, 1(2), 237-245. — Practical guide to applying GARCH models.

### Markov Chain & Regime-Switching Models
10. **Hamilton, J. D.** (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica*, 57(2), 357-384. — Introduced Markov regime-switching models for financial time series.
11. **Ang, A., & Bekaert, G.** (2002). "International asset allocation with regime shifts." *Review of Financial Studies*, 15(4), 1137-1187. — Applied regime switching to international portfolio allocation.

### Feynman Path Integral in Finance
12. **Kleinert, H.** (2004). *Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets* (4th ed.). World Scientific, Singapore. — Comprehensive treatment of Feynman path integrals, including financial applications.
13. **Baaquie, B. E.** (2004). *Quantum Finance: Path Integrals and Hamiltonians for Options and Interest Rates*. Cambridge University Press. — Applies quantum field theory and path integrals to financial modelling.

### Kelly Criterion & Optimal Growth
14. **Kelly, J. L.** (1956). "A new interpretation of information rate." *Bell System Technical Journal*, 35(4), 917-926. — Original paper introducing the Kelly Criterion.
15. **Thorp, E. O.** (1969). "Optimal gambling systems for favorable games." *Revue de l'Institut International de Statistique*, 37(3), 273-293. — First application of Kelly to financial markets.
16. **MacLean, L. C., Thorp, E. O., & Ziemba, W. T.** (2011). *The Kelly Capital Growth Investment Criterion: Theory and Practice*. World Scientific. — Comprehensive book on Kelly Criterion applications in investing.
17. **Nekrasov, V.** (2014). "Kelly criterion for multivariate portfolios: A model-free approach." Available at SSRN: https://ssrn.com/abstract=2259133. — Foundation for Portfolio Kelly / Diversification Bonus calculations used in this platform.

### Modern Portfolio Theory & Diversification
18. **Markowitz, H.** (1952). "Portfolio selection." *Journal of Finance*, 7(1), 77-91. — Seminal work on mean-variance optimization (Nobel Prize 1990).
19. **Sharpe, W. F.** (1966). "Mutual fund performance." *Journal of Business*, 39(1), 119-138. — Introduced the Sharpe ratio.
20. **Fernholz, R., & Shay, B.** (1982). "Stochastic portfolio theory and stock market equilibrium." *Journal of Finance*, 37(2), 615-624. — Mathematical basis for the rebalancing bonus.
21. **Booth, D. G., & Fama, E. F.** (1992). "Diversification returns and asset contributions." *Financial Analysts Journal*, 48(3), 26-32. — Empirical evidence for the diversification / rebalancing bonus.

### Volatility Drag & Leverage
22. **Mindlin, D.** (2011). "On the relationship between arithmetic and geometric returns." *CDI Advisors Research*. — Formal treatment of variance drag: G ~ mu - (1/2)*sigma^2.
23. **Cheng, M., & Madhavan, A.** (2009). "The dynamics of leveraged and inverse exchange-traded funds." *Journal of Investment Management*, 7(4), 43-62. — Analysis of volatility drag in leveraged ETFs.

### Stochastic Calculus Reference
24. **Oksendal, B.** (2003). *Stochastic Differential Equations: An Introduction with Applications* (6th ed.). Springer. — Standard reference for GBM and Ito calculus.
25. **Hull, J. C.** (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson. — Industry-standard textbook covering GBM, Black-Scholes, and Monte Carlo in finance.

---

*This platform is built for educational purposes. Citations are provided to support further academic study.*
