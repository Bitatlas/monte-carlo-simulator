# 🛠️ Use Cases — Multi-Asset Monte Carlo Simulator

This platform covers the full investment analysis lifecycle: from projecting a single asset's outcomes to optimising an entire portfolio using Kelly Criterion mathematics. Below are practical scenarios organised by goal.

---

## 1. 🏖️ Retirement Planning

**Goal:** Determine whether your current savings rate and asset allocation will meet your retirement target.

**How to use this platform:**
1. Open the **Dashboard** tab. Set *Initial Investment* to your current savings, *Time Horizon* to your years-to-retirement.
2. Select your primary asset (e.g. S&P 500 as a proxy for a diversified equity portfolio).
3. Run 500–1000 simulations. Observe the 5th–95th percentile band — this is your realistic range of outcomes.
4. Switch to **Simulation Details** to see the ruin probability (% of paths ending below 1% of initial).
5. Use the **Portfolio Simulator** to test a 60/40 (equity/bond) split: does adding bonds reduce the worst-case outcome while sacrificing acceptable upside?
6. Enable **DCA** in the Portfolio Simulator to model monthly salary contributions on top of your initial lump sum.

**Key questions answered:**
- What is the probability my portfolio reaches $X by retirement?
- How much worse is a bear-market start to retirement vs. a bull-market start?
- Does rebalancing improve or hurt outcomes in my specific asset mix?

---

## 2. 📊 Investment Strategy Testing

**Goal:** Compare different leverage levels, asset classes, and rebalancing rules on an even playing field.

**Scenarios to test in the Dashboard:**

| Strategy | Settings |
|----------|----------|
| Unlevered index | S&P 500, Manual 1× leverage |
| Kelly-optimal equity | S&P 500, Kelly Criterion leverage |
| Half Kelly (conservative) | S&P 500, Fractional Kelly 0.5 |
| Leveraged ETF proxy | S&P 500, Manual 2× or 3× leverage |
| Bond allocation | US 10-Yr Treasury, 1× leverage |

- The **Kelly Analysis** tab shows the growth curve: visually identify how much expected growth is sacrificed by under- or over-betting Kelly.
- Run the same asset with **different simulation models** (e.g. Monte Carlo vs. GARCH vs. Markov Chain) — do the risk estimates diverge? If so, model uncertainty is significant for that asset.

---

## 3. 🔍 Portfolio Diversification & Optimisation

**Goal:** Find multi-asset combinations that offer the best *diversification bonus* — extra growth achieved by holding uncorrelated assets together.

**Workflow using the Portfolio Optimizer:**
1. Open the **Portfolio Optimizer** tab.
2. Select a preset universe, e.g. *Balanced Mix (Recommended)* (SPY, TLT, GLD, EEM, QQQ, IEF, VNQ, SLV, EWJ, HYG) or *US Equity ETFs*.
3. Choose portfolio size: 2, 3, or 4 assets.
4. Set lookback period (10 years gives a full cycle including COVID crash and 2022 rate shock).
5. Click **Run Optimizer**.

**Reading the results:**
- Sort by **Diversification Bonus** (K\* − best individual Kelly): a positive bonus means the combination grows faster than the best single asset alone.
- **Portfolio Kelly (K\*)** is the theoretical maximum growth rate given optimal weights.
- The **Best Diversifier** column shows which asset contributes most to lowering correlation risk.
- Compare **Sharpe Ratio** across combinations to identify risk-adjusted winners.

**Tip:** Combinations with low or negative correlations (e.g. equity + bonds + gold) tend to have the highest diversification bonuses — the Kelly math rewards uncorrelated assets disproportionately.

---

## 4. 🛡️ Risk Assessment & Stress Testing

**Goal:** Understand downside exposure, maximum drawdown expectations, and tail risks before committing capital.

**Workflow:**
1. Select a high-volatility asset (e.g. Nasdaq 100, individual tech stock, or Emerging Markets ETF).
2. Set leverage to 2× or 3× to simulate a leveraged position.
3. Choose the **GARCH(1,1)** model — it produces fat-tailed distributions more consistent with real crash dynamics.
4. Run 1000+ simulations.
5. In **Simulation Details**: check the *Max Drawdown (mean)*, *Ruin Probability*, and the 5th percentile CAGR.

**Key stress tests:**
- What is the maximum drawdown I should expect at 2× leverage on Nasdaq over 10 years?
- How does Markov Chain (regime-switching) compare to GBM for the same asset — is regime risk material?
- At what leverage does the ruin probability exceed 5%? (Use the Kelly Growth Curve in the **Kelly Analysis** tab to find this visually.)

---

## 5. 🎓 Financial Education

**Goal:** Build intuition about compounding, volatility, leverage, and the Kelly Criterion.

### Understanding Compounding
- Set leverage to 1×, time horizon to 30 years, run any equity index.
- Compare median final value to naive projection (starting value × (1 + annual return)^30).
- The difference illustrates **volatility drag**: high volatility reduces compound growth even with the same arithmetic mean return.

### Understanding Leverage Risk
- Run the **Kelly Game** tab (🎮) — choose any asset, then try betting 2× Kelly for 20 rounds.
- Observe how overbetting eventually destroys the portfolio even though individual bets have positive expected value.
- Then try Half Kelly for 20 rounds — smoother, more survivable, still competitive.

### Understanding the Kelly Criterion
- In the **Kelly Analysis** tab, study the growth curve shape:
  - Left of the peak → under-betting (safe but suboptimal growth)
  - At the peak → Full Kelly (maximum long-run growth)
  - Right of the peak → over-betting (eventual ruin territory)
- Compare Full Kelly vs. Half Kelly CAGR difference: typically only 10–15% less growth for 50% less volatility.

### Model Comparison
- Run the same asset with all 5 models and compare the distribution of final values.
- Notice how GARCH and Markov Chain produce fatter left tails (more crash scenarios) than simple Monte Carlo or GBM.

---

## 6. 💼 Financial Advising & Client Education

**Goal:** Illustrate portfolio projections and risk to clients in a visual, intuitive way.

**Suggested workflows:**

**For retirement savers:**
1. Run a simulation with the client's actual portfolio size and time horizon.
2. Show the Simulation Paths chart: clients immediately grasp the range of possible outcomes.
3. Overlay the Distribution chart: the median and worst-case values make abstract statistics concrete.

**For leverage/options discussions:**
1. Show the Kelly Growth Curve: explain that Full Kelly doubles growth in theory but triples volatility.
2. Recommend Half Kelly as the institutional standard for leveraged strategies.

**For diversification conversations:**
1. Run the Portfolio Optimizer on a mix of the client's current holdings.
2. Show the Diversification Bonus table: prove mathematically that uncorrelated assets outperform concentration.
3. Compare a 100% equity portfolio vs. a 70/30 equity/bond portfolio using the Portfolio Simulator with rebalancing enabled.

---

## 7. 🔬 Quantitative Research

**Goal:** Explore model assumptions, fit parameters to specific assets, and test theoretical predictions.

**Research questions this platform supports:**

| Question | Tool |
|----------|------|
| Is GARCH a better fit than GBM for this asset? | Compare Simulation Details → Model Parameters across both models |
| How many Markov states best capture this market's regimes? | Vary "States" parameter (2–10) and compare distribution shapes |
| Does the empirical Kelly match the theoretical formula? | Compare f\* in Kelly Analysis vs. Leverage that maximises CAGR in simulations |
| What is the real diversification benefit of gold in a US equity + bond portfolio? | Portfolio Optimizer: compare SPY+TLT vs. SPY+TLT+GLD combinations |
| How sensitive is Kelly to the lookback window? | Run the same asset with 3yr, 10yr, 20yr historical data; compare f\* values |

---

## 8. 💸 DCA & Contribution Strategy Optimisation

**Goal:** Decide between lump-sum investing and systematic contributions.

**Workflow using Portfolio Simulator:**
1. Set a 2-asset portfolio (e.g. S&P 500 + 10-Yr Treasury, 70/30 split).
2. **Without DCA, without rebalancing** → record median final value.
3. **Enable rebalancing (quarterly)** → compare: does rebalancing add value for this asset pair?
4. **Enable DCA** at e.g. $500/month → compare the final distribution shape.
5. Note: DCA reduces the variance of outcomes (flattening the distribution) at the cost of lower median final value vs. lump-sum when markets trend upward.

**Insight:** DCA outperforms lump-sum in volatile, mean-reverting markets. Lump-sum outperforms DCA in trending bull markets ~70% of the time historically. The Portfolio Simulator lets you test this for your specific scenario.

---

## 9. 🏦 Institutional & Fund-Level Analysis

**Goal:** Apply Kelly + MPT framework to a fund-level asset allocation decision.

**Workflow:**
1. Open **Portfolio Optimizer**. Select *Factor ETFs* or *Global / International ETFs* universe.
2. Run 3- or 4-asset combinations with a 10-year lookback.
3. Identify portfolios with the highest Portfolio Kelly (K\*) and highest Diversification Bonus.
4. For the top-ranked combination, go to the **Portfolio Simulator** and model the exact allocation with annual rebalancing.
5. Run the same 4-asset portfolio with monthly DCA to simulate systematic fund inflows.
6. Export the statistics from Simulation Details for due diligence reporting.

**Key metrics to report:**
- Portfolio Kelly K\* (theoretical maximum growth rate at optimal weights)
- Diversification Bonus (marginal value of combining these specific assets)
- Historical Sharpe Ratio vs. Simulated Median Sharpe
- 5th Percentile CAGR (worst-case growth rate)
- Maximum Drawdown (mean and worst across simulations)
