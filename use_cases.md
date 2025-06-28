# Monte Carlo Simulator Use Cases Guide

This guide explains the primary use cases for the Monte Carlo Simulator and how to interpret the results of different models.

## Primary Use Case: Optimal Kelly Leverage to Maximize Risk-Adjusted Returns

The most important application of this simulator is to determine the optimal leverage that maximizes your risk-adjusted returns over the long term.

### How to Find Optimal Kelly Leverage

1. **Select your asset** in the sidebar
2. **Set the historical data period** to a timeframe you believe represents the future behavior of the asset
3. **Choose your risk-free rate** (typically the yield on short-term Treasury bills)
4. **Set leverage method** to "Kelly Criterion" or "Numerical Optimization"
5. **Run the simulation**
6. **Navigate to the "Kelly Analysis" tab** to see the full Kelly curve and optimal leverage values

### Interpreting Kelly Results

- **Full Kelly Leverage**: The mathematically optimal leverage to maximize long-term growth rate
- **Half Kelly Leverage**: A more conservative approach (50% of full Kelly) that sacrifices some growth for lower risk
- **Optimal Leverage (Numerical)**: Often more robust than the standard Kelly formula, as it accounts for the actual distribution of returns

Remember that Kelly assumes you can maintain the exact leverage indefinitely through rebalancing. In practice, most professionals use "Half Kelly" or even "Quarter Kelly" to account for parameter uncertainty.

## Secondary Use Cases

### 1. Portfolio Drawdown Risk Assessment

- Run simulations at different leverage levels to evaluate maximum drawdowns
- Use multiple models (especially GARCH for volatile periods) to get different perspectives on tail risk
- Focus on the "Simulation Details" tab to see the full range of potential drawdown scenarios

### 2. Retirement Planning

- Set your initial investment amount to your current savings
- Choose a time horizon matching your retirement timeline
- Run simulations with different asset allocations (represented by different leverage values)
- The "Distribution of Final Values" chart helps visualize the range of possible outcomes

### 3. Comparing Asset Classes

- Run simulations on different assets (stocks, bonds, cryptocurrencies)
- Compare their risk-return profiles under various leverage assumptions
- Use the "Simulation Dashboard" tab to quickly compare key metrics

## Advanced Use Case: Asset Allocation Optimization

For sophisticated users, the simulator can help determine optimal asset allocation:

1. Run separate simulations for each asset class
2. Note the optimal Kelly leverage for each asset
3. Allocate your portfolio inversely proportional to the assets' variance and proportional to their excess returns
4. Re-run simulations with the combined portfolio to validate the approach

## Real-World Example

Imagine you're considering investing in the S&P 500 with leverage:

1. Select "S&P 500" as your asset
2. Set historical data to 10 years (capturing a full market cycle)
3. Use a risk-free rate of 2%
4. Run the simulation with "Numerical Optimization" as the leverage method
5. The simulator might suggest an optimal leverage of 1.5x
6. Check the drawdown risk at this leverage level
7. Consider using a more conservative 0.75x leverage (Half Kelly) to account for uncertainty

## Common Mistakes to Avoid

1. **Using too short historical periods**: Make sure your data captures different market regimes
2. **Applying full Kelly blindly**: Parameter estimation errors can lead to excessive risk
3. **Ignoring drawdowns**: Even optimal Kelly strategies can experience significant temporary losses
4. **Not considering correlation between assets**: When leveraging multiple assets, account for their correlations

By using the Monte Carlo Simulator thoughtfully, you can develop a more robust understanding of the risk-reward tradeoffs in your investment decisions.
