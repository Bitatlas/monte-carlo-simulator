"""
Demo script to show the core functionality of the Monte Carlo simulator.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import our components
from data.fetchers import EquityIndexFetcher
from models import MonteCarloModel, GeometricBrownianMotionModel, HAS_GARCH
from optimization import KellyCalculator
from visualization import ChartGenerator

print("Multi-Asset Monte Carlo Simulator Demo")
print("-" * 50)

# Check available dependencies
print("\nChecking dependencies...")
try:
    import sklearn
    print("✓ scikit-learn is installed")
except ImportError:
    print("✗ scikit-learn is not installed (needed for Markov Chain model)")

try:
    from arch import arch_model
    print("✓ arch package is installed (GARCH models available)")
except ImportError:
    print("✗ arch package is not installed (GARCH models not available)")
    
print(f"GARCH models available: {'Yes' if HAS_GARCH else 'No'}")

def main():
    print("Multi-Asset Monte Carlo Simulator Demo")
    print("-" * 50)
    
    # 1. Fetch S&P 500 data
    print("\nFetching S&P 500 data...")
    fetcher = EquityIndexFetcher(index_type="SP500", period="5y")
    data = fetcher.fetch_data()
    returns_data = fetcher.calculate_returns()
    stats = fetcher.get_statistics()
    
    print(f"Data loaded for {fetcher.name}")
    print(f"Mean Annual Return: {stats['mean_annual']*100:.2f}%")
    print(f"Annual Volatility: {stats['std_annual']*100:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    
    # 2. Calculate Kelly Criterion
    print("\nCalculating Kelly Criterion...")
    kelly_calc = KellyCalculator(returns_data['daily'], risk_free_rate=0.02/252)  # 2% annual risk-free rate
    full_kelly = kelly_calc.calculate_full_kelly()
    half_kelly = full_kelly / 2
    
    print(f"Full Kelly Leverage: {full_kelly:.2f}x")
    print(f"Half Kelly Leverage: {half_kelly:.2f}x")
    
    # Generate leverage curve
    leverage_values, growth_rates = kelly_calc.generate_leverage_curve(max_leverage=3.0, points=50)
    
    # 3. Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation...")
    
    # Parameters
    investment_amount = 10000
    time_horizon_years = 10
    num_simulations = 1000
    leverage = half_kelly  # Use half Kelly
    
    # Create model
    model = GeometricBrownianMotionModel(
        returns_data=returns_data['daily'],
        investment_amount=investment_amount,
        time_horizon_years=time_horizon_years,
        num_simulations=num_simulations
    )
    
    # Run simulation
    result = model.simulate(leverage=leverage)
    
    # 4. Print results
    print(f"\nSimulation completed for {fetcher.name} using {leverage:.2f}x leverage")
    print(f"Initial Investment: ${investment_amount:,.0f}")
    print(f"Median Final Value: ${result['stats']['median']:,.0f}")
    print(f"Mean Final Value: ${result['stats']['mean']:,.0f}")
    print(f"5th Percentile: ${result['stats']['percentiles']['5%']:,.0f}")
    print(f"95th Percentile: ${result['stats']['percentiles']['95%']:,.0f}")
    print(f"Median CAGR: {result['stats']['cagr']['median']*100:.2f}%")
    print(f"Median Max Drawdown: {result['stats']['max_drawdown']['median']*100:.2f}%")
    print(f"Probability of Ruin: {result['stats']['ruin_probability']*100:.2f}%")
    
    # 5. Generate charts
    print("\nGenerating charts...")
    chart_gen = ChartGenerator()
    
    # Kelly curve
    kelly_fig = chart_gen.plot_kelly_curve(
        leverage_values, 
        growth_rates, 
        optimal_leverage=full_kelly
    )
    kelly_fig.savefig('kelly_curve.png')
    print("Kelly curve saved as 'kelly_curve.png'")
    
    # Simulation paths
    paths_fig = chart_gen.plot_simulation_paths(
        result['paths'],
        title=f"{fetcher.name} Simulation Paths (Leverage: {leverage:.2f}x)",
        num_paths=50
    )
    paths_fig.savefig('simulation_paths.png')
    print("Simulation paths saved as 'simulation_paths.png'")
    
    # Final distribution
    dist_fig = chart_gen.plot_final_distribution(result)
    dist_fig.savefig('final_distribution.png')
    print("Final distribution saved as 'final_distribution.png'")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
