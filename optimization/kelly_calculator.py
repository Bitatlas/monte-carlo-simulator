import numpy as np
import pandas as pd
from scipy.optimize import minimize

class KellyCalculator:
    """
    Kelly Criterion calculator for determining optimal leverage.
    
    The Kelly Criterion is a formula used to determine the optimal size of a series
    of bets or investments to maximize the logarithm of wealth over the long run.
    """
    
    def __init__(self, returns, risk_free_rate=0.0):
        """
        Initialize the Kelly calculator.
        
        Parameters:
        -----------
        returns : pandas.Series or numpy.ndarray
            Historical returns data
        risk_free_rate : float
            Annual risk-free rate (default: 0.0) - will be converted to daily
        """
        self.returns = returns
        
        # Convert annual risk-free rate to daily equivalent
        # Use continuous compounding formula: daily_rate = (1 + annual_rate)^(1/252) - 1
        self.annual_risk_free_rate = risk_free_rate
        self.risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        print(f"INFO - Converting annual risk-free rate {risk_free_rate*100:.2f}% to daily: {self.risk_free_rate*100:.6f}%")
        
        # Calculate key statistics
        self.mean_return = float(np.mean(returns))
        self.variance = float(np.var(returns))
        self.std_dev = float(np.std(returns))
        self.excess_return = self.mean_return - self.risk_free_rate
        
        # Annualized metrics
        self.annual_mean_return = ((1 + self.mean_return) ** 252) - 1
        self.annual_std_dev = self.std_dev * np.sqrt(252)
    
    def calculate_full_kelly(self):
        """
        Calculate the full Kelly leverage.
        
        Returns:
        --------
        float
            Full Kelly leverage
        """
        # Print diagnostic information
        print(f"DEBUG - Kelly Calculation for asset:")
        print(f"  Mean return: {self.mean_return * 100:.6f}% daily, {self.mean_return * 252 * 100:.2f}% annually")
        print(f"  Volatility: {self.std_dev * 100:.6f}% daily, {self.std_dev * np.sqrt(252) * 100:.2f}% annually")
        print(f"  Risk-free rate: {self.risk_free_rate * 100:.6f}% daily")
        print(f"  Excess return: {self.excess_return * 100:.6f}% daily")
        print(f"  Variance: {self.variance * 100:.6f}%")
        
        # Classical Kelly formula: f* = (μ - r) / σ²
        if self.variance > 0:
            # Calculate Kelly formula
            kelly = self.excess_return / self.variance
            
            # Handle negative excess returns more gracefully
            if self.excess_return <= 0:
                print(f"  WARNING: Negative excess return ({self.excess_return:.6f}). Kelly formula gives negative leverage: {kelly:.4f}x")
                print(f"  Returning 0.0 leverage since negative leverage isn't typically used")
                return 0.0
            else:
                # Cap at a reasonable maximum to avoid extreme values
                max_reasonable_kelly = 5.0
                capped_kelly = min(kelly, max_reasonable_kelly)
                
                if kelly > max_reasonable_kelly:
                    print(f"  NOTE: Kelly formula gave very high leverage ({kelly:.4f}x), capped at {max_reasonable_kelly:.1f}x")
                
                print(f"  Kelly formula result: {capped_kelly:.4f}x")
                return capped_kelly
        else:
            # Fallback for zero variance (should not happen with real data)
            print(f"  WARNING: Zero or negative variance. Using fallback leverage of 1.0x")
            return 1.0
    
    def calculate_fractional_kelly(self, fraction=0.5):
        """
        Calculate a fractional Kelly leverage.
        
        Parameters:
        -----------
        fraction : float
            Fraction of full Kelly to use (default: 0.5)
            
        Returns:
        --------
        float
            Fractional Kelly leverage
        """
        full_kelly = self.calculate_full_kelly()
        return full_kelly * fraction
    
    def calculate_growth_rate(self, leverage):
        """
        Calculate expected logarithmic growth rate for a given leverage.
        
        Parameters:
        -----------
        leverage : float
            Leverage value
            
        Returns:
        --------
        float
            Expected growth rate
        """
        # Calculate using continuous compounding formula
        # Use the classical formula E[log(1 + f*R)] ≈ μ*f - (σ²*f²)/2
        # But scale it to reflect the annual growth rate rather than daily rate
        
        # Convert daily expected return and variance to annualized values
        annual_mean = self.mean_return * 252  # Approximately 252 trading days in a year
        annual_variance = self.variance * 252
        
        # Calculate using Kelly formula
        # First calculate daily growth rate
        daily_growth_rate = leverage * self.excess_return - (leverage**2 * self.variance) / 2
        
        # Annualize (approximately)
        annual_growth_rate = daily_growth_rate * 252
        
        # For more accuracy with real returns, also compute using actual data
        # This better accounts for non-normal return distributions
        log_wealth = np.log(1 + leverage * self.returns)
        empirical_daily_growth = np.mean(log_wealth)
        empirical_annual_growth = empirical_daily_growth * 252
        
        # Print debugging information
        print(f"DEBUG - Leverage: {leverage:.2f}, Formula Growth: {annual_growth_rate:.4%}, Empirical Growth: {empirical_annual_growth:.4%}")
        
        # Use the empirical calculation as it's more accurate for real data
        return empirical_annual_growth
    
    def find_optimal_leverage_numerical(self, max_leverage=5.0):
        """
        Find optimal leverage using numerical optimization.
        
        This is more accurate than the simple formula, especially for non-normal returns.
        
        Parameters:
        -----------
        max_leverage : float
            Maximum leverage to consider
            
        Returns:
        --------
        float
            Optimal leverage
        """
        # Print some statistics about the returns for debugging
        print(f"DEBUG - Kelly Optimization for asset with:")
        print(f"  Mean return: {self.mean_return * 100:.6f}% daily, {self.annual_mean_return * 100:.2f}% annually")
        print(f"  Volatility: {self.std_dev * 100:.6f}% daily, {self.annual_std_dev * 100:.2f}% annually")
        print(f"  Risk-free rate: {self.risk_free_rate * 100:.6f}% daily, {self.annual_risk_free_rate * 100:.2f}% annually")
        print(f"  Simple Kelly formula result: {self.calculate_full_kelly():.4f}x")
        
        # First, perform a grid search to find the area with highest growth
        num_grid_points = 50
        grid_leverages = np.linspace(0.01, max_leverage, num_grid_points)
        grid_growth_rates = np.zeros(num_grid_points)
        
        print(f"DEBUG - Performing grid search with {num_grid_points} points from 0.01 to {max_leverage:.2f}x")
        
        # Calculate growth rate at each grid point
        for i, lev in enumerate(grid_leverages):
            # Define a safety check function for potential ruin
            # Avoid leverage that would cause bankruptcy on any historical return
            min_return = np.min(self.returns)
            if lev * min_return <= -1:
                # This leverage could cause ruin with historical data
                grid_growth_rates[i] = -100.0  # Severe penalty
                continue
                
            # Calculate log wealth for each period
            log_wealth = np.log(1 + lev * self.returns)
            
            # Expected growth rate is the mean of log wealth
            growth_rate = np.mean(log_wealth)
            
            # Scale to annual rate
            annual_growth = growth_rate * 252
            grid_growth_rates[i] = annual_growth
        
        # Find best grid point
        best_grid_index = np.argmax(grid_growth_rates)
        best_grid_leverage = grid_leverages[best_grid_index]
        best_grid_growth = grid_growth_rates[best_grid_index]
        
        print(f"DEBUG - Grid search found best leverage at {best_grid_leverage:.4f}x with growth {best_grid_growth:.6f}")
        
        # Define negative growth rate function for optimization
        def negative_growth_rate(leverage):
            # Convert to array for compatibility with minimize
            lev = leverage[0]
            
            # Check for potential ruin
            min_return = np.min(self.returns)
            if lev * min_return <= -1:
                return 1000.0  # Heavy penalty
            
            # Calculate growth rate using log wealth
            log_wealth = np.log(1 + lev * self.returns)
            growth_rate = np.mean(log_wealth)
            
            # Apply a penalty for high leverage that scales with volatility
            # This prevents excessive leverage for highly volatile assets
            volatility_penalty = 0.01 * lev * self.annual_std_dev
            
            return -(growth_rate * 252 - volatility_penalty)  # Negative for minimization
        
        # Make initial guess based on grid search and Kelly formula
        kelly_f = self.calculate_full_kelly()
        
        # Use the best of: grid search result, Kelly formula, or 1.0x
        # as the starting point for fine-tuned optimization
        candidate_starting_points = [
            best_grid_leverage,
            kelly_f if kelly_f > 0 else 1.0,
            1.0  # Always include 1.0x as a safe starting point
        ]
        
        # Set reasonable bounds based on asset characteristics
        # More volatile assets should have tighter bounds
        if self.annual_std_dev > 0.3:  # Very volatile (>30% annual volatility)
            safe_max = min(3.0, max_leverage)
            print(f"DEBUG - Limiting max leverage to {safe_max:.1f}x due to high volatility ({self.annual_std_dev*100:.1f}%)")
        else:
            safe_max = max_leverage
            
        bounds = [(0.0, safe_max)]
        
        # Try multiple starting points and optimization methods
        best_result = None
        best_growth = -np.inf
        
        print(f"DEBUG - Trying optimization with multiple starting points: {[f'{x:.2f}x' for x in candidate_starting_points]}")
        
        for start_point in candidate_starting_points:
            # Skip invalid starting points
            if start_point <= 0 or start_point > safe_max:
                continue
                
            initial_guess = [start_point]
            
            # Try multiple optimization methods
            methods = ['L-BFGS-B', 'SLSQP', 'TNC']
            
            for method in methods:
                try:
                    # Minimize negative growth rate
                    result = minimize(negative_growth_rate, initial_guess, bounds=bounds, method=method)
                    
                    if result.success and -result.fun > best_growth:
                        best_result = result
                        best_growth = -result.fun
                        
                    print(f"DEBUG - From start {start_point:.2f}x, method {method} result: {result.x[0]:.4f}x with growth rate: {-result.fun:.6f}")
                except Exception as e:
                    print(f"DEBUG - Method {method} failed from start point {start_point:.2f}x: {e}")
        
        # Process the result
        if best_result is not None:
            optimal_lev = float(best_result.x[0])
            
            # Apply reasonableness check based on asset characteristics
            if optimal_lev > 0 and kelly_f > 0:
                # If optimization result is more than 3x the Kelly formula, cap it
                if optimal_lev > kelly_f * 3:
                    print(f"DEBUG - Optimization result ({optimal_lev:.4f}x) is much higher than Kelly ({kelly_f:.4f}x). Capping at {kelly_f * 2:.4f}x")
                    optimal_lev = kelly_f * 2
            
            print(f"DEBUG - Final optimal leverage: {optimal_lev:.4f}x with growth rate: {best_growth:.6f}")
            return optimal_lev
        else:
            # Fallback to grid search result
            print(f"DEBUG - Optimization failed, falling back to grid search result: {best_grid_leverage:.4f}x")
            return best_grid_leverage
    
    def generate_leverage_curve(self, max_leverage=5.0, points=100):
        """
        Generate growth rate curve for different leverage values.
        
        Parameters:
        -----------
        max_leverage : float
            Maximum leverage to consider
        points : int
            Number of points to calculate
            
        Returns:
        --------
        tuple
            (leverage_values, growth_rates)
        """
        leverage_values = np.linspace(0, max_leverage, points)
        growth_rates = np.zeros_like(leverage_values)
        
        for i, lev in enumerate(leverage_values):
            growth_rates[i] = self.calculate_growth_rate(lev)
        
        return leverage_values, growth_rates
    
    def analyze_leveraged_returns(self, leverages=[0.0, 0.5, 1.0, 1.5, 2.0]):
        """
        Analyze returns with different leverage values.
        
        Parameters:
        -----------
        leverages : list
            List of leverage values to analyze
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with analysis results
        """
        results = []
        
        for lev in leverages:
            # Calculate leveraged returns
            lev_returns = self.returns * lev
            
            # Check for ruin
            ruin_mask = lev_returns <= -1
            if np.any(ruin_mask):
                lev_returns[ruin_mask] = -1
            
            # Calculate statistics
            mean_return = np.mean(lev_returns)
            std_dev = np.std(lev_returns)
            sharpe = mean_return / std_dev if std_dev > 0 else 0
            
            # Calculate terminal wealth multiple
            terminal_wealth = np.prod(1 + lev_returns)
            
            # Calculate max drawdown
            cum_returns = np.cumprod(1 + lev_returns)
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns / peak) - 1
            max_drawdown = np.min(drawdown)
            
            # Calculate growth rate
            growth_rate = np.mean(np.log(1 + lev_returns))
            
            results.append({
                'leverage': lev,
                'mean_return': mean_return,
                'std_dev': std_dev,
                'sharpe_ratio': sharpe,
                'terminal_wealth': terminal_wealth,
                'max_drawdown': max_drawdown,
                'growth_rate': growth_rate,
                'ruin_probability': np.mean(ruin_mask)
            })
        
        return pd.DataFrame(results)
