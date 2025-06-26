import numpy as np
import pandas as pd
from .base_model import BaseModel

class GeometricBrownianMotionModel(BaseModel):
    """
    Geometric Brownian Motion model for asset price simulation.
    
    This model assumes that asset prices follow a geometric Brownian motion:
    dS = μS dt + σS dW
    
    Where:
    - S is the asset price
    - μ is the drift (expected return)
    - σ is the volatility
    - dW is a Wiener process (Brownian motion)
    """
    
    def __init__(self, returns_data, investment_amount=10000, time_horizon_years=10, 
                 num_simulations=10000, trading_days_per_year=252):
        """
        Initialize the Geometric Brownian Motion model.
        
        Parameters:
        -----------
        returns_data : pandas.Series or dict
            Historical returns data or dict containing returns and statistics
        investment_amount : float
            Initial investment amount in dollars
        time_horizon_years : int
            Number of years to simulate
        num_simulations : int
            Number of simulation paths to generate
        trading_days_per_year : int
            Number of trading days in a year (default: 252)
        """
        super().__init__(returns_data, investment_amount, time_horizon_years, 
                        num_simulations, trading_days_per_year)
        
        # Calculate mean and standard deviation if not provided in statistics
        if not self.statistics or 'mean_daily' not in self.statistics:
            mean_daily = float(self.returns.mean())
            std_daily = float(self.returns.std())
            self.statistics['mean_daily'] = mean_daily
            self.statistics['std_daily'] = std_daily
            self.statistics['mean_annual'] = mean_daily * trading_days_per_year
            self.statistics['std_annual'] = std_daily * np.sqrt(trading_days_per_year)
        
        # Compute drift and volatility for GBM
        self.dt = 1 / trading_days_per_year  # Time step (in years)
        self.mu = self.statistics['mean_daily']  # Drift
        self.sigma = self.statistics['std_daily']  # Volatility
        
        # Print diagnostics
        print(f"DEBUG - GBM Asset: {self.asset_name}")
        print(f"DEBUG - Mean daily return: {self.mu * 100:.6f}%")
        print(f"DEBUG - Annual return (approx): {self.mu * trading_days_per_year * 100:.2f}%")
        print(f"DEBUG - Daily volatility: {self.sigma * 100:.6f}%")
        
        # Adjust drift for GBM (Ito's correction)
        # We use the historical mean directly to ensure we match the expected returns
        # The correction -0.5*sigma^2 is applied to account for the log-normal distribution
        self.drift = self.mu - 0.5 * self.sigma**2
        
        # For very low volatility assets, the drift might be negative after Ito's correction
        # In that case, use a minimum threshold to ensure we capture the expected returns
        min_drift = self.mu / 2  # Ensure we preserve at least half of the expected return
        if self.drift < min_drift:
            print(f"DEBUG - Adjusting drift from {self.drift:.6f} to {min_drift:.6f} to preserve expected returns")
            self.drift = min_drift
    
    def generate_returns(self):
        """
        Generate price paths using Geometric Brownian Motion and convert to returns.
        
        Returns:
        --------
        numpy.ndarray
            Array of shape (num_simulations, total_days) with random returns
        """
        # Generate random Brownian motion increments
        dW = np.random.normal(0, np.sqrt(self.dt), size=(self.num_simulations, self.total_days))
        
        # Initialize price paths array (starting at 1.0)
        S = np.zeros((self.num_simulations, self.total_days + 1))
        S[:, 0] = 1.0
        
        # Generate price paths
        for t in range(1, self.total_days + 1):
            S[:, t] = S[:, t-1] * np.exp(self.drift * self.dt + self.sigma * dW[:, t-1])
        
        # Convert price paths to returns
        returns = np.diff(S, axis=1) / S[:, :-1]
        
        return returns
    
    def simulate(self, leverage=1.0):
        """
        Run Geometric Brownian Motion simulation with specified leverage.
        
        Parameters:
        -----------
        leverage : float
            Leverage to apply to returns
            
        Returns:
        --------
        dict
            Simulation results including statistics and paths
        """
        # Generate returns
        returns = self.generate_returns()
        
        # Apply leverage
        levered_returns = returns * leverage
        
        # Check for ruin cases (when levered return <= -100%)
        ruin_mask = levered_returns <= -1
        if np.any(ruin_mask):
            # Set return to -100% for ruin cases
            levered_returns[ruin_mask] = -1
        
        # Calculate cumulative returns
        cumulative_factor = np.cumprod(1 + levered_returns, axis=1)
        
        # Calculate portfolio values
        portfolio_values = self.investment_amount * cumulative_factor
        
        # Final portfolio values
        final_values = portfolio_values[:, -1]
        
        # Store for later use
        self.paths = portfolio_values
        self.final_values = final_values
        
        # Calculate statistics
        stats = self.calculate_statistics(final_values, portfolio_values, ruin_mask)
        
        # Create paths DataFrame
        paths_df = self.create_paths_dataframe(portfolio_values)
        
        # Store results
        result = {
            'stats': stats,
            'leverage': leverage,
            'investment_amount': self.investment_amount,
            'time_horizon_years': self.time_horizon_years,
            'num_simulations': self.num_simulations,
            'paths': paths_df,
            'asset_name': self.asset_name,
            'model_parameters': {
                'drift': float(self.drift),
                'volatility': float(self.sigma),
                'dt': self.dt
            }
        }
        
        # Store in instance variable and return
        self.simulation_results[leverage] = result
        return result
    
    def simulate_multiple_leverages(self, leverages=[0.5, 1.0, 1.5, 2.0]):
        """
        Run simulations with multiple leverage values.
        
        Parameters:
        -----------
        leverages : list
            List of leverage values to simulate
            
        Returns:
        --------
        dict
            Dictionary with leverage as keys and simulation results as values
        """
        results = {}
        for leverage in leverages:
            results[leverage] = self.simulate(leverage=leverage)
        
        return results
