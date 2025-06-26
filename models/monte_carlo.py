import numpy as np
import pandas as pd
from .base_model import BaseModel

class MonteCarloModel(BaseModel):
    """
    Standard Monte Carlo model that simulates asset returns using a normal distribution.
    """
    
    def __init__(self, returns_data, investment_amount=10000, time_horizon_years=10, 
                 num_simulations=10000, trading_days_per_year=252):
        """
        Initialize the Monte Carlo model.
        
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
    
    def generate_returns(self):
        """
        Generate random returns from normal distribution using historical mean and std.
        
        Returns:
        --------
        numpy.ndarray
            Array of shape (num_simulations, total_days) with random returns
        """
        mean_daily = self.statistics['mean_daily']
        std_daily = self.statistics['std_daily']
        
        # Generate returns from normal distribution
        returns = np.random.normal(
            loc=mean_daily,
            scale=std_daily,
            size=(self.num_simulations, self.total_days)
        )
        
        return returns
    
    def simulate(self, leverage=1.0):
        """
        Run Monte Carlo simulation with specified leverage.
        
        Parameters:
        -----------
        leverage : float
            Leverage to apply to returns
            
        Returns:
        --------
        dict
            Simulation results including statistics and paths
        """
        # For debugging: Print the mean and std of the returns being used
        if hasattr(self.returns, 'mean'):
            print(f"DEBUG - Asset: {self.asset_name}")
            
            # Get mean as float value
            mean_daily = float(self.returns.iloc[0]) if hasattr(self.returns, 'iloc') else float(self.returns.mean())
            std_daily = float(self.returns.iloc[0]) if hasattr(self.returns.std(), 'iloc') else float(self.returns.std())
            
            print(f"DEBUG - Mean daily return: {mean_daily * 100:.6f}%")
            print(f"DEBUG - Annual return (approx): {mean_daily * 252 * 100:.2f}%")
            print(f"DEBUG - Daily volatility: {std_daily * 100:.6f}%")
            print(f"DEBUG - Annual volatility (approx): {std_daily * np.sqrt(252) * 100:.2f}%")
        
        # Generate random returns
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
            'asset_name': self.asset_name
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
