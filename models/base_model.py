import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all simulation models.
    
    Defines the interface that all specific models must implement.
    """
    
    def __init__(self, returns_data, investment_amount=10000, time_horizon_years=10, 
                 num_simulations=10000, trading_days_per_year=252):
        """
        Initialize the base model.
        
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
        # Store raw returns data
        if isinstance(returns_data, dict):
            self.returns = returns_data.get('returns', returns_data.get('daily', None))
            self.statistics = returns_data.get('statistics', {})
            self.asset_name = returns_data.get('name', "Unknown Asset")
        else:
            self.returns = returns_data
            self.statistics = {}
            self.asset_name = "Unknown Asset"
        
        # Configuration parameters
        self.investment_amount = investment_amount
        self.time_horizon_years = time_horizon_years
        self.num_simulations = num_simulations
        self.trading_days_per_year = trading_days_per_year
        self.total_days = self.time_horizon_years * self.trading_days_per_year
        
        # Output containers
        self.simulation_results = {}
        self.paths = None
        self.final_values = None
    
    @abstractmethod
    def generate_returns(self):
        """
        Generate random returns for simulation.
        Must be implemented by subclasses.
        
        Returns:
        --------
        numpy.ndarray
            Array of shape (num_simulations, total_days) with random returns
        """
        pass
    
    @abstractmethod
    def simulate(self, leverage=1.0):
        """
        Run simulation with specified parameters.
        Must be implemented by subclasses.
        
        Parameters:
        -----------
        leverage : float
            Leverage to apply to returns
            
        Returns:
        --------
        dict
            Simulation results including statistics and paths
        """
        pass
    
    def calculate_max_drawdowns(self, portfolio_values):
        """
        Calculate maximum drawdown for each simulation path.
        
        Parameters:
        -----------
        portfolio_values : numpy.ndarray
            Array of portfolio values for each simulation path
            
        Returns:
        --------
        numpy.ndarray
            Array of maximum drawdowns for each path
        """
        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values, axis=1)
        
        # Calculate drawdowns
        drawdowns = portfolio_values / running_max - 1
        
        # Get maximum drawdown for each path
        max_drawdowns = np.abs(np.min(drawdowns, axis=1))
        
        return max_drawdowns
    
    def calculate_statistics(self, final_values, portfolio_values, ruin_mask=None):
        """
        Calculate statistics from simulation results.
        
        Parameters:
        -----------
        final_values : numpy.ndarray
            Array of final portfolio values
        portfolio_values : numpy.ndarray
            Array of portfolio values for each simulation path
        ruin_mask : numpy.ndarray, optional
            Boolean mask indicating which paths experienced ruin
            
        Returns:
        --------
        dict
            Dictionary of statistics
        """
        # Calculate ruin probability if not provided
        if ruin_mask is None:
            ruin_mask = portfolio_values <= 0
            
        ruin_probability = np.mean(np.any(ruin_mask, axis=1))
        
        # Calculate bust counters (paths ending below thresholds)
        total_paths = len(final_values)
        
        # Define complete ruin as a loss of over 99% (final value < 1% of initial)
        ruin_threshold = self.investment_amount * 0.01
        bust_total_ruin = np.sum(final_values < ruin_threshold)  # Loss of over 99%
        
        # Paths below initial investment but not in ruin
        bust_below_initial = np.sum((final_values >= ruin_threshold) & (final_values < self.investment_amount))
        
        # Paths above initial investment (successful paths)
        paths_above_initial = np.sum(final_values >= self.investment_amount)
        
        # Calculate benchmark based on asset's historical return from the specific historical period
        if hasattr(self, 'statistics') and 'mean_annual' in self.statistics:
            # Use historical return to calculate benchmark for the simulation time horizon
            asset_annual_return = self.statistics['mean_annual']
            benchmark_multiplier = (1 + asset_annual_return) ** self.time_horizon_years
            benchmark_value = self.investment_amount * benchmark_multiplier
            benchmark_name = f"Historical Return ({asset_annual_return*100:.1f}% annually)"
            
            # Store historical return details for comparison
            historical_return = asset_annual_return
            historical_period_years = self.statistics.get('data_period_years', 
                                                           'unknown period')
            historical_period_name = f"{historical_period_years} years"
        else:
            # Fallback to a simple 2x benchmark if historical data not available
            benchmark_multiplier = 2.0
            benchmark_value = self.investment_amount * benchmark_multiplier
            benchmark_name = "2x Initial Investment"
            historical_return = None
            historical_period_years = 'unknown'
            historical_period_name = 'unknown'
            
        # Count paths above the asset-based benchmark
        paths_above_benchmark = np.sum(final_values >= benchmark_value)
        
        # Calculate percentages
        bust_total_ruin_pct = bust_total_ruin / total_paths
        bust_below_initial_pct = bust_below_initial / total_paths
        paths_above_initial_pct = paths_above_initial / total_paths
        paths_above_benchmark_pct = paths_above_benchmark / total_paths
        
        # Calculate CAGR (Compound Annual Growth Rate)
        # CAGR = (Final Value / Initial Value)^(1/years) - 1
        cagr_values = (final_values / self.investment_amount) ** (1/self.time_horizon_years) - 1
        
        # Calculate max drawdowns
        max_drawdowns = self.calculate_max_drawdowns(portfolio_values)
        
        # Calculate Sharpe ratios for simulation paths
        # First, calculate returns from portfolio values for each path
        sim_returns = np.diff(portfolio_values, axis=1) / portfolio_values[:, :-1]
        
        # Explicitly account for the effect of leverage
        # Leverage affects returns but also increases volatility
        # We need to make sure this is properly reflected in the Sharpe ratio
        
        # Risk-free rate for excess return calculation - assume standard 2% if not available
        risk_free_daily = 0.02 / 252  # Approx daily risk-free rate from 2% annual
        
        # Calculate Sharpe ratio for each path with correct risk adjustment
        path_sharpe_ratios = []
        for i in range(sim_returns.shape[0]):
            # Get returns for this path
            path_return = np.mean(sim_returns[i])
            path_std = np.std(sim_returns[i])
            
            # Only calculate if we have meaningful volatility
            if path_std > 0:
                # Excess return over risk-free rate
                excess_return = path_return - risk_free_daily
                
                # Sharpe ratio is excess return divided by risk
                path_sharpe = excess_return / path_std
            else:
                path_sharpe = 0
                
            path_sharpe_ratios.append(path_sharpe)
        
        # Get historical Sharpe ratio if available
        if hasattr(self, 'statistics') and 'sharpe_ratio' in self.statistics:
            historical_sharpe = self.statistics['sharpe_ratio']
        else:
            historical_sharpe = None
        
        # Calculate annualized Sharpe (multiply by sqrt(252) to annualize)
        path_sharpe_ratios = np.array(path_sharpe_ratios) * np.sqrt(self.trading_days_per_year)
        
        # Debug logging
        print(f"DEBUG - Sharpe calculation:")
        print(f"  Average path return: {np.mean(sim_returns) * 100:.6f}%")
        print(f"  Average path volatility: {np.mean([np.std(sim_returns[i]) for i in range(sim_returns.shape[0])]) * 100:.6f}%")
        # Fix f-string formatting issue
        if historical_sharpe is not None:
            print(f"  Historical Sharpe: {historical_sharpe:.4f}")
        else:
            print("  Historical Sharpe: N/A")
        print(f"  Path Sharpe ratios range: {np.min(path_sharpe_ratios):.4f} to {np.max(path_sharpe_ratios):.4f}")
        print(f"  Leverage: {leverage if 'leverage' in locals() else 'unknown'}")
        
        # Calculate median and mean
        median_sharpe = float(np.median(path_sharpe_ratios))
        mean_sharpe = float(np.mean(path_sharpe_ratios))
        
        # Compile statistics
        stats = {
            'mean': float(np.mean(final_values)),
            'median': float(np.median(final_values)),
            'std': float(np.std(final_values)),
            'min': float(np.min(final_values)),
            'max': float(np.max(final_values)),
            'ruin_probability': float(ruin_probability),
            'bust_counters': {
                'total_ruin': int(bust_total_ruin),
                'below_initial': int(bust_below_initial),
                'above_initial': int(paths_above_initial),
                'above_benchmark': int(paths_above_benchmark),
                'total_ruin_pct': float(bust_total_ruin_pct),
                'below_initial_pct': float(bust_below_initial_pct),
                'above_initial_pct': float(paths_above_initial_pct),
                'above_benchmark_pct': float(paths_above_benchmark_pct),
                'total_paths': total_paths,
                'benchmark_value': float(benchmark_value),
                'benchmark_name': benchmark_name,
                'historical_return': historical_return,
                'historical_period': historical_period_name,
                'ruin_threshold': float(ruin_threshold)
            },
            'sharpe_ratio': {
                'median': median_sharpe,
                'mean': mean_sharpe,
                'min': float(np.min(path_sharpe_ratios)),
                'max': float(np.max(path_sharpe_ratios)),
                'historical': historical_sharpe,
                'percentiles': {
                    '5%': float(np.percentile(path_sharpe_ratios, 5)),
                    '25%': float(np.percentile(path_sharpe_ratios, 25)),
                    '75%': float(np.percentile(path_sharpe_ratios, 75)),
                    '95%': float(np.percentile(path_sharpe_ratios, 95))
                }
            },
            'percentiles': {
                '1%': float(np.percentile(final_values, 1)),
                '5%': float(np.percentile(final_values, 5)),
                '10%': float(np.percentile(final_values, 10)),
                '25%': float(np.percentile(final_values, 25)),
                '50%': float(np.percentile(final_values, 50)),
                '75%': float(np.percentile(final_values, 75)),
                '90%': float(np.percentile(final_values, 90)),
                '95%': float(np.percentile(final_values, 95)),
                '99%': float(np.percentile(final_values, 99))
            },
            'cagr': {
                'mean': float(np.mean(cagr_values)),
                'median': float(np.median(cagr_values)),
                'percentiles': {
                    '5%': float(np.percentile(cagr_values, 5)),
                    '95%': float(np.percentile(cagr_values, 95))
                }
            },
            'max_drawdown': {
                'mean': float(np.mean(max_drawdowns)),
                'median': float(np.median(max_drawdowns)),
                'max': float(np.max(max_drawdowns)),
                'percentiles': {
                    '95%': float(np.percentile(max_drawdowns, 95))
                }
            }
        }
        
        return stats
    
    def create_paths_dataframe(self, portfolio_values):
        """
        Create a DataFrame with dates for the simulated paths.
        
        Parameters:
        -----------
        portfolio_values : numpy.ndarray
            Array of portfolio values
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with dates as index and paths as columns
        """
        from datetime import datetime, timedelta
        
        # Create dates
        start_date = datetime.now()
        dates = [start_date + timedelta(days=i*365.25/self.trading_days_per_year) 
                for i in range(self.total_days)]
        
        # Select subset of paths for visualization
        num_paths_to_include = min(100, self.num_simulations)
        random_indices = np.random.choice(portfolio_values.shape[0], num_paths_to_include, replace=False)
        selected_paths = portfolio_values[random_indices, :]
        
        # Ensure selected_paths is 2D
        if len(selected_paths.shape) > 2:
            selected_paths = selected_paths.reshape(selected_paths.shape[1], selected_paths.shape[0]).T
        
        # Create DataFrame with non-numeric column names (to avoid showing numbers in legends)
        column_names = [f"Path_{i}" for i in range(len(random_indices))]
        paths_df = pd.DataFrame(selected_paths, columns=dates, index=column_names).T
        
        return paths_df
