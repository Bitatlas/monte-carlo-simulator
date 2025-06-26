import numpy as np
import pandas as pd
from scipy import stats
from .base_model import BaseModel

class FeynmanPathIntegralModel(BaseModel):
    """
    Feynman Path Integral model for asset returns simulation.
    
    This model applies concepts from quantum mechanics to finance, treating
    asset price paths as quantum paths weighted by an "action" function.
    It uses path sampling to generate realistic market scenarios that may
    include non-Gaussian and rare events.
    """
    
    def __init__(self, returns_data, investment_amount=10000, time_horizon_years=10, 
                 num_simulations=10000, trading_days_per_year=252, num_paths=1000, 
                 num_time_steps=50, num_price_levels=100):
        """
        Initialize the Feynman Path Integral model.
        
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
        num_paths : int
            Number of paths to sample in the path integral (default: 1000)
        num_time_steps : int
            Number of time steps for discretization (default: 50)
        num_price_levels : int
            Number of price levels for discretization (default: 100)
        """
        super().__init__(returns_data, investment_amount, time_horizon_years, 
                        num_simulations, trading_days_per_year)
        
        # Path integral parameters
        self.num_paths = num_paths
        self.num_time_steps = num_time_steps
        self.num_price_levels = num_price_levels
        
        # Derived parameters
        if not self.statistics or 'mean_daily' not in self.statistics:
            mean_daily = float(self.returns.mean())
            std_daily = float(self.returns.std())
            self.statistics['mean_daily'] = mean_daily
            self.statistics['std_daily'] = std_daily
            self.statistics['mean_annual'] = mean_daily * trading_days_per_year
            self.statistics['std_annual'] = std_daily * np.sqrt(trading_days_per_year)
        
        # Compute drift and volatility
        self.dt = 1 / trading_days_per_year  # Time step (in years)
        self.mu = self.statistics['mean_daily']  # Drift
        self.sigma = self.statistics['std_daily']  # Volatility
        
        # Print diagnostics
        print(f"DEBUG - Path Integral Asset: {self.asset_name}")
        print(f"DEBUG - Mean daily return: {self.mu * 100:.6f}%")
        print(f"DEBUG - Annual return (approx): {self.mu * trading_days_per_year * 100:.2f}%")
        print(f"DEBUG - Daily volatility: {self.sigma * 100:.6f}%")
        
        # For discretization
        self.time_step = self.time_horizon_years / self.num_time_steps
        
        # Compute price grid
        total_vol = self.sigma * np.sqrt(self.time_horizon_years)
        self.log_price_min = -5 * total_vol
        self.log_price_max = 5 * total_vol
        self.log_price_step = (self.log_price_max - self.log_price_min) / (self.num_price_levels - 1)
        
        # Precompute transition probabilities
        self.precompute_transition_probs()
    
    def precompute_transition_probs(self):
        """
        Precompute transition probabilities between price levels.
        """
        # Create price grid
        self.log_price_grid = np.linspace(self.log_price_min, self.log_price_max, self.num_price_levels)
        
        # Compute transition probability matrix
        self.trans_probs = np.zeros((self.num_price_levels, self.num_price_levels))
        
        # Time step for discretized process
        dt = self.time_step
        
        # Drift and diffusion parameters for the transition probabilities
        # Adjust the drift to better preserve the expected return characteristics
        # The -0.5*sigma^2 term is the Ito correction for GBM
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        
        # For very low volatility assets, the drift might become too small after Ito's correction
        # In that case, enforce a minimum drift based on the expected return
        min_drift = self.mu * dt * 0.5  # At least half of the expected return
        if drift < min_drift:
            print(f"DEBUG - Adjusting path integral drift from {drift:.6f} to {min_drift:.6f} to preserve expected returns")
            drift = min_drift
            
        # Set the diffusion parameter based on volatility
        diffusion = self.sigma * np.sqrt(dt)
        
        # Compute transition probabilities
        for i, log_price_i in enumerate(self.log_price_grid):
            for j, log_price_j in enumerate(self.log_price_grid):
                # Calculate log return
                log_return = log_price_j - log_price_i
                
                # Calculate probability density using normal distribution
                prob = stats.norm.pdf(log_return, loc=drift, scale=diffusion)
                
                # Scale by step size to get approximate probability
                self.trans_probs[i, j] = prob * self.log_price_step
        
        # Normalize each row to ensure valid probability distribution
        row_sums = self.trans_probs.sum(axis=1, keepdims=True)
        self.trans_probs = self.trans_probs / row_sums
    
    def calculate_action(self, path):
        """
        Calculate the action (negative log probability) for a given path.
        
        Parameters:
        -----------
        path : numpy.ndarray
            Array of price level indices representing a path
            
        Returns:
        --------
        float
            Action value (lower is more probable)
        """
        action = 0.0
        
        for t in range(len(path) - 1):
            from_level = path[t]
            to_level = path[t + 1]
            
            # Get transition probability
            prob = self.trans_probs[from_level, to_level]
            
            # Add negative log probability to action
            action -= np.log(prob + 1e-10)  # Add small epsilon to avoid log(0)
        
        return action
    
    def metropolis_path_sampling(self):
        """
        Sample paths using Metropolis-Hastings algorithm.
        
        Returns:
        --------
        list
            List of sampled paths as price level indices
        """
        # Start with a simple path: middle price level throughout
        mid_level = self.num_price_levels // 2
        current_path = np.full(self.num_time_steps + 1, mid_level)
        current_action = self.calculate_action(current_path)
        
        sampled_paths = []
        
        # Metropolis-Hastings sampling
        for _ in range(self.num_paths):
            # Copy current path
            proposed_path = current_path.copy()
            
            # Choose a random time point to modify (except the first one)
            t = np.random.randint(1, self.num_time_steps + 1)
            
            # Propose a new price level (randomly move up or down by 1-3 levels)
            step = np.random.choice([-3, -2, -1, 1, 2, 3])
            proposed_path[t] = np.clip(proposed_path[t] + step, 0, self.num_price_levels - 1)
            
            # Calculate new action
            proposed_action = self.calculate_action(proposed_path)
            
            # Accept/reject based on action difference
            acceptance_ratio = np.exp(current_action - proposed_action)
            
            if np.random.random() < acceptance_ratio:
                # Accept the proposed path
                current_path = proposed_path
                current_action = proposed_action
            
            # Store the current path
            sampled_paths.append(current_path.copy())
        
        return sampled_paths
    
    def path_to_returns(self, path):
        """
        Convert a path of price levels to a sequence of returns.
        
        Parameters:
        -----------
        path : numpy.ndarray
            Array of price level indices representing a path
            
        Returns:
        --------
        numpy.ndarray
            Array of returns
        """
        # Convert price level indices to log prices
        log_prices = np.array([self.log_price_grid[idx] for idx in path])
        
        # Convert log prices to returns
        returns = np.diff(log_prices)
        
        # Scale returns to match the desired time step
        returns = returns / self.time_step * self.dt
        
        return returns
    
    def generate_returns(self):
        """
        Generate returns using the Feynman Path Integral approach.
        
        Returns:
        --------
        numpy.ndarray
            Array of shape (num_simulations, total_days) with random returns
        """
        # Sample paths using Metropolis-Hastings
        sampled_paths = self.metropolis_path_sampling()
        
        # Convert paths to returns
        path_returns = [self.path_to_returns(path) for path in sampled_paths]
        
        # Now we need to interpolate these returns to get the full time series
        full_returns = np.zeros((self.num_simulations, self.total_days))
        
        # For each simulation, randomly select a path and interpolate
        for i in range(self.num_simulations):
            # Randomly select a path
            path_idx = np.random.randint(0, len(path_returns))
            path_ret = path_returns[path_idx]
            
            # Interpolate to get returns for all days
            # We use piecewise constant interpolation for simplicity
            days_per_step = self.total_days // len(path_ret)
            
            for j, ret in enumerate(path_ret):
                start_idx = j * days_per_step
                end_idx = min((j + 1) * days_per_step, self.total_days)
                
                if start_idx < self.total_days:
                    # Add a small amount of noise to avoid identical returns
                    noise = np.random.normal(0, self.sigma * 0.1, end_idx - start_idx)
                    full_returns[i, start_idx:end_idx] = ret + noise
        
        return full_returns
    
    def simulate(self, leverage=1.0):
        """
        Run Feynman Path Integral simulation with specified leverage.
        
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
                'num_paths': self.num_paths,
                'num_time_steps': self.num_time_steps,
                'num_price_levels': self.num_price_levels,
                'mu': float(self.mu),
                'sigma': float(self.sigma)
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
