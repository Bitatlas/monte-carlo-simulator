import numpy as np
import pandas as pd
from arch import arch_model
from .base_model import BaseModel

class GARCHModel(BaseModel):
    """
    GARCH model for asset returns simulation with time-varying volatility.
    
    This model captures volatility clustering in financial time series by allowing
    the conditional variance to depend on past squared returns and past variances.
    """
    
    def __init__(self, returns_data, investment_amount=10000, time_horizon_years=10, 
                 num_simulations=10000, trading_days_per_year=252, p=1, q=1):
        """
        Initialize the GARCH model.
        
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
        p : int
            GARCH lag order (default: 1)
        q : int
            ARCH lag order (default: 1)
        """
        super().__init__(returns_data, investment_amount, time_horizon_years, 
                        num_simulations, trading_days_per_year)
        
        # GARCH model parameters
        self.p = p
        self.q = q
        self.garch_model = None
        self.garch_result = None
        
        # Fit GARCH model to historical data
        self.fit_garch_model()
    
    def fit_garch_model(self):
        """
        Fit GARCH model to historical returns.
        """
        try:
            # Convert returns to numpy array if it's a pandas Series
            returns_array = self.returns.values if hasattr(self.returns, 'values') else self.returns
            
            # Fit GARCH(p,q) model
            self.garch_model = arch_model(returns_array, vol='GARCH', p=self.p, q=self.q)
            self.garch_result = self.garch_model.fit(disp='off')
            
            # Store fitted parameters
            self.omega = self.garch_result.params['omega']
            self.alpha = self.garch_result.params['alpha[1]'] if self.q >= 1 else 0
            self.beta = self.garch_result.params['beta[1]'] if self.p >= 1 else 0
            
            # Use historical mean instead of GARCH mu parameter to avoid overly optimistic returns
            # The GARCH mu parameter can sometimes be biased upward
            self.mu = float(self.returns.mean()) if hasattr(self.returns, 'mean') else 0
            
            # Get the GARCH mu for comparison
            garch_mu = self.garch_result.params['mu']
            
            # Calculate expected annual return based on both means
            annual_return_garch = garch_mu * 252
            annual_return_hist = self.mu * 252
            
            # Store fitted conditional volatility
            self.conditional_vol = self.garch_result.conditional_volatility
            self.last_vol = self.conditional_vol[-1]
            
            # Log successful fit with both means for comparison
            print(f"GARCH({self.p},{self.q}) model fit successfully")
            print(f"Parameters: ω={self.omega:.6f}, α={self.alpha:.6f}, β={self.beta:.6f}")
            print(f"GARCH μ={garch_mu:.6f} (annual: {annual_return_garch:.2%})")
            print(f"Historical μ={self.mu:.6f} (annual: {annual_return_hist:.2%})")
            print(f"Using historical mean for more realistic projections")
            
        except Exception as e:
            print(f"Error fitting GARCH model: {e}")
            print("Falling back to standard deviation as constant volatility")
            
            # Fall back to using standard deviation as constant volatility
            self.mu = float(self.returns.mean()) if hasattr(self.returns, 'mean') else 0
            self.omega = float(self.returns.var()) if hasattr(self.returns, 'var') else 0.0001
            self.alpha = 0
            self.beta = 0
            self.last_vol = float(self.returns.std()) if hasattr(self.returns, 'std') else 0.01
    
    def generate_returns(self):
        """
        Generate returns using the fitted GARCH model.
        
        Returns:
        --------
        numpy.ndarray
            Array of shape (num_simulations, total_days) with random returns
        """
        # Initialize arrays
        returns = np.zeros((self.num_simulations, self.total_days))
        variances = np.zeros((self.num_simulations, self.total_days))
        
        # Set initial variance for all simulations
        h = self.last_vol**2
        
        # Generate returns for each simulation path
        for i in range(self.num_simulations):
            # Previous return (start with 0)
            prev_return = 0
            
            for t in range(self.total_days):
                # Current variance
                variances[i, t] = h
                
                # Generate random return with current variance
                z = np.random.normal(0, 1)
                returns[i, t] = self.mu + np.sqrt(h) * z
                
                # Update variance for next period using GARCH formula
                h = self.omega + self.alpha * (returns[i, t] - self.mu)**2 + self.beta * h
        
        return returns
    
    def simulate(self, leverage=1.0):
        """
        Run GARCH simulation with specified leverage.
        
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
                'mu': float(self.mu),
                'omega': float(self.omega),
                'alpha': float(self.alpha),
                'beta': float(self.beta),
                'garch_p': self.p,
                'garch_q': self.q
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
