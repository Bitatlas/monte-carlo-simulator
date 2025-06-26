import numpy as np
import pandas as pd
from .kelly_calculator import KellyCalculator

class LeverageOptimizer:
    """
    Optimizer for determining optimal leverage across different assets and models.
    
    This class provides tools to analyze and recommend leverage levels based on
    simulation results and Kelly criterion calculations.
    """
    
    def __init__(self, simulation_results=None, historical_returns=None, risk_free_rate=0.0):
        """
        Initialize the leverage optimizer.
        
        Parameters:
        -----------
        simulation_results : dict, optional
            Dictionary of simulation results from different models
        historical_returns : pandas.Series, optional
            Historical returns data for Kelly criterion calculations
        risk_free_rate : float
            Daily risk-free rate (default: 0.0)
        """
        self.simulation_results = simulation_results or {}
        self.historical_returns = historical_returns
        self.risk_free_rate = risk_free_rate
        
        # Create Kelly calculator if historical returns are provided
        self.kelly_calculator = None
        if historical_returns is not None:
            self.kelly_calculator = KellyCalculator(historical_returns, risk_free_rate)
    
    def set_historical_returns(self, returns, risk_free_rate=None):
        """
        Set historical returns for Kelly criterion calculations.
        
        Parameters:
        -----------
        returns : pandas.Series
            Historical returns data
        risk_free_rate : float, optional
            Daily risk-free rate (if None, use the current value)
        """
        self.historical_returns = returns
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate
        
        # Create or update Kelly calculator
        self.kelly_calculator = KellyCalculator(returns, self.risk_free_rate)
    
    def add_simulation_results(self, model_name, results):
        """
        Add simulation results for a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        results : dict
            Simulation results from the model
        """
        self.simulation_results[model_name] = results
    
    def calculate_optimal_leverage(self, method="kelly", fractional=0.5, max_leverage=10.0):
        """
        Calculate optimal leverage using specified method.
        
        Parameters:
        -----------
        method : str
            Method to use for calculating optimal leverage:
            - "kelly": Use Kelly criterion
            - "numerical": Use numerical optimization
            - "simulation": Use simulation results
        fractional : float
            Fraction of full Kelly to use (default: 0.5)
        max_leverage : float
            Maximum leverage to consider
            
        Returns:
        --------
        float
            Optimal leverage
        """
        if self.kelly_calculator is None and method in ["kelly", "numerical"]:
            raise ValueError("Historical returns must be provided for Kelly calculations")
        
        if method == "kelly":
            return self.kelly_calculator.calculate_fractional_kelly(fractional)
        elif method == "numerical":
            return self.kelly_calculator.find_optimal_leverage_numerical(max_leverage)
        elif method == "simulation":
            return self._find_optimal_leverage_from_simulations()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _find_optimal_leverage_from_simulations(self):
        """
        Find optimal leverage from simulation results.
        
        Returns:
        --------
        float
            Optimal leverage based on median growth rate
        """
        if not self.simulation_results:
            raise ValueError("No simulation results provided")
        
        # Use the first model's results
        model_name = next(iter(self.simulation_results))
        results = self.simulation_results[model_name]
        
        # Extract leverage and growth rate information
        leverages = []
        growth_rates = []
        
        for leverage, result in results.items():
            if isinstance(leverage, (int, float)):  # Ensure it's a numeric leverage value
                leverages.append(leverage)
                # Use CAGR as growth rate
                growth_rates.append(result['stats']['cagr']['median'])
        
        if not leverages:
            return 1.0  # Default to no leverage if no results
        
        # Find leverage with highest median growth rate
        best_idx = np.argmax(growth_rates)
        return leverages[best_idx]
    
    def get_leverage_recommendations(self, risk_profile="moderate"):
        """
        Get leverage recommendations based on risk profile.
        
        Parameters:
        -----------
        risk_profile : str
            Risk profile to use for recommendations:
            - "conservative": Lower leverage (25% of Kelly)
            - "moderate": Moderate leverage (50% of Kelly)
            - "aggressive": Higher leverage (75% of Kelly)
            - "full_kelly": Full Kelly leverage
            
        Returns:
        --------
        dict
            Dictionary with leverage recommendations
        """
        if self.kelly_calculator is None:
            raise ValueError("Historical returns must be provided for Kelly calculations")
        
        # Calculate Kelly leverage
        full_kelly = self.kelly_calculator.calculate_full_kelly()
        numerical_kelly = self.kelly_calculator.find_optimal_leverage_numerical()
        
        # Apply risk profile
        fractions = {
            "conservative": 0.25,
            "moderate": 0.5,
            "aggressive": 0.75,
            "full_kelly": 1.0
        }
        
        fraction = fractions.get(risk_profile, 0.5)
        recommended_leverage = full_kelly * fraction
        
        # If we have simulation results, also provide model-based recommendation
        simulation_leverage = None
        if self.simulation_results:
            try:
                simulation_leverage = self._find_optimal_leverage_from_simulations()
            except:
                pass
        
        return {
            "risk_profile": risk_profile,
            "full_kelly": full_kelly,
            "numerical_kelly": numerical_kelly,
            "recommended_leverage": recommended_leverage,
            "simulation_leverage": simulation_leverage,
            "fraction_of_kelly": fraction
        }
    
    def analyze_leverage_impact(self, leverage_range=None, points=10):
        """
        Analyze the impact of different leverage values.
        
        Parameters:
        -----------
        leverage_range : tuple, optional
            (min_leverage, max_leverage) range to analyze
        points : int
            Number of leverage points to analyze
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with analysis results
        """
        if self.kelly_calculator is None:
            raise ValueError("Historical returns must be provided for Kelly calculations")
        
        # Default leverage range from 0 to 2x Kelly
        if leverage_range is None:
            full_kelly = self.kelly_calculator.calculate_full_kelly()
            leverage_range = (0.0, max(2.0, 2 * full_kelly))
        
        # Generate leverage values
        leverage_values = np.linspace(leverage_range[0], leverage_range[1], points)
        
        # Analyze historical returns with different leverage values
        historical_analysis = self.kelly_calculator.analyze_leveraged_returns(leverage_values)
        
        # If we have simulation results, add simulation-based metrics
        if self.simulation_results:
            # Use the first model's results
            model_name = next(iter(self.simulation_results))
            results = self.simulation_results[model_name]
            
            # Find the closest leverage values in the simulation results
            for leverage in leverage_values:
                if leverage in results:
                    # Exact match
                    sim_result = results[leverage]
                else:
                    # Find closest leverage
                    sim_leverages = [lev for lev in results.keys() if isinstance(lev, (int, float))]
                    closest_lev = min(sim_leverages, key=lambda x: abs(x - leverage))
                    sim_result = results[closest_lev]
                
                # Add simulation-based metrics to the corresponding row
                idx = historical_analysis.index[historical_analysis['leverage'] == leverage].tolist()
                if idx:
                    historical_analysis.loc[idx[0], 'sim_median_return'] = sim_result['stats']['cagr']['median']
                    historical_analysis.loc[idx[0], 'sim_95th_percentile'] = sim_result['stats']['cagr']['percentiles']['95%']
                    historical_analysis.loc[idx[0], 'sim_5th_percentile'] = sim_result['stats']['cagr']['percentiles']['5%']
                    historical_analysis.loc[idx[0], 'sim_max_drawdown'] = sim_result['stats']['max_drawdown']['median']
                    historical_analysis.loc[idx[0], 'sim_ruin_probability'] = sim_result['stats']['ruin_probability']
        
        return historical_analysis
