import numpy as np
import pandas as pd
from .base_model import BaseModel
from sklearn.cluster import KMeans

class MarkovChainModel(BaseModel):
    """
    Markov Chain model for asset returns simulation.
    
    This model discretizes returns into states and uses transition probabilities
    between states to generate future return sequences.
    """
    
    def __init__(self, returns_data, investment_amount=10000, time_horizon_years=10, 
                 num_simulations=10000, trading_days_per_year=252, num_states=5):
        """
        Initialize the Markov Chain model.
        
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
        num_states : int
            Number of discrete states to use (default: 5)
        """
        super().__init__(returns_data, investment_amount, time_horizon_years, 
                        num_simulations, trading_days_per_year)
        
        # Markov model parameters
        self.num_states = num_states
        self.state_returns = None  # Mean return for each state
        self.state_stds = None     # Standard deviation for each state
        self.transition_matrix = None  # Probability of transitioning between states
        self.initial_state_probs = None  # Initial state probabilities
        
        # Compute states and transition probabilities
        self.discretize_states()
        self.compute_transition_matrix()
    
    def discretize_states(self):
        """
        Discretize historical returns into states using clustering.
        """
        # Convert returns to numpy array if it's a pandas Series
        returns_array = self.returns.values.reshape(-1, 1) if hasattr(self.returns, 'values') else self.returns.reshape(-1, 1)
        
        # Use K-means clustering to discretize returns into states
        kmeans = KMeans(n_clusters=self.num_states, random_state=42)
        states = kmeans.fit_predict(returns_array)
        
        # Compute mean return and std for each state
        self.state_returns = np.zeros(self.num_states)
        self.state_stds = np.zeros(self.num_states)
        
        for i in range(self.num_states):
            state_returns = returns_array[states == i].flatten()
            self.state_returns[i] = np.mean(state_returns)
            self.state_stds[i] = np.std(state_returns)
        
        # Sort states by mean return
        sort_idx = np.argsort(self.state_returns)
        self.state_returns = self.state_returns[sort_idx]
        self.state_stds = self.state_stds[sort_idx]
        
        # Remap states to sorted order
        new_states = np.zeros_like(states)
        for i, old_idx in enumerate(sort_idx):
            new_states[states == old_idx] = i
        
        # Store state sequence
        self.state_sequence = new_states
        
        # Compute initial state probabilities (frequency of each state)
        self.initial_state_probs = np.bincount(new_states) / len(new_states)
        
        # Ensure all states have a non-zero probability
        self.initial_state_probs = np.maximum(self.initial_state_probs, 0.01)
        self.initial_state_probs = self.initial_state_probs / np.sum(self.initial_state_probs)
    
    def compute_transition_matrix(self):
        """
        Compute transition probabilities between states.
        """
        # Initialize transition counts matrix
        transition_counts = np.zeros((self.num_states, self.num_states))
        
        # Count transitions
        for i in range(len(self.state_sequence) - 1):
            from_state = self.state_sequence[i]
            to_state = self.state_sequence[i + 1]
            transition_counts[from_state, to_state] += 1
        
        # Convert counts to probabilities
        self.transition_matrix = np.zeros((self.num_states, self.num_states))
        
        for i in range(self.num_states):
            row_sum = np.sum(transition_counts[i, :])
            if row_sum > 0:
                self.transition_matrix[i, :] = transition_counts[i, :] / row_sum
            else:
                # If a state never occurred or has no transitions, assign uniform probabilities
                self.transition_matrix[i, :] = 1.0 / self.num_states
    
    def generate_returns(self):
        """
        Generate returns using the Markov Chain model.
        
        Returns:
        --------
        numpy.ndarray
            Array of shape (num_simulations, total_days) with random returns
        """
        # Initialize returns array
        returns = np.zeros((self.num_simulations, self.total_days))
        
        # Generate state sequences for each simulation
        for i in range(self.num_simulations):
            # Initialize with a random state based on initial probabilities
            current_state = np.random.choice(self.num_states, p=self.initial_state_probs)
            
            for t in range(self.total_days):
                # Generate return from current state distribution
                state_mean = self.state_returns[current_state]
                state_std = self.state_stds[current_state]
                returns[i, t] = np.random.normal(state_mean, state_std)
                
                # Transition to next state based on transition probabilities
                current_state = np.random.choice(self.num_states, p=self.transition_matrix[current_state, :])
        
        return returns
    
    def simulate(self, leverage=1.0):
        """
        Run Markov Chain simulation with specified leverage.
        
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
                'num_states': self.num_states,
                'state_returns': self.state_returns.tolist(),
                'state_stds': self.state_stds.tolist(),
                'transition_matrix': self.transition_matrix.tolist()
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
