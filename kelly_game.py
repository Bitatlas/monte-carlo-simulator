import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional, Union, Any

# Import from existing modules
from data.fetchers import EquityIndexFetcher, StockFetcher, CryptoFetcher, BondFetcher
from optimization import KellyCalculator

class BaseKellyGameSimulator:
    """Base class for Kelly betting game simulators."""
    
    def __init__(self, returns_data: pd.Series, asset_name: str, initial_investment: float = 10000):
        """
        Initialize the simulator.
        
        Parameters:
        -----------
        returns_data : pd.Series
            Historical returns data for the asset
        asset_name : str
            Name of the asset being simulated
        initial_investment : float
            Initial investment amount
        """
        self.returns_data = returns_data
        self.asset_name = asset_name
        self.initial_investment = float(initial_investment)  # Ensure it's a float
        self.current_investment = float(initial_investment)  # Ensure it's a float
        self.benchmark_value = float(initial_investment)     # Ensure it's a float
        self.weeks_elapsed = 0
        self.portfolio_history = [float(initial_investment)]  # Ensure list of floats
        self.benchmark_history = [float(initial_investment)]  # Ensure list of floats
        self.return_history = []
        self.max_portfolio_value = float(initial_investment)  # Ensure it's a float
        self.optimal_kelly = self._calculate_optimal_kelly()
        self.dates = [datetime.now()]
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.game_over = False
        
    def _calculate_optimal_kelly(self) -> float:
        """
        Calculate the optimal Kelly fraction for the asset.
        
        Returns:
        --------
        float
            Optimal Kelly fraction
        """
        # Calculate using risk-free rate of 0 for simplicity
        kelly_calc = KellyCalculator(self.returns_data, risk_free_rate=0.0)
        return kelly_calc.calculate_full_kelly()
    
    def advance_week(self, kelly_fraction: float) -> Dict[str, Any]:
        """
        Advance the simulation by one week.
        
        Parameters:
        -----------
        kelly_fraction : float
            Kelly fraction to apply to the investment
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary of updated statistics
        """
        # Generate a return (to be implemented by subclasses)
        weekly_return = self._generate_return()
        
        # Apply Kelly fraction (leverage)
        leveraged_return = weekly_return * kelly_fraction
        
        # Update investment value
        self.current_investment *= (1 + leveraged_return)
        
        # Update benchmark value (asset without leverage)
        self.benchmark_value *= (1 + weekly_return)
        
        # Update history
        self.portfolio_history.append(self.current_investment)
        self.benchmark_history.append(self.benchmark_value)
        self.return_history.append(weekly_return)
        
        # Update weeks elapsed
        self.weeks_elapsed += 1
        
        # Update dates (weekly increments)
        self.dates.append(self.dates[-1] + timedelta(days=7))
        
        # Update max portfolio value
        self.max_portfolio_value = max(self.max_portfolio_value, self.current_investment)
        
        # Calculate current drawdown with explicit float conversions
        self.current_drawdown = 1.0 - float(self.current_investment) / float(self.max_portfolio_value)
        self.max_drawdown = max(float(self.max_drawdown), float(self.current_drawdown))
        
        # Check for game over (99% loss)
        if self.current_investment < 0.01 * self.initial_investment:
            self.game_over = True
        
        # Return updated statistics
        return self.get_statistics()
    
    def _generate_return(self) -> float:
        """
        Generate a return for the next time step.
        
        To be implemented by subclasses.
        
        Returns:
        --------
        float
            Generated return
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current game statistics.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary of statistics
        """
        # Calculate total return
        total_return = (self.current_investment / self.initial_investment) - 1
        
        # Calculate annualized return (if weeks_elapsed > 0)
        if self.weeks_elapsed > 0:
            annualized_return = ((1 + total_return) ** (52 / self.weeks_elapsed)) - 1
        else:
            annualized_return = 0
        
        # Calculate Sharpe ratio (if return_history not empty)
        if len(self.return_history) > 0:
            returns_array = np.array(self.return_history)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            if std_return > 0:
                sharpe_ratio = mean_return / std_return * np.sqrt(52)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate Kelly ratio (ratio of user's Kelly to optimal Kelly)
        kelly_ratio = 0  # Will be set by the calling code
        
        # Improved risk of ruin calculation that responds to portfolio value changes
        # The risk decreases as portfolio value increases relative to initial investment
        # and increases with drawdown and volatility
        
        # Portfolio ratio factor: how much has the portfolio grown from initial investment
        portfolio_ratio = max(0.01, self.current_investment / self.initial_investment)
        
        # Volatility factor (using standard deviation of returns)
        volatility_factor = 1.0
        if len(self.return_history) > 5:  # Only calculate if we have enough history
            volatility_factor = max(0.5, np.std(self.return_history) * np.sqrt(52) * 2)
        
        # Risk of ruin that decreases as portfolio grows and increases with drawdown
        risk_of_ruin = (1 - np.exp(-2 * self.current_drawdown)) / np.sqrt(portfolio_ratio)
        
        # Cap the risk of ruin at 99%
        risk_of_ruin = min(0.99, risk_of_ruin)
        
        return {
            'current_value': self.current_investment,
            'benchmark_value': self.benchmark_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'weeks_elapsed': self.weeks_elapsed,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'optimal_kelly': self.optimal_kelly,
            'kelly_ratio': kelly_ratio,
            'risk_of_ruin': risk_of_ruin,
            'portfolio_history': self.portfolio_history,
            'benchmark_history': self.benchmark_history,
            'dates': self.dates,
            'game_over': self.game_over
        }
    
    def reset(self, initial_investment: float = None) -> None:
        """
        Reset the simulation.
        
        Parameters:
        -----------
        initial_investment : float, optional
            New initial investment amount. If None, uses the original amount.
        """
        if initial_investment is not None:
            self.initial_investment = initial_investment
        
        self.current_investment = self.initial_investment
        self.benchmark_value = self.initial_investment
        self.weeks_elapsed = 0
        self.portfolio_history = [self.initial_investment]
        self.benchmark_history = [self.initial_investment]
        self.return_history = []
        self.max_portfolio_value = self.initial_investment
        self.dates = [datetime.now()]
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.game_over = False


class BootstrapSimulator(BaseKellyGameSimulator):
    """Simulator that uses bootstrap sampling from historical returns."""
    
    def _generate_return(self) -> float:
        """
        Generate a return by randomly sampling from historical returns.
        
        Returns:
        --------
        float
            Sampled return
        """
        # Randomly sample a return from historical data
        # Convert the numpy value to a Python float to avoid formatting issues
        return float(random.choice(self.returns_data.dropna().values))


class MarkovChainSimulator(BaseKellyGameSimulator):
    """Simulator that uses a Markov Chain model."""
    
    def __init__(self, returns_data: pd.Series, asset_name: str, 
                 initial_investment: float = 10000, num_states: int = 5):
        """
        Initialize the Markov Chain simulator.
        
        Parameters:
        -----------
        returns_data : pd.Series
            Historical returns data for the asset
        asset_name : str
            Name of the asset being simulated
        initial_investment : float
            Initial investment amount
        num_states : int
            Number of states in the Markov Chain
        """
        super().__init__(returns_data, asset_name, initial_investment)
        
        self.num_states = num_states
        self.state_bounds = None
        self.transition_matrix = None
        self.state_returns = None
        self.current_state = None
        
        # Initialize the Markov Chain
        self._initialize_markov_chain()
        
    def _initialize_markov_chain(self) -> None:
        """Initialize the Markov Chain model."""
        returns = self.returns_data.dropna().values
        
        # Determine state boundaries based on return quantiles
        quantiles = np.linspace(0, 1, self.num_states + 1)
        self.state_bounds = np.quantile(returns, quantiles)
        
        # Assign states to returns
        states = np.zeros(len(returns), dtype=int)
        for i in range(len(returns)):
            for j in range(self.num_states):
                if returns[i] <= self.state_bounds[j + 1]:
                    states[j] = j
                    break
            else:
                states[i] = self.num_states - 1
        
        # Calculate transition matrix
        self.transition_matrix = np.zeros((self.num_states, self.num_states))
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            self.transition_matrix[current_state, next_state] += 1
        
        # Normalize transition matrix
        row_sums = self.transition_matrix.sum(axis=1)
        for i in range(self.num_states):
            if row_sums[i] > 0:
                self.transition_matrix[i, :] /= row_sums[i]
            else:
                # If a state was never observed, use uniform distribution
                self.transition_matrix[i, :] = 1.0 / self.num_states
        
        # Calculate average return for each state
        self.state_returns = np.zeros(self.num_states)
        for i in range(self.num_states):
            mask = (states == i)
            if np.any(mask):
                self.state_returns[i] = np.mean(returns[mask])
        
        # Set initial state (middle state)
        self.current_state = self.num_states // 2
    
    def _generate_return(self) -> float:
        """
        Generate a return based on the current state and transition matrix.
        
        Returns:
        --------
        float
            Generated return
        """
        # Sample next state based on transition probabilities
        next_state = np.random.choice(self.num_states, p=self.transition_matrix[self.current_state, :])
        
        # Update current state
        self.current_state = next_state
        
        # Generate return based on the new state (with some noise)
        base_return = self.state_returns[next_state]
        noise = np.random.normal(0, 0.01)  # Small noise term
        
        # Explicitly convert to float to avoid numpy formatting issues
        return float(base_return + noise)
    
    def get_state_description(self) -> str:
        """
        Get a description of the current market state.
        
        Returns:
        --------
        str
            Description of the current market state
        """
        if self.current_state == 0:
            return "Strong Bear Market"
        elif self.current_state == self.num_states - 1:
            return "Strong Bull Market"
        elif self.current_state < self.num_states // 2:
            return "Mild Bear Market"
        elif self.current_state > self.num_states // 2:
            return "Mild Bull Market"
        else:
            return "Sideways Market"


def fetch_asset_data(asset_type: str, asset: str, period: str = "10y"):
    """
    Fetch historical data for the selected asset.
    
    Parameters:
    -----------
    asset_type : str
        Type of asset (Equity Index, Individual Stock, Cryptocurrency, Bond)
    asset : str
        Specific asset identifier
    period : str
        Time period to fetch data for
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing asset data and statistics
    """
    try:
        if asset_type == "📊 Equity Index" or asset_type == "Equity Index":
            fetcher = EquityIndexFetcher(index_type=asset, period=period)
        elif asset_type == "🏢 Individual Stock" or asset_type == "Individual Stock":
            fetcher = StockFetcher(ticker=asset, period=period)
        elif asset_type == "₿ Cryptocurrency" or asset_type == "Cryptocurrency":
            fetcher = CryptoFetcher(crypto_type=asset, period=period)
        else:  # Bond (🔒 Bond or Bond)
            fetcher = BondFetcher(bond_type=asset, period=period)
        
        # Fetch data and calculate returns
        data = fetcher.fetch_data()
        returns_data = fetcher.calculate_returns()
        stats = fetcher.get_statistics()
        
        # Get name for display
        asset_name = fetcher.name
        
        return {
            'data': data,
            'returns': returns_data,
            'stats': stats,
            'name': asset_name,
            'fetcher': fetcher
        }
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def kelly_game_tab():
    """Render the Kelly betting game tab in the Streamlit app."""
    st.markdown('<div class="main-header">Kelly Betting Game</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This interactive game demonstrates the Kelly criterion in action. Select an asset, 
    set your initial investment and leverage amount, then advance week-by-week to see how 
    your portfolio performs.
    
    The goal is to maximize your returns without going bankrupt. Try different leverage 
    amounts to see which works best for each asset!
    """)
    
    # Initialize session state for the game
    if 'kg_simulator' not in st.session_state:
        st.session_state.kg_simulator = None
    if 'kg_asset_data' not in st.session_state:
        st.session_state.kg_asset_data = None
    if 'kg_simulator_type' not in st.session_state:
        st.session_state.kg_simulator_type = "Bootstrap"
    if 'kg_kelly_fraction' not in st.session_state:
        st.session_state.kg_kelly_fraction = 1.0
    if 'kg_initial_investment' not in st.session_state:
        st.session_state.kg_initial_investment = 10000
    
    # Determine where to display controls - support both sidebar and column layouts
    # First check if we're in a column layout (set by app.py)
    use_column_layout = False
    if hasattr(st.session_state, 'kg_controls_location') and st.session_state.kg_controls_location == "column":
        use_column_layout = True
        controls_container = st
    else:
        # Default to sidebar controls for backward compatibility and deployments
        controls_container = st.sidebar
    
    # Create controls container - handle sidebar differently for Python 3.13 compatibility
    # Don't use 'with' statement for sidebar as it doesn't support context manager protocol in Python 3.13
    if controls_container == st.sidebar:
        # Direct access for sidebar
        controls_container.markdown('<div class="sub-header">Game Controls</div>', unsafe_allow_html=True)
    else:
        # Use context manager for normal streamlit or columns
        with controls_container:
            st.markdown('<div class="sub-header">Game Controls</div>', unsafe_allow_html=True)
        
        # Asset selection
        asset_type = st.selectbox(
            "Asset Type",
            options=["Equity Index", "Individual Stock", "Cryptocurrency", "Bond"],
            key="kg_asset_type",
            help="Select the type of asset to simulate"
        )
        
        # Specific asset selection based on type
        if asset_type == "Equity Index":
            asset = st.selectbox(
                "Equity Index",
                options=["SP500", "NASDAQ"],
                index=0,
                format_func=lambda x: {"SP500": "S&P 500", "NASDAQ": "Nasdaq 100"}.get(x, x),
                key="kg_asset",
                help="Select the equity index to simulate"
            )
        elif asset_type == "Individual Stock":
            asset = st.selectbox(
                "Stock Ticker",
                options=["AAPL", "PLTR", "BRK.A"],
                index=0,
                format_func=lambda x: {
                    "AAPL": "Apple (AAPL)",
                    "PLTR": "Palantir (PLTR)",
                    "BRK.A": "Berkshire Hathaway (BRK.A)"
                }.get(x, x),
                key="kg_stock",
                help="Select the stock to simulate"
            )
        elif asset_type == "Cryptocurrency":
            asset = st.selectbox(
                "Cryptocurrency",
                options=["BTC", "ETH"],
                index=0,
                format_func=lambda x: {"BTC": "Bitcoin (BTC)", "ETH": "Ethereum (ETH)"}.get(x, x),
                key="kg_crypto",
                help="Select the cryptocurrency to simulate"
            )
        else:  # Bond
            asset = st.selectbox(
                "Bond",
                options=["TLT", "IEF"],
                index=0,
                format_func=lambda x: {
                    "TLT": "iShares 20+ Year Treasury Bond ETF (TLT)",
                    "IEF": "iShares 7-10 Year Treasury Bond ETF (IEF)"
                }.get(x, x),
                key="kg_bond",
                help="Select the bond to simulate"
            )
        
        # Simulation method
        simulator_type = st.radio(
            "Simulation Method",
            options=["Bootstrap", "Markov Chain"],
            index=0,
            key="kg_simulator_type",
            help="Choose the simulation method"
        )
        
        # Initialize button
        if st.button("Initialize Game", key="kg_initialize"):
            with st.spinner("Fetching asset data..."):
                # Fetch asset data
                asset_data = fetch_asset_data(asset_type, asset, period="10y")
                
                if asset_data:
                    st.session_state.kg_asset_data = asset_data
                    
                    # Create simulator
                    if simulator_type == "Bootstrap":
                        st.session_state.kg_simulator = BootstrapSimulator(
                            returns_data=asset_data['returns']['daily'],
                            asset_name=asset_data['name'],
                            initial_investment=st.session_state.kg_initial_investment
                        )
                    else:  # Markov Chain
                        st.session_state.kg_simulator = MarkovChainSimulator(
                            returns_data=asset_data['returns']['daily'],
                            asset_name=asset_data['name'],
                            initial_investment=st.session_state.kg_initial_investment,
                            num_states=5
                        )
                    
                    st.success(f"Game initialized for {asset_data['name']}!")
                else:
                    st.error("Failed to fetch asset data. Please try again.")
        
        # Game parameters (only show when game is initialized)
        if st.session_state.kg_simulator:
            st.markdown('<div class="sub-header">Game Parameters</div>', unsafe_allow_html=True)
            
            # Initial investment
            initial_investment = st.number_input(
                "Initial Investment ($)",
                min_value=1000,
                max_value=1000000,
                value=st.session_state.kg_initial_investment,
                step=1000,
                key="kg_initial_investment_input",
                help="Your initial investment amount"
            )
            
            # Update initial investment if changed
            if initial_investment != st.session_state.kg_initial_investment:
                st.session_state.kg_initial_investment = initial_investment
                st.session_state.kg_simulator.reset(initial_investment=initial_investment)
            
            # Leverage slider (renamed from Kelly Fraction)
            optimal_kelly = st.session_state.kg_simulator.optimal_kelly
            kelly_fraction = st.slider(
                "Leverage",
                min_value=0.0,
                max_value=5.0,
                value=st.session_state.kg_kelly_fraction,
                step=0.1,
                key="kg_kelly_fraction_slider",
                help="Amount of leverage to apply (Kelly fraction)"
            )
            st.session_state.kg_kelly_fraction = kelly_fraction
            
            # Show Kelly ratio with humorous comments
            kelly_ratio = kelly_fraction / optimal_kelly if optimal_kelly > 0 else float('inf')
            
            if kelly_ratio < 0.25:
                st.info(f"🐢 You're being too cautious! ({kelly_ratio:.2f}x optimal Kelly) Even my grandma takes more risk than this.")
            elif kelly_ratio < 0.5:
                st.info(f"🐌 Playing it safe, huh? ({kelly_ratio:.2f}x optimal Kelly) At least your money's growing... very... slowly...")
            elif kelly_ratio < 0.75:
                st.success(f"🦔 Respectable conservatism. ({kelly_ratio:.2f}x optimal Kelly) Smart money territory.")
            elif kelly_ratio < 0.9:
                st.success(f"🦊 Almost perfect! ({kelly_ratio:.2f}x optimal Kelly) You've got risk management skills.")
            elif kelly_ratio < 1.1:
                st.success(f"🦁 Perfect balance! ({kelly_ratio:.2f}x optimal Kelly) You're a Kelly Criterion master!")
            elif kelly_ratio < 1.5:
                st.warning(f"🦅 Getting aggressive... ({kelly_ratio:.2f}x optimal Kelly) Hope you can handle the volatility!")
            elif kelly_ratio < 2.0:
                st.warning(f"🐆 Bold strategy, Cotton! ({kelly_ratio:.2f}x optimal Kelly) Let's see if it pays off.")
            elif kelly_ratio < 3.0:
                st.error(f"🦍 Living dangerously! ({kelly_ratio:.2f}x optimal Kelly) Your risk of ruin is climbing fast.")
            else:
                st.error(f"🦖 YOLO mode activated! ({kelly_ratio:.2f}x optimal Kelly) Hope you enjoy rollercoasters and sleepless nights.")
            
            # Reset button
            if st.button("Reset Game", key="kg_reset"):
                st.session_state.kg_simulator.reset()
                st.success("Game reset!")
    
    # Main content area
    if st.session_state.kg_simulator:
        simulator = st.session_state.kg_simulator
        
        # Game stats area
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Portfolio Value",
                f"${float(simulator.current_investment):,.2f}",
                f"{(float(simulator.current_investment) / float(simulator.initial_investment) - 1) * 100:.2f}%"
            )
        
        with col2:
            st.metric(
                "Benchmark Value",
                f"${float(simulator.benchmark_value):,.2f}",
                f"{(float(simulator.benchmark_value) / float(simulator.initial_investment) - 1) * 100:.2f}%"
            )
        
        with col3:
            st.metric(
                "Weeks Elapsed",
                f"{simulator.weeks_elapsed}"
            )
        
        # Advance week button
        if st.button("Advance Week", key="kg_advance"):
            # Update statistics with Kelly fraction
            stats = simulator.advance_week(kelly_fraction=st.session_state.kg_kelly_fraction)
            
            # Check for game over
            if stats['game_over']:
                st.error("GAME OVER! Your portfolio has lost 99% of its value.")
        
        # Display current state for Markov Chain
        if st.session_state.kg_simulator_type == "Markov Chain" and isinstance(simulator, MarkovChainSimulator):
            st.info(f"Current Market State: {simulator.get_state_description()}")
        
        # Plot portfolio value over time
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Format the dates
        dates = simulator.dates
        
        # Convert lists to numpy arrays to ensure they're properly formatted
        portfolio_history = np.array([float(x) for x in simulator.portfolio_history])
        benchmark_history = np.array([float(x) for x in simulator.benchmark_history])
        
        # Plot portfolio value
        ax.plot(dates, portfolio_history, 'b-', linewidth=2, label=f'Portfolio (Leverage: {kelly_fraction:.1f}x)')
        
        # Plot benchmark
        ax.plot(dates, benchmark_history, 'g--', linewidth=1.5, label='Benchmark (No Leverage)')
        
        # Add fill for drawdown periods
        if len(simulator.portfolio_history) > 1:
            # Convert to proper numpy array ensuring all elements are float
            portfolio_history_np = np.array([float(x) for x in simulator.portfolio_history])
            
            # Calculate peaks and drawdowns
            peaks = np.maximum.accumulate(portfolio_history_np)
            drawdowns = 1 - portfolio_history_np / peaks
            
            # Add shading for significant drawdowns
            for i in range(1, len(drawdowns)):
                if drawdowns[i] > 0.1:  # Only shade significant drawdowns (>10%)
                    ax.axvspan(dates[i-1], dates[i], alpha=0.2, color='red')
        
        # Customize plot
        ax.set_title(f"Kelly Betting Game: {simulator.asset_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Display the chart
        st.pyplot(fig)
        
        # Display additional statistics
        st.markdown('<div class="sub-header">Performance Statistics</div>', unsafe_allow_html=True)
        
        # Create stat columns
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            # Calculate using explicit float conversions to avoid numpy formatting issues
            total_return = (float(simulator.current_investment) / float(simulator.initial_investment) - 1) * 100
            benchmark_return = (float(simulator.benchmark_value) / float(simulator.initial_investment) - 1) * 100
            
            st.metric(
                "Total Return",
                f"{total_return:.2f}%"
            )
            st.metric(
                "Benchmark Return",
                f"{benchmark_return:.2f}%"
            )
        
        with stat_col2:
            st.metric(
                "Current Drawdown",
                f"{simulator.current_drawdown * 100:.2f}%",
                delta_color="inverse"
            )
            st.metric(
                "Maximum Drawdown",
                f"{simulator.max_drawdown * 100:.2f}%",
                delta_color="inverse"
            )
        
        with stat_col3:
            # Get stats
            stats = simulator.get_statistics()
            
            st.metric(
                "Annualized Return",
                f"{stats['annualized_return'] * 100:.2f}%"
            )
            st.metric(
                "Sharpe Ratio",
                f"{stats['sharpe_ratio']:.2f}"
            )
        
        with stat_col4:
            st.metric(
                "Optimal Kelly",
                f"{optimal_kelly:.2f}x"
            )
            st.metric(
                "Risk of Ruin",
                f"{stats['risk_of_ruin'] * 100:.2f}%",
                delta_color="inverse"
            )
        
        # Educational tips
        with st.expander("Kelly Betting Tips", expanded=False):
            st.markdown("""
            ### Kelly Betting Strategy Tips
            
            - **Full Kelly (1.0x optimal)** maximizes long-term growth but has high volatility
            - **Half Kelly (0.5x optimal)** gives ~75% of the optimal growth rate with much less volatility
            - **Quarter Kelly (0.25x optimal)** gives ~90% of Half Kelly's growth rate with even less risk
            
            Most professional investors use Fractional Kelly (25-50% of full Kelly) due to:
            1. Uncertainty in parameter estimation
            2. Risk aversion and client concerns about drawdowns
            3. The asymmetric risk of overestimating vs. underestimating optimal Kelly
            
            Remember: Going over optimal Kelly is much more dangerous than going under!
            """)
    else:
        # Show instructions when game is not initialized
        st.info("Select an asset and click 'Initialize Game' to start playing!")
        
        with st.expander("How to Play", expanded=True):
            st.markdown("""
            ### How to Play the Kelly Betting Game
            
            1. **Select an asset** from the sidebar dropdown
            2. **Initialize the game** by clicking the 'Initialize Game' button
            3. **Set your leverage** using the slider (0 = no leverage, 1 = full Kelly, etc.)
            4. **Click 'Advance Week'** to simulate one week of market movement
            5. **Watch your portfolio** grow or shrink based on the market and your leverage
            6. **Adjust your leverage** as needed based on performance
            
            The game continues until you lose 99% of your initial investment or choose to reset.
            
            ### Simulation Methods
            
            - **Bootstrap:** Randomly samples from actual historical returns
            - **Markov Chain:** Uses a state-based model that captures market regimes
            
            Try both methods to see how they affect your optimal betting strategy!
            """)
