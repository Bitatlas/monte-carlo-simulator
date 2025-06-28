import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Import our game module
from kelly_game import BootstrapSimulator, MarkovChainSimulator, fetch_asset_data

# Set page configuration
st.set_page_config(
    page_title="Kelly Betting Game",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Clean Theme Custom CSS
st.markdown("""
<style>
    /* Clean, modern styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        color: #1E88E5;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
        color: #0277BD;
    }
    
    /* Text styles */
    .info-text {
        font-size: 1.05rem;
    }
    
    .highlight {
        background-color: #f8f9fa;
        border-left: 3px solid #1E88E5;
        padding: 1rem;
        border-radius: 0.3rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Metrics */
    .css-1wivap2 {
        background-color: #ffffff;
        border-radius: 6px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    
    .css-1wivap2:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Improve button styling */
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 4px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #0277BD;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Kelly slider styling */
    div[data-testid="stSlider"] {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Game parameter box */
    .game-params {
        padding: 1rem;
        background-color: #f1f8fe;
        border-radius: 8px;
        border: 1px solid #90caf9;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
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

# Split the page into two columns for better layout
left_col, right_col = st.columns([1, 3])

# Game controls in the left column
with left_col:
    st.markdown('<div class="sub-header">Game Setup</div>', unsafe_allow_html=True)
    
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
    st.session_state.kg_initial_investment = initial_investment
    
    # Initialize button (more prominent)
    if st.button("Initialize Game", key="kg_initialize", use_container_width=True):
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
    
    # Education section at the bottom of the left column
    with st.expander("How to Play", expanded=False):
        st.markdown("""
        ### How to Play the Kelly Betting Game
        
        1. **Select an asset** from the dropdown menus
        2. **Set your initial investment** amount
        3. **Initialize the game** by clicking the button
        4. **Adjust your leverage** using the slider
        5. **Click 'Advance Week'** to simulate one week of market movement
        6. **Watch your portfolio** grow or shrink based on your leverage
        7. **Repeat steps 4-6** as you navigate the simulation
        
        The game continues until you lose 99% of your initial investment or choose to reset.
        
        ### Simulation Methods
        
        - **Bootstrap:** Randomly samples from actual historical returns, preserving their exact statistical properties
        - **Markov Chain:** Uses a state-based model that captures market regimes and trend persistence
        
        Try both methods to see how they affect your optimal betting strategy!
        """)
    
    with st.expander("Understanding Kelly Criterion", expanded=False):
        st.markdown("""
        ### What is the Kelly Criterion?
        
        The Kelly Criterion is a formula that determines the optimal size of a series of bets to maximize long-term growth.
        
        **Mathematical formula:**
        ```
        f* = (p × b - q) ÷ b
        ```
        
        Where:
        - f* is the optimal fraction of your capital to bet
        - p is the probability of winning
        - q is the probability of losing (1-p)
        - b is the odds received on the bet
        
        In finance, this is often simplified to:
        ```
        f* = (μ - r) ÷ σ²
        ```
        
        Where:
        - μ is the expected return
        - r is the risk-free rate
        - σ² is the variance of returns
        
        ### Kelly Strategies
        
        - **Full Kelly (1.0x)**: Maximizes long-term growth but can be very volatile
        - **Half Kelly (0.5x)**: Gives ~75% of optimal growth with much less risk
        - **Quarter Kelly (0.25x)**: Even more conservative approach
        
        Most professionals use fractional Kelly (25-50%) due to uncertainty in the input parameters.
        """)

# Main game area in the right column
with right_col:
    # Game controls area - only shown when game is initialized
    if st.session_state.kg_simulator:
        simulator = st.session_state.kg_simulator
        
        # Game control box
        st.markdown('<div class="sub-header">Game Controls</div>', unsafe_allow_html=True)
        
        # Create two columns for the Kelly slider and advance button
        slider_col, button_col = st.columns([3, 1])
        
        with slider_col:
            # Prominently display the Kelly fraction slider
            optimal_kelly = simulator.optimal_kelly
            
            # Make a more prominent container for the leverage slider
            st.markdown('<div class="game-params">', unsafe_allow_html=True)
            st.markdown(f"### Leverage Selector (Optimal: {optimal_kelly:.2f}x)")
            kelly_fraction = st.slider(
                "Select your leverage amount",
                min_value=0.0,
                max_value=5.0,
                value=st.session_state.kg_kelly_fraction,
                step=0.1,
                key="kg_kelly_fraction_slider",
                help="Amount of leverage to apply (Kelly fraction)"
            )
            st.session_state.kg_kelly_fraction = kelly_fraction
            
            # Show Kelly ratio with color-coded feedback
            kelly_ratio = kelly_fraction / optimal_kelly if optimal_kelly > 0 else float('inf')
            
            if kelly_ratio < 0.5:
                st.info(f"🔵 Conservative: {kelly_ratio:.2f}x optimal Kelly")
            elif kelly_ratio < 0.9:
                st.success(f"🟢 Moderate: {kelly_ratio:.2f}x optimal Kelly")
            elif kelly_ratio < 1.1:
                st.success(f"🟢 Optimal: {kelly_ratio:.2f}x optimal Kelly")
            elif kelly_ratio < 2:
                st.warning(f"🟠 Aggressive: {kelly_ratio:.2f}x optimal Kelly")
            else:
                st.error(f"🔴 Very aggressive: {kelly_ratio:.2f}x optimal Kelly")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with button_col:
            st.markdown('<div style="margin-top:60px;"></div>', unsafe_allow_html=True)
            # Reset button
            if st.button("Reset Game", key="kg_reset", use_container_width=True):
                st.session_state.kg_simulator.reset()
                st.success("Game reset!")
            
            # Add some space
            st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
            
            # Advance week button - make it larger and more prominent
            if st.button("Advance Week", key="kg_advance", use_container_width=True):
                # Update statistics with Kelly fraction
                stats = simulator.advance_week(kelly_fraction=st.session_state.kg_kelly_fraction)
                
                # Check for game over
                if stats['game_over']:
                    st.error("GAME OVER! Your portfolio has lost 99% of its value.")
        
        # Display current state for Markov Chain
        if st.session_state.kg_simulator_type == "Markov Chain" and isinstance(simulator, MarkovChainSimulator):
            st.info(f"Current Market State: {simulator.get_state_description()}")
        
        # Game stats area
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Portfolio Value",
                f"${simulator.current_investment:,.2f}",
                f"{(simulator.current_investment / simulator.initial_investment - 1) * 100:.2f}%"
            )
        
        with col2:
            st.metric(
                "Benchmark Value",
                f"${simulator.benchmark_value:,.2f}",
                f"{(simulator.benchmark_value / simulator.initial_investment - 1) * 100:.2f}%"
            )
        
        with col3:
            st.metric(
                "Weeks Elapsed",
                f"{simulator.weeks_elapsed}"
            )
        
        # Plot portfolio value over time
        st.markdown('<div class="sub-header">Portfolio Performance</div>', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Format the dates
        dates = simulator.dates
        
        # Plot portfolio value
        ax.plot(dates, simulator.portfolio_history, 'b-', linewidth=2, label=f'Portfolio (Leverage: {kelly_fraction:.1f}x)')
        
        # Plot benchmark
        ax.plot(dates, simulator.benchmark_history, 'g--', linewidth=1.5, label='Benchmark (No Leverage)')
        
        # Add fill for drawdown periods
        if len(simulator.portfolio_history) > 1:
            peaks = np.maximum.accumulate(simulator.portfolio_history)
            drawdowns = 1 - np.array(simulator.portfolio_history) / peaks
            
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
            st.metric(
                "Total Return",
                f"{(simulator.current_investment / simulator.initial_investment - 1) * 100:.2f}%"
            )
            st.metric(
                "Benchmark Return",
                f"{(simulator.benchmark_value / simulator.initial_investment - 1) * 100:.2f}%"
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
        st.info("👈 Select an asset and click 'Initialize Game' to start playing!")
        
        # Display some example imagery or information about the game
        st.markdown("""
        ### What You'll Experience
        
        The Kelly Betting Game simulates how different leverage strategies perform when investing in real assets. You'll see:
        
        - **Real-time portfolio value** changing based on your leverage
        - **Performance comparison** against the unleveraged benchmark
        - **Risk metrics** like drawdown and probability of ruin
        - **Market regimes** when using the Markov Chain simulation
        
        This hands-on experience helps develop intuition about the Kelly Criterion and optimal leverage.
        """)
        
        # Instead of placeholder image, show some text describing the game
        st.markdown("""
        ### Example Visualization
        
        Once you initialize the game, you'll see a chart here showing your portfolio value over time compared to the benchmark.
        You'll also see detailed statistics about your performance, including returns, drawdowns, and risk metrics.
        
        Initialize the game to get started!
        """)

# Footer
st.markdown("---")
st.markdown("""
<div class="info-text">
This game is for educational purposes only and does not constitute investment advice.
Past performance is not indicative of future results. Leveraged investing involves significant risks.
</div>
""", unsafe_allow_html=True)

# Link back to main app
st.markdown("""
<div style="text-align:center; margin-top: 20px;">
<a href="./app.py">Return to Main Monte Carlo Simulator</a>
</div>
""", unsafe_allow_html=True)
