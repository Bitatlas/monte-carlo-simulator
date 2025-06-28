import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

# Import our modules
from data.fetchers import (
    EquityIndexFetcher, 
    StockFetcher, 
    CryptoFetcher, 
    BondFetcher
)
from kelly_game import kelly_game_tab
from models import (
    MonteCarloModel,
    GeometricBrownianMotionModel,
    GARCHModel,
    MarkovChainModel,
    FeynmanPathIntegralModel,
    HAS_GARCH
)
from optimization import KellyCalculator, LeverageOptimizer
from visualization import ChartGenerator

# Define helper function for financial jargon tooltips
def financial_tooltip(term, explanation):
    """
    Create a tooltip for financial terms with explanation.
    
    Parameters:
    -----------
    term : str
        The financial term to explain
    explanation : str
        The explanation of the term
        
    Returns:
    --------
    str
        HTML for the tooltip
    """
    return f"""<span class="tooltip">{term}<span class="tooltiptext">{explanation}</span></span>"""

# Set page configuration
st.set_page_config(
    page_title="Multi-Asset Monte Carlo Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Modern Theme Custom CSS with Dark Mode Support
st.markdown("""
<style>
    /* Modern sophisticated styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        color: #1E88E5;
        text-shadow: 0 2px 4px rgba(0,0,0,0.05);
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
        color: #0277BD;
        position: relative;
        padding-bottom: 8px;
    }
    
    .sub-header::after {
        content: '';
        position: absolute;
        left: 0;
        bottom: 0;
        width: 40px;
        height: 3px;
        background-color: #1E88E5;
        border-radius: 3px;
    }
    
    /* Text styles */
    .info-text {
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    .highlight {
        background-color: #f8f9fa;
        border-left: 3px solid #1E88E5;
        padding: 1rem;
        border-radius: 0.3rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .highlight:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Metrics with enhanced styling */
    .css-1wivap2, div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        padding: 12px !important;
        border: 1px solid rgba(30, 136, 229, 0.1);
    }
    
    .css-1wivap2:hover, div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        border-color: rgba(30, 136, 229, 0.3);
    }
    
    /* Data frames with improved styling */
    .dataframe {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 3px 8px rgba(0,0,0,0.05);
    }
    
    .dataframe th {
        background-color: #f2f7ff;
        padding: 12px 15px !important;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .dataframe td {
        padding: 10px 15px !important;
    }
    
    /* Enhanced button styling */
    .stButton>button {
        background-color: #1E88E5;
        background-image: linear-gradient(135deg, #1E88E5, #1976D2);
        color: white;
        border: none;
        border-radius: 6px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.12);
        transition: all 0.3s;
        font-weight: 500;
        letter-spacing: 0.3px;
        padding: 0.5rem 1.2rem;
    }
    
    .stButton>button:hover {
        background-image: linear-gradient(135deg, #1976D2, #0D47A1);
        box-shadow: 0 5px 12px rgba(0,0,0,0.18);
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 6px;
        border: 1px solid #e0e0e0;
        transition: all 0.2s;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #f0f4f8;
        border-color: #1E88E5;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #1E88E5;
        cursor: help;
        color: #1E88E5;
        font-weight: 500;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 280px;
        background-color: #323232;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px 15px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -140px;
        opacity: 0;
        transition: opacity 0.3s;
        font-weight: normal;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #323232 transparent transparent transparent;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Card layout styling */
    .card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e6e6e6;
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .card:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-color: #bbb;
    }
    
    .card-header {
        font-weight: 600;
        font-size: 1.25rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.75rem;
    }
    
    /* Animation for page transitions */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stApp {
        animation: fadeIn 0.4s ease-out;
    }
    
    /* Hover effects for select boxes */
    .stSelectbox:hover {
        border-color: #1E88E5;
    }
    
    /* Gradient accents */
    .gradient-accent {
        background: linear-gradient(135deg, #1E88E5, #1976D2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# App title with OptiFolio Simulator branding
st.markdown('<div style="display: flex; justify-content: space-between; align-items: center;">', unsafe_allow_html=True)
st.markdown('<div style="font-weight: bold; color: #1E88E5; font-size: 1.2rem;">OptiFolio Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Multi-Asset Monte Carlo Simulator with Advanced Models</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
This application simulates future returns for various assets using multiple mathematical models.
It also calculates the optimal leverage based on the Kelly Criterion to maximize long-term growth.
""")

with st.expander("📚 How to Use & Mathematical Background", expanded=False):
    st.markdown("""
    ### How to Use This Simulator
    
    1. **Select Asset & Parameters**: Choose an asset type, specific asset, and historical data period in the sidebar
    2. **Set Investment Parameters**: Specify initial investment amount and time horizon
    3. **Choose Model**: Select from different mathematical models (each with different assumptions)
    4. **Set Leverage Method**: Choose how leverage is determined (manual, Kelly criterion, etc.)
    5. **Run Simulation**: Click the button to generate thousands of possible future scenarios
    6. **Analyze Results**: Examine statistics, charts, and risk metrics across all simulation tabs
    
    ### What is Monte Carlo Simulation?
    
    A Monte Carlo simulation is like rolling dice thousands of times to see what might happen. Instead of just making one prediction about the future, we create thousands of possible scenarios based on historical patterns. This helps us understand not just what *might* happen, but how *likely* different outcomes are.
    
    Think of it like this: If you want to know your chances of getting to your destination on time, you could check a single traffic report. But a Monte Carlo simulation would be like running through thousands of commute scenarios with different traffic patterns, weather conditions, and departure times to give you a complete picture of possible outcomes.
    
    ### Mathematical Background
    
    #### Return Calculation
    - **Simple Returns**: $R_t = \\frac{P_t - P_{t-1}}{P_{t-1}}$
      * *In plain English*: The percentage change in price from one day to the next. If a stock goes from $100 to $110, that's a 10% return.
    
    - **Log Returns**: $r_t = \\ln(\\frac{P_t}{P_{t-1}})$
      * *In plain English*: A special way of calculating returns that works better for mathematical models. They're slightly smaller than simple returns but have useful statistical properties.
    
    - **Annualized Return**: $R_{annual} = (1 + R)^{252} - 1$ (assuming 252 trading days)
      * *In plain English*: What your return would be over a full year if the current rate continued. Like saying "at this pace, you'd make X% per year."
    
    #### Risk Measures
    - **Volatility**: $\\sigma = \\sqrt{\\frac{\\sum_{i=1}^{n}(r_i - \\bar{r})^2}{n-1}}$
      * *In plain English*: How much prices bounce around. Higher volatility means more dramatic price swings and typically more risk. Like measuring the bumpiness of a road.
    
    - **CAGR**: $CAGR = (\\frac{FV}{PV})^{\\frac{1}{n}} - 1$
      * *In plain English*: The smoothed-out yearly growth rate. If $10,000 becomes $16,105 after 5 years, the CAGR is 10% (because $10,000 × 1.10⁵ = $16,105).
    
    - **Maximum Drawdown**: $MaxDD = \\max_t\\{1 - \\frac{V_t}{\\max_{s \\leq t}V_s}\\}$
      * *In plain English*: The worst peak-to-trough decline you would have experienced. Like watching your $1,000 investment drop to $600 before recovering - that's a 40% drawdown.
    
    - **Sharpe Ratio**: $Sharpe = \\frac{R_p - R_f}{\\sigma_p}$
      * *In plain English*: Return per unit of risk - higher is better. Like miles per gallon for your investment. A Sharpe ratio of 1.0 means you're getting 1% of extra return for each 1% of volatility you accept.
    
    #### Simulation Models
    Most models simulate price paths using the formula:
    $S_{t+1} = S_t \\cdot (1 + R_t)$
    
    Where $R_t$ is generated according to the specific model's assumptions:
    
    - **Monte Carlo**: $R_t \\sim N(\\mu, \\sigma^2)$
      * *In plain English*: Imagine repeatedly flipping a weighted coin where the odds match historical returns. Each flip is completely independent of previous flips.
    
    - **GBM**: $\\frac{dS}{S} = \\mu dt + \\sigma dW_t$
      * *In plain English*: Like a random walk where each step depends partly on where you currently stand. The bigger your investment grows, the larger the dollar swings (while percentage swings stay similar).
    
    - **GARCH**: Time-varying volatility where $\\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$
      * *In plain English*: Models periods of calm followed by periods of turbulence, similar to how real markets behave. When markets get volatile, they tend to stay volatile for a while.
    
    #### Kelly Criterion
    The optimal leverage that maximizes long-term growth:
    $f^* = \\frac{\\mu - r}{\\sigma^2}$
    
    Where:
    - $f^*$ is the optimal leverage
    - $\\mu$ is the expected return
    - $r$ is the risk-free rate
    - $\\sigma^2$ is the variance of returns
    
    **Kelly Criterion Made Simple**:
    The Kelly Criterion helps find the "sweet spot" for investment sizing:
    - Too little invested = leaving money on the table
    - Too much invested = risk of ruin
    - Kelly finds the optimal middle ground for long-term growth

    It's like driving a car - go too slow and you'll never reach your destination quickly, go too fast and you risk a crash. Kelly finds the speed that gets you there fastest on average.
    
    ### Practical Interpretation of Results
    
    - **Median Outcome**: The middle result - half of simulations did better, half did worse
    - **Confidence Intervals**: Think of these as the "likely range" - not guaranteed, but probable
    - **Maximum Drawdown**: The worst decline you might face - like planning for the worst storm that might hit your house
    - **Probability of Major Loss**: The chance of losing a catastrophic amount (over 99% of initial investment)
    """)

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")

# Add a reset button at the top of the sidebar
if st.sidebar.button("🔄 Reset Cache", help="Clear all cached calculations to get fresh results"):
    # Clear all st.cache_data
    st.cache_data.clear()
    st.sidebar.success("Cache cleared! Results will be recalculated.")
    
# Add instructional text under the button
st.sidebar.caption("Click here every time you run a new simulation")

# Asset selection
st.sidebar.markdown('<div class="sub-header">Asset Selection</div>', unsafe_allow_html=True)

asset_type = st.sidebar.selectbox(
    "Asset Type",
    options=["Equity Index", "Individual Stock", "Cryptocurrency", "Bond"],
    index=0,
    help="Select the type of asset to simulate. Different asset classes have different historical return patterns, volatility characteristics, and risk profiles."
)

# Specific asset selection based on type
if asset_type == "Equity Index":
    asset = st.sidebar.selectbox(
        "Equity Index",
        options=["SP500", "NASDAQ", "EURO_STOXX50", "STOXX600"],
        index=0,
        format_func=lambda x: {
            "SP500": "S&P 500",
            "NASDAQ": "Nasdaq 100",
            "EURO_STOXX50": "Euro Stoxx 50",
            "STOXX600": "STOXX Europe 600"
        }.get(x, x),
        help="Select the equity index to simulate"
    )
elif asset_type == "Individual Stock":
    # Popular stock tickers organized by sector
    popular_stocks = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "INTC", "AMD", "CRM"],
        "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "C", "BLK"],
        "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV", "BMY", "ABT", "TMO", "LLY", "AMGN"],
        "Consumer": ["KO", "PEP", "MCD", "NKE", "PG", "WMT", "DIS", "SBUX", "HD", "TGT"],
        "Industrial": ["GE", "BA", "CAT", "MMM", "HON", "UPS", "FDX", "LMT", "RTX", "DE"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "BP", "RDS.A", "TOT"],
        "Other": ["Custom Ticker"]
    }
    
    # First select sector
    sector = st.sidebar.selectbox(
        "Sector",
        options=list(popular_stocks.keys()),
        index=0,
        help="Select a market sector"
    )
    
    # Then select stock from that sector
    if sector == "Other":
        asset = st.sidebar.text_input(
            "Custom Stock Ticker",
            value="",
            help="Enter any ticker symbol (e.g., AAPL for Apple Inc.)"
        )
        if not asset:  # If empty, default to AAPL
            asset = "AAPL"
    else:
        asset = st.sidebar.selectbox(
            "Stock Ticker",
            options=popular_stocks[sector],
            index=0,
            format_func=lambda x: {
                "AAPL": "Apple (AAPL)", 
                "MSFT": "Microsoft (MSFT)", 
                "GOOGL": "Alphabet (GOOGL)",
                "AMZN": "Amazon (AMZN)",
                "META": "Meta Platforms (META)",
                "NVDA": "NVIDIA (NVDA)",
                "TSLA": "Tesla (TSLA)",
                "INTC": "Intel (INTC)",
                "AMD": "AMD (AMD)",
                "CRM": "Salesforce (CRM)",
                # Finance
                "JPM": "JPMorgan Chase (JPM)",
                "BAC": "Bank of America (BAC)",
                "WFC": "Wells Fargo (WFC)",
                "GS": "Goldman Sachs (GS)",
                "MS": "Morgan Stanley (MS)",
                "V": "Visa (V)",
                "MA": "Mastercard (MA)",
                "AXP": "American Express (AXP)",
                "C": "Citigroup (C)",
                "BLK": "BlackRock (BLK)",
                # Healthcare
                "JNJ": "Johnson & Johnson (JNJ)",
                "PFE": "Pfizer (PFE)",
                "MRK": "Merck (MRK)",
                "UNH": "UnitedHealth (UNH)",
                "ABBV": "AbbVie (ABBV)",
                "BMY": "Bristol Myers Squibb (BMY)",
                "ABT": "Abbott Laboratories (ABT)",
                "TMO": "Thermo Fisher (TMO)",
                "LLY": "Eli Lilly (LLY)",
                "AMGN": "Amgen (AMGN)",
                # Consumer
                "KO": "Coca-Cola (KO)",
                "PEP": "PepsiCo (PEP)",
                "MCD": "McDonald's (MCD)",
                "NKE": "Nike (NKE)",
                "PG": "Procter & Gamble (PG)",
                "WMT": "Walmart (WMT)",
                "DIS": "Disney (DIS)",
                "SBUX": "Starbucks (SBUX)",
                "HD": "Home Depot (HD)",
                "TGT": "Target (TGT)",
                # Industrial
                "GE": "General Electric (GE)",
                "BA": "Boeing (BA)",
                "CAT": "Caterpillar (CAT)",
                "MMM": "3M (MMM)",
                "HON": "Honeywell (HON)",
                "UPS": "UPS (UPS)",
                "FDX": "FedEx (FDX)",
                "LMT": "Lockheed Martin (LMT)",
                "RTX": "Raytheon Technologies (RTX)",
                "DE": "Deere & Company (DE)",
                # Energy
                "XOM": "ExxonMobil (XOM)",
                "CVX": "Chevron (CVX)",
                "COP": "ConocoPhillips (COP)",
                "SLB": "Schlumberger (SLB)",
                "EOG": "EOG Resources (EOG)",
                "OXY": "Occidental Petroleum (OXY)",
                "PSX": "Phillips 66 (PSX)",
                "BP": "BP (BP)",
                "RDS.A": "Royal Dutch Shell (RDS.A)",
                "TOT": "Total (TOT)"
            }.get(x, x),
            help="Select a stock ticker symbol"
        )
elif asset_type == "Cryptocurrency":
    asset = st.sidebar.selectbox(
        "Cryptocurrency",
        options=["BTC", "ETH"],
        index=0,
        format_func=lambda x: {
            "BTC": "Bitcoin (BTC)",
            "ETH": "Ethereum (ETH)"
        }.get(x, x),
        help="Select the cryptocurrency to simulate"
    )
else:  # Bond
    asset = st.sidebar.selectbox(
        "Bond Type",
        options=["US10Y", "US30Y", "US3M", "TLT", "IEF", "SHY"],
        index=0,
        format_func=lambda x: {
            "US10Y": "10-Year US Treasury Yield",
            "US30Y": "30-Year US Treasury Yield",
            "US3M": "3-Month US Treasury Yield",
            "TLT": "iShares 20+ Year Treasury Bond ETF",
            "IEF": "iShares 7-10 Year Treasury Bond ETF",
            "SHY": "iShares 1-3 Year Treasury Bond ETF"
        }.get(x, x),
        help="Select the bond type to simulate"
    )

# Historical Data Period as a slider for years
historical_years = st.sidebar.slider(
    "Historical Data Years",
    min_value=1,
    max_value=100,  # Allow up to 100 years to cover all available data
    value=10,
    step=1,
    help="Number of years of historical data to use for simulation parameters"
)

# Convert slider value to appropriate period format for data fetchers
# Using a more granular mapping to ensure different periods get different data
if historical_years == 1:
    data_period = "1y"
elif historical_years == 2:
    data_period = "2y"  
elif historical_years == 3:
    data_period = "3y"
elif historical_years == 4:
    data_period = "4y"
elif historical_years == 5:
    data_period = "5y"
elif historical_years <= 7:
    data_period = "7y"
elif historical_years <= 10:
    data_period = "10y"
elif historical_years <= 15:
    data_period = "15y"  # Custom period
elif historical_years <= 20:
    data_period = "20y"  # Custom period
else:
    data_period = "max"  # Use max for longer periods

# Display actual period being used for transparency
st.sidebar.caption(f"Using data period: {data_period}")

# Investment parameters
st.sidebar.markdown('<div class="sub-header">Investment Parameters</div>', unsafe_allow_html=True)

investment_amount = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=1000,
    max_value=10000000,
    value=10000,
    step=1000,
    help="Your initial investment amount in dollars"
)

time_horizon = st.sidebar.slider(
    "Time Horizon (Years)",
    min_value=1,
    max_value=30,
    value=10,
    step=1,
    help="Number of years to simulate"
)

# Simulation parameters
st.sidebar.markdown('<div class="sub-header">Simulation Parameters</div>', unsafe_allow_html=True)

# Define available models based on installed packages
available_models = [
    "Monte Carlo", 
    "Geometric Brownian Motion"
]

if HAS_GARCH:
    available_models.append("GARCH(1,1)")
else:
    st.sidebar.warning("GARCH model is not available. Install the 'arch' package with `pip install arch`.")

available_models.extend([
    "Markov Chain",
    "Feynman Path Integral"
])

model_type = st.sidebar.selectbox(
    "Simulation Model",
    options=available_models,
    index=0,
    help="Mathematical model to use for simulating returns. Each model makes different assumptions about the distribution and behavior of returns. Standard Monte Carlo is simplest, while GARCH and Markov Chain can capture more complex market dynamics."
)

num_simulations = st.sidebar.slider(
    "Number of Simulations",
    min_value=10,
    max_value=3000,
    value=200,
    step=10,
    help="More simulations = more accurate results but slower performance. Recommended range: 200-1000 for balance between speed and accuracy."
)

risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="Annual risk-free interest rate (e.g., Treasury yield)"
) / 100  # Convert to decimal

# Leverage parameters
st.sidebar.markdown('<div class="sub-header">Leverage Parameters</div>', unsafe_allow_html=True)

leverage_method = st.sidebar.selectbox(
    "Leverage Method",
    options=["Manual", "Kelly Criterion", "Fractional Kelly", "Numerical Optimization"],
    index=0,
    help="Method to determine leverage applied to returns. Manual: set leverage directly. Kelly Criterion: optimal leverage to maximize long-term growth rate. Fractional Kelly: a more conservative fraction of the Kelly value. Numerical Optimization: finds optimal leverage using simulation data."
)

if leverage_method == "Manual":
    leverage = st.sidebar.slider(
        "Leverage",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Leverage to apply (1.0 = no leverage)"
    )
elif leverage_method == "Fractional Kelly":
    kelly_fraction = st.sidebar.slider(
        "Kelly Fraction",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Fraction of full Kelly to use (0.5 = Half Kelly, more conservative)"
    )

# Additional model-specific parameters
st.sidebar.markdown('<div class="sub-header">Model-Specific Parameters</div>', unsafe_allow_html=True)


model_params = {}

if model_type == "GARCH(1,1)" and HAS_GARCH:
    model_params['p'] = st.sidebar.slider(
        "GARCH Lag (p)",
        min_value=1,
        max_value=3,
        value=1,
        step=1,
        help="GARCH model lag parameter"
    )
    model_params['q'] = st.sidebar.slider(
        "ARCH Lag (q)",
        min_value=1,
        max_value=3,
        value=1,
        step=1,
        help="ARCH model lag parameter"
    )
elif model_type == "Markov Chain":
    model_params['num_states'] = st.sidebar.slider(
        "Number of States",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help="Number of discrete states in the Markov chain"
    )
elif model_type == "Feynman Path Integral":
    model_params['num_paths'] = st.sidebar.slider(
        "Number of Paths",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Number of paths to sample in the path integral"
    )
    model_params['num_time_steps'] = st.sidebar.slider(
        "Number of Time Steps",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="Number of time steps for discretization"
    )

# Initialize tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Dashboard", "Simulation Details", "Kelly Analysis", "Use Cases", "About Models", "Kelly Game"])

# Function to fetch asset data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_asset_data(asset_type, asset, period):
    """Fetch historical data for the selected asset."""
    try:
        if asset_type == "Equity Index":
            fetcher = EquityIndexFetcher(index_type=asset, period=period)
        elif asset_type == "Individual Stock":
            fetcher = StockFetcher(ticker=asset, period=period)
        elif asset_type == "Cryptocurrency":
            fetcher = CryptoFetcher(crypto_type=asset, period=period)
        else:  # Bond
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
        st.error(f"Error fetching data: {e}")
        return None

# Function to run the simulation
@st.cache_data
def run_simulation(_asset_data, model_type, investment_amount, time_horizon, num_simulations, 
                  risk_free_rate, leverage, model_params=None):
    """Run simulation with the selected model and parameters."""
    if _asset_data is None:
        return None
    
    # Get returns data
    returns = _asset_data['returns']['daily']
    
    # Create model instance based on selected model type
    if model_type == "Monte Carlo":
        model = MonteCarloModel(
            returns_data=returns,
            investment_amount=investment_amount,
            time_horizon_years=time_horizon,
            num_simulations=num_simulations,
            trading_days_per_year=252
        )
    elif model_type == "Geometric Brownian Motion":
        model = GeometricBrownianMotionModel(
            returns_data=returns,
            investment_amount=investment_amount,
            time_horizon_years=time_horizon,
            num_simulations=num_simulations,
            trading_days_per_year=252
        )
    elif model_type == "GARCH(1,1)":
        if not HAS_GARCH:
            st.error("GARCH model is not available. Please install the 'arch' package with `pip install arch`.")
            return None
            
        p = model_params.get('p', 1) if model_params else 1
        q = model_params.get('q', 1) if model_params else 1
        
        try:
            model = GARCHModel(
                returns_data=returns,
                investment_amount=investment_amount,
                time_horizon_years=time_horizon,
                num_simulations=num_simulations,
                trading_days_per_year=252,
                p=p,
                q=q
            )
        except ImportError as e:
            st.error(f"Error creating GARCH model: {e}")
            return None
    elif model_type == "Markov Chain":
        num_states = model_params.get('num_states', 5) if model_params else 5
        model = MarkovChainModel(
            returns_data=returns,
            investment_amount=investment_amount,
            time_horizon_years=time_horizon,
            num_simulations=num_simulations,
            trading_days_per_year=252,
            num_states=num_states
        )
    elif model_type == "Feynman Path Integral":
        num_paths = model_params.get('num_paths', 1000) if model_params else 1000
        num_time_steps = model_params.get('num_time_steps', 50) if model_params else 50
        model = FeynmanPathIntegralModel(
            returns_data=returns,
            investment_amount=investment_amount,
            time_horizon_years=time_horizon,
            num_simulations=num_simulations,
            trading_days_per_year=252,
            num_paths=num_paths,
            num_time_steps=num_time_steps
        )
    else:
        st.error(f"Unknown model type: {model_type}")
        return None
    
    # Run simulation with the specified leverage
    result = model.simulate(leverage=leverage)
    
    return result

# Function to calculate Kelly criterion
# NOTE: No caching to ensure fresh calculations for each asset
def calculate_kelly(_returns, risk_free_rate, _asset_name):
    """Calculate Kelly criterion for the given returns."""
    kelly_calc = KellyCalculator(_returns, risk_free_rate)
    
    # Add asset name to logs for debugging
    print(f"DEBUG - Calculating Kelly for asset: {_asset_name}")
    
    # Calculate full Kelly
    full_kelly = kelly_calc.calculate_full_kelly()
    
    # Generate leverage curve data
    leverage_values, growth_rates = kelly_calc.generate_leverage_curve(max_leverage=5.0, points=100)
    
    # Find optimal leverage numerically - pass asset-specific max leverage
    # Use lower max_leverage for volatile assets
    volatility_val = float(np.std(_returns) * np.sqrt(252))  # Annualized volatility as scalar
    
    print(f"DEBUG - Asset volatility: {volatility_val*100:.2f}% annualized")
    
    if volatility_val > 0.5:  # >50% annual volatility (crypto)
        max_lev = 2.0
        print(f"DEBUG - Using reduced max leverage of {max_lev}x for highly volatile asset ({volatility_val*100:.1f}%)")
    elif volatility_val > 0.3:  # >30% annual volatility
        max_lev = 3.0
        print(f"DEBUG - Using reduced max leverage of {max_lev}x for volatile asset ({volatility_val*100:.1f}%)")
    else:
        max_lev = 5.0
        
    optimal_leverage = kelly_calc.find_optimal_leverage_numerical(max_leverage=max_lev)
    
    return {
        'full_kelly': full_kelly,
        'optimal_leverage': optimal_leverage,
        'leverage_curve': (leverage_values, growth_rates, optimal_leverage)
    }

# Add separator and Kelly Game Controls section
st.sidebar.markdown('---')
st.sidebar.markdown('<div class="sub-header">Kelly Game Controls</div>', unsafe_allow_html=True)
st.sidebar.markdown('*These controls are only for the Kelly Game tab*')

# Run simulation when user clicks the button
current_tab_index = st.query_params.get("tab", ["0"])[0]
is_kelly_game_tab = (current_tab_index == "5")  # Kelly Game is the 6th tab (index 5)

if st.button("Run Simulation"):
    # Show progress spinner
    with st.spinner("Fetching asset data and running simulation..."):
        # Fetch asset data
        asset_data = fetch_asset_data(asset_type, asset, data_period)
        
        if asset_data:
            # Display asset information with years of data
            data_years = asset_data['stats'].get('data_period_years', historical_years)
            st.markdown(f"<div class='sub-header'>Analysis for: {asset_data['name']} ({data_years} years of data)</div>", unsafe_allow_html=True)
            
            # Calculate Kelly criterion if needed
            if leverage_method in ["Kelly Criterion", "Fractional Kelly", "Numerical Optimization"]:
                kelly_result = calculate_kelly(
                    _returns=asset_data['returns']['daily'], 
                    risk_free_rate=risk_free_rate,
                    _asset_name=asset_data['name']
                )
                
                # Determine leverage based on method
                if leverage_method == "Kelly Criterion":
                    leverage = kelly_result['full_kelly']
                elif leverage_method == "Fractional Kelly":
                    leverage = kelly_result['full_kelly'] * kelly_fraction
                else:  # Numerical Optimization
                    leverage = kelly_result['optimal_leverage']
                
                st.info(f"Using {leverage_method} leverage: {leverage:.2f}x")
            
            # Run simulation
            result = run_simulation(
                asset_data, 
                model_type, 
                investment_amount, 
                time_horizon, 
                num_simulations, 
                risk_free_rate, 
                leverage, 
                model_params
            )
            
            if result:
                # Create chart generator
                chart_gen = ChartGenerator()
                
                # Dashboard tab content
                with tab1:
                    st.markdown('<div class="sub-header">Simulation Dashboard</div>', unsafe_allow_html=True)
                    
                    # Show key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Initial Investment", f"${investment_amount:,.0f}")
                    with col2:
                        # Add confidence interval to Final Median Value
                        median_value = result['stats']['median']
                        p5 = result['stats']['percentiles']['5%']
                        p95 = result['stats']['percentiles']['95%']
                        st.metric(
                            "Final Median Value", 
                            f"${median_value:,.0f}",
                            f"95% CI: ${p5:,.0f} to ${p95:,.0f}"
                        )
                    with col3:
                        # Add confidence interval to Median Annual Return
                        cagr = result['stats']['cagr']['median']
                        cagr_p5 = result['stats']['cagr']['percentiles']['5%']
                        cagr_p95 = result['stats']['cagr']['percentiles']['95%']
                        st.metric(
                            "Median Annual Return", 
                            f"{cagr*100:.2f}%",
                            f"95% CI: {cagr_p5*100:.2f}% to {cagr_p95*100:.2f}%"
                        )
                    with col4:
                        st.metric("Leverage", f"{leverage:.2f}x")
                    
                    # Add bust counter metrics with expanded information
                    st.subheader("Path Analysis")
                    st.markdown("Analysis of simulation paths based on performance thresholds")
                    
                    # Create two rows for better organization
                    bust_col1, bust_col2 = st.columns(2)
                    with bust_col1:
                        st.markdown("##### Underperforming Paths")
                        under_col1, under_col2 = st.columns(2)
                        
                        # Get the ruin threshold and format it
                        ruin_threshold = result['stats']['bust_counters']['ruin_threshold']
                        
                        with under_col1:
                            st.metric(
                                f"Major Loss (>99%)", 
                                f"{result['stats']['bust_counters']['total_ruin']} paths",
                                f"{result['stats']['bust_counters']['total_ruin_pct']*100:.2f}%",
                                delta_color="inverse"
                            )
                            st.caption(f"Final value below ${ruin_threshold:.2f}")
                        
                        with under_col2:
                            st.metric(
                                "Below Initial Investment", 
                                f"{result['stats']['bust_counters']['below_initial']} paths",
                                f"{result['stats']['bust_counters']['below_initial_pct']*100:.2f}%",
                                delta_color="inverse"
                            )
                            
                    with bust_col2:
                        st.markdown("##### Outperforming Paths")
                        over_col1, over_col2 = st.columns(2)
                        
                        with over_col1:
                            st.metric(
                                "Above Initial Investment", 
                                f"{result['stats']['bust_counters']['above_initial']} paths",
                                f"{result['stats']['bust_counters']['above_initial_pct']*100:.2f}%"
                            )
                        
                        with over_col2:
                            # Get benchmark name and value
                            benchmark_name = result['stats']['bust_counters']['benchmark_name']
                            benchmark_value = result['stats']['bust_counters']['benchmark_value']
                            
                            st.metric(
                                f"Above Benchmark", 
                                f"{result['stats']['bust_counters']['above_benchmark']} paths",
                                f"{result['stats']['bust_counters']['above_benchmark_pct']*100:.2f}%"
                            )
                            st.caption(f"{benchmark_name} (${benchmark_value:,.0f})")
                    
                    # Show simulation paths
                    st.subheader("Simulation Paths")
                    paths_fig = chart_gen.plot_simulation_paths(
                        result['paths'], 
                        title=f"{asset_data['name']} Simulation Paths (Leverage: {leverage:.2f}x)",
                        num_paths=50
                    )
                    st.pyplot(paths_fig)
                    
                    # Show final distribution
                    st.subheader("Distribution of Final Values")
                    dist_fig = chart_gen.plot_final_distribution(result)
                    st.pyplot(dist_fig)
                
                # Simulation Details tab content
                with tab2:
                    st.markdown('<div class="sub-header">Historical Data Analysis</div>', unsafe_allow_html=True)
                    
                    # Show historical statistics with period info
                    st.info(f"Analysis based on {historical_years} years of historical data")
                    
                    hist_col1, hist_col2, hist_col3, hist_col4 = st.columns(4)
                    with hist_col1:
                        st.metric(
                            "Historical Annual Return", 
                            f"{asset_data['stats']['mean_annual']*100:.2f}%",
                            help=f"Based on {historical_years} years of historical data"
                        )
                    with hist_col2:
                        st.metric(
                            "Historical Volatility", 
                            f"{asset_data['stats']['std_annual']*100:.2f}%",
                            help=f"Annualized standard deviation of returns"
                        )
                    with hist_col3:
                        st.metric(
                            "Historical Sharpe Ratio", 
                            f"{asset_data['stats']['sharpe_ratio']:.2f}",
                            help="Ratio of excess returns to volatility (higher is better)"
                        )
                    with hist_col4:
                        if 'max_drawdown' in asset_data['stats']:
                            st.metric(
                                "Historical Max Drawdown", 
                                f"{asset_data['stats']['max_drawdown']*100:.2f}%",
                                help="Largest peak-to-trough decline during the historical period"
                            )
                    
                    # Show historical price plot
                    st.markdown('<div class="sub-header">Historical Price Performance</div>', unsafe_allow_html=True)
                    
                    # Completely rewritten historical price chart with guaranteed dimension matching
                    try:
                        # Create figure and axis
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Set modern style for plot
                        ax.set_facecolor('#f8f9fa')
                        fig.patch.set_facecolor('#ffffff')
                        
                        # Get price column from data
                        price_col = 'Adj Close' if 'Adj Close' in asset_data['data'].columns else 'Close'
                        
                        # Get historical data
                        hist_dates = asset_data['data'].index
                        hist_prices = asset_data['data'][price_col].values
                        
                        # Plot historical price data
                        ax.plot(hist_dates, hist_prices, label='Historical Price', color='#1E88E5', linewidth=2)
                        
                        # Get the last historical date and price
                        start_date = hist_dates[-1]
                        last_price = hist_prices[-1]
                        
                        # Extract simulation data safely
                        sim_data = {}
                        
                        # First convert result paths to numpy for consistent handling
                        if isinstance(result['paths'], pd.DataFrame):
                            # Get dates as list
                            sim_dates = result['paths'].index.tolist()
                            
                            # Calculate metrics
                            sim_data['median'] = result['paths'].median(axis=1).values
                            sim_data['mean'] = result['paths'].mean(axis=1).values
                            sim_data['p5'] = result['paths'].quantile(0.05, axis=1).values
                            sim_data['p95'] = result['paths'].quantile(0.95, axis=1).values
                        else:
                            # If it's not a DataFrame, handle as numpy array
                            sim_dates = [start_date + pd.Timedelta(days=i*365.25/252) for i in range(result['paths'].shape[1])]
                            sim_data['median'] = np.median(result['paths'], axis=0)
                            sim_data['mean'] = np.mean(result['paths'], axis=0)
                            sim_data['p5'] = np.percentile(result['paths'], 5, axis=0)
                            sim_data['p95'] = np.percentile(result['paths'], 95, axis=0)
                        
                        # Ensure all arrays are exactly the same length
                        min_length = min(len(sim_dates), 
                                         len(sim_data['median']),
                                         len(sim_data['mean']), 
                                         len(sim_data['p5']), 
                                         len(sim_data['p95']))
                        
                        # Verify array dimensions match
                        print(f"DEBUG - Confirmed array length: {min_length}")
                        
                        # Use only the first min_length elements
                        sim_dates = sim_dates[:min_length]
                        for key in sim_data:
                            sim_data[key] = sim_data[key][:min_length]
                        
                        # Normalize the simulation paths to start at the last historical price
                        init_portfolio = result['investment_amount']
                        scale_factor = last_price / init_portfolio
                        
                        # Plot simulation overlays with modern colors
                        ax.plot(sim_dates, sim_data['median'] * scale_factor, 
                                color='#0277BD', linestyle='--', linewidth=2, label='Simulation Median')
                        ax.plot(sim_dates, sim_data['mean'] * scale_factor, 
                                color='#26A69A', linestyle=':', linewidth=2, label='Simulation Mean')
                        
                        # Add confidence interval
                        ax.fill_between(sim_dates, 
                                       sim_data['p5'] * scale_factor, 
                                       sim_data['p95'] * scale_factor, 
                                       color='#90CAF9', alpha=0.3, label='90% Confidence Interval')
                        
                        # Add grid and styling
                        ax.grid(True, linestyle='-', alpha=0.2)
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#cccccc')
                        
                        # Customize plot
                        ax.set_title(f'{asset_data["name"]} Historical Price with Simulation Projections', fontsize=14)
                        ax.set_ylabel('Price ($)', fontsize=12)
                        ax.set_xlabel('Date', fontsize=12)
                        
                        # Add marker for simulation start
                        ax.annotate('Simulation begins →', 
                                  xy=(start_date, last_price),
                                  xytext=(-100, 30),
                                  textcoords='offset points',
                                  arrowprops=dict(arrowstyle='->', color='#0277BD', lw=1.5),
                                  fontsize=10,
                                  color='#0277BD')
                        
                        # Add legend
                        legend = ax.legend(loc='upper left', framealpha=0.9)
                    
                    except Exception as e:
                        st.error(f"Error creating historical price chart: {str(e)}")
                        print(f"ERROR - Historical chart: {str(e)}")
                        # Continue with other parts of the app instead of crashing
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.text(0.5, 0.5, "Chart could not be rendered due to dimension mismatch.", 
                               ha='center', va='center', fontsize=14, color='#ff2a6d')
                        ax.set_facecolor('#14142a')
                        fig.patch.set_facecolor('#0c0c14')
                    
                    st.pyplot(fig)
                    
                    # Show simulation statistics
                    st.markdown('<div class="sub-header">Simulation Statistics</div>', unsafe_allow_html=True)
                    
                    # Create two columns for statistics
                    stat_col1, stat_col2 = st.columns(2)
                    
                    with stat_col1:
                        st.markdown("#### Value Statistics")
                        st.markdown(f"""
                        - **Median Value**: ${result['stats']['median']:,.2f}
                        - **Mean Value**: ${result['stats']['mean']:,.2f}
                        - **Minimum Value**: ${result['stats']['min']:,.2f}
                        - **Maximum Value**: ${result['stats']['max']:,.2f}
                        - **Standard Deviation**: ${result['stats']['std']:,.2f}
                        """)
                        
                        st.markdown("#### Percentiles")
                        st.markdown(f"""
                        - **5th Percentile**: ${result['stats']['percentiles']['5%']:,.2f}
                        - **25th Percentile**: ${result['stats']['percentiles']['25%']:,.2f}
                        - **50th Percentile (Median)**: ${result['stats']['percentiles']['50%']:,.2f}
                        - **75th Percentile**: ${result['stats']['percentiles']['75%']:,.2f}
                        - **95th Percentile**: ${result['stats']['percentiles']['95%']:,.2f}
                        """)
                    
                    with stat_col2:
                        st.markdown("#### Risk Metrics")
                        st.markdown(f"""
                        - **Median CAGR**: {result['stats']['cagr']['median']*100:.2f}%
                        - **Mean CAGR**: {result['stats']['cagr']['mean']*100:.2f}%
                        - **5th Percentile CAGR**: {result['stats']['cagr']['percentiles']['5%']*100:.2f}%
                        - **95th Percentile CAGR**: {result['stats']['cagr']['percentiles']['95%']*100:.2f}%
                        """)
                        
                        st.markdown("#### Drawdown Risk")
                        st.markdown(f"""
                        - **Median Max Drawdown**: {result['stats']['max_drawdown']['median']*100:.2f}%
                        - **Mean Max Drawdown**: {result['stats']['max_drawdown']['mean']*100:.2f}%
                        - **Maximum Drawdown**: {result['stats']['max_drawdown']['max']*100:.2f}%
                        """)
                        
                        st.markdown(f"**Probability of Major Loss (>99%)**: {result['stats']['ruin_probability']*100:.2f}%")
                    
                    # Add Sharpe Ratio comparison section
                    st.markdown('<div class="sub-header">Sharpe Ratio Comparison</div>', unsafe_allow_html=True)
                    
                    # Create columns for Sharpe ratio comparison
                    sharpe_col1, sharpe_col2, sharpe_col3, sharpe_col4 = st.columns(4)
                    
                    with sharpe_col1:
                        # Historical Sharpe
                        historical_sharpe = asset_data['stats']['sharpe_ratio']
                        st.metric(
                            "Historical Sharpe Ratio",
                            f"{historical_sharpe:.2f}",
                            help=f"Based on {historical_years} years of historical data"
                        )
                    
                    with sharpe_col2:
                        # Simulated median Sharpe
                        sim_median_sharpe = result['stats']['sharpe_ratio']['median']
                        st.metric(
                            "Simulation Median Sharpe",
                            f"{sim_median_sharpe:.2f}",
                            f"{(sim_median_sharpe - historical_sharpe):.2f}",
                            help="Median Sharpe ratio across all simulation paths"
                        )
                        
                    with sharpe_col3:
                        # Simulated mean Sharpe
                        sim_mean_sharpe = result['stats']['sharpe_ratio']['mean']
                        st.metric(
                            "Simulation Mean Sharpe",
                            f"{sim_mean_sharpe:.2f}",
                            f"{(sim_mean_sharpe - historical_sharpe):.2f}",
                            help="Average Sharpe ratio across all simulation paths"
                        )
                    
                    with sharpe_col4:
                        # Simulation Sharpe range
                        sharpe_p5 = result['stats']['sharpe_ratio']['percentiles']['5%']
                        sharpe_p95 = result['stats']['sharpe_ratio']['percentiles']['95%']
                        st.metric(
                            "Simulation Sharpe (95% CI)",
                            f"{sharpe_p5:.2f} to {sharpe_p95:.2f}",
                            help="5th to 95th percentile range of Sharpe ratios"
                        )
                    
                    # Create a bar chart to compare historical vs simulated Sharpe ratios with modern style
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.set_facecolor('#f8f9fa')
                    fig.patch.set_facecolor('#ffffff')
                    
                    # Data for the bar chart - now including mean
                    sharpe_values = [historical_sharpe, result['stats']['sharpe_ratio']['median'], result['stats']['sharpe_ratio']['mean']]
                    labels = ['Historical', 'Simulation Median', 'Simulation Mean']
                    colors = ['#1E88E5', '#26A69A', '#AB47BC']  # Blue, teal, and purple - modern color scheme
                    
                    # Plot bars
                    bars = ax.bar(labels, sharpe_values, color=colors, width=0.5)
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height:.2f}', ha='center', va='bottom', color='#333333')
                    
                    # Add error bar for simulation (5th to 95th percentile range)
                    ax.errorbar(1, result['stats']['sharpe_ratio']['median'], 
                                yerr=[[result['stats']['sharpe_ratio']['median'] - sharpe_p5], 
                                      [sharpe_p95 - result['stats']['sharpe_ratio']['median']]], 
                                fmt='o', color='#0277BD', capsize=10, capthick=2)
                    
                    # Add note about leverage effect
                    leverage_note = f"Note: Simulation Sharpe reflects {leverage:.1f}x leverage"
                    ax.annotate(leverage_note, xy=(0.98, 0.02), xycoords='axes fraction',
                                ha='right', va='bottom', fontsize=10, style='italic')
                    
                    # Customize plot
                    ax.set_title('Sharpe Ratio Comparison', fontsize=14)
                    ax.set_ylabel('Sharpe Ratio')
                    ax.grid(True, alpha=0.2, linestyle='-')
                    
                    # Modern border styling
                    for spine in ax.spines.values():
                        spine.set_color('#cccccc')
                        spine.set_linewidth(0.8)
                    
                    st.pyplot(fig)
                    
                    # Model parameters if available
                    if 'model_parameters' in result:
                        st.markdown('<div class="sub-header">Model Parameters</div>', unsafe_allow_html=True)
                        st.json(result['model_parameters'])
                
                # Kelly Analysis tab content
                with tab3:
                    st.markdown('<div class="sub-header">Kelly Criterion Analysis</div>', unsafe_allow_html=True)
                    
                    # Calculate Kelly if not already done
                    if 'kelly_result' not in locals():
                        kelly_result = calculate_kelly(
                            _returns=asset_data['returns']['daily'], 
                            risk_free_rate=risk_free_rate,
                            _asset_name=asset_data['name']
                        )
                    
                    # Show Kelly metrics
                    kelly_col1, kelly_col2, kelly_col3 = st.columns(3)
                    with kelly_col1:
                        st.metric("Full Kelly Leverage", f"{kelly_result['full_kelly']:.2f}x")
                    with kelly_col2:
                        st.metric("Half Kelly Leverage", f"{kelly_result['full_kelly']/2:.2f}x")
                    with kelly_col3:
                        st.metric("Optimal Leverage (Numerical)", f"{kelly_result['optimal_leverage']:.2f}x")
                    
                    # Plot Kelly curve
                    st.markdown('<div class="sub-header">Kelly Criterion Growth Curve</div>', unsafe_allow_html=True)
                    kelly_fig = chart_gen.plot_kelly_curve(
                        kelly_result['leverage_curve'][0],
                        kelly_result['leverage_curve'][1],
                        kelly_result['leverage_curve'][2]
                    )
                    st.pyplot(kelly_fig)
                    
                    # Explanation of Kelly Criterion with tooltips
                    st.markdown('<div class="sub-header">Understanding the Kelly Criterion</div>', unsafe_allow_html=True)
                    
                    kelly_tooltip = financial_tooltip("Kelly Criterion", 
                        "A mathematical formula that helps investors determine the optimal size of investments to maximize long-term growth. It balances risk and reward by calculating how much to invest based on the probability of success and the risk/reward ratio.")
                    
                    leverage_tooltip = financial_tooltip("leverage", 
                        "Using borrowed capital to increase potential returns. For example, 2x leverage means you're controlling $20,000 worth of assets with just $10,000 of your own capital.")
                    
                    log_utility_tooltip = financial_tooltip("logarithmic utility", 
                        "A way of measuring the satisfaction or value an investor gets from wealth. It increases with wealth but at a decreasing rate, which means gaining $1,000 means more to someone with $10,000 than to someone with $100,000.")
                    
                    geometric_growth_tooltip = financial_tooltip("geometric growth rate", 
                        "The compound growth rate that accounts for compounding effects over time. Unlike arithmetic averages, it properly represents the growth of investments over multiple periods.")
                    
                    probability_ruin_tooltip = financial_tooltip("probability of ruin", 
                        "The likelihood that an investor will lose all or nearly all of their capital, making recovery impossible.")
                    
                    fractional_kelly_tooltip = financial_tooltip("Fractional Kelly", 
                        "Using a fraction (e.g., 50%) of the Kelly-suggested allocation to reduce risk. This approach sacrifices some expected return to gain significant reduction in volatility.")
                    
                    st.markdown(f"""
                    The {kelly_tooltip} is a formula for determining the optimal size of a series of bets or investments
                    to maximize the logarithm of wealth over the long run. For continuous returns, the formula is:
                    
                    **f* = (μ - r) / σ²**
                    
                    Where:
                    - **f*** is the optimal {leverage_tooltip}
                    - **μ** is the expected return
                    - **r** is the risk-free rate
                    - **σ²** is the variance of returns
                    
                    The Kelly criterion has several key properties:
                    
                    1. **Maximizes {log_utility_tooltip}**: It provides the highest expected {geometric_growth_tooltip}
                    2. **No {probability_ruin_tooltip}**: When strictly followed, it ensures you never lose everything
                    3. **Long-term optimality**: Any strategy using more or less than Kelly will underperform in the long run
                    
                    However, many investors use a **{fractional_kelly_tooltip}** approach (e.g., Half Kelly) to reduce risk,
                    acknowledging that we don't know the true parameters of the return distribution.
                    """, unsafe_allow_html=True)
                
                # Use Cases tab content
                with tab4:
                    st.markdown('<div class="sub-header">Use Cases & Interpretation Guide</div>', unsafe_allow_html=True)
                    
                    # Load and display the use cases markdown file
                    with open("use_cases.md", "r") as f:
                        use_cases_content = f.read()
                    st.markdown(use_cases_content)
                    
                    # Load and display the model interpretations markdown file directly
                    st.markdown('<div class="sub-header">Practical Interpretation of Results by Model Type</div>', unsafe_allow_html=True)
                    with open("model_interpretations.md", "r") as f:
                        model_interpretations_content = f.read()
                    st.markdown(model_interpretations_content)
                
                # About Models tab content
                with tab5:
                    st.markdown('<div class="sub-header">About Simulation Models</div>', unsafe_allow_html=True)
                    
                    # Create expanders for each model
                    with st.expander("Standard Monte Carlo"):
                        st.markdown("""
                        **Standard Monte Carlo** simulates returns by randomly sampling from a normal distribution,
                        using the historical mean and standard deviation of returns. This is the simplest model but assumes
                        that returns are independent and identically distributed (i.i.d.) and follow a normal distribution.
                        
                        **Key Assumptions:**
                        - Returns follow a normal distribution
                        - Returns are independent from day to day
                        - Parameters (mean, volatility) remain constant
                        
                        **When to Use:**
                        - For simple, quick simulations
                        - When you believe markets are relatively efficient
                        - For educational purposes and basic planning
                        """)
                    
                    with st.expander("Geometric Brownian Motion (GBM)"):
                        st.markdown("""
                        **Geometric Brownian Motion** is a continuous-time stochastic process where the logarithm of the
                        asset price follows a Brownian motion with drift. This is the model underlying the Black-Scholes
                        options pricing formula.
                        
                        The GBM model is described by the stochastic differential equation:
                        
                        dS = μS dt + σS dW
                        
                        Where:
                        - S is the asset price
                        - μ is the drift (expected return)
                        - σ is the volatility
                        - dW is a Wiener process (Brownian motion)
                        
                        **Key Assumptions:**
                        - Returns are log-normally distributed
                        - Volatility is constant
                        - No autocorrelation in returns
                        
                        **When to Use:**
                        - For modeling most financial assets
                        - When you want a more theoretically sound model than simple Monte Carlo
                        - For options pricing and risk management
                        """)
                    
                    with st.expander("GARCH(1,1)"):
                        st.markdown("""
                        **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** models capture the fact that
                        volatility in financial markets tends to cluster (periods of high volatility tend to persist).
                        GARCH(1,1) is the simplest form, where current volatility depends on the previous period's
                        volatility and squared return.
                        
                        The GARCH(1,1) model is defined as:
                        
                        σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁
                        
                        Where:
                        - σ²ₜ is the variance at time t
                        - r²ₜ₋₁ is the squared return in the previous period
                        - ω, α, β are parameters estimated from historical data
                        
                        **Key Advantages:**
                        - Captures volatility clustering
                        - More realistic risk estimates during turbulent markets
                        - Better suited for measuring tail risks
                        
                        **When to Use:**
                        - During periods of changing volatility
                        - For risk management in complex markets
                        - When you need more accurate Value-at-Risk (VaR) estimates
                        """)
                    
                    with st.expander("Markov Chain"):
                        st.markdown("""
                        **Markov Chain** models discretize returns into a finite number of states and use transition
                        probabilities between these states to generate future return sequences. This approach can capture
                        regime-switching behavior in markets.
                        
                        **Key Features:**
                        - Captures different market regimes (bull, bear, sideways)
                        - Models state persistence (tendency to stay in the same state)
                        - Can represent non-normal return distributions
                        
                        **When to Use:**
                        - When markets show distinct regimes
                        - For modeling assets with cyclical behavior
                        - When simple normal distribution assumptions are inadequate
                        """)
                    
                    with st.expander("Feynman Path Integral"):
                        st.markdown("""
                        **Feynman Path Integral** approaches, borrowed from quantum mechanics, treat asset price evolution
                        as a sum over all possible paths, weighted by an "action" function. This allows for more complex
                        dynamics and rare events.
                        
                        **Key Features:**
                        - Can model complex, non-Gaussian dynamics
                        - Better representation of extreme market events
                        - Accounts for path-dependency in price evolution
                        
                        **When to Use:**
                        - For sophisticated risk analysis
                        - When concerned about tail risks and black swan events
                        - For research into complex market dynamics
                        
                        This is an advanced, experimental approach to financial modeling inspired by quantum physics.
                        """)
            else:
                st.error("Failed to run simulation. Please check the parameters and try again.")
        else:
            st.error("Failed to fetch asset data. Please check the asset selection and try again.")

# Display Use Cases tab content (always visible regardless of simulation)
with tab4:
    st.markdown('<div class="sub-header">Use Cases & Interpretation Guide</div>', unsafe_allow_html=True)
    
    # Load and display the use cases markdown file
    with open("use_cases.md", "r") as f:
        use_cases_content = f.read()
    st.markdown(use_cases_content)
    
    # Load and display the model interpretations markdown file directly
    st.markdown('<div class="sub-header">Practical Interpretation of Results by Model Type</div>', unsafe_allow_html=True)
    with open("model_interpretations.md", "r") as f:
        model_interpretations_content = f.read()
    st.markdown(model_interpretations_content)

# Initial message if simulation hasn't been run
if "result" not in locals():
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to start.")
    
    # Show explanation in the About Models tab
    with tab5:
        st.markdown('<div class="sub-header">About Simulation Models</div>', unsafe_allow_html=True)
        
        st.markdown("""
        This application provides several mathematical models for simulating asset returns:
        
        1. **Standard Monte Carlo**: Simple sampling from a normal distribution
        2. **Geometric Brownian Motion (GBM)**: The classic continuous-time model for asset prices
        3. **GARCH(1,1)**: Captures volatility clustering and time-varying volatility
        4. **Markov Chain**: Models regime-switching behavior using discrete states
        5. **Feynman Path Integral**: Quantum-inspired approach for complex market dynamics
        
        Each model has different assumptions and is suitable for different market conditions.
        Select a model from the sidebar and click "Run Simulation" to see detailed explanations.
        """)

# Kelly Game tab content
with tab6:
    # Create two columns for better organization: game controls on left, game display on right
    game_col1, game_col2 = st.columns([1, 3])
    
    with game_col1:
        st.markdown('<div class="sub-header">Kelly Game Controls</div>', unsafe_allow_html=True)
        
        # Add explanation
        st.markdown("""
        This interactive game demonstrates the Kelly criterion in action. 
        
        Select an asset and configure your game parameters, then click "Initialize Game" to start.
        """)
        
        # Add controls in the left column (these will be passed to the kelly_game_tab function)
        st.session_state.kg_controls_location = "column"
    
    with game_col2:
        # Game display area
        st.markdown('<div class="main-header">Kelly Betting Game</div>', unsafe_allow_html=True)
        
        # Help text about separation
        st.info("Game controls are in the left column ←")
        
        # Main game display
        kelly_game_tab()

# About the Author Section
st.markdown("---")
with st.expander("About the Author", expanded=False):
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <div style="flex: 1;">
            <h3>Henrique Centieiro</h3>
            <p class="info-text">
                Henrique Centieiro is a financial engineer and quantitative analyst specializing in investment optimization 
                and risk management. With expertise in Monte Carlo simulations and portfolio theory, he developed OptiFolio 
                Simulator to help investors understand complex investment concepts through interactive visualization and analysis.
            </p>
            <p class="info-text">
                His work combines mathematical rigor with practical financial applications, making sophisticated investment 
                strategies accessible to both individual and institutional investors. Henrique is passionate about financial 
                education and empowering investors through data-driven decision making.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="info-text">
This application is for educational purposes only and does not constitute investment advice.
Past performance is not indicative of future results. Leveraged investing involves significant risks.
</div>
""", unsafe_allow_html=True)
