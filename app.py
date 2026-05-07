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
    BondFetcher,
    SectorETFFetcher
)
from kelly_game import kelly_game_tab
from portfolio_optimizer import portfolio_optimizer_tab
from models import (
    MonteCarloModel,
    PortfolioMonteCarloModel,
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
    initial_sidebar_state="collapsed"
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
    
    /* Metrics with enhanced styling - Fixed for dark mode */
    .css-1wivap2, div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: inherit !important;
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        padding: 12px !important;
        border: 1px solid rgba(30, 136, 229, 0.1);
    }
    
    /* Fix for metric values in dark mode */
    div[data-testid="stMetric"] > div {
        color: inherit !important;
    }
    
    div[data-testid="stMetric"] > div > div {
        color: inherit !important;
    }
    
    div[data-testid="stMetric"] label {
        color: inherit !important;
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
    
    /* Enhanced tab visibility */
    button[data-baseweb="tab"] {
        font-weight: 800 !important; /* Increased font weight for bolder text */
        padding: 10px 20px !important;
        margin: 0 3px !important;
        border-radius: 5px 5px 0 0 !important;
        transition: all 0.2s ease !important;
        font-size: 1.05rem !important;
        border: 1px solid #e0e0e0 !important;
        border-bottom: none !important;
    }
    
    /* Make tab text and icons stand out more */
    button[data-baseweb="tab"] span {
        font-weight: 800 !important;
    }
    
    button[data-baseweb="tab"]:hover {
        background-color: rgba(30, 136, 229, 0.15) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1) !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(30, 136, 229, 0.2) !important;
        border-bottom: 4px solid #1E88E5 !important;
        box-shadow: 0 -3px 7px rgba(0,0,0,0.12) !important;
    }
    
    div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        border: 1px solid #e0e0e0 !important;
    }
    
    /* Adjusted top padding */
    .main .block-container {
        padding-top: 2rem !important;
    }
    
    /* Aggressively reduced sidebar spacing */
    .stSelectbox, .stSlider, section[data-testid="stSidebar"] .stNumberInput {
        margin-bottom: 0.2rem !important;
        padding-bottom: 0 !important;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem !important;
    }
    
    section[data-testid="stSidebar"] hr {
        margin: 0.4rem 0 !important;
    }
    
    /* Reduce spacing between sidebar elements */
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0.1rem !important;
        padding-bottom: 0 !important;
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Compress sidebar labels */
    section[data-testid="stSidebar"] label {
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
        line-height: 1.2 !important;
    }
    
    /* Logo styling */
    .app-logo {
        width: 150px;
        height: auto;
        margin-bottom: 10px;
    }
    
    /* Bold sidebar instruction */
    .reset-instruction {
        font-weight: bold;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Fix the Kelly Game layout specifically */
    
    /* 1. Make the horizontal layout more compact - reduce gap between columns */
    div[data-testid="tabs"] > div:nth-child(8) div[data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        column-gap: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* 2. Remove left padding from content column to move text closer to controls */
    div[data-testid="tabs"] > div:nth-child(8) div[data-testid="column"]:nth-child(2) {
        padding-left: 0 !important;
        margin-left: -70px !important; /* Even larger negative margin to pull content much closer to left panel */
    }
    
    /* 3. Compact the left panel's internal elements - controls */
    div[data-testid="tabs"] > div:nth-child(8) div[data-testid="column"]:first-child {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    div[data-testid="tabs"] > div:nth-child(8) div[data-testid="column"]:first-child > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* 4. Reduce space between each control in the left panel */
    div[data-testid="tabs"] > div:nth-child(8) div[data-testid="column"]:first-child .stSelectbox {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    div[data-testid="tabs"] > div:nth-child(8) div[data-testid="column"]:first-child .stRadio {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    div[data-testid="tabs"] > div:nth-child(8) div[data-testid="column"]:first-child .element-container {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* 5. Target selection boxes and radio labels specifically to reduce their padding */
    div[data-testid="tabs"] > div:nth-child(8) div[data-testid="column"]:first-child label {
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
        font-size: 0.9em !important;
    }
    
    /* 6. Target the control inputs themselves */
    div[data-testid="tabs"] > div:nth-child(8) div[data-testid="column"]:first-child select,
    div[data-testid="tabs"] > div:nth-child(8) div[data-testid="column"]:first-child input {
        margin-top: 0 !important;
        margin-bottom: 2px !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Base64 encoded logo image - simplified version
logo_base64 = ""

# Create app header with improved title and branding
st.markdown('<div class="main-header">Henrique Wealth Academy OptiFolio Simulator</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 1.2rem; margin-bottom: 1rem;">📈 Multi-Asset Monte Carlo Simulator with Advanced Models</div>', unsafe_allow_html=True)

with st.expander("📚 How to Use & Mathematical Background", expanded=False):
    st.markdown("""
    ## Platform Overview — 8 Tabs

    | Tab | Purpose |
    |-----|---------|
    | 📊 **Dashboard** | Configure all parameters, run a single-asset simulation, view headline results |
    | 🔬 **Simulation Details** | Deep-dive statistics, historical price chart with projections, Sharpe comparison |
    | 📈 **Kelly Analysis** | Optimal leverage, Kelly growth curve, Full / Half / Numerical Kelly |
    | 🗂️ **Portfolio Simulator** | 2–3 asset correlated simulation with rebalancing & DCA |
    | 🔍 **Portfolio Optimizer** | Exhaustive Kelly + MPT ranking across asset universes |
    | 🛠️ **Use Cases** | Practical guided scenarios (retirement, stress testing, advising…) |
    | ℹ️ **About Models** | Full model documentation, formulas, and when to use each |
    | 🎮 **Kelly Game** | Interactive game — learn Kelly Criterion by playing with real data |

    ---

    ## How to Run a Simulation (Dashboard)

    1. **Expand ⚙️ Simulation Parameters** — four columns:
       - **Asset** — pick type (Equity Index, Stock, ETF, Bond), select the specific asset, set historical lookback years
       - **Investment** — initial capital, time horizon (1–30 yr), risk-free rate
       - **Model** — choose from 5 stochastic models; optional model-specific parameters appear automatically
       - **Leverage** — Manual / Kelly Criterion / Fractional Kelly / Numerical Optimization
    2. **Click ▶ Run Simulation** — results persist as you switch tabs
    3. **Switch to other tabs** to explore deeper statistics, Kelly analysis, and model parameters

    ---

    ## How to Use the Portfolio Simulator

    1. Choose 2 or 3 assets and set weights (auto-normalised to 100%)
    2. Set investment amount, time horizon, simulations, and data years
    3. Optionally enable **Rebalancing** (monthly / quarterly / annually) + transaction cost
    4. Optionally enable **DCA** (Dollar-Cost Averaging) — fixed contribution each month or quarter
    5. Click ▶ Run Portfolio Simulation — get correlated paths, correlation matrix, and Portfolio Kelly analysis

    ---

    ## How to Use the Portfolio Optimizer

    1. Select a **preset universe** (US Large Cap, ETFs, Global, Bonds, Commodities, Factor, Thematic, Balanced Mix)
       or enter custom tickers
    2. Choose **portfolio size** (2, 3, or 4 assets)
    3. Set lookback period and risk-free rate
    4. Click **Run Optimizer** — every N-choose-K combination is ranked by Diversification Bonus, Portfolio Kelly, and Sharpe

    ---

    ## Mathematical Background

    ### Return Formulas
    - **Simple Return**: $R_t = (P_t - P_{t-1}) / P_{t-1}$
    - **Log Return**: $r_t = \\ln(P_t / P_{t-1})$ — additive over time; preferred for modelling
    - **Annualised Return**: $(1 + R_{daily})^{252} - 1$

    ### Risk Metrics
    - **Volatility** $\\sigma$: standard deviation of daily log-returns × √252 (annualised)
    - **CAGR**: $(FV / PV)^{1/T} - 1$ — the compound annual growth rate
    - **Max Drawdown**: largest peak-to-trough decline; $\\max_t\\{1 - V_t / \\max_{s \\le t} V_s\\}$
    - **Sharpe Ratio**: $(\\mu_p - r_f) / \\sigma_p$ — excess return per unit of risk
    - **Ruin Probability**: fraction of paths ending below 1% of initial capital

    ### Simulation Models
    | Model | Core Formula | Key Feature |
    |-------|-------------|-------------|
    | 🎲 Monte Carlo | $r_t \\sim N(\\mu, \\sigma^2)$ | Fast baseline; constant volatility |
    | 📉 GBM | $dS = \\mu S\\,dt + \\sigma S\\,dW_t$ | Log-normal prices; Black-Scholes foundation |
    | 📊 GARCH(1,1) | $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$ | Volatility clustering; fat tails |
    | ⛓️ Markov Chain | $P_{ij} = P(\\text{state}_j \\mid \\text{state}_i)$ | Regime switching (bull/bear/crash) |
    | 🔄 Feynman Path | $K = \\int D[S]\\,e^{iS[\\text{path}]/\\hbar}$ | Quantum-inspired; path dependencies |

    ### Kelly Criterion
    The leverage $f^*$ that maximises expected log-growth $G(f) = f\\mu - \\tfrac{1}{2}f^2\\sigma^2$:

    $$f^* = \\frac{\\mu - r_f}{\\sigma^2}$$

    - $f^* > 1$ → borrow to invest (positive risk-premium justifies leverage)
    - $f^* > 2$ → rare; implies very high Sharpe ratio
    - Overbetting ($f > 2f^*$) → expected log-growth turns **negative** (ruin territory)
    - **Half Kelly** ($f^*/2$): ~75% of max growth, ~50% of variance — institutional standard

    **Portfolio Kelly (Nekrasov):**
    $$K^* = (\\mu - r_f)^T \\Sigma^{-1} (\\mu - r_f)$$
    **Diversification Bonus** = $K^* - \\max(f^*_i)$ — extra growth from combining low-correlation assets.

    ---

    ### Practical Interpretation of Results

    | Metric | What it means in plain English |
    |--------|-------------------------------|
    | Median outcome | The middle result — half of simulations ended better, half worse |
    | 5th–95th percentile band | Your "realistic range" — not guaranteed, but covers 90% of scenarios |
    | Max Drawdown (mean) | The average worst decline you should expect to weather before recovering |
    | Ruin Probability | Chance of losing almost everything — should be near 0% for any serious strategy |
    | Diversification Bonus | Mathematical proof that combining low-correlation assets beats the best individual asset |
    """)

# Sidebar: Reset Cache only
if st.sidebar.button("🔄 Reset Cache",
                     help="Clear all cached calculations to get fresh results",
                     use_container_width=True):
    st.cache_data.clear()
    for k in ["_sim_result","_asset_data","_leverage","_chart_gen","_kelly_result",
              "_hist_years","_lev_method","_rf_rate"]:
        st.session_state.pop(k, None)
    st.sidebar.success("✅ Cache cleared!")
st.sidebar.caption("Click here every time you run a new simulation")

# Initialize tabs (using CSS to make them bold rather than HTML tags)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Dashboard",
    "🔬 Simulation Details",
    "📈 Kelly Analysis",
    "🗂️ Portfolio Simulator",
    "🔍 Portfolio Optimizer",
    "🛠️ Use Cases",
    "ℹ️ About Models",
    "🎮 Kelly Game",
])

# Add even stronger CSS to make tab text bold
st.markdown("""
<style>
/* Ensure tab text is extra bold */
button[data-baseweb="tab"] p {
    font-weight: 900 !important;
    font-size: 1.1rem !important;
}
button[data-baseweb="tab"] {
    font-weight: 900 !important;
    font-size: 1.1rem !important;
}
/* Make emojis stand out */
button[data-baseweb="tab"] span {
    font-weight: 900 !important;
    font-size: 1.1rem !important;
}
</style>
""", unsafe_allow_html=True)

# Function to fetch asset data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_asset_data(asset_type, asset, period):
    """Fetch historical data for the selected asset."""
    try:
        if asset_type == "📊 Equity Index" or asset_type == "Equity Index":
            fetcher = EquityIndexFetcher(index_type=asset, period=period)
        elif asset_type == "🏢 Individual Stock" or asset_type == "Individual Stock":
            fetcher = StockFetcher(ticker=asset, period=period)
        elif asset_type == "📈 Sector ETF" or asset_type == "Sector ETF":
            fetcher = SectorETFFetcher(etf_ticker=asset, period=period)
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
    if model_type == "🎲 Monte Carlo" or model_type == "Monte Carlo":
        model = MonteCarloModel(
            returns_data=returns,
            investment_amount=investment_amount,
            time_horizon_years=time_horizon,
            num_simulations=num_simulations,
            trading_days_per_year=252
        )
    elif model_type == "📉 Geometric Brownian Motion" or model_type == "Geometric Brownian Motion":
        model = GeometricBrownianMotionModel(
            returns_data=returns,
            investment_amount=investment_amount,
            time_horizon_years=time_horizon,
            num_simulations=num_simulations,
            trading_days_per_year=252
        )
    elif model_type == "📊 GARCH(1,1)" or model_type == "GARCH(1,1)":
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
    elif model_type == "⛓️ Markov Chain" or model_type == "Markov Chain":
        num_states = model_params.get('num_states', 5) if model_params else 5
        model = MarkovChainModel(
            returns_data=returns,
            investment_amount=investment_amount,
            time_horizon_years=time_horizon,
            num_simulations=num_simulations,
            trading_days_per_year=252,
            num_states=num_states
        )
    elif model_type == "🔄 Feynman Path Integral" or model_type == "Feynman Path Integral":
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

# ─── Dashboard tab: parameters + Run Simulation + results ──────────────────
with tab1:
    # ── Simulation Parameters (moved from sidebar) ──────────────────────────
    with st.expander("⚙️ Simulation Parameters", expanded=True):
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)

        with p_col1:
            st.markdown("**🔍 Asset**")
            asset_type = st.selectbox(
                "Asset Type",
                ["📊 Equity Index","🏢 Individual Stock","📈 Sector ETF","🔒 Bond"],
                index=0, key="dash_asset_type"
            )
            if asset_type == "📊 Equity Index":
                asset = st.selectbox("Equity Index",
                    ["SP500","NASDAQ","DOW_JONES","RUSSELL2000","EURO_STOXX50","STOXX600",
                     "FTSE100","DAX","CAC40","SMI","NIKKEI225","HANG_SENG","ASX200",
                     "KOSPI","STI","TSX","BOVESPA","MEXICO_IPC","MSCI_WORLD","EMERGING","MSCI_ACWI"],
                    index=0, key="dash_eq_idx",
                    format_func=lambda x: {
                        "SP500":"🇺🇸 S&P 500","NASDAQ":"🇺🇸 Nasdaq 100","DOW_JONES":"🇺🇸 Dow Jones",
                        "RUSSELL2000":"🇺🇸 Russell 2000","EURO_STOXX50":"🇪🇺 Euro Stoxx 50",
                        "STOXX600":"🇪🇺 STOXX Europe 600","FTSE100":"🇬🇧 FTSE 100","DAX":"🇩🇪 DAX",
                        "CAC40":"🇫🇷 CAC 40","SMI":"🇨🇭 Swiss SMI","NIKKEI225":"🇯🇵 Nikkei 225",
                        "HANG_SENG":"🇭🇰 Hang Seng","ASX200":"🇦🇺 ASX 200","KOSPI":"🇰🇷 KOSPI",
                        "STI":"🇸🇬 Straits Times","TSX":"🇨🇦 S&P/TSX","BOVESPA":"🇧🇷 Ibovespa",
                        "MEXICO_IPC":"🇲🇽 IPC","MSCI_WORLD":"🌍 MSCI World ETF",
                        "EMERGING":"🌏 Emerging Markets ETF","MSCI_ACWI":"🌐 MSCI ACWI ETF"
                    }.get(x, x)
                )
            elif asset_type == "🏢 Individual Stock":
                asset = st.text_input("Stock Ticker", value="AAPL", key="dash_stock",
                                      help="Enter any ticker, e.g. AAPL")
            elif asset_type == "📈 Sector ETF":
                asset = st.selectbox("Sector / Bond ETF",
                    ["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLU","XLB","XLC","CLRE","TLT"],
                    key="dash_etf",
                    format_func=lambda x: {
                        "XLK":"Technology (XLK)","XLF":"Financials (XLF)","XLE":"Energy (XLE)",
                        "XLV":"Health Care (XLV)","XLY":"Consumer Disc. (XLY)",
                        "XLP":"Consumer Staples (XLP)","XLI":"Industrials (XLI)",
                        "XLU":"Utilities (XLU)","XLB":"Materials (XLB)",
                        "XLC":"Comm. Services (XLC)",
                        "CLRE":"Return Stacked Bonds & Futures (CLRE)",
                        "TLT":"20+ Yr Treasury ETF (TLT)"
                    }.get(x, x)
                )
            else:
                asset = st.selectbox("Bond Type",
                    ["US10Y","US30Y","US3M","TLT","IEF","SHY"], key="dash_bond",
                    format_func=lambda x: {
                        "US10Y":"10-Yr US Treasury","US30Y":"30-Yr US Treasury",
                        "US3M":"3-Mo US Treasury","TLT":"iShares 20+ Yr Treasury ETF",
                        "IEF":"iShares 7-10 Yr Treasury ETF","SHY":"iShares 1-3 Yr Treasury ETF"
                    }.get(x, x)
                )
            historical_years = st.slider("Historical Data Years", 1, 100, 10, key="dash_hist_yrs")
            if   historical_years <= 1:  data_period = "1y"
            elif historical_years <= 2:  data_period = "2y"
            elif historical_years <= 3:  data_period = "3y"
            elif historical_years <= 4:  data_period = "4y"
            elif historical_years <= 5:  data_period = "5y"
            elif historical_years <= 7:  data_period = "7y"
            elif historical_years <= 10: data_period = "10y"
            elif historical_years <= 15: data_period = "15y"
            elif historical_years <= 20: data_period = "20y"
            else:                        data_period = "max"
            st.caption(f"Using data period: {data_period}")

        with p_col2:
            st.markdown("**💰 Investment**")
            investment_amount = st.number_input("Initial Investment ($)",
                min_value=1000, max_value=10_000_000, value=10_000, step=1_000, key="dash_invest")
            time_horizon  = st.slider("Time Horizon (Years)", 1, 30, 10, key="dash_horizon")
            risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1,
                                       key="dash_rf") / 100

        with p_col3:
            st.markdown("**⚙️ Model**")
            _avail_models = ["🎲 Monte Carlo", "📉 Geometric Brownian Motion"]
            if HAS_GARCH:
                _avail_models.append("📊 GARCH(1,1)")
            _avail_models.extend(["⛓️ Markov Chain", "🔄 Feynman Path Integral"])
            model_type = st.selectbox("Simulation Model", _avail_models, index=0, key="dash_model")
            num_simulations = st.slider("Simulations", 10, 3000, 200, 10, key="dash_sims")
            model_params = {}
            if "GARCH" in model_type and HAS_GARCH:
                model_params["p"] = st.slider("GARCH Lag (p)", 1, 3, 1, key="dash_gp")
                model_params["q"] = st.slider("ARCH Lag (q)",  1, 3, 1, key="dash_gq")
            elif "Markov" in model_type:
                model_params["num_states"] = st.slider("States", 2, 10, 5, key="dash_states")
            elif "Feynman" in model_type:
                model_params["num_paths"]      = st.slider("Paths",      100, 2000, 1000, 100, key="dash_fp")
                model_params["num_time_steps"] = st.slider("Time Steps",  10,  100,   50,  10, key="dash_ft")

        with p_col4:
            st.markdown("**📊 Leverage**")
            leverage_method = st.selectbox("Leverage Method",
                ["Manual","Kelly Criterion","Fractional Kelly","Numerical Optimization"],
                index=0, key="dash_lev_method")
            leverage = 1.0
            kelly_fraction = 0.5
            if leverage_method == "Manual":
                leverage = st.slider("Leverage", 0.0, 5.0, 1.0, 0.1, key="dash_leverage")
            elif leverage_method == "Fractional Kelly":
                kelly_fraction = st.slider("Kelly Fraction", 0.1, 1.0, 0.5, 0.1, key="dash_kf")

    # ── Run Simulation button ────────────────────────────────────────────────
    if st.button("▶ Run Simulation", type="primary", use_container_width=True, key="dash_run"):
        with st.spinner("Fetching asset data and running simulation..."):
            _ad = fetch_asset_data(asset_type, asset, data_period)
            if _ad:
                _kr = calculate_kelly(_returns=_ad["returns"]["daily"],
                                      risk_free_rate=risk_free_rate,
                                      _asset_name=_ad["name"])
                _lev = leverage
                if leverage_method == "Kelly Criterion":
                    _lev = _kr["full_kelly"]
                elif leverage_method == "Fractional Kelly":
                    _lev = _kr["full_kelly"] * kelly_fraction
                elif leverage_method == "Numerical Optimization":
                    _lev = _kr["optimal_leverage"]
                if leverage_method != "Manual":
                    st.info(f"Using {leverage_method} leverage: {_lev:.2f}x")
                _res = run_simulation(_ad, model_type, investment_amount, time_horizon,
                                      num_simulations, risk_free_rate, _lev, model_params)
                if _res:
                    _cg = ChartGenerator()
                    st.session_state["_sim_result"]   = _res
                    st.session_state["_asset_data"]   = _ad
                    st.session_state["_leverage"]     = _lev
                    st.session_state["_chart_gen"]    = _cg
                    st.session_state["_kelly_result"] = _kr
                    st.session_state["_hist_years"]   = historical_years
                    st.session_state["_lev_method"]   = leverage_method
                    st.session_state["_rf_rate"]      = risk_free_rate

    # ── Dashboard results ────────────────────────────────────────────────────
    if "_sim_result" in st.session_state:
        result       = st.session_state["_sim_result"]
        asset_data   = st.session_state["_asset_data"]
        leverage     = st.session_state["_leverage"]
        chart_gen    = st.session_state["_chart_gen"]
        historical_years = st.session_state.get("_hist_years", 10)
        leverage_method  = st.session_state.get("_lev_method", "Manual")

        st.markdown(f'''<div class="sub-header">Analysis for: {asset_data["name"]}</div>''',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Initial Investment", f"${result['investment_amount']:,.0f}")
        col2.metric("Final Median Value",
                    f"${result['stats']['median']:,.0f}",
                    f"95% CI: ${result['stats']['percentiles']['5%']:,.0f} – "
                    f"${result['stats']['percentiles']['95%']:,.0f}")
        col3.metric("Median Annual Return",
                    f"{result['stats']['cagr']['median']*100:.2f}%",
                    f"95% CI: {result['stats']['cagr']['percentiles']['5%']*100:.2f}% – "
                    f"{result['stats']['cagr']['percentiles']['95%']*100:.2f}%")
        col4.metric("Leverage", f"{leverage:.2f}x")

        st.subheader("Path Analysis")
        bc = result['stats']['bust_counters']
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("##### Underperforming Paths")
            u1, u2 = st.columns(2)
            u1.metric("Major Loss (>99%)", f"{bc['total_ruin']} paths",
                      f"{bc['total_ruin_pct']*100:.2f}%", delta_color="inverse")
            u1.caption(f"Final value below ${bc['ruin_threshold']:.2f}")
            u2.metric("Below Initial Investment", f"{bc['below_initial']} paths",
                      f"{bc['below_initial_pct']*100:.2f}%", delta_color="inverse")
        with bc2:
            st.markdown("##### Outperforming Paths")
            o1, o2 = st.columns(2)
            o1.metric("Above Initial Investment", f"{bc['above_initial']} paths",
                      f"{bc['above_initial_pct']*100:.2f}%")
            o2.metric("Above Benchmark", f"{bc['above_benchmark']} paths",
                      f"{bc['above_benchmark_pct']*100:.2f}%")
            o2.caption(f"{bc['benchmark_name']} (${bc['benchmark_value']:,.0f})")

        st.subheader("Simulation Paths")
        paths_fig = chart_gen.plot_simulation_paths(
            result['paths'],
            title=f"{asset_data['name']} Simulation Paths (Leverage: {leverage:.2f}x)",
            num_paths=50
        )
        st.pyplot(paths_fig)

        st.subheader("Distribution of Final Values")
        dist_fig = chart_gen.plot_final_distribution(result)
        st.pyplot(dist_fig)
    else:
        st.info("Configure parameters above and click **▶ Run Simulation** to start.")

# ── Simulation Details tab ────────────────────────────────────────────────────
with tab2:
    if "_sim_result" in st.session_state:
        result       = st.session_state["_sim_result"]
        asset_data   = st.session_state["_asset_data"]
        leverage     = st.session_state["_leverage"]
        chart_gen    = st.session_state["_chart_gen"]
        historical_years = st.session_state.get("_hist_years", 10)

        st.markdown('<div class="sub-header">Historical Data Analysis</div>', unsafe_allow_html=True)
        st.info(f"Analysis based on {historical_years} years of historical data")

        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("Historical Annual Return",
                   f"{asset_data['stats']['mean_annual']*100:.2f}%")
        hc2.metric("Historical Volatility",
                   f"{asset_data['stats']['std_annual']*100:.2f}%")
        hc3.metric("Historical Sharpe Ratio",
                   f"{asset_data['stats']['sharpe_ratio']:.2f}")
        if 'max_drawdown' in asset_data['stats']:
            hc4.metric("Historical Max Drawdown",
                       f"{asset_data['stats']['max_drawdown']*100:.2f}%")

        st.markdown('<div class="sub-header">Historical Price Performance</div>', unsafe_allow_html=True)
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_facecolor('#f8f9fa'); fig.patch.set_facecolor('#ffffff')
            price_col   = 'Adj Close' if 'Adj Close' in asset_data['data'].columns else 'Close'
            hist_dates  = asset_data['data'].index
            hist_prices = asset_data['data'][price_col].values
            ax.plot(hist_dates, hist_prices, label='Historical Price', color='#1E88E5', linewidth=2)
            start_date  = hist_dates[-1]; last_price = hist_prices[-1]
            sim_d = {}
            if isinstance(result['paths'], pd.DataFrame):
                sim_dates      = result['paths'].index.tolist()
                sim_d['median'] = result['paths'].median(axis=1).values
                sim_d['mean']   = result['paths'].mean(axis=1).values
                sim_d['p5']     = result['paths'].quantile(0.05, axis=1).values
                sim_d['p95']    = result['paths'].quantile(0.95, axis=1).values
            else:
                sim_dates      = [start_date + pd.Timedelta(days=i*365.25/252)
                                  for i in range(result['paths'].shape[1])]
                sim_d['median'] = np.median(result['paths'], axis=0)
                sim_d['mean']   = np.mean(result['paths'], axis=0)
                sim_d['p5']     = np.percentile(result['paths'], 5, axis=0)
                sim_d['p95']    = np.percentile(result['paths'], 95, axis=0)
            mn = min(len(sim_dates), *[len(v) for v in sim_d.values()])
            sim_dates = sim_dates[:mn]
            for k in sim_d: sim_d[k] = sim_d[k][:mn]
            sf = last_price / result['investment_amount']
            ax.plot(sim_dates, sim_d['median']*sf, color='#0277BD', linestyle='--',
                    linewidth=2, label='Simulation Median')
            ax.plot(sim_dates, sim_d['mean']*sf, color='#26A69A', linestyle=':',
                    linewidth=2, label='Simulation Mean')
            ax.fill_between(sim_dates, sim_d['p5']*sf, sim_d['p95']*sf,
                            color='#90CAF9', alpha=0.3, label='90% CI')
            ax.grid(True, linestyle='-', alpha=0.2)
            for sp in ax.spines.values(): sp.set_edgecolor('#cccccc')
            ax.set_title(f"{asset_data['name']} Historical Price with Simulation Projections", fontsize=14)
            ax.set_ylabel('Price ($)', fontsize=12); ax.set_xlabel('Date', fontsize=12)
            ax.annotate('Simulation begins →', xy=(start_date, last_price),
                        xytext=(-100,30), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='#0277BD', lw=1.5),
                        fontsize=10, color='#0277BD')
            ax.legend(loc='upper left', framealpha=0.9)
        except Exception as e:
            st.error(f"Error creating historical price chart: {e}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "Chart could not be rendered.", ha='center', va='center', fontsize=14)
        st.pyplot(fig)

        st.markdown('<div class="sub-header">Simulation Statistics</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1:
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
- **50th Percentile**: ${result['stats']['percentiles']['50%']:,.2f}
- **75th Percentile**: ${result['stats']['percentiles']['75%']:,.2f}
- **95th Percentile**: ${result['stats']['percentiles']['95%']:,.2f}
""")
        with sc2:
            st.markdown("#### Risk Metrics")
            st.markdown(f"""
- **Median CAGR**: {result['stats']['cagr']['median']*100:.2f}%
- **Mean CAGR**: {result['stats']['cagr']['mean']*100:.2f}%
- **5th Pct CAGR**: {result['stats']['cagr']['percentiles']['5%']*100:.2f}%
- **95th Pct CAGR**: {result['stats']['cagr']['percentiles']['95%']*100:.2f}%
""")
            st.markdown("#### Drawdown Risk")
            st.markdown(f"""
- **Median Max Drawdown**: {result['stats']['max_drawdown']['median']*100:.2f}%
- **Mean Max Drawdown**: {result['stats']['max_drawdown']['mean']*100:.2f}%
- **Maximum Drawdown**: {result['stats']['max_drawdown']['max']*100:.2f}%
""")
            st.markdown(f"**Probability of Major Loss (>99%)**: "
                        f"{result['stats']['ruin_probability']*100:.2f}%")

        st.markdown('<div class="sub-header">Sharpe Ratio Comparison</div>', unsafe_allow_html=True)
        sh1, sh2, sh3, sh4 = st.columns(4)
        hist_sharpe = asset_data['stats']['sharpe_ratio']
        sim_med_sh  = result['stats']['sharpe_ratio']['median']
        sim_mn_sh   = result['stats']['sharpe_ratio']['mean']
        sh_p5       = result['stats']['sharpe_ratio']['percentiles']['5%']
        sh_p95      = result['stats']['sharpe_ratio']['percentiles']['95%']
        sh1.metric("Historical Sharpe",   f"{hist_sharpe:.2f}")
        sh2.metric("Sim Median Sharpe",   f"{sim_med_sh:.2f}", f"{sim_med_sh-hist_sharpe:.2f}")
        sh3.metric("Sim Mean Sharpe",     f"{sim_mn_sh:.2f}",  f"{sim_mn_sh-hist_sharpe:.2f}")
        sh4.metric("Sim Sharpe (95% CI)", f"{sh_p5:.2f} – {sh_p95:.2f}")

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.set_facecolor('#f8f9fa'); fig2.patch.set_facecolor('#ffffff')
        bars = ax2.bar(['Historical','Sim Median','Sim Mean'],
                       [hist_sharpe, sim_med_sh, sim_mn_sh],
                       color=['#1E88E5','#26A69A','#AB47BC'], width=0.5)
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x()+b.get_width()/2, h+0.05, f'{h:.2f}',
                     ha='center', va='bottom', color='#333333')
        ax2.errorbar(1, sim_med_sh,
                     yerr=[[sim_med_sh-sh_p5],[sh_p95-sim_med_sh]],
                     fmt='o', color='#0277BD', capsize=10, capthick=2)
        ax2.set_title('Sharpe Ratio Comparison', fontsize=14)
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, alpha=0.2, linestyle='-')
        for sp in ax2.spines.values(): sp.set_color('#cccccc'); sp.set_linewidth(0.8)
        ax2.annotate(f"Note: reflects {leverage:.1f}x leverage",
                     xy=(0.98,0.02), xycoords='axes fraction', ha='right', va='bottom',
                     fontsize=10, style='italic')
        st.pyplot(fig2)

        if 'model_parameters' in result:
            st.markdown('<div class="sub-header">Model Parameters</div>', unsafe_allow_html=True)
            st.json(result['model_parameters'])
    else:
        st.info("▶ Run a simulation from the **Dashboard** tab first.")

# ── Kelly Analysis tab ────────────────────────────────────────────────────────
with tab3:
    if "_sim_result" in st.session_state:
        result       = st.session_state["_sim_result"]
        asset_data   = st.session_state["_asset_data"]
        leverage     = st.session_state["_leverage"]
        chart_gen    = st.session_state["_chart_gen"]
        kelly_result = st.session_state.get("_kelly_result")
        risk_free_rate = st.session_state.get("_rf_rate", 0.02)

        st.markdown('<div class="sub-header">Kelly Criterion Analysis</div>', unsafe_allow_html=True)
        if kelly_result is None:
            kelly_result = calculate_kelly(_returns=asset_data['returns']['daily'],
                                           risk_free_rate=risk_free_rate,
                                           _asset_name=asset_data['name'])

        kc1, kc2, kc3 = st.columns(3)
        kc1.metric("Full Kelly Leverage",         f"{kelly_result['full_kelly']:.2f}x")
        kc2.metric("Half Kelly Leverage",         f"{kelly_result['full_kelly']/2:.2f}x")
        kc3.metric("Optimal Leverage (Numerical)",f"{kelly_result['optimal_leverage']:.2f}x")

        st.markdown('<div class="sub-header">Kelly Criterion Growth Curve</div>', unsafe_allow_html=True)
        kelly_fig = chart_gen.plot_kelly_curve(
            kelly_result['leverage_curve'][0],
            kelly_result['leverage_curve'][1],
            kelly_result['leverage_curve'][2]
        )
        st.plotly_chart(kelly_fig, use_container_width=True)

        st.markdown('<div class="sub-header">Understanding the Kelly Criterion</div>', unsafe_allow_html=True)
        kelly_tt  = financial_tooltip("Kelly Criterion",
            "A mathematical formula to determine the optimal investment size to maximize long-term growth.")
        lev_tt    = financial_tooltip("leverage",
            "Using borrowed capital to increase potential returns.")
        logu_tt   = financial_tooltip("logarithmic utility",
            "Investor satisfaction from wealth — increases at a decreasing rate with wealth.")
        geom_tt   = financial_tooltip("geometric growth rate",
            "The compound growth rate accounting for compounding effects over time.")
        ruin_tt   = financial_tooltip("probability of ruin",
            "The likelihood of losing all or nearly all capital.")
        fkelly_tt = financial_tooltip("Fractional Kelly",
            "Using a fraction (e.g. 50%) of the Kelly-suggested allocation to reduce risk.")
        st.markdown(f"""
The {kelly_tt} maximises long-run log wealth. Formula: **f\* = (μ - r) / σ²**

Where **f\*** = optimal {lev_tt}, **μ** = expected return, **r** = risk-free rate, **σ²** = variance.

**Key properties:**
1. Maximises {logu_tt} → highest expected {geom_tt}
2. No {ruin_tt} when strictly followed
3. More or less than Kelly underperforms long-run

Many investors use **{fkelly_tt}** (e.g. Half Kelly) for a smoother ride.
""", unsafe_allow_html=True)
    else:
        st.info("▶ Run a simulation from the **Dashboard** tab first.")

# Use Cases tab content
with tab6:
    # Load and display the use cases markdown file with path handling for different environments
    try:
        # Try direct path first (for deployed environments)
        with open("use_cases.md", "r", encoding="utf-8") as f:
            use_cases_content = f.read()
    except FileNotFoundError:
        try:
            # Try relative path from current directory
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(current_dir, "use_cases.md"), "r", encoding="utf-8") as f:
                use_cases_content = f.read()
        except FileNotFoundError:
            # Fallback to hardcoded path
            with open("monte_carlo_simulator/use_cases.md", "r", encoding="utf-8") as f:
                use_cases_content = f.read()
    st.markdown(use_cases_content)

# About Models tab content
with tab7:
    try:
        with open("model_interpretations.md", "r", encoding="utf-8") as _f:
            _about_content = _f.read()
    except FileNotFoundError:
        try:
            import os as _os
            _cdir = _os.path.dirname(_os.path.abspath(__file__))
            with open(_os.path.join(_cdir, "model_interpretations.md"), "r", encoding="utf-8") as _f:
                _about_content = _f.read()
        except FileNotFoundError:
            _about_content = "# About Models\n\n*model_interpretations.md not found.*"
    st.markdown(_about_content)


# Kelly Game tab content
with tab8:
    # Create two columns for better organization: game controls on left, game display on right
    game_col1, game_col2 = st.columns([1, 3])
    
    with game_col1:
        # Add controls in the left column (these will be passed to the kelly_game_tab function)
        st.session_state.kg_controls_location = "column"
    
    with game_col2:
        # Main game display
        kelly_game_tab()

# ─────────────────────────────────────────────────────────────
# Portfolio Simulator tab (tab4)
# ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sub-header">🗂️ Portfolio Simulator</div>', unsafe_allow_html=True)
    st.markdown("Build a multi-asset portfolio (up to 3 assets), simulate it with correlated returns, "
                "optional rebalancing, and Dollar-Cost Averaging.")

    # ── helper to build one asset selector inside a column ──────────────
    ASSET_TYPE_OPTIONS = ["📊 Equity Index", "🏢 Individual Stock", "📈 Sector ETF", "🔒 Bond"]

    def _asset_selector(prefix, col):
        """Render asset-type + asset dropdowns inside a Streamlit column."""
        with col:
            atype = st.selectbox(
                f"Asset Type",
                options=ASSET_TYPE_OPTIONS,
                key=f"{prefix}_type"
            )
            if atype == "📊 Equity Index":
                asset = st.selectbox(
                    "Index",
                    options=[
                        "SP500","NASDAQ","DOW_JONES","RUSSELL2000",
                        "EURO_STOXX50","STOXX600","FTSE100","DAX","CAC40","SMI",
                        "NIKKEI225","HANG_SENG","ASX200","KOSPI","STI",
                        "TSX","BOVESPA","MEXICO_IPC",
                        "MSCI_WORLD","EMERGING","MSCI_ACWI",
                    ],
                    format_func=lambda x: {
                        "SP500":"🇺🇸 S&P 500","NASDAQ":"🇺🇸 Nasdaq 100",
                        "DOW_JONES":"🇺🇸 Dow Jones","RUSSELL2000":"🇺🇸 Russell 2000",
                        "EURO_STOXX50":"🇪🇺 Euro Stoxx 50","STOXX600":"🇪🇺 STOXX Europe 600",
                        "FTSE100":"🇬🇧 FTSE 100","DAX":"🇩🇪 DAX",
                        "CAC40":"🇫🇷 CAC 40","SMI":"🇨🇭 Swiss SMI",
                        "NIKKEI225":"🇯🇵 Nikkei 225","HANG_SENG":"🇭🇰 Hang Seng",
                        "ASX200":"🇦🇺 ASX 200","KOSPI":"🇰🇷 KOSPI","STI":"🇸🇬 Straits Times",
                        "TSX":"🇨🇦 S&P/TSX","BOVESPA":"🇧🇷 Ibovespa","MEXICO_IPC":"🇲🇽 IPC Mexico",
                        "MSCI_WORLD":"🌍 MSCI World ETF","EMERGING":"🌏 Emerging Markets ETF",
                        "MSCI_ACWI":"🌐 MSCI All Country World ETF",
                    }.get(x, x),
                    key=f"{prefix}_asset"
                )
            elif atype == "🏢 Individual Stock":
                asset = st.text_input("Ticker symbol", value="AAPL", key=f"{prefix}_asset")
            elif atype == "📈 Sector ETF":
                asset = st.selectbox(
                    "ETF",
                    options=["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLU","XLB","XLC","CLRE","TLT"],
                    format_func=lambda x: {"XLK":"Technology (XLK)","XLF":"Financials (XLF)",
                        "XLE":"Energy (XLE)","XLV":"Health Care (XLV)","XLY":"Consumer Disc. (XLY)",
                        "XLP":"Consumer Staples (XLP)","XLI":"Industrials (XLI)","XLU":"Utilities (XLU)",
                        "XLB":"Materials (XLB)","XLC":"Comm. Services (XLC)",
                        "CLRE":"Return Stacked Bonds & Futures (CLRE)","TLT":"20+ Yr Treasury ETF (TLT)"}.get(x, x),
                    key=f"{prefix}_asset"
                )
            else:  # Bond
                asset = st.selectbox(
                    "Bond",
                    options=["US10Y","US30Y","US3M","TLT","IEF","SHY"],
                    format_func=lambda x: {"US10Y":"10-Yr US Treasury","US30Y":"30-Yr US Treasury",
                        "US3M":"3-Mo US Treasury","TLT":"20+ Yr Treasury ETF",
                        "IEF":"7-10 Yr Treasury ETF","SHY":"1-3 Yr Treasury ETF"}.get(x, x),
                    key=f"{prefix}_asset"
                )
        return atype, asset

    # ── number of assets ─────────────────────────────────────────────────
    n_pf_assets = st.radio("Number of assets in portfolio", [2, 3], horizontal=True, key="pf_n_assets")

    # ── asset selectors ───────────────────────────────────────────────────
    pf_cols = st.columns(n_pf_assets)
    pf_assets = []
    for i in range(n_pf_assets):
        atype, asset = _asset_selector(f"pf_a{i}", pf_cols[i])
        pf_assets.append((atype, asset))

    # ── weights ───────────────────────────────────────────────────────────
    st.markdown("#### 📐 Portfolio Weights")
    weight_cols = st.columns(n_pf_assets)
    raw_weights = []
    for i, col in enumerate(weight_cols):
        default = round(100 / n_pf_assets)
        w = col.number_input(f"Asset {i+1} weight (%)", min_value=1, max_value=99,
                             value=default, step=1, key=f"pf_w{i}")
        raw_weights.append(w)
    total_w = sum(raw_weights)
    weights_norm = [w / total_w for w in raw_weights]
    if abs(total_w - 100) > 0.5:
        st.warning(f"Weights sum to {total_w}% — will be auto-normalised to 100%.")
    else:
        st.caption(f"✅ Weights sum to {total_w}%")

    # ── investment / horizon params ───────────────────────────────────────
    st.markdown("#### 💰 Investment Parameters")
    pc1, pc2, pc3 = st.columns(3)
    pf_investment = pc1.number_input("Initial Investment ($)", min_value=1000, max_value=10_000_000,
                                      value=10_000, step=1000, key="pf_investment")
    pf_horizon    = pc2.slider("Time Horizon (years)", 1, 30, 10, key="pf_horizon")
    pf_sims       = pc3.slider("Simulations", 100, 2000, 500, step=100, key="pf_sims")
    pf_data_yrs   = st.slider("Historical Data Years", 1, 30, 10, key="pf_data_yrs",
                               help="Years of history used to estimate correlations and returns")

    # ── rebalancing ───────────────────────────────────────────────────────
    st.markdown("#### 🔄 Rebalancing")
    st.caption("Rebalancing periodically resets weights back to targets by selling over-weight assets "
               "and buying under-weight ones — this can add a 'rebalancing bonus' in volatile markets.")
    rb_on = st.toggle("Enable rebalancing", key="pf_rb_on")
    rebal_freq = None
    txn_cost   = 0.001
    if rb_on:
        rbc1, rbc2 = st.columns(2)
        rebal_freq = rbc1.selectbox("Rebalancing frequency",
                                     ["monthly", "quarterly", "annually"], index=1, key="pf_rb_freq")
        txn_cost_pct = rbc2.slider(
            "Transaction cost (%)", min_value=0.0, max_value=0.5, value=0.1, step=0.05,
            key="pf_txn",
            help="Cost per dollar traded at each rebalancing event (e.g. 0.1% covers ETF spread + brokerage). "
                 "Applied to the absolute value of each trade, so a rebalance moving $500 incurs $0.50 at 0.1%."
        )
        txn_cost = txn_cost_pct / 100

    # ── DCA ───────────────────────────────────────────────────────────────
    st.markdown("#### 💸 Dollar-Cost Averaging (DCA)")
    st.caption("DCA adds a fixed amount at regular intervals, regardless of market conditions, "
               "reducing the impact of market timing.")
    dca_on = st.toggle("Enable DCA", key="pf_dca_on")
    dca_amount = 0
    dca_freq   = None
    if dca_on:
        dcac1, dcac2 = st.columns(2)
        dca_amount = dcac1.number_input("Contribution per period ($)", min_value=100, max_value=100_000,
                                         value=500, step=100, key="pf_dca_amt")
        dca_freq   = dcac2.selectbox("Contribution frequency",
                                      ["monthly", "quarterly"], index=0, key="pf_dca_freq")

    # ── run ───────────────────────────────────────────────────────────────
    if st.button("▶ Run Portfolio Simulation", use_container_width=True, key="pf_run"):
        pf_period = f"{pf_data_yrs}y" if pf_data_yrs <= 10 else "max"

        with st.spinner("Fetching data & running portfolio simulation…"):
            try:
                # Fetch all assets
                pf_asset_data_list = []
                fetch_ok = True
                for atype, asset in pf_assets:
                    d = fetch_asset_data(atype, asset, pf_period)
                    if d is None:
                        st.error(f"Could not fetch data for {asset}. Please check the ticker.")
                        fetch_ok = False
                        break
                    pf_asset_data_list.append(d.get('fetcher').get_data_for_simulation()
                                               if hasattr(d.get('fetcher', None), 'get_data_for_simulation')
                                               else {'returns': d['returns']['daily'], 'name': d['name']})

                if fetch_ok:
                    model = PortfolioMonteCarloModel(
                        assets_data=pf_asset_data_list,
                        weights=weights_norm,
                        investment_amount=pf_investment,
                        time_horizon_years=pf_horizon,
                        num_simulations=pf_sims,
                        rebalancing_frequency=rebal_freq,
                        transaction_cost=txn_cost,
                        dca_amount=dca_amount,
                        dca_frequency=dca_freq,
                    )
                    pf_result = model.simulate()

                    # ── dashboard ─────────────────────────────────────────
                    st.markdown("---")
                    st.markdown('<div class="sub-header">📊 Portfolio Simulation Results</div>',
                                unsafe_allow_html=True)

                    # Allocation summary
                    alloc_str = "  |  ".join(
                        f"**{pf_assets[i][1]}**: {weights_norm[i]*100:.1f}%"
                        for i in range(n_pf_assets)
                    )
                    st.caption(f"Portfolio: {alloc_str}")

                    avg_invested = pf_result['total_invested']
                    stats = pf_result['stats']

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Invested (avg)", f"${avg_invested:,.0f}")
                    m2.metric("Median Final Value",
                              f"${stats['median']:,.0f}",
                              f"95% CI: ${stats['percentiles']['5%']:,.0f}–${stats['percentiles']['95%']:,.0f}")
                    m3.metric("Median CAGR",
                              f"{stats['cagr']['median']*100:.2f}%",
                              f"95% CI: {stats['cagr']['percentiles']['5%']*100:.2f}%–{stats['cagr']['percentiles']['95%']*100:.2f}%")
                    m4.metric("Avg Transaction Costs",
                              f"${stats['avg_transaction_costs']:,.0f}" if rb_on else "N/A")

                    # Path analysis
                    st.markdown("#### Path Analysis")
                    bc = stats['bust_counters']
                    pa1, pa2, pa3 = st.columns(3)
                    pa1.metric("Major Loss (>99%)",
                               f"{bc['total_ruin']} paths",
                               f"{bc['total_ruin_pct']*100:.2f}%", delta_color="inverse")
                    pa2.metric("Below Total Invested",
                               f"{bc['below_initial']} paths",
                               f"{bc['below_initial_pct']*100:.2f}%", delta_color="inverse")
                    pa3.metric("Above Total Invested",
                               f"{bc['above_initial']} paths",
                               f"{bc['above_initial_pct']*100:.2f}%")

                    # Paths chart
                    st.markdown("#### Simulation Paths")
                    import plotly.graph_objects as go
                    paths_df = pf_result['paths']
                    fig_paths = go.Figure()
                    for col in paths_df.columns[:50]:
                        fig_paths.add_trace(go.Scatter(
                            x=paths_df.index, y=paths_df[col],
                            mode='lines', line=dict(width=0.6, color='rgba(30,136,229,0.25)'),
                            showlegend=False
                        ))
                    # Median path
                    fig_paths.add_trace(go.Scatter(
                        x=paths_df.index, y=paths_df.median(axis=1),
                        mode='lines', line=dict(width=2.5, color='#FF7043'),
                        name='Median'
                    ))
                    fig_paths.update_layout(
                        title="Portfolio Value Paths",
                        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
                        template="plotly_dark", height=420,
                        margin=dict(l=40, r=20, t=50, b=40)
                    )
                    st.plotly_chart(fig_paths, use_container_width=True)

                    # Correlation matrix
                    st.markdown("#### 🔗 Historical Correlation Matrix")
                    st.caption("Used to generate realistically correlated returns in the simulation.")
                    corr_df = pd.DataFrame(
                        pf_result['correlation_matrix'],
                        index=pf_result['asset_names'],
                        columns=pf_result['asset_names']
                    ).round(3)
                    st.dataframe(corr_df.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1))

                    # ── Portfolio Kelly Analysis ──────────────────────────────────
                    st.markdown("---")
                    st.markdown("#### 📈 Portfolio Kelly Analysis")
                    st.caption(
                        "Kelly Criterion applied to each asset individually and to the blended portfolio "
                        "return series (weighted by your target allocations)."
                    )

                    import plotly.graph_objects as go_kelly_pf

                    # Compute individual Kelly values
                    pf_kelly_cols = st.columns(n_pf_assets + 1)
                    individual_fks = []
                    _r_daily = 0.02 / 252  # daily risk-free rate
                    for i in range(n_pf_assets):
                        asset_rets_i = model.aligned_returns.iloc[:, i]
                        # Analytical Kelly: f* = max(0, (μ - r_daily) / σ²)
                        _mu_i = float(asset_rets_i.mean())
                        _var_i = float(asset_rets_i.var())
                        fk_i = max(0.0, (_mu_i - _r_daily) / _var_i) if _var_i > 0 else 0.0
                        individual_fks.append(fk_i)
                        pf_kelly_cols[i].metric(
                            f"Kelly: {pf_result['asset_names'][i]}",
                            f"{fk_i:.2f}×",
                            help=(f"Analytical Full Kelly for {pf_result['asset_names'][i]} in isolation. "
                                  f"Formula: (μ - r) / σ²  where μ={_mu_i*252*100:.1f}% annual, "
                                  f"σ={np.sqrt(_var_i*252)*100:.1f}% annual")
                        )

                    # Blended portfolio return series & analytical Kelly
                    pf_blend = model.aligned_returns @ np.array(pf_result['weights'])
                    _mu_pf   = float(pf_blend.mean())
                    _var_pf  = float(pf_blend.var())
                    pf_fk    = max(0.0, (_mu_pf - _r_daily) / _var_pf) if _var_pf > 0 else 0.0
                    pf_kelly_cols[-1].metric(
                        "Portfolio Kelly (blended)",
                        f"{pf_fk:.2f}×",
                        help=(f"Analytical Full Kelly for the weighted-average portfolio. "
                              f"μ={_mu_pf*252*100:.1f}% annual, σ={np.sqrt(_var_pf*252)*100:.1f}% annual. "
                              f"Diversification reduces portfolio variance, which can boost Kelly above "
                              f"any individual asset's Kelly.")
                    )

                    # Diversification bonus explanation
                    max_ind_fk = max(individual_fks) if individual_fks else 0.0
                    if pf_fk > max_ind_fk + 0.1:
                        st.info(
                            f"ℹ️ **Diversification Bonus detected** — Portfolio Kelly ({pf_fk:.2f}×) exceeds "
                            f"the highest individual Kelly ({max_ind_fk:.2f}×). This is mathematically valid: "
                            f"when assets are negatively or lowly correlated, combining them **reduces portfolio "
                            f"variance more than it reduces expected return**, boosting the Kelly ratio. "
                            f"This is the mathematical expression of Modern Portfolio Theory's diversification benefit."
                        )

                    # Kelly growth curve — set x-range to show the full peak and decline
                    chart_max_lev = max(8.0, pf_fk * 2.5)
                    _lev_arr = np.linspace(0, chart_max_lev, 200)
                    # Expected log growth: E[log(1 + f*r)] ≈ f*μ - 0.5*f²*σ²
                    _gr_arr  = _lev_arr * _mu_pf - 0.5 * _lev_arr**2 * _var_pf

                    fig_pf_kelly = go_kelly_pf.Figure()
                    fig_pf_kelly.add_trace(go_kelly_pf.Scatter(
                        x=_lev_arr, y=_gr_arr,
                        mode='lines', line=dict(color='#1E88E5', width=2.5),
                        name='Expected Log Growth Rate'
                    ))
                    # Add zero-growth reference line
                    fig_pf_kelly.add_hline(y=0, line_dash='dot', line_color='gray',
                                           annotation_text="Break-even", annotation_position="right")
                    fig_pf_kelly.add_vline(
                        x=pf_fk, line_dash='dash', line_color='#FF7043',
                        annotation_text=f"Full Kelly: {pf_fk:.2f}×",
                        annotation_position="top right"
                    )
                    fig_pf_kelly.add_vline(
                        x=pf_fk / 2, line_dash='dot', line_color='#26A69A',
                        annotation_text=f"Half Kelly: {pf_fk/2:.2f}×",
                        annotation_position="top left"
                    )
                    fig_pf_kelly.update_layout(
                        title="Portfolio Kelly Growth Curve (parabolic — peaks at Full Kelly, falls to zero at 2× Kelly)",
                        xaxis_title="Leverage (×)", yaxis_title="Expected Log Growth Rate",
                        template="plotly_dark", height=400,
                        margin=dict(l=40, r=20, t=60, b=40)
                    )
                    st.plotly_chart(fig_pf_kelly, use_container_width=True)
                    st.caption(
                        f"💡 **Full Kelly** ({pf_fk:.2f}×) maximises long-term growth. "
                        f"**Half Kelly** ({pf_fk/2:.2f}×) gives ~75% of max growth with far lower drawdowns. "
                        f"Beyond **2× Kelly** ({pf_fk*2:.2f}×) expected growth turns negative — you destroy wealth."
                    )

            except Exception as e:
                st.error(f"Portfolio simulation error: {e}")

# Portfolio Optimizer tab
with tab5:
    portfolio_optimizer_tab()

# About the Author Section
st.markdown("---")
with st.expander("About the Author", expanded=False):
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <div style="flex: 1;">
            <h3>Henrique Centieiro</h3>
            <p class="info-text">
                Henrique Centieiro is a financial engineer, quantitative analyst, Hedge Fund Manager at
                <strong>Maverick Capital</strong>, and Founder of <strong>Henrique Wealth Academy</strong>.
                With expertise spanning Monte Carlo simulations, portfolio theory, and quantitative finance,
                he has been working in the finance and tech industry since 2004 and is frequently invited to
                teach at leading universities, including the University of Hong Kong business school.
            </p>
            <p class="info-text">
                As the developer of OptiFolio Simulator, Henrique combines mathematical rigor with practical
                financial applications, making sophisticated investment strategies accessible to both individual
                and institutional investors. He is renowned for his pioneering work on leveraged ETFs, having
                developed the "Optimal Leverage Indicator" to help investors determine appropriate leverage
                levels based on asset volatility and returns. His extensive research on leveraged investing
                strategies has been published in multiple articles on Limitless Investor and is taught through
                his "Leveraged ETFs Masterclass" course.
            </p>
            <p class="info-text">
                Beyond his financial research, Henrique is an 8X Medium Top Writer who shares insights on
                quantitative finance, investing, and financial mindsets. He is also the Founder of the
                "Be Limitless" community, dedicated to financial education and empowering investors through
                data-driven decision making.
            </p>
            <p class="info-text">
                <i>For more alpha, follow <a href="https://linktr.ee/cryptohenri" target="_blank">https://linktr.ee/cryptohenri</a></i>
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
