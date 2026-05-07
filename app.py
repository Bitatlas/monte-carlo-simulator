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
    /* ── Semantic colour tokens ──────────────────────────────── */
    :root {
        --c-primary:  #1E88E5;   /* blue  — interactive / brand */
        --c-positive: #26A69A;   /* teal  — gains / above-benchmark */
        --c-caution:  #FFA726;   /* amber — moderate risk */
        --c-danger:   #EF5350;   /* red   — losses / ruin */
        --c-accent:   #7E57C2;   /* purple — Kelly peak / special */
    }

    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        background: linear-gradient(135deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    .app-subtitle {
        font-size: 1.0rem;
        color: #888;
        margin-bottom: 1.4rem;
        letter-spacing: 0.2px;
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
        background-image: linear-gradient(135deg, #1E88E5, #7E57C2);
        color: white;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(30,136,229,0.35);
        transition: all 0.3s;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 0.65rem 1.4rem;
        font-size: 1.0rem;
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
        background: rgba(255,255,255,0.06);
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
st.markdown("""
<div class="main-header">📈 OptiFolio Simulator</div>
<div class="app-subtitle">Henrique Wealth Academy &nbsp;·&nbsp; Multi-Asset Monte Carlo &amp; Portfolio Optimizer</div>
""", unsafe_allow_html=True)

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
                    st.session_state["_model_label"]  = model_type

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

        # ── Simulation Context Banner (tab1) ─────────────────────────────────
        import datetime as _dt
        _ctx_html = (
            f'<div style="background:rgba(30,136,229,0.10);border:1px solid rgba(30,136,229,0.25);'
            f'border-radius:8px;padding:8px 16px;margin-bottom:12px;font-size:0.88rem;color:inherit;">'
            f'<b>\U0001f4ca {asset_data["name"]}</b>&nbsp;\u00b7&nbsp;'
            f'\u2699\ufe0f {st.session_state.get("_model_label", "Monte Carlo")}&nbsp;\u00b7&nbsp;'
            f'\U0001f4d0 {leverage:.2f}\u00d7 leverage&nbsp;\u00b7&nbsp;'
            f'\U0001f550 {historical_years}yr data&nbsp;\u00b7&nbsp;'
            f'\U0001f4c5 {_dt.date.today()}'
            '</div>'
        )
        st.markdown(_ctx_html, unsafe_allow_html=True)

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

        st.markdown('<div class="sub-header">Path Analysis</div>', unsafe_allow_html=True)
        bc = result['stats']['bust_counters']
        _n = result['paths'].shape[0] if hasattr(result['paths'], 'shape') else num_simulations
        _ruin_pct  = bc['total_ruin_pct'] * 100
        _below_pct = (bc['below_initial_pct'] - bc['total_ruin_pct']) * 100
        _bench_pct = bc['above_benchmark_pct'] * 100
        _above_pct = (bc['above_initial_pct'] - bc['above_benchmark_pct']) * 100

        # Summary sentence
        st.markdown(
            f"Of **{_n} simulated paths**: "
            f"<span style='color:#EF5350;font-weight:600'>{bc['total_ruin_pct']*100:.1f}% ruin</span> · "
            f"<span style='color:#FFA726;font-weight:600'>{(bc['below_initial_pct']-bc['total_ruin_pct'])*100:.1f}% below start</span> · "
            f"<span style='color:#26A69A;font-weight:600'>{bc['above_initial_pct']*100:.1f}% profitable</span> · "
            f"<span style='color:#7E57C2;font-weight:600'>{bc['above_benchmark_pct']*100:.1f}% beat {bc['benchmark_name']}</span>",
            unsafe_allow_html=True
        )

        # Stacked horizontal bar
        import plotly.graph_objects as _go_pa
        _pa_fig = _go_pa.Figure()
        _pa_fig.add_trace(_go_pa.Bar(
            y=["Paths"], x=[_ruin_pct], name=f"Ruin (<1% left) {_ruin_pct:.1f}%",
            orientation='h', marker_color='#EF5350',
            hovertemplate=f"Ruin: {bc['total_ruin']} paths ({_ruin_pct:.1f}%)<extra></extra>"
        ))
        _pa_fig.add_trace(_go_pa.Bar(
            y=["Paths"], x=[max(0,_below_pct)], name=f"Below initial {max(0,_below_pct):.1f}%",
            orientation='h', marker_color='#FFA726',
            hovertemplate=f"Below start: {bc['below_initial']-bc['total_ruin']} paths ({max(0,_below_pct):.1f}%)<extra></extra>"
        ))
        _pa_fig.add_trace(_go_pa.Bar(
            y=["Paths"], x=[max(0,_above_pct)], name=f"Profitable {max(0,_above_pct):.1f}%",
            orientation='h', marker_color='#26A69A',
            hovertemplate=f"Profitable: {bc['above_initial']-bc['above_benchmark']} paths ({max(0,_above_pct):.1f}%)<extra></extra>"
        ))
        _pa_fig.add_trace(_go_pa.Bar(
            y=["Paths"], x=[_bench_pct], name=f"Beat {bc['benchmark_name']} {_bench_pct:.1f}%",
            orientation='h', marker_color='#7E57C2',
            hovertemplate=f"Beat benchmark: {bc['above_benchmark']} paths ({_bench_pct:.1f}%)<extra></extra>"
        ))
        _pa_fig.update_layout(
            barmode='stack', template='plotly_dark', height=120,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation='h', y=-0.5, x=0),
            xaxis=dict(range=[0,100], ticksuffix='%', showgrid=False),
            yaxis=dict(showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(_pa_fig, use_container_width=True)

        st.markdown('<div class="sub-header">Simulation Paths — Fan Chart</div>', unsafe_allow_html=True)
        import plotly.graph_objects as _go_fan
        _paths = result['paths']
        if hasattr(_paths, 'shape'):
            _arr = _paths if not hasattr(_paths, 'values') else _paths.values
            _xs = list(range(_arr.shape[1]))
            _p5  = np.percentile(_arr, 5,  axis=0)
            _p25 = np.percentile(_arr, 25, axis=0)
            _p50 = np.percentile(_arr, 50, axis=0)
            _p75 = np.percentile(_arr, 75, axis=0)
            _p95 = np.percentile(_arr, 95, axis=0)
        else:
            _arr = _paths.values
            _xs  = list(range(_arr.shape[1]))
            _p5  = np.percentile(_arr, 5,  axis=0)
            _p25 = np.percentile(_arr, 25, axis=0)
            _p50 = np.percentile(_arr, 50, axis=0)
            _p75 = np.percentile(_arr, 75, axis=0)
            _p95 = np.percentile(_arr, 95, axis=0)

        _fan = _go_fan.Figure()
        # P5–P95 outer band
        _fan.add_trace(_go_fan.Scatter(
            x=_xs + _xs[::-1], y=list(_p95) + list(_p5[::-1]),
            fill='toself', fillcolor='rgba(30,136,229,0.10)',
            line=dict(color='rgba(0,0,0,0)'), name='P5–P95 (90% CI)',
            hoverinfo='skip'
        ))
        # P25–P75 inner band
        _fan.add_trace(_go_fan.Scatter(
            x=_xs + _xs[::-1], y=list(_p75) + list(_p25[::-1]),
            fill='toself', fillcolor='rgba(30,136,229,0.22)',
            line=dict(color='rgba(0,0,0,0)'), name='P25–P75 (IQR)',
            hoverinfo='skip'
        ))
        # P5 line
        _fan.add_trace(_go_fan.Scatter(x=_xs, y=_p5, mode='lines',
            line=dict(color='#EF5350', width=1.2, dash='dot'), name='P5 (worst 5%)'))
        # P95 line
        _fan.add_trace(_go_fan.Scatter(x=_xs, y=_p95, mode='lines',
            line=dict(color='#26A69A', width=1.2, dash='dot'), name='P95 (best 5%)'))
        # Median
        _fan.add_trace(_go_fan.Scatter(x=_xs, y=_p50, mode='lines',
            line=dict(color='#FFA726', width=2.5), name='Median (P50)'))
        # Starting value reference
        _fan.add_hline(y=result['investment_amount'], line_dash='dash', line_color='#888',
                       annotation_text='Initial', annotation_position='right')
        _fan.update_layout(
            title=f"{asset_data['name']} — Simulation Fan Chart ({leverage:.2f}× leverage)",
            xaxis_title='Trading Days', yaxis_title='Portfolio Value ($)',
            template='plotly_dark', height=460,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation='h', y=-0.15),
            yaxis=dict(tickprefix='$', tickformat=',.0f'),
            hovermode='x unified'
        )
        st.plotly_chart(_fan, use_container_width=True)

        st.markdown('<div class="sub-header">Distribution of Final Values</div>', unsafe_allow_html=True)
        import plotly.graph_objects as _go_dist
        from scipy.stats import gaussian_kde as _kde
        _finals = result['final_values'] if 'final_values' in result else (
            _arr[:, -1] if '_arr' in dir() else np.array([]))
        if len(_finals) == 0 and hasattr(result['paths'], 'shape'):
            _tmp = result['paths']
            _finals = (_tmp.values if hasattr(_tmp, 'values') else _tmp)[:, -1]
        if len(_finals) > 0:
            _med_f  = float(np.median(_finals))
            _p5_f   = float(np.percentile(_finals, 5))
            _p95_f  = float(np.percentile(_finals, 95))
            _kde_fn = _kde(_finals)
            _xrng   = np.linspace(_finals.min(), _finals.max(), 300)
            _yrng   = _kde_fn(_xrng)

            _dist_fig = _go_dist.Figure()
            # Left-tail shading (below P5) — red danger zone
            _mask_l = _xrng <= _p5_f
            _dist_fig.add_trace(_go_dist.Scatter(
                x=_xrng[_mask_l], y=_yrng[_mask_l],
                fill='tozeroy', fillcolor='rgba(239,83,80,0.25)',
                line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
            ))
            # Main KDE curve
            _dist_fig.add_trace(_go_dist.Scatter(
                x=_xrng, y=_yrng, mode='lines',
                line=dict(color='#1E88E5', width=2.5), name='Density (KDE)'
            ))
            # Histogram (normalised)
            _dist_fig.add_trace(_go_dist.Histogram(
                x=_finals, histnorm='probability density', nbinsx=60,
                marker_color='rgba(30,136,229,0.18)',
                marker_line=dict(color='rgba(30,136,229,0.4)', width=0.5),
                name='Simulated outcomes', showlegend=True
            ))
            # Annotation lines
            for _val, _lbl, _col in [(_p5_f, 'P5', '#EF5350'),
                                     (_med_f, 'Median', '#FFA726'),
                                     (_p95_f, 'P95', '#26A69A')]:
                _dist_fig.add_vline(x=_val, line_dash='dash', line_color=_col,
                    annotation_text=f"{_lbl}<br>${_val:,.0f}",
                    annotation_font_color=_col, annotation_position='top')
            # Initial investment line
            _dist_fig.add_vline(x=result['investment_amount'], line_dash='dot', line_color='#888',
                annotation_text='Initial', annotation_position='bottom right')
            _dist_fig.update_layout(
                title='Distribution of Final Portfolio Values',
                xaxis_title='Final Value ($)', yaxis_title='Density',
                template='plotly_dark', height=400, barmode='overlay',
                margin=dict(l=40, r=40, t=60, b=40),
                xaxis=dict(tickprefix='$', tickformat=',.0f'),
                legend=dict(orientation='h', y=-0.15)
            )
            st.plotly_chart(_dist_fig, use_container_width=True)
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

        # ── Context Banner ────────────────────────────────────────────────────
        _ctx2 = (
            f'<div style="background:rgba(30,136,229,0.10);border:1px solid rgba(30,136,229,0.25);'
            f'border-radius:8px;padding:8px 16px;margin-bottom:12px;font-size:0.88rem;color:inherit;">'
            f'<b>📊 {asset_data["name"]}</b>&nbsp;·&nbsp;'
            f'⚙️ {st.session_state.get("_model_label","Monte Carlo")}&nbsp;·&nbsp;'
            f'📐 {leverage:.2f}× leverage'
            '</div>'
        )
        st.markdown(_ctx2, unsafe_allow_html=True)

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
            import plotly.graph_objects as _go_hist
            _price_col  = 'Adj Close' if 'Adj Close' in asset_data['data'].columns else 'Close'
            _hist_dates = asset_data['data'].index.tolist()
            _hist_px    = asset_data['data'][_price_col].values
            _last_px    = float(_hist_px[-1])
            _inv_amt    = float(result['investment_amount'])
            _sf         = _last_px / _inv_amt  # scale factor from sim-$ to price

            _hfig = _go_hist.Figure()
            # Historical price
            _hfig.add_trace(_go_hist.Scatter(
                x=_hist_dates, y=_hist_px, mode='lines',
                line=dict(color='#1E88E5', width=2), name='Historical Price'
            ))
            # Simulation projections
            if isinstance(result['paths'], pd.DataFrame):
                _sdates = result['paths'].index.tolist()
                _smed   = result['paths'].median(axis=1).values * _sf
                _smn    = result['paths'].mean(axis=1).values * _sf
                _sp5    = result['paths'].quantile(0.05, axis=1).values * _sf
                _sp95   = result['paths'].quantile(0.95, axis=1).values * _sf
            else:
                import pandas as _pd2
                _start = _hist_dates[-1]
                _sdates = [_start + _pd2.Timedelta(days=i*365.25/252)
                           for i in range(result['paths'].shape[1])]
                _smed  = np.median(result['paths'], axis=0) * _sf
                _smn   = np.mean(result['paths'], axis=0) * _sf
                _sp5   = np.percentile(result['paths'], 5,  axis=0) * _sf
                _sp95  = np.percentile(result['paths'], 95, axis=0) * _sf
            _mn2 = min(len(_sdates), len(_smed))
            _sdates = _sdates[:_mn2]
            _smed = _smed[:_mn2]; _smn = _smn[:_mn2]
            _sp5  = _sp5[:_mn2];  _sp95 = _sp95[:_mn2]

            # 90% CI band
            _hfig.add_trace(_go_hist.Scatter(
                x=_sdates + _sdates[::-1],
                y=list(_sp95) + list(_sp5[::-1]),
                fill='toself', fillcolor='rgba(30,136,229,0.12)',
                line=dict(color='rgba(0,0,0,0)'), name='90% CI', hoverinfo='skip'
            ))
            _hfig.add_trace(_go_hist.Scatter(
                x=_sdates, y=_sp5, mode='lines',
                line=dict(color='#EF5350', width=1, dash='dot'), name='P5'
            ))
            _hfig.add_trace(_go_hist.Scatter(
                x=_sdates, y=_sp95, mode='lines',
                line=dict(color='#26A69A', width=1, dash='dot'), name='P95'
            ))
            _hfig.add_trace(_go_hist.Scatter(
                x=_sdates, y=_smed, mode='lines',
                line=dict(color='#FFA726', width=2.2, dash='dash'), name='Sim Median'
            ))
            _hfig.add_trace(_go_hist.Scatter(
                x=_sdates, y=_smn, mode='lines',
                line=dict(color='#7E57C2', width=1.8, dash='dot'), name='Sim Mean'
            ))
            # Vertical "simulation begins" marker
            _hfig.add_vline(x=str(_hist_dates[-1]), line_dash='dash', line_color='#888',
                            annotation_text='Simulation begins →', annotation_position='top right')
            _hfig.update_layout(
                title=f"{asset_data['name']} — Historical Price & Simulation Projections",
                xaxis_title='Date', yaxis_title='Price ($)',
                template='plotly_dark', height=480,
                margin=dict(l=40, r=40, t=60, b=40),
                hovermode='x unified',
                legend=dict(orientation='h', y=-0.15),
                yaxis=dict(tickprefix='$', tickformat=',.2f')
            )
        except Exception as _e:
            st.error(f"Error creating historical chart: {_e}")
            _hfig = None
        if _hfig: st.plotly_chart(_hfig, use_container_width=True)

        st.markdown('<div class="sub-header">Simulation Statistics</div>', unsafe_allow_html=True)
        _stats_data = {
            'Metric': [
                'Median Final Value', 'Mean Final Value', 'Std Deviation',
                'Minimum', 'Maximum',
                'P5 Value', 'P25 Value', 'P75 Value', 'P95 Value',
                'Median CAGR', 'Mean CAGR', 'P5 CAGR', 'P95 CAGR',
                'Median Max Drawdown', 'Mean Max Drawdown', 'Worst Drawdown',
                '🔴 Ruin Probability (>99% loss)',
            ],
            'Value': [
                f"${result['stats']['median']:,.2f}",
                f"${result['stats']['mean']:,.2f}",
                f"${result['stats']['std']:,.2f}",
                f"${result['stats']['min']:,.2f}",
                f"${result['stats']['max']:,.2f}",
                f"${result['stats']['percentiles']['5%']:,.2f}",
                f"${result['stats']['percentiles']['25%']:,.2f}",
                f"${result['stats']['percentiles']['75%']:,.2f}",
                f"${result['stats']['percentiles']['95%']:,.2f}",
                f"{result['stats']['cagr']['median']*100:.2f}%",
                f"{result['stats']['cagr']['mean']*100:.2f}%",
                f"{result['stats']['cagr']['percentiles']['5%']*100:.2f}%",
                f"{result['stats']['cagr']['percentiles']['95%']*100:.2f}%",
                f"{result['stats']['max_drawdown']['median']*100:.2f}%",
                f"{result['stats']['max_drawdown']['mean']*100:.2f}%",
                f"{result['stats']['max_drawdown']['max']*100:.2f}%",
                f"{result['stats']['ruin_probability']*100:.2f}%",
            ],
        }
        _sdf = pd.DataFrame(_stats_data)
        st.dataframe(_sdf, use_container_width=True, hide_index=True,
                     column_config={
                         "Metric": st.column_config.TextColumn("Metric", width="medium"),
                         "Value":  st.column_config.TextColumn("Value",  width="small"),
                     })

        st.markdown('<div class="sub-header">Sharpe Ratio Comparison</div>', unsafe_allow_html=True)
        import plotly.graph_objects as _go_sh
        hist_sharpe = asset_data['stats']['sharpe_ratio']
        sim_med_sh  = result['stats']['sharpe_ratio']['median']
        sim_mn_sh   = result['stats']['sharpe_ratio']['mean']
        sh_p5       = result['stats']['sharpe_ratio']['percentiles']['5%']
        sh_p95      = result['stats']['sharpe_ratio']['percentiles']['95%']
        sh1, sh2, sh3, sh4 = st.columns(4)
        sh1.metric("Historical Sharpe",   f"{hist_sharpe:.2f}")
        sh2.metric("Sim Median Sharpe",   f"{sim_med_sh:.2f}", f"{sim_med_sh-hist_sharpe:+.2f}")
        sh3.metric("Sim Mean Sharpe",     f"{sim_mn_sh:.2f}",  f"{sim_mn_sh-hist_sharpe:+.2f}")
        sh4.metric("Sim Sharpe 90% CI",   f"{sh_p5:.2f} – {sh_p95:.2f}")
        _sh_fig = _go_sh.Figure()
        _sh_fig.add_trace(_go_sh.Bar(
            x=['Historical', 'Sim Median', 'Sim Mean'],
            y=[hist_sharpe, sim_med_sh, sim_mn_sh],
            marker_color=['#1E88E5', '#26A69A', '#7E57C2'],
            text=[f"{v:.2f}" for v in [hist_sharpe, sim_med_sh, sim_mn_sh]],
            textposition='outside',
            error_y=dict(type='data', symmetric=False,
                         array=[0, sh_p95-sim_med_sh, 0],
                         arrayminus=[0, sim_med_sh-sh_p5, 0],
                         color='#FFA726', thickness=2, width=8),
            name='Sharpe Ratio',
        ))
        _sh_fig.add_hline(y=0, line_color='#888', line_dash='dot')
        _sh_fig.update_layout(
            title=f'Sharpe Ratio Comparison  (at {leverage:.1f}× leverage)',
            yaxis_title='Sharpe Ratio', template='plotly_dark', height=380,
            margin=dict(l=40, r=40, t=60, b=40), showlegend=False
        )
        st.plotly_chart(_sh_fig, use_container_width=True)

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
        _ctx3 = (
            f'<div style="background:rgba(126,87,194,0.10);border:1px solid rgba(126,87,194,0.25);'
            f'border-radius:8px;padding:8px 16px;margin-bottom:12px;font-size:0.88rem;color:inherit;">'
            f'<b>📊 {asset_data["name"]}</b>&nbsp;·&nbsp;Kelly analysis at {leverage:.2f}× leverage'
            '</div>'
        )
        st.markdown(_ctx3, unsafe_allow_html=True)
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
        # Zone shading: green (0→Full Kelly), amber (Full→2×), red (2×→max)
        _fk = kelly_result['full_kelly']
        _max_lev = max(kelly_result['leverage_curve'][0]) if len(kelly_result['leverage_curve'][0]) else _fk * 3
        kelly_fig.add_vrect(x0=0, x1=_fk,
            fillcolor='rgba(38,166,154,0.10)', layer='below', line_width=0,
            annotation_text='✅ Optimal Zone', annotation_position='top left',
            annotation_font_color='#26A69A')
        kelly_fig.add_vrect(x0=_fk, x1=min(_fk*2, _max_lev),
            fillcolor='rgba(255,167,38,0.10)', layer='below', line_width=0,
            annotation_text='⚠️ Caution', annotation_position='top left',
            annotation_font_color='#FFA726')
        if _fk * 2 < _max_lev:
            kelly_fig.add_vrect(x0=_fk*2, x1=_max_lev,
                fillcolor='rgba(239,83,80,0.10)', layer='below', line_width=0,
                annotation_text='☠️ Ruin Territory', annotation_position='top left',
                annotation_font_color='#EF5350')
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
