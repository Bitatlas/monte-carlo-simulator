# Multi-Asset Monte Carlo Simulator

A sophisticated financial simulator for analyzing and projecting returns across multiple asset classes, with advanced mathematical models and visualization tools.

![Monte Carlo Simulation](https://img.shields.io/badge/Monte%20Carlo-Simulation-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)

## Features

- **Multi-Asset Support**: Simulate equity indices, individual stocks, cryptocurrencies, and bonds
- **Advanced Mathematical Models**:
  - Standard Monte Carlo
  - Geometric Brownian Motion (GBM)
  - GARCH(1,1) for volatility clustering
  - Markov Chain for regime-switching
  - Feynman Path Integral for complex dynamics
- **Kelly Criterion Analysis**: Calculate optimal leverage for long-term growth
- **Comprehensive Statistics**:
  - Return distributions
  - Maximum drawdown analysis
  - Sharpe ratio comparisons
  - Confidence intervals
- **Interactive Visualizations**:
  - Dynamic price charts
  - Distribution plots
  - Sharpe ratio comparisons

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/monte-carlo-simulator.git
cd monte-carlo-simulator

# Install dependencies
pip install -r requirements.txt

# Optional: Install GARCH dependencies
pip install arch
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

Then open your browser to http://localhost:8501

## How to Use

1. **Select Asset & Parameters** in the sidebar
   - Choose an asset type and specific asset
   - Set the historical data period
   - Configure investment amount and time horizon

2. **Choose a Mathematical Model**
   - Different models have different assumptions and use cases

3. **Set Leverage Method**
   - Manual: Set leverage directly
   - Kelly Criterion: Optimal leverage for long-term growth
   - Fractional Kelly: More conservative approach
   - Numerical Optimization: Finds best leverage through simulation

4. **Run the Simulation**
   - Generate thousands of possible future scenarios
   - Analyze the results across different tabs

## Mathematical Background

The simulator implements various stochastic processes to model asset price movements:

- **Standard Monte Carlo**: Samples returns from a normal distribution
- **Geometric Brownian Motion**: Uses the SDE dS = μS dt + σS dW
- **GARCH(1,1)**: Models volatility clustering with σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁
- **Markov Chain**: Captures regime-switching behavior
- **Feynman Path Integral**: Quantum-inspired approach for complex dynamics

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- yfinance
- scikit-learn
- Optional: arch (for GARCH models)

## Deployment

This application can be deployed using:

- Streamlit Cloud
- Docker containers
- Firebase with Cloud Run
- Heroku or other PaaS providers

## License

MIT License

## Disclaimer

This application is for educational purposes only and does not constitute investment advice. Past performance is not indicative of future results. Leveraged investing involves significant risks.
