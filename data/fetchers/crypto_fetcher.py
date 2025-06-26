import yfinance as yf
import pandas as pd
import numpy as np
from .base_fetcher import BaseFetcher

class CryptoFetcher(BaseFetcher):
    """
    Data fetcher for cryptocurrencies like Bitcoin.
    """
    
    def __init__(self, crypto_type="BTC", period="max"):
        """
        Initialize the cryptocurrency fetcher.
        
        Parameters:
        -----------
        crypto_type : str
            The cryptocurrency to fetch. Options:
            - "BTC": Bitcoin
            - "ETH": Ethereum
        period : str
            The time period to fetch data for (default: "max")
        """
        super().__init__(period=period)
        self.crypto_type = crypto_type
        
        # Define crypto tickers and names
        self.crypto_map = {
            "BTC": {"ticker": "BTC-USD", "name": "Bitcoin"},
            "ETH": {"ticker": "ETH-USD", "name": "Ethereum"},
        }
        
        # Set ticker and name based on selected cryptocurrency
        if crypto_type in self.crypto_map:
            self.ticker = self.crypto_map[crypto_type]["ticker"]
            self.name = self.crypto_map[crypto_type]["name"]
        else:
            raise ValueError(f"Unsupported cryptocurrency type: {crypto_type}. "
                             f"Supported types are: {list(self.crypto_map.keys())}")
    
    def fetch_data(self):
        """
        Fetch historical data for the selected cryptocurrency.
        
        Returns:
        --------
        pandas.DataFrame
            Historical price data
        """
        self.data = yf.download(self.ticker, period=self.period, auto_adjust=False)
        return self.data
    
    def get_statistics(self):
        """
        Calculate key statistics from the returns data.
        Override to handle the higher volatility of cryptocurrencies.
        
        Returns:
        --------
        dict
            Dictionary of statistics
        """
        stats = super().get_statistics()
        
        # Add specific crypto metrics if needed
        stats['max_drawdown'] = self.calculate_max_drawdown()
        stats['volatility_ratio'] = stats['std_annual'] / abs(stats['mean_annual'])
        
        return stats
    
    def calculate_max_drawdown(self):
        """
        Calculate the historical maximum drawdown.
        
        Returns:
        --------
        float
            Maximum drawdown as a positive decimal
        """
        if self.returns is None:
            self.calculate_returns()
            
        # Get price column
        price_col = self.get_price_column()
        
        # Calculate cumulative returns
        cum_returns = (1 + self.returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdowns
        drawdowns = (cum_returns / running_max) - 1
        
        # Get maximum drawdown
        max_dd = float(drawdowns.min())
        
        return abs(max_dd)
