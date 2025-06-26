import yfinance as yf
import pandas as pd
import numpy as np
from .base_fetcher import BaseFetcher

class EquityIndexFetcher(BaseFetcher):
    """
    Data fetcher for equity indices like S&P 500, Nasdaq 100, etc.
    """
    
    def __init__(self, index_type="SP500", period="max"):
        """
        Initialize the equity index fetcher.
        
        Parameters:
        -----------
        index_type : str
            The index to fetch. Options:
            - "SP500": S&P 500 Index
            - "NASDAQ": Nasdaq 100 Index
            - "EURO_STOXX50": Euro Stoxx 50 Index
            - "STOXX600": STOXX Europe 600
        period : str
            The time period to fetch data for (default: "max")
        """
        super().__init__(period=period)
        self.index_type = index_type
        
        # Define index tickers and names
        self.index_map = {
            "SP500": {"ticker": "^GSPC", "name": "S&P 500"},
            "NASDAQ": {"ticker": "^NDX", "name": "Nasdaq 100"},
            "EURO_STOXX50": {"ticker": "^STOXX50E", "name": "Euro Stoxx 50"},
            "STOXX600": {"ticker": "^STOXX", "name": "STOXX Europe 600"},
        }
        
        # Set ticker and name based on selected index
        if index_type in self.index_map:
            self.ticker = self.index_map[index_type]["ticker"]
            self.name = self.index_map[index_type]["name"]
        else:
            raise ValueError(f"Unsupported index type: {index_type}. "
                             f"Supported types are: {list(self.index_map.keys())}")
    
    def fetch_data(self):
        """
        Fetch historical data for the selected equity index.
        
        Returns:
        --------
        pandas.DataFrame
            Historical price data
        """
        self.data = yf.download(self.ticker, period=self.period, auto_adjust=False)
        return self.data

class StockFetcher(BaseFetcher):
    """
    Data fetcher for individual stocks.
    """
    
    def __init__(self, ticker, period="max"):
        """
        Initialize the stock fetcher.
        
        Parameters:
        -----------
        ticker : str
            The stock ticker symbol (e.g., "AAPL" for Apple Inc.)
        period : str
            The time period to fetch data for (default: "max")
        """
        super().__init__(period=period)
        self.ticker = ticker
        
        # Get the company name from yfinance
        try:
            ticker_info = yf.Ticker(ticker).info
            if 'shortName' in ticker_info:
                self.name = ticker_info['shortName']
            elif 'longName' in ticker_info:
                self.name = ticker_info['longName']
            else:
                self.name = ticker
        except:
            # If we can't get the name, just use the ticker
            self.name = ticker
    
    def fetch_data(self):
        """
        Fetch historical data for the stock.
        
        Returns:
        --------
        pandas.DataFrame
            Historical price data
        """
        self.data = yf.download(self.ticker, period=self.period, auto_adjust=False)
        return self.data
