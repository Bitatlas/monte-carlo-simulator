import yfinance as yf
import pandas as pd
import numpy as np
from .base_fetcher import BaseFetcher

class BondFetcher(BaseFetcher):
    """
    Data fetcher for bonds, focused on US Treasury bonds.
    """
    
    def __init__(self, bond_type="US10Y", period="max"):
        """
        Initialize the bond fetcher.
        
        Parameters:
        -----------
        bond_type : str
            The bond type to fetch. Options:
            - "US10Y": 10-Year US Treasury Yield
            - "US30Y": 30-Year US Treasury Yield
            - "US3M": 3-Month US Treasury Yield
            - "TLT": iShares 20+ Year Treasury Bond ETF
            - "IEF": iShares 7-10 Year Treasury Bond ETF
            - "SHY": iShares 1-3 Year Treasury Bond ETF
        period : str
            The time period to fetch data for (default: "max")
        """
        super().__init__(period=period)
        self.bond_type = bond_type
        
        # Define bond tickers and names
        self.bond_map = {
            "US10Y": {"ticker": "^TNX", "name": "10-Year US Treasury Yield"},
            "US30Y": {"ticker": "^TYX", "name": "30-Year US Treasury Yield"},
            "US3M": {"ticker": "^IRX", "name": "3-Month US Treasury Yield"},
            "TLT": {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond ETF"},
            "IEF": {"ticker": "IEF", "name": "iShares 7-10 Year Treasury Bond ETF"},
            "SHY": {"ticker": "SHY", "name": "iShares 1-3 Year Treasury Bond ETF"},
        }
        
        # Set ticker and name based on selected bond
        if bond_type in self.bond_map:
            self.ticker = self.bond_map[bond_type]["ticker"]
            self.name = self.bond_map[bond_type]["name"]
        else:
            raise ValueError(f"Unsupported bond type: {bond_type}. "
                             f"Supported types are: {list(self.bond_map.keys())}")
        
        # Flag to determine if this is a yield or price series
        self.is_yield = self.ticker.startswith('^')
    
    def fetch_data(self):
        """
        Fetch historical data for the selected bond.
        
        Returns:
        --------
        pandas.DataFrame
            Historical price or yield data
        """
        self.data = yf.download(self.ticker, period=self.period, auto_adjust=False)
        return self.data
    
    def calculate_returns(self):
        """
        Calculate returns, handling the differences between yields and prices.
        For yields, we use the negative of the yield changes for returns calculation.
        
        Returns:
        --------
        dict
            Dictionary containing different return series
        """
        if self.data is None:
            self.fetch_data()
            
        # Identify the price column
        price_col = self.get_price_column()
        
        if self.is_yield:
            # For yields, we use yield changes but with opposite sign
            # (bond prices move opposite to yields)
            yield_changes = self.data[price_col].diff()
            # Normalize by dividing by yield level to get something like returns
            self.returns = -yield_changes / self.data[price_col].shift(1)
            self.returns = self.returns.dropna()
        else:
            # For bond ETFs, calculate returns normally
            self.returns = self.data[price_col].pct_change().dropna()
        
        # Calculate log returns
        self.log_returns = np.log(1 + self.returns).dropna()
        
        # Calculate monthly returns
        if self.is_yield:
            monthly_changes = self.data[price_col].resample('ME').ffill().diff()
            monthly_levels = self.data[price_col].resample('ME').ffill().shift(1)
            self.monthly_returns = -monthly_changes / monthly_levels
            self.monthly_returns = self.monthly_returns.dropna()
        else:
            self.monthly_returns = self.data[price_col].resample('ME').ffill().pct_change().dropna()
        
        # Calculate annual returns
        if self.is_yield:
            annual_changes = self.data[price_col].resample('YE').ffill().diff()
            annual_levels = self.data[price_col].resample('YE').ffill().shift(1)
            self.annual_returns = -annual_changes / annual_levels
            self.annual_returns = self.annual_returns.dropna()
        else:
            self.annual_returns = self.data[price_col].resample('YE').ffill().pct_change().dropna()
        
        return {
            'daily': self.returns,
            'monthly': self.monthly_returns,
            'annual': self.annual_returns,
            'log_daily': self.log_returns
        }
    
    def get_statistics(self):
        """
        Calculate key statistics, with considerations for bond-specific metrics.
        
        Returns:
        --------
        dict
            Dictionary of statistics
        """
        stats = super().get_statistics()
        
        # Add bond-specific metrics
        if not self.is_yield:
            # For bond ETFs, calculate yield-like metrics
            price_col = self.get_price_column()
            current_price = float(self.data[price_col].iloc[-1])
            stats['current_price'] = current_price
            
        return stats
    
    def get_current_yield(self):
        """
        Get the most recent yield value for yield series.
        
        Returns:
        --------
        float
            Current yield as a percentage
        """
        if not self.is_yield:
            return None
            
        price_col = self.get_price_column()
        return float(self.data[price_col].iloc[-1])
