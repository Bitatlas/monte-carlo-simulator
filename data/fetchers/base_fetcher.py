import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

class BaseFetcher(ABC):
    """
    Abstract base class for all data fetchers.
    
    Defines the interface that all specific asset fetchers must implement.
    """
    
    def __init__(self, period="max"):
        """
        Initialize the base fetcher.
        
        Parameters:
        -----------
        period : str
            The time period to fetch data for (default: "max")
            Valid options include: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
        """
        self.period = period
        self.data = None
        self.returns = None
        self.log_returns = None
        self.name = "Base Asset"  # Should be overridden by subclasses
        self.ticker = None  # Should be overridden by subclasses
        
    @abstractmethod
    def fetch_data(self):
        """
        Fetch historical price data for the asset.
        Must be implemented by subclasses.
        
        Returns:
        --------
        pandas.DataFrame
            Historical price data
        """
        pass
    
    def calculate_returns(self):
        """
        Calculate daily, monthly, and annual returns from the price data.
        
        Returns:
        --------
        dict
            Dictionary containing different return series
        """
        if self.data is None:
            self.fetch_data()
            
        # Identify the price column
        price_col = self.get_price_column()
        
        # Calculate daily returns
        self.returns = self.data[price_col].pct_change().dropna()
        
        # Calculate log returns (useful for statistical analysis)
        self.log_returns = np.log(1 + self.returns).dropna()
        
        # Calculate monthly returns
        self.monthly_returns = self.data[price_col].resample('ME').ffill().pct_change().dropna()
        
        # Calculate annual returns
        self.annual_returns = self.data[price_col].resample('YE').ffill().pct_change().dropna()
        
        return {
            'daily': self.returns,
            'monthly': self.monthly_returns,
            'annual': self.annual_returns,
            'log_daily': self.log_returns
        }
    
    def get_statistics(self):
        """
        Calculate key statistics from the returns data.
        
        Returns:
        --------
        dict
            Dictionary of statistics
        """
        if self.returns is None:
            self.calculate_returns()
            
        # Convert pandas Series values to floats
        mean_daily = float(self.returns.mean())
        std_daily = float(self.returns.std())
        skew = float(self.returns.skew())
        kurtosis = float(self.returns.kurtosis())
        
        # Extract data period years for benchmark calculations
        data_period_years = self._extract_data_period_years()
        
        stats = {
            'mean_daily': mean_daily,
            'std_daily': std_daily,
            'mean_annual': mean_daily * 252,  # Approximate trading days in a year
            'std_annual': std_daily * np.sqrt(252),
            'skew': skew,
            'kurtosis': kurtosis,
            'sharpe_ratio': (mean_daily * 252) / (std_daily * np.sqrt(252)),
            'data_period_years': data_period_years  # Add period information for benchmark
        }
        
        return stats
        
    def _extract_data_period_years(self):
        """
        Extract the number of years in the data period.
        
        Returns:
        --------
        float or int
            Number of years in the data period
        """
        if self.period == "max" and self.data is not None and len(self.data) > 0:
            # Calculate years from actual data timespan
            start_date = self.data.index[0]
            end_date = self.data.index[-1]
            days = (end_date - start_date).days
            return round(days / 365.25, 1)  # Round to 1 decimal place
        elif self.period == "ytd":
            # Year to date - fraction of current year
            today = datetime.now()
            start_of_year = datetime(today.year, 1, 1)
            days = (today - start_of_year).days
            return round(days / 365.25, 1)
        elif self.period.endswith('d'):
            # Days period (e.g., "5d")
            try:
                days = int(self.period[:-1])
                return round(days / 365.25, 2)
            except:
                return 0.01  # Default to a very small period
        elif self.period.endswith('mo'):
            # Months period (e.g., "3mo")
            try:
                months = int(self.period[:-2])
                return round(months / 12, 1)
            except:
                return 0.25  # Default to 3 months
        elif self.period.endswith('y'):
            # Years period (e.g., "5y")
            try:
                return int(self.period[:-1])
            except:
                return 1  # Default to 1 year
        else:
            # Default fallback
            return 5  # Assume 5 years if unknown format
    
    def get_data_for_simulation(self):
        """
        Prepare data for Monte Carlo simulation.
        
        Returns:
        --------
        dict
            Dictionary containing data for simulation
        """
        if self.returns is None:
            self.calculate_returns()
            
        return {
            'returns': self.returns,
            'log_returns': self.log_returns,
            'statistics': self.get_statistics(),
            'name': self.name,
            'ticker': self.ticker
        }
    
    def get_price_column(self):
        """
        Identify the price column in the data.
        
        Returns:
        --------
        str
            Name of the price column
        """
        # Default implementation - override if needed
        if 'Adj Close' in self.data.columns:
            return 'Adj Close'
        elif 'Close' in self.data.columns:
            return 'Close'
        elif 'Price' in self.data.columns:
            return 'Price'
        else:
            # If none of the expected columns are found, use the first numeric column
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return numeric_cols[0]
            else:
                raise ValueError("No suitable price column found in the data")
