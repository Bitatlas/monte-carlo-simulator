import yfinance as yf
import pandas as pd
import numpy as np
from .base_fetcher import BaseFetcher


class SectorETFFetcher(BaseFetcher):
    """
    Data fetcher for US Sector ETFs and bond ETFs.
    """

    SECTOR_MAP = {
        "XLK":  {"name": "Technology Select Sector (XLK)"},
        "XLF":  {"name": "Financials Select Sector (XLF)"},
        "XLE":  {"name": "Energy Select Sector (XLE)"},
        "XLV":  {"name": "Health Care Select Sector (XLV)"},
        "XLY":  {"name": "Consumer Discretionary (XLY)"},
        "XLP":  {"name": "Consumer Staples (XLP)"},
        "XLI":  {"name": "Industrials Select Sector (XLI)"},
        "XLU":  {"name": "Utilities Select Sector (XLU)"},
        "XLB":  {"name": "Materials Select Sector (XLB)"},
        "XLC":  {"name": "Communication Services (XLC)"},
        "CLRE": {"name": "Return Stacked Bonds & Managed Futures (CLRE)"},
        "TLT":  {"name": "iShares 20+ Year Treasury Bond ETF (TLT)"},
    }

    def __init__(self, etf_ticker="XLK", period="max"):
        super().__init__(period=period)
        if etf_ticker not in self.SECTOR_MAP:
            raise ValueError(
                f"Unsupported ETF: {etf_ticker}. "
                f"Supported tickers: {list(self.SECTOR_MAP.keys())}"
            )
        self.ticker = etf_ticker
        self.name = self.SECTOR_MAP[etf_ticker]["name"]

    def fetch_data(self):
        """
        Fetch historical price data for the ETF.
        """
        self.data = yf.download(self.ticker, period=self.period, auto_adjust=False)
        # Handle newer yfinance MultiIndex columns (single ticker download)
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
        return self.data
