import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class PortfolioMonteCarloModel:
    """
    Monte Carlo simulation for a multi-asset portfolio with optional
    periodic rebalancing and dollar-cost averaging (DCA).

    Uses Cholesky decomposition of the historical covariance matrix to
    generate correlated daily returns, so assets move together in a
    realistic way.
    """

    # Approximate trading days per calendar period
    PERIOD_DAYS = {
        'monthly':   21,
        'quarterly': 63,
        'annually':  252,
    }

    def __init__(
        self,
        assets_data,
        weights,
        investment_amount=10_000,
        time_horizon_years=10,
        num_simulations=500,
        trading_days_per_year=252,
        rebalancing_frequency=None,   # None | 'monthly' | 'quarterly' | 'annually'
        transaction_cost=0.001,       # 0.001 = 0.1 % per dollar traded
        dca_amount=0,                 # periodic contribution in $
        dca_frequency=None,           # None | 'monthly' | 'quarterly'
    ):
        """
        Parameters
        ----------
        assets_data : list[dict]
            Each dict is the output of fetcher.get_data_for_simulation()
            and must contain at least 'returns' (pd.Series) and 'name'.
        weights : list[float]
            Target allocation for each asset (must sum to 1.0).
        """
        self.assets_data = assets_data
        self.weights = np.array(weights, dtype=float)
        self.weights /= self.weights.sum()          # normalise just in case

        self.investment_amount = float(investment_amount)
        self.time_horizon_years = int(time_horizon_years)
        self.num_simulations = int(num_simulations)
        self.trading_days_per_year = trading_days_per_year
        self.total_days = self.time_horizon_years * trading_days_per_year

        self.rebalancing_frequency = rebalancing_frequency
        self.transaction_cost = float(transaction_cost)
        self.dca_amount = float(dca_amount)
        self.dca_frequency = dca_frequency

        self._prepare_returns()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_returns(self):
        """Align returns by date and compute covariance / Cholesky."""
        series_list = [ad['returns'] for ad in self.assets_data]
        names = [ad['name'] for ad in self.assets_data]

        combined = pd.concat(series_list, axis=1, join='inner')
        combined.columns = names
        combined = combined.dropna()

        if len(combined) < 30:
            raise ValueError(
                "Insufficient overlapping history between the selected assets. "
                "Try a longer data period or different assets."
            )

        self.aligned_returns = combined
        self.n_assets = len(self.assets_data)
        self.asset_names = names

        self.means = combined.mean().values           # daily mean returns
        self.stds = combined.std().values             # daily std devs
        self.corr_matrix = combined.corr().values     # correlation matrix
        cov = combined.cov().values

        # Ensure positive-definite (add tiny diagonal if needed)
        for _ in range(5):
            try:
                self.cholesky = np.linalg.cholesky(cov)
                break
            except np.linalg.LinAlgError:
                cov += np.eye(self.n_assets) * 1e-7

    def _event_days(self, frequency):
        """Return a set of day indices when an event fires."""
        if frequency is None:
            return set()
        period = self.PERIOD_DAYS.get(frequency, 0)
        if period == 0:
            return set()
        return set(range(period, self.total_days, period))

    # ------------------------------------------------------------------
    # Public simulation entry point
    # ------------------------------------------------------------------

    def simulate(self):
        """
        Run the portfolio simulation.

        Returns
        -------
        dict with keys:
            stats, paths, total_invested, correlation_matrix,
            asset_names, weights, investment_amount, time_horizon_years,
            num_simulations
        """
        n = self.n_assets
        S = self.num_simulations
        rebal_days = self._event_days(self.rebalancing_frequency)
        dca_days   = self._event_days(self.dca_frequency)

        # Initialise holdings matrix: (S, n_assets)
        portfolio = np.outer(
            np.ones(S),
            self.investment_amount * self.weights
        )  # shape (S, n)

        # Record total portfolio value at each step: (S, total_days+1)
        portfolio_values = np.empty((S, self.total_days + 1))
        portfolio_values[:, 0] = self.investment_amount

        # Track total cash put in (for CAGR vs total-invested)
        total_invested = np.full(S, self.investment_amount)

        # Track total transaction costs paid
        total_costs = np.zeros(S)

        for day in range(self.total_days):
            # --- generate correlated daily returns ---
            Z = np.random.standard_normal((S, n))          # uncorrelated
            corr_returns = Z @ self.cholesky.T              # correlated

            portfolio *= (1.0 + corr_returns)

            # --- DCA: inject new cash at target weights ---
            if day in dca_days and self.dca_amount > 0:
                contrib = self.dca_amount * self.weights    # shape (n,)
                portfolio += contrib[np.newaxis, :]
                total_invested += self.dca_amount

            # --- Rebalancing ---
            if day in rebal_days:
                row_totals = portfolio.sum(axis=1)          # (S,)
                targets = row_totals[:, np.newaxis] * self.weights[np.newaxis, :]

                if self.transaction_cost > 0:
                    # Cost = transaction_cost * sum of absolute trades
                    traded = np.abs(portfolio - targets)    # (S, n)
                    cost   = traded.sum(axis=1) * self.transaction_cost  # (S,)
                    total_costs  += cost
                    row_totals   -= cost
                    targets = row_totals[:, np.newaxis] * self.weights[np.newaxis, :]

                portfolio = targets

            portfolio_values[:, day + 1] = portfolio.sum(axis=1)

        final_values = portfolio_values[:, -1]
        stats = self._calc_stats(final_values, portfolio_values, total_invested, total_costs)
        paths_df = self._make_paths_df(portfolio_values)

        return {
            'stats':             stats,
            'paths':             paths_df,
            'total_invested':    float(total_invested.mean()),
            'correlation_matrix': self.corr_matrix,
            'asset_names':       self.asset_names,
            'weights':           self.weights.tolist(),
            'investment_amount': self.investment_amount,
            'time_horizon_years': self.time_horizon_years,
            'num_simulations':   self.num_simulations,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _calc_stats(self, final_values, portfolio_values, total_invested, total_costs):
        avg_invested = float(total_invested.mean())
        total_paths  = len(final_values)

        pcts = {k: float(np.percentile(final_values, v))
                for k, v in [('1%', 1), ('5%', 5), ('10%', 10), ('25%', 25),
                              ('50%', 50), ('75%', 75), ('90%', 90), ('95%', 95), ('99%', 99)]}

        cagr_vals = (final_values / avg_invested) ** (1.0 / self.time_horizon_years) - 1.0

        ruin_threshold   = avg_invested * 0.01
        bust_ruin        = int(np.sum(final_values < ruin_threshold))
        bust_below       = int(np.sum((final_values >= ruin_threshold) & (final_values < avg_invested)))
        paths_above      = int(np.sum(final_values >= avg_invested))

        running_max  = np.maximum.accumulate(portfolio_values, axis=1)
        drawdowns    = portfolio_values / running_max - 1.0
        max_dd       = np.abs(np.min(drawdowns, axis=1))

        return {
            'mean':   float(np.mean(final_values)),
            'median': float(np.median(final_values)),
            'std':    float(np.std(final_values)),
            'min':    float(np.min(final_values)),
            'max':    float(np.max(final_values)),
            'total_invested':         avg_invested,
            'avg_transaction_costs':  float(total_costs.mean()),
            'percentiles': pcts,
            'cagr': {
                'mean':   float(np.mean(cagr_vals)),
                'median': float(np.median(cagr_vals)),
                'percentiles': {
                    '5%':  float(np.percentile(cagr_vals, 5)),
                    '95%': float(np.percentile(cagr_vals, 95)),
                },
            },
            'bust_counters': {
                'total_ruin':         bust_ruin,
                'below_initial':      bust_below,
                'above_initial':      paths_above,
                'total_ruin_pct':     bust_ruin  / total_paths,
                'below_initial_pct':  bust_below / total_paths,
                'above_initial_pct':  paths_above / total_paths,
                'total_paths':        total_paths,
                'ruin_threshold':     float(ruin_threshold),
            },
            'max_drawdown': {
                'mean':   float(np.mean(max_dd)),
                'median': float(np.median(max_dd)),
                'max':    float(np.max(max_dd)),
                'percentiles': {'95%': float(np.percentile(max_dd, 95))},
            },
        }

    # ------------------------------------------------------------------
    # Paths dataframe for charts
    # ------------------------------------------------------------------

    def _make_paths_df(self, portfolio_values):
        start = datetime.now()
        dates = [
            start + timedelta(days=i * 365.25 / self.trading_days_per_year)
            for i in range(self.total_days + 1)
        ]
        n_show = min(100, self.num_simulations)
        idx    = np.random.choice(portfolio_values.shape[0], n_show, replace=False)
        cols   = [f"Path_{i}" for i in range(n_show)]
        return pd.DataFrame(portfolio_values[idx, :], columns=dates, index=cols).T
