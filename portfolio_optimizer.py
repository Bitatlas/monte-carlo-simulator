"""
Portfolio Optimization Engine
─────────────────────────────
Searches all N-asset combinations from a user-defined universe and ranks them
by Kelly + MPT metrics (Diversification Bonus, Sharpe, Portfolio Kelly, etc.).

Formulas used
─────────────
• Individual Kelly (long-only, analytical): f*ᵢ = max(0, (μᵢ − r) / σᵢ²)
• Nekrasov Portfolio Kelly (unconstrained): w* = Σ⁻¹μ  →  K* = μᵀ Σ⁻¹ μ
• Diversification Bonus: K* − max(f*ᵢ)
• Sharpe Ratio (annualised): (μ_p − r) / σ_p
"""

from __future__ import annotations

import itertools
import math
from datetime import date, timedelta
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ──────────────────────────────────────────────────────────────────────────────
# PRESET ASSET UNIVERSES
# ──────────────────────────────────────────────────────────────────────────────
PRESET_UNIVERSES: Dict[str, List[str]] = {
    "🇺🇸 US Large Cap Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V", "JNJ", "BRK-B",
        "XOM", "UNH", "PG", "MA", "HD",
    ],
    "📈 US Equity ETFs": [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "IJH", "VUG", "VTV", "SCHD", "ARKK",
    ],
    "🌍 Global / International ETFs": [
        "EWJ", "EWZ", "EWG", "EWU", "FXI", "EEM", "VEA", "EWA", "EWC", "INDA",
    ],
    "🔒 Bonds & Fixed Income": [
        "TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "BND", "VCIT", "MUB", "TIPS",
    ],
    "🥇 Commodities & Alternatives": [
        "GLD", "SLV", "USO", "UNG", "PDBC", "VNQ", "CORN", "WEAT", "DBA", "IAU",
    ],
    "🌈 Balanced Mix (Recommended)": [
        "SPY", "TLT", "GLD", "EEM", "QQQ", "IEF", "VNQ", "SLV", "EWJ", "HYG",
    ],
    "🔬 Factor ETFs": [
        "MTUM", "QUAL", "USMV", "VLUE", "SIZE", "IVW", "IVE", "SPHQ", "XSLV", "LRGF",
    ],
    "🚀 High Growth / Thematic": [
        "NVDA", "AMD", "TSLA", "PLTR", "SOFI", "SNOW", "DDOG", "NET", "CRWD", "ZS",
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# CORE CALCULATION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_universe_data(
    tickers: Tuple[str, ...],
    start_date: date,
    end_date: date,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Download adjusted-close prices for a list of tickers.

    Returns
    -------
    prices_df : pd.DataFrame  — 'Adj Close' for each valid ticker
    failed    : list[str]     — tickers that could not be downloaded
    """
    data: Dict[str, pd.Series] = {}
    failed: List[str] = []

    for ticker in tickers:
        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
            )
            if raw.empty:
                failed.append(ticker)
                continue

            # Handle MultiIndex columns (newer yfinance)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            col = "Adj Close" if "Adj Close" in raw.columns else "Close"
            series = raw[col].dropna()

            if len(series) < 60:   # need at least ~3 months of data
                failed.append(ticker)
                continue

            data[ticker] = series
        except Exception as exc:
            failed.append(ticker)
            print(f"[portfolio_optimizer] Failed {ticker}: {exc}")

    prices_df = pd.DataFrame(data)
    # Align all series to the common date index
    prices_df = prices_df.dropna(how="all")
    return prices_df, failed


def evaluate_portfolio_combination(
    asset_subset: List[str],
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> Dict[str, Any] | None:
    """
    Evaluate a portfolio using Kelly + MPT metrics.

    Nekrasov unconstrained Kelly weights: w* = Σ⁻¹μ
    Portfolio Kelly leverage:             K* = μᵀ Σ⁻¹ μ

    Returns None if the covariance matrix is degenerate or data insufficient.
    """
    subset_returns = returns_df[list(asset_subset)].dropna()
    if len(subset_returns) < 60:
        return None

    r_daily = risk_free_rate / 252.0

    # ── Daily excess returns (annualised for Sharpe; daily for Kelly) ──────
    mu_daily  = subset_returns.mean().values          # shape (n,)
    mu_excess = mu_daily - r_daily                    # excess over rf

    # ── Covariance matrix ──────────────────────────────────────────────────
    sigma = subset_returns.cov().values               # (n, n)

    # ── Nekrasov Kelly (unconstrained) ─────────────────────────────────────
    try:
        cond = np.linalg.cond(sigma)
        if cond > 1e12:
            sigma_inv = np.linalg.pinv(sigma)
        else:
            sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        sigma_inv = np.linalg.pinv(sigma)

    kelly_weights_raw = sigma_inv @ mu_excess         # can be negative
    portfolio_kelly   = float(mu_excess @ kelly_weights_raw)  # μᵀ Σ⁻¹ μ

    if not np.isfinite(portfolio_kelly):
        return None

    # ── Normalised (long-only) weights for Sharpe / display ────────────────
    w_pos = np.maximum(kelly_weights_raw, 0.0)
    w_sum = w_pos.sum()
    if w_sum < 1e-10:
        # Fall back to equal weight if all weights are ≤ 0
        w_pos = np.ones(len(asset_subset))
        w_sum = float(len(asset_subset))
    normalized_weights = w_pos / w_sum

    # ── Individual K* (same quadratic-form units as Portfolio K*) ─────────────
    # K*ᵢ = (μᵢ − r)² / σᵢ²  (Nekrasov single-asset case, Σ⁻¹ reduces to 1/σ²)
    # Annualised × 252 so values are O(0.1–2.0) rather than O(0.001)
    individual_kellys: Dict[str, float] = {}
    for i, ticker in enumerate(asset_subset):
        var_i = float(subset_returns[ticker].var())
        k_star_i = (mu_excess[i] ** 2) / var_i if var_i > 0 else 0.0
        individual_kellys[ticker] = round(max(0.0, k_star_i) * 252, 4)
    max_individual_kelly = max(individual_kellys.values(), default=0.0)

    # Annualise Portfolio K* to same scale
    portfolio_kelly_ann = max(0.0, portfolio_kelly) * 252

    # ── Diversification bonus (annualised K* units) ─────────────────────────
    div_bonus = portfolio_kelly_ann - max_individual_kelly

    # ── Annualised Sharpe for the normalised-weight portfolio ───────────────
    mu_ann    = (subset_returns.mean().values @ normalized_weights) * 252
    rf_ann    = risk_free_rate
    cov_ann   = subset_returns.cov().values * 252
    vol_ann   = float(np.sqrt(normalized_weights @ cov_ann @ normalized_weights))
    sharpe    = (mu_ann - rf_ann) / vol_ann if vol_ann > 0 else 0.0

    # ── Average pairwise correlation ────────────────────────────────────────
    corr_matrix = subset_returns.corr().values
    n = len(asset_subset)
    avg_correlation = (corr_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 0.0

    # ── Composite metrics (use annualised K* so numbers are meaningful) ──────
    kelly_sharpe_product  = portfolio_kelly_ann * sharpe
    risk_adjusted_kelly   = portfolio_kelly_ann / (1.0 + vol_ann) if vol_ann >= 0 else 0.0

    return {
        "assets":                tuple(asset_subset),
        "portfolio_kelly":       round(portfolio_kelly_ann, 4),
        "max_individual_kelly":  round(max_individual_kelly, 4),
        "diversification_bonus": round(div_bonus, 4),
        "sharpe_ratio":          round(sharpe, 4),
        "avg_correlation":       round(avg_correlation, 4),
        "kelly_sharpe_product":  round(kelly_sharpe_product, 4),
        "risk_adjusted_kelly":   round(risk_adjusted_kelly, 4),
        "kelly_weights":         kelly_weights_raw.tolist(),
        "normalized_weights":    normalized_weights.tolist(),
        "expected_return":       round(mu_ann, 4),
        "volatility":            round(vol_ann, 4),
        "individual_kellys":     individual_kellys,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN UI FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def portfolio_optimizer_tab() -> None:
    """Render the full Portfolio Optimization Engine UI."""

    st.title("🔍 Portfolio Optimization Engine")
    st.markdown("""
    Find the best Kelly + MPT asset combinations from your universe.  
    This tool searches **every possible N-asset portfolio** and ranks them by:
    - 🎯 **Diversification Bonus** — Portfolio Kelly > Individual Kelly (negative correlation benefit)
    - 📈 **Sharpe Ratio** — risk-adjusted return
    - ⚡ **Kelly Leverage** — how aggressively the math says to bet
    - 🔗 **Low Correlation** — assets that move independently
    """)

    # ── SECTION 1: ASSET UNIVERSE ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 1️⃣ Build Your Asset Universe")

    input_mode = st.radio(
        "How to define your universe:",
        ["🏷️ Preset Categories", "✏️ Custom Tickers", "📁 Upload CSV"],
        horizontal=True,
        key="po_input_mode",
    )

    asset_universe: List[str] = []

    if input_mode == "🏷️ Preset Categories":
        selected_presets = st.multiselect(
            "Select one or more preset categories:",
            options=list(PRESET_UNIVERSES.keys()),
            default=["🌈 Balanced Mix (Recommended)"],
            key="po_presets",
        )
        for cat in selected_presets:
            asset_universe += PRESET_UNIVERSES[cat]
        # Deduplicate while preserving order
        seen: set = set()
        asset_universe = [t for t in asset_universe if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]
        if asset_universe:
            st.info(f"✅ {len(asset_universe)} unique tickers from selected categories")

    elif input_mode == "✏️ Custom Tickers":
        ticker_input = st.text_area(
            "Enter tickers (comma or newline separated):",
            placeholder="SPY, TLT, GLD, QQQ, EEM\nor one per line",
            height=100,
            key="po_custom_tickers",
        )
        if ticker_input.strip():
            # Accept both comma and newline separators
            raw = ticker_input.replace("\n", ",").split(",")
            asset_universe = [t.strip().upper() for t in raw if t.strip()]
            st.info(f"✅ {len(asset_universe)} custom tickers entered")

    else:  # CSV upload
        uploaded_file = st.file_uploader(
            "Upload CSV with a 'ticker' column:",
            type=["csv"],
            key="po_csv_upload",
        )
        if uploaded_file:
            try:
                df_csv = pd.read_csv(uploaded_file)
                if "ticker" in df_csv.columns:
                    asset_universe = df_csv["ticker"].str.upper().dropna().tolist()
                    st.success(f"✅ Loaded {len(asset_universe)} tickers from CSV")
                else:
                    st.error("❌ CSV must have a column named 'ticker'")
            except Exception as exc:
                st.error(f"❌ Could not read CSV: {exc}")

    if asset_universe:
        with st.expander("📋 View Selected Assets", expanded=False):
            st.write(", ".join(asset_universe))

    # ── SECTION 2: CONFIGURATION & DATA DOWNLOAD ─────────────────────────────
    st.markdown("---")
    st.markdown("## 2️⃣ Configure & Download Data")

    cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
    end_date_default   = date.today()
    start_date_default = end_date_default - timedelta(days=5 * 365)

    with cfg_c1:
        start_date = st.date_input("Start date", value=start_date_default, key="po_start")
    with cfg_c2:
        end_date   = st.date_input("End date",   value=end_date_default,   key="po_end")
    with cfg_c3:
        po_rf_rate = st.slider(
            "Risk-free rate (%)", 0.0, 10.0, 2.0, 0.1, key="po_rf"
        ) / 100.0

    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        return

    if st.button("📊 Download & Analyze Data", type="primary", key="po_download"):
        if not asset_universe:
            st.error("❌ Please define your asset universe first (Section 1).")
        else:
            with st.spinner(f"Downloading data for {len(asset_universe)} assets…"):
                prices_df, failed = get_universe_data(
                    tuple(asset_universe), start_date, end_date
                )

            if failed:
                st.warning(f"⚠️ Could not download {len(failed)} asset(s): {', '.join(failed)}")

            if prices_df.empty:
                st.error("❌ No data downloaded. Check your tickers and date range.")
                return

            returns_df = prices_df.pct_change().dropna(how="all")

            # Drop columns with too many NaNs
            min_obs = 60
            valid_cols = [c for c in returns_df.columns if returns_df[c].notna().sum() >= min_obs]
            returns_df = returns_df[valid_cols].dropna()

            if len(valid_cols) < 2:
                st.error("❌ Need at least 2 assets with sufficient data.")
                return

            # Store in session state
            st.session_state["po_prices"]   = prices_df[valid_cols]
            st.session_state["po_returns"]  = returns_df
            st.session_state["po_tickers"]  = valid_cols
            st.session_state["po_rf_rate"]  = po_rf_rate
            # Clear old search results when data changes
            st.session_state.pop("po_results", None)

            st.success(
                f"✅ Downloaded data for **{len(valid_cols)} assets** "
                f"({returns_df.index[0].date()} → {returns_df.index[-1].date()}, "
                f"{len(returns_df):,} trading days)"
            )
            if len(valid_cols) < len(asset_universe) - len(failed):
                dropped = set(asset_universe) - set(valid_cols) - set(failed)
                if dropped:
                    st.info(f"ℹ️ Dropped {len(dropped)} asset(s) with < {min_obs} observations: {', '.join(dropped)}")

    # ── SECTION 3: CORRELATION ANALYSIS ──────────────────────────────────────
    if "po_returns" not in st.session_state:
        st.info("👆 Complete Section 2 to enable correlation analysis and portfolio search.")
        return

    returns_df  : pd.DataFrame = st.session_state["po_returns"]
    valid_tickers: List[str]   = st.session_state["po_tickers"]
    rf_rate      : float       = st.session_state.get("po_rf_rate", 0.02)

    st.markdown("---")
    st.markdown("## 3️⃣ Correlation Analysis")

    corr_full = returns_df.corr()

    # Heatmap
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_full.values,
        x=corr_full.columns.tolist(),
        y=corr_full.index.tolist(),
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_full.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig_heat.update_layout(
        title=f"Full Correlation Matrix ({len(valid_tickers)} assets)",
        height=max(400, 30 * len(valid_tickers)),
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_dark",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Pairs tables
    pairs: List[Dict] = []
    for i in range(len(corr_full.columns)):
        for j in range(i + 1, len(corr_full.columns)):
            pairs.append({
                "Asset A":     corr_full.columns[i],
                "Asset B":     corr_full.columns[j],
                "Correlation": corr_full.iloc[i, j],
            })
    pairs_df = pd.DataFrame(pairs).sort_values("Correlation")

    col_neg, col_pos = st.columns(2)
    with col_neg:
        st.subheader("📉 Most Negative Correlations")
        neg10 = pairs_df.head(10)
        st.dataframe(
            neg10.style
                .format({"Correlation": "{:.3f}"})
                .background_gradient(subset=["Correlation"], cmap="RdYlGn_r"),
            hide_index=True, use_container_width=True,
        )
        st.caption("💡 Negative pairs → maximum diversification bonus")

    with col_pos:
        st.subheader("🔗 Lowest Positive Correlations")
        low_pos = pairs_df[pairs_df["Correlation"] > 0].head(10)
        st.dataframe(
            low_pos.style
                .format({"Correlation": "{:.3f}"})
                .background_gradient(subset=["Correlation"], cmap="RdYlGn_r"),
            hide_index=True, use_container_width=True,
        )
        st.caption("💡 Low positive correlation also boosts the portfolio Kelly")

    # ── SECTION 4: PORTFOLIO SEARCH ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 4️⃣ Portfolio Search")

    s4c1, s4c2, s4c3 = st.columns(3)
    with s4c1:
        combo_size = st.number_input(
            "Portfolio size (N assets per combo)",
            min_value=2, max_value=min(5, len(valid_tickers)),
            value=min(3, len(valid_tickers)),
            step=1, key="po_combo_size",
        )
    with s4c2:
        max_corr_filter = st.slider(
            "Max avg pairwise correlation threshold",
            min_value=-1.0, max_value=1.0, value=1.0, step=0.05,
            key="po_max_corr",
            help="Only keep portfolios with avg pairwise correlation ≤ this value. "
                 "Start at 1.0 (no filter) and lower to focus on well-diversified combos.",
        )
    with s4c3:
        min_div_bonus = st.slider(
            "Min diversification bonus (annualised K*)",
            min_value=-5.0, max_value=10.0, value=-5.0, step=0.1,
            key="po_min_div",
            help="Diversification Bonus = Portfolio K* − Max Individual K* (annualised). "
                 "Positive means the portfolio beats the best single-asset Kelly. "
                 "Default -5 shows all portfolios; raise to filter for true diversification winners.",
        )

    optimization_metric = st.selectbox(
        "Rank portfolios by:",
        options=[
            "Diversification Bonus",
            "Portfolio Kelly Leverage",
            "Sharpe Ratio",
            "Kelly-Sharpe Product",
            "Risk-Adjusted Kelly",
        ],
        key="po_metric",
    )

    top_n = st.slider("Number of top portfolios to display", 5, 50, 10, key="po_top_n")

    # Combination count warning
    n_tickers  = len(valid_tickers)
    n_combos   = math.comb(n_tickers, int(combo_size))
    if n_combos > 100_000:
        st.error(
            f"⚠️ {n_combos:,} combinations would be tested — this is too large (> 100,000). "
            f"Please reduce universe size or portfolio size."
        )
    elif n_combos > 50_000:
        st.warning(f"⚠️ {n_combos:,} combinations — this may take a while. Consider reducing universe size.")
    else:
        st.info(f"🔢 Will test **{n_combos:,}** combinations of size {int(combo_size)}")

    if st.button("🔍 Search All Combinations", type="primary", key="po_search",
                 disabled=(n_combos > 100_000)):
        metric_key_map = {
            "Diversification Bonus":   "diversification_bonus",
            "Portfolio Kelly Leverage": "portfolio_kelly",
            "Sharpe Ratio":            "sharpe_ratio",
            "Kelly-Sharpe Product":    "kelly_sharpe_product",
            "Risk-Adjusted Kelly":     "risk_adjusted_kelly",
        }
        sort_key = metric_key_map[optimization_metric]

        all_combos = list(itertools.combinations(valid_tickers, int(combo_size)))
        results: List[Dict] = []

        # Debug counters — shown after search so we know where failures come from
        n_none = 0
        n_corr_fail = 0
        n_div_fail = 0
        first_sample: Dict | None = None   # keep one raw result for debugging

        progress_bar = st.progress(0)
        status_text  = st.empty()

        for idx, combo in enumerate(all_combos):
            if idx % max(1, n_combos // 200) == 0:
                pct = int(idx / n_combos * 100)
                progress_bar.progress(pct)
                status_text.text(f"Testing combination {idx:,} / {n_combos:,}…")

            res = evaluate_portfolio_combination(list(combo), returns_df, rf_rate)
            if res is None:
                n_none += 1
                continue
            if first_sample is None:
                first_sample = res        # capture first non-None result for debug

            if res["avg_correlation"] > max_corr_filter:
                n_corr_fail += 1
                continue
            if res["diversification_bonus"] < min_div_bonus:
                n_div_fail += 1
                continue
            results.append(res)

        progress_bar.empty()
        status_text.empty()

        # Always show diagnostic breakdown
        with st.expander("🔬 Search Diagnostics (click to expand)", expanded=(len(results) == 0)):
            st.markdown(f"""
| Stage | Count |
|---|---|
| Total combos tested | `{n_combos:,}` |
| Returned `None` (bad data) | `{n_none:,}` |
| Filtered out — avg correlation > {max_corr_filter} | `{n_corr_fail:,}` |
| Filtered out — div bonus < {min_div_bonus} | `{n_div_fail:,}` |
| **Passed all filters** | **`{len(results):,}`** |
""")
            if first_sample:
                st.markdown("**First non-None result (before filters):**")
                st.json({k: v for k, v in first_sample.items()
                         if k not in ("kelly_weights", "normalized_weights", "individual_kellys")})
            else:
                st.warning("All combinations returned None — check data quality below:")
                st.write(f"returns_df shape: {returns_df.shape}")
                st.write(f"returns_df NaN count: {returns_df.isna().sum().to_dict()}")
                st.write(f"returns_df first row: {returns_df.iloc[0].to_dict()}")

        if not results:
            st.warning(
                "⚠️ No portfolios met your criteria. "
                "See the Search Diagnostics expander above to understand why."
            )
            st.session_state.pop("po_results", None)
            return

        results_sorted = sorted(results, key=lambda x: x[sort_key], reverse=True)
        # Use distinct keys (_saved) to avoid collision with widget keys po_metric / po_top_n
        st.session_state["po_results"]       = results_sorted
        st.session_state["po_metric_saved"]  = optimization_metric
        st.session_state["po_top_n_saved"]   = top_n

        st.success(f"✅ Found **{len(results_sorted):,}** qualifying portfolios — showing top {top_n}!")

    # ── SECTION 5: RESULTS DISPLAY ────────────────────────────────────────────
    if "po_results" not in st.session_state:
        return

    results_sorted: List[Dict]  = st.session_state["po_results"]
    display_n                   = st.session_state.get("po_top_n_saved", top_n)
    top_results                 = results_sorted[:display_n]

    st.markdown("---")
    st.markdown(f"## 5️⃣ Results — Top {len(top_results)} Portfolios")

    # ── Scatter: Sharpe vs Diversification Bonus ─────────────────────────────
    scatter_df = pd.DataFrame([{
        "Rank":            i + 1,
        "Assets":          " + ".join(res["assets"]),
        "Div Bonus":       res["diversification_bonus"],
        "Sharpe":          res["sharpe_ratio"],
        "Kelly":           res["portfolio_kelly"],
        "Correlation":     res["avg_correlation"],
    } for i, res in enumerate(top_results)])

    fig_scatter = px.scatter(
        scatter_df,
        x="Sharpe", y="Div Bonus",
        size=[max(0.01, abs(k)) for k in scatter_df["Kelly"]],
        color="Correlation",
        hover_data=["Assets", "Rank"],
        title=f"Sharpe vs Diversification Bonus (Top {len(top_results)} Portfolios)",
        labels={"Div Bonus": "Diversification Bonus →", "Sharpe": "Sharpe Ratio →"},
        color_continuous_scale="RdYlGn_r",
        template="plotly_dark",
    )
    fig_scatter.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="white")))
    fig_scatter.update_layout(
        height=480,
        coloraxis_colorbar=dict(title="Avg Corr"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = []
    for i, res in enumerate(top_results):
        summary_rows.append({
            "Rank":           i + 1,
            "Assets":         " | ".join(res["assets"]),
            "Port. Kelly":    f"{res['portfolio_kelly']:.2f}×",
            "Div Bonus":      f"{res['diversification_bonus']:.2f}",
            "Sharpe":         f"{res['sharpe_ratio']:.2f}",
            "Exp. Return":    f"{res['expected_return']*100:.1f}%",
            "Volatility":     f"{res['volatility']*100:.1f}%",
            "Avg Corr":       f"{res['avg_correlation']:.3f}",
            "Kelly×Sharpe":   f"{res['kelly_sharpe_product']:.2f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # ── Detailed expander per portfolio ───────────────────────────────────────
    st.markdown("### 🔎 Detailed View")
    for i, res in enumerate(top_results[:min(10, len(top_results))]):
        label = f"#{i+1}  {' + '.join(res['assets'])}  —  Div Bonus: {res['diversification_bonus']:.2f}  |  Sharpe: {res['sharpe_ratio']:.2f}  |  Kelly: {res['portfolio_kelly']:.2f}×"
        with st.expander(label, expanded=(i == 0)):
            dc1, dc2, dc3, dc4, dc5 = st.columns(5)
            dc1.metric("Portfolio Kelly", f"{res['portfolio_kelly']:.2f}×",
                       help="Nekrasov K* = μᵀ Σ⁻¹ μ")
            dc2.metric("Div Bonus",       f"{res['diversification_bonus']:.2f}",
                       help="Portfolio Kelly − Max Individual Kelly")
            dc3.metric("Sharpe Ratio",    f"{res['sharpe_ratio']:.2f}")
            dc4.metric("Exp. Return",     f"{res['expected_return']*100:.1f}%")
            dc5.metric("Volatility",      f"{res['volatility']*100:.1f}%")

            # Individual Kellys
            st.markdown("**Individual Kelly Ratios (long-only normalised weights):**")
            raw_kw = res.get("kelly_weights", [])
            ik_rows = []
            for j, (k, v) in enumerate(res["individual_kellys"].items()):
                w_pct = res["normalized_weights"][j] * 100
                raw_w = raw_kw[j] if j < len(raw_kw) else 0.0
                note = ""
                if raw_w < 0:
                    note = "⚠️ Model suggests short; set to 0 in long-only mode"
                elif w_pct < 0.1:
                    note = "Near-zero contribution"
                ik_rows.append({
                    "Asset": k,
                    "Kelly K* (individual)": f"{v:.2f}",
                    "Unconstrained Weight": f"{raw_w:.2f}×",
                    "Long-Only Weight": f"{w_pct:.1f}%",
                    "Note": note,
                })
            ik_df = pd.DataFrame(ik_rows)
            st.dataframe(ik_df, hide_index=True, use_container_width=True)
            if any(r.get("kelly_weights", [i])[i] < 0 for i, r in [(j, res)] for j in range(len(res["assets"]))):
                st.caption(
                    "⚠️ **Short position detected**: the Nekrasov unconstrained formula says to short "
                    "one or more assets (negative weight). In long-only mode these are set to 0%. "
                    "The Portfolio Kelly metric still reflects the full unconstrained opportunity — "
                    "if you can short, the benefit is real."
                )

            # Kelly-weights bar chart
            asset_names = list(res["assets"])
            norm_w      = res["normalized_weights"]
            fig_bar = go.Figure(go.Bar(
                x=asset_names, y=[w * 100 for w in norm_w],
                marker_color="#1E88E5",
                text=[f"{w*100:.1f}%" for w in norm_w],
                textposition="auto",
            ))
            fig_bar.update_layout(
                title="Kelly-Derived Portfolio Weights (long-only normalised)",
                yaxis_title="Weight (%)", template="plotly_dark",
                height=280, margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Correlation heatmap for this portfolio
            sub_corr = returns_df[list(asset_names)].corr()
            fig_sub_heat = go.Figure(data=go.Heatmap(
                z=sub_corr.values,
                x=sub_corr.columns.tolist(),
                y=sub_corr.index.tolist(),
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                text=np.round(sub_corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 14},
            ))
            fig_sub_heat.update_layout(
                title=f"Correlation Matrix (avg: {res['avg_correlation']:.3f})",
                height=350, template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_sub_heat, use_container_width=True)

    # ── SECTION 6: EXPORT ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 6️⃣ Export Results")

    export_df = pd.DataFrame([{
        "Rank":                  i + 1,
        "Assets":                "|".join(res["assets"]),
        "Portfolio_Kelly":       res["portfolio_kelly"],
        "Max_Individual_Kelly":  res["max_individual_kelly"],
        "Diversification_Bonus": res["diversification_bonus"],
        "Sharpe_Ratio":          res["sharpe_ratio"],
        "Kelly_Sharpe_Product":  res["kelly_sharpe_product"],
        "Risk_Adjusted_Kelly":   res["risk_adjusted_kelly"],
        "Expected_Return_Ann":   res["expected_return"],
        "Volatility_Ann":        res["volatility"],
        "Avg_Correlation":       res["avg_correlation"],
        "Normalized_Weights":    "|".join(f"{w:.4f}" for w in res["normalized_weights"]),
    } for i, res in enumerate(results_sorted[:200])])

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Top Portfolios as CSV",
        data=csv_bytes,
        file_name="portfolio_optimization_results.csv",
        mime="text/csv",
        key="po_export",
    )
    st.caption(
        f"Exporting top {min(200, len(results_sorted))} portfolios out of {len(results_sorted):,} qualifying results. "
        f"Kelly values are annualised K* = μᵀ Σ⁻¹ μ × 252 (same scale for individual and portfolio)."
    )
