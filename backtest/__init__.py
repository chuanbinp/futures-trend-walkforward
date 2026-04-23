"""Backtest package (refactor of `Backtest.py`)."""

from .core import (
    BacktestRunParams,
    GridSearchResult,
    compute_stats,
    rolling_hh_ll,
    run_channel_with_dd_control,
    run_grid_search,
    run_grid_search_with_params,
)
from .data import (
    calculate_bars_back,
    filter_session,
    get_period_indices,
    load_ohlcv_csv,
    prepare_market_dataframe,
)

__all__ = [
    "BacktestRunParams",
    "GridSearchResult",
    "calculate_bars_back",
    "compute_stats",
    "filter_session",
    "get_period_indices",
    "load_ohlcv_csv",
    "prepare_market_dataframe",
    "rolling_hh_ll",
    "run_channel_with_dd_control",
    "run_grid_search",
    "run_grid_search_with_params",
]
