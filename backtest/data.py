"""Data loading, session times, in/out sample index ranges."""

from __future__ import annotations

from datetime import datetime, time
from math import ceil

import numpy as np
import pandas as pd

from MarketConfig import MarketConfig


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    required = ["date", "time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dt = pd.to_datetime(
        df[cols["date"]].astype(str) + " " + df[cols["time"]].astype(str),
        errors="coerce",
    )
    if dt.isna().any():
        bad_idx = np.where(dt.isna())[0][:10]
        raise ValueError(f"Failed to parse Date/Time rows, sample bad indices: {bad_idx}")

    out = pd.DataFrame(
        {
            "DateTime": dt,
            "Open": pd.to_numeric(df[cols["open"]], errors="coerce"),
            "High": pd.to_numeric(df[cols["high"]], errors="coerce"),
            "Low": pd.to_numeric(df[cols["low"]], errors="coerce"),
            "Close": pd.to_numeric(df[cols["close"]], errors="coerce"),
        }
    )

    if out[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("Found NaNs in OHLC columns after parsing")

    out = out.sort_values("DateTime").reset_index(drop=True)
    return out


def filter_session(df: pd.DataFrame, time_open: time, time_close: time) -> pd.DataFrame:
    temp = df.set_index("DateTime")
    temp = temp.between_time(time_open, time_close, inclusive="both")
    return temp.reset_index()


def prepare_market_dataframe(csv_path: str, market: MarketConfig) -> pd.DataFrame:
    df = load_ohlcv_csv(csv_path)
    return filter_session(df, market.time_open, market.time_close)


def calculate_bars_back(
    min_per_session: int,
    min_per_bar: int = 5,
    years: float = 4.0,
    trading_days_per_year: float = 252.0,
) -> int:
    bars_per_day = min_per_session / min_per_bar
    bars_back = ceil(years * trading_days_per_year * bars_per_day)
    return int(bars_back)


def get_period_indices(
    dt: pd.Series,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
    bars_back: int,
) -> tuple[int, int]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    start_idx = max(int((dt < start_ts).sum()), bars_back)
    end_idx = max(int((dt < (end_ts + pd.Timedelta(days=1))).sum()) - 1, bars_back)

    if end_idx < start_idx:
        raise ValueError(
            f"Invalid period: start={start_ts}, end={end_ts}, "
            f"start_idx={start_idx}, end_idx={end_idx}"
        )
    return start_idx, end_idx
