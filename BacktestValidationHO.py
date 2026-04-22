# This gives the same results as the matlab code main.m provided

from MarketConfig import MarketConfig
import pandas as pd
import numpy as np
from datetime import time, datetime
from math import ceil
from dataclasses import dataclass
from numba import njit

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

@dataclass
class GridSearchResult:
    result_in_sample: np.ndarray
    result_out_sample: np.ndarray
    chn_len_values: np.ndarray
    stop_pct_values: np.ndarray
    labels: tuple[str, ...]
    last_equity: np.ndarray
    last_drawdown: np.ndarray
    last_trades: np.ndarray
    last_hh: np.ndarray
    last_ll: np.ndarray
    last_chn_len: int
    last_stop_pct: float
    ind_in_sample: tuple[int, int]
    ind_out_sample: tuple[int, int]
    df: pd.DataFrame

@njit
def rolling_hh_ll(
    high: np.ndarray,
    low: np.ndarray,
    bars_back: int,
    chn_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = high.shape[0]
    hh = np.zeros(n, dtype=np.float64)
    ll = np.zeros(n, dtype=np.float64)

    for k in range(bars_back, n):
        start = k - chn_len
        hmax = high[start]
        lmin = low[start]

        for i in range(start + 1, k):
            if high[i] > hmax:
                hmax = high[i]
            if low[i] < lmin:
                lmin = low[i]

        hh[k] = hmax
        ll[k] = lmin

    return hh, ll


@njit
def run_channel_with_dd_control(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    hh: np.ndarray,
    ll: np.ndarray,
    bars_back: int,
    slpg: float,
    pv: float,
    stop_pct: float,
    e0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]

    equity = np.empty(n, dtype=np.float64)
    drawdown = np.zeros(n, dtype=np.float64)
    trades = np.zeros(n, dtype=np.float64)

    for i in range(n):
        equity[i] = e0

    position = 0
    equity_max = e0
    benchmark_long = 0.0
    benchmark_short = 0.0

    for k in range(bars_back, n):
        traded = False
        delta = pv * (close[k] - close[k - 1]) * position

        if position == 0:
            buy = high[k] >= hh[k]
            sell = low[k] <= ll[k]

            if buy and sell:
                delta = -slpg + pv * (ll[k] - hh[k])
                trades[k] = 1.0
            else:
                if buy:
                    delta = -slpg / 2.0 + pv * (close[k] - hh[k])
                    position = 1
                    traded = True
                    benchmark_long = high[k]
                    trades[k] = 0.5

                if sell:
                    delta = -slpg / 2.0 - pv * (close[k] - ll[k])
                    position = -1
                    traded = True
                    benchmark_short = low[k]
                    trades[k] = 0.5

        if position == 1 and not traded:
            sell_short = low[k] <= ll[k]
            sell = low[k] <= benchmark_long * (1.0 - stop_pct)

            if sell_short and sell:
                delta = delta - slpg - 2.0 * pv * (close[k] - ll[k])
                position = -1
                benchmark_short = low[k]
                trades[k] = 1.0
            else:
                if sell:
                    delta = delta - slpg / 2.0 - pv * (close[k] - benchmark_long * (1.0 - stop_pct))
                    position = 0
                    trades[k] = 0.5

                if sell_short:
                    delta = delta - slpg - 2.0 * pv * (close[k] - ll[k])
                    position = -1
                    benchmark_short = low[k]
                    trades[k] = 1.0

            if high[k] > benchmark_long:
                benchmark_long = high[k]

        if position == -1 and not traded:
            buy_long = high[k] >= hh[k]
            buy = high[k] >= benchmark_short * (1.0 + stop_pct)

            if buy_long and buy:
                delta = delta - slpg + 2.0 * pv * (close[k] - hh[k])
                position = 1
                benchmark_long = high[k]
                trades[k] = 1.0
            else:
                if buy:
                    delta = delta - slpg / 2.0 + pv * (close[k] - benchmark_short * (1.0 + stop_pct))
                    position = 0
                    trades[k] = 0.5

                if buy_long:
                    delta = delta - slpg + 2.0 * pv * (close[k] - hh[k])
                    position = 1
                    benchmark_long = high[k]
                    trades[k] = 1.0

            if low[k] < benchmark_short:
                benchmark_short = low[k]

        equity[k] = equity[k - 1] + delta

        if equity[k] > equity_max:
            equity_max = equity[k]
        drawdown[k] = equity[k] - equity_max

    pnl = np.zeros(n, dtype=np.float64)
    for k in range(bars_back, n):
        pnl[k] = equity[k] - equity[k - 1]

    return equity, drawdown, trades, pnl


@njit
def compute_stats(
    equity: np.ndarray,
    drawdown: np.ndarray,
    pnl: np.ndarray,
    trades: np.ndarray,
    i1: int,
    i2: int,
) -> tuple[float, float, float, float]:
    profit = equity[i2] - equity[i1]

    worst_dd = drawdown[i1]
    for k in range(i1 + 1, i2 + 1):
        if drawdown[k] < worst_dd:
            worst_dd = drawdown[k]

    count = i2 - i1 + 1
    mean = 0.0
    for k in range(i1, i2 + 1):
        mean += pnl[k]
    mean /= count

    var = 0.0
    for k in range(i1, i2 + 1):
        diff = pnl[k] - mean
        var += diff * diff
    if count > 1:
        var /= (count - 1)
    else:
        var = 0.0
    stdev = np.sqrt(var)

    ntrades = 0.0
    for k in range(i1, i2 + 1):
        ntrades += trades[k]

    return profit, worst_dd, stdev, ntrades


def run_grid_search(
    df: pd.DataFrame,
    in_sample: tuple[datetime, datetime],
    out_sample: tuple[datetime, datetime],
    bars_back: int,
    slpg: float,
    pv: float,
    e0: float,
    chn_len_values: np.ndarray,
    stop_pct_values: np.ndarray,
) -> GridSearchResult:
    dt = pd.to_datetime(df["DateTime"])
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)
    close = df["Close"].to_numpy(dtype=np.float64)

    if len(df) <= bars_back:
        raise ValueError(f"Not enough rows ({len(df)}) for bars_back={bars_back}")

    if bars_back < int(np.max(chn_len_values)):
        raise ValueError("bars_back must be at least max(chn_len_values)")

    ind_in_1, ind_in_2 = get_period_indices(dt, in_sample[0], in_sample[1], bars_back)
    ind_out_1, ind_out_2 = get_period_indices(dt, out_sample[0], out_sample[1], bars_back)

    labels = ("Profit", "WorstDrawDown", "StDev", "#Trades")
    result_in_sample = np.zeros((len(chn_len_values), len(stop_pct_values), len(labels)), dtype=np.float64)
    result_out_sample = np.zeros((len(chn_len_values), len(stop_pct_values), len(labels)), dtype=np.float64)

    last_equity = np.array([], dtype=np.float64)
    last_drawdown = np.array([], dtype=np.float64)
    last_trades = np.array([], dtype=np.float64)
    last_hh = np.array([], dtype=np.float64)
    last_ll = np.array([], dtype=np.float64)
    last_chn_len = -1
    last_stop_pct = np.nan

    print("\nGrid search results")

    row_fmt = (
        "{:<10} | "
        "{:>14} {:>14} {:>14} {:>10} | "
        "{:>14} {:>14} {:>14} {:>10}"
    )

    line = "-" * 130

    print(line)
    print(
        row_fmt.format(
            "StopPct",
            "IN Profit", "IN DD", "IN Std", "IN Trades",
            "OUT Profit", "OUT DD", "OUT Std", "OUT Trades",
        )
    )
    print(line)

    for i, chn_len in enumerate(chn_len_values):
        print(f"ChnLen = {chn_len:,}")
        print(line)

        hh, ll = rolling_hh_ll(high, low, bars_back, int(chn_len))

        for j, stop_pct in enumerate(stop_pct_values):
            equity, drawdown, trades, pnl = run_channel_with_dd_control(
                close=close,
                high=high,
                low=low,
                hh=hh,
                ll=ll,
                bars_back=bars_back,
                slpg=slpg,
                pv=pv,
                stop_pct=float(stop_pct),
                e0=e0,
            )

            result_in_sample[i, j, :] = compute_stats(
                equity, drawdown, pnl, trades, ind_in_1, ind_in_2
            )
            result_out_sample[i, j, :] = compute_stats(
                equity, drawdown, pnl, trades, ind_out_1, ind_out_2
            )

            in_profit, in_dd, in_std, in_trades = result_in_sample[i, j, :]
            out_profit, out_dd, out_std, out_trades = result_out_sample[i, j, :]

            print(
                row_fmt.format(
                    f"{stop_pct:.4%}",
                    f"{in_profit:,.2f}",
                    f"{in_dd:,.2f}",
                    f"{in_std:,.2f}",
                    f"{in_trades:.1f}",
                    f"{out_profit:,.2f}",
                    f"{out_dd:,.2f}",
                    f"{out_std:,.2f}",
                    f"{out_trades:.1f}",
                )
            )

            last_equity = equity
            last_drawdown = drawdown
            last_trades = trades
            last_hh = hh
            last_ll = ll
            last_chn_len = int(chn_len)
            last_stop_pct = float(stop_pct)

    return GridSearchResult(
        result_in_sample=result_in_sample,
        result_out_sample=result_out_sample,
        chn_len_values=chn_len_values,
        stop_pct_values=stop_pct_values,
        labels=labels,
        last_equity=last_equity,
        last_drawdown=last_drawdown,
        last_trades=last_trades,
        last_hh=last_hh,
        last_ll=last_ll,
        last_chn_len=last_chn_len,
        last_stop_pct=last_stop_pct,
        ind_in_sample=(ind_in_1, ind_in_2),
        ind_out_sample=(ind_out_1, ind_out_2),
        df=df,
    )

if __name__ == "__main__":
    # Read in config
    cfg = MarketConfig.from_yaml("./configs/HO.yml")
    print(cfg)

    # Read in data
    df = load_ohlcv_csv("./data/HO-5minHLV.csv")
    print("Loaded dataframe with shape:", df.shape)
    print("--------------------------------")
    print(df.head())
    print(df.tail())

    # Ensure only confined to trading session timings
    df = filter_session(df, cfg.time_open, cfg.time_close)
    print("Filtered trading session to shape:", df.shape)
    print("--------------------------------")
    print(df.head())
    print(df.tail())

    # Calculate bars back
    bars_back = 17001 #HARDCODED

    # Initialize variables
    in_sample = (datetime(1980, 1, 1), datetime(2000, 1, 1)) #HARDCODED
    out_sample = (datetime(2000, 1, 1), datetime(2023, 3, 23)) #HARDCODED
    slpg = 47 #HARDCODED
    pv = 42000 #HARDCODED
    e0 = 100000.0 #HARDCODED
    print("Backtest variables:")
    print("--------------------------------")
    print(f"In-sample: {in_sample}")
    print(f"Out-sample: {out_sample}")
    print(f"Bars back: {bars_back}")
    print(f"Slippage: {slpg}")
    print(f"Point value: {pv}")
    print(f"Initial equity: {e0}")

    # Run grid search
    chn_len_values = np.arange(10000, 11001, 100, dtype=np.int64)
    stop_pct_values = np.arange(0.010, 0.0201, 0.002, dtype=np.float64)

    result = run_grid_search(
        df=df,
        in_sample=in_sample,
        out_sample=out_sample,
        bars_back=bars_back,
        slpg=slpg,
        pv=pv,
        e0=e0,
        chn_len_values=chn_len_values,
        stop_pct_values=stop_pct_values,
    )

    print("--------------------------------")
    print("Completed grid search")
    print("In-sample shape:", result.result_in_sample.shape)
    print("Out-of-sample shape:", result.result_out_sample.shape)
    print("In-sample indices:", result.ind_in_sample)
    print("Out-of-sample indices:", result.ind_out_sample)
