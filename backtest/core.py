"""Grid search, numba backtest, and run parameter bundle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from numba import njit

from MarketConfig import MarketConfig
from OptimConfig import OptimConfig

from .data import calculate_bars_back, get_period_indices


# --- result -----------------------------------------------------------------


@dataclass
class GridSearchResult:
    result_in_sample: np.ndarray
    result_out_sample: np.ndarray
    target_in_sample: np.ndarray
    chn_len_values: np.ndarray
    stop_pct_values: np.ndarray
    labels: tuple[str, ...]
    target_label: str
    best_i: int
    best_j: int
    best_chn_len: int
    best_stop_pct: float
    best_target: float
    best_in_sample_stats: np.ndarray
    best_out_sample_stats: np.ndarray
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


@dataclass(frozen=True, slots=True)
class BacktestRunParams:
    bars_back: int
    slpg: float
    pv: float
    e0: float
    chn_len_values: np.ndarray
    stop_pct_values: np.ndarray

    @classmethod
    def from_configs(
        cls,
        market: MarketConfig,
        optim: OptimConfig,
        e0: float = 100_000.0,
    ) -> BacktestRunParams:
        bars_back = calculate_bars_back(
            market.minutes_per_session, years=optim.bars_back_years
        )
        slpg = float(market.slippage)
        pv = float(market.point_value * market.pv_multiplier)
        return cls(
            bars_back=bars_back,
            slpg=slpg,
            pv=pv,
            e0=e0,
            chn_len_values=optim.chn_len_values(),
            stop_pct_values=optim.stop_pct_values(),
        )


# --- numba ------------------------------------------------------------------


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

    maxdq = np.empty(n, dtype=np.int64)
    mindq = np.empty(n, dtype=np.int64)

    max_head = 0
    max_tail = 0
    min_head = 0
    min_tail = 0

    for t in range(n):
        while max_tail > max_head and high[t] >= high[maxdq[max_tail - 1]]:
            max_tail -= 1
        maxdq[max_tail] = t
        max_tail += 1

        while min_tail > min_head and low[t] <= low[mindq[min_tail - 1]]:
            min_tail -= 1
        mindq[min_tail] = t
        min_tail += 1

        if t >= chn_len:
            window_start = t - chn_len + 1

            while max_tail > max_head and maxdq[max_head] < window_start:
                max_head += 1
            while min_tail > min_head and mindq[min_head] < window_start:
                min_head += 1

            k = t + 1
            if k < n and k >= bars_back:
                hh[k] = high[maxdq[max_head]]
                ll[k] = low[mindq[min_head]]

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


# --- grid search -------------------------------------------------------------


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
    *,
    verbose: bool = True,
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

    row_fmt = (
        "{:<10} | "
        "{:>14} {:>14} {:>14} {:>10} | "
        "{:>14} {:>14} {:>14} {:>10}"
    )
    line = "-" * 130
    if verbose:
        print("\nGrid search results")
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
        if verbose:
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

            if verbose:
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

    target_label = "ProfitToAbsWorstDD"

    target_in_sample = np.full(
        (len(chn_len_values), len(stop_pct_values)),
        -np.inf,
        dtype=np.float64,
    )

    for i in range(len(chn_len_values)):
        for j in range(len(stop_pct_values)):
            profit = result_in_sample[i, j, 0]
            worst_dd = result_in_sample[i, j, 1]

            if worst_dd < 0.0:
                target_in_sample[i, j] = profit / (-worst_dd)
            elif profit > 0.0:
                target_in_sample[i, j] = np.inf
            else:
                target_in_sample[i, j] = -np.inf

    best_flat = int(np.argmax(target_in_sample))
    best_i, best_j = np.unravel_index(best_flat, target_in_sample.shape)

    best_chn_len = int(chn_len_values[best_i])
    best_stop_pct = float(stop_pct_values[best_j])
    best_target = float(target_in_sample[best_i, best_j])

    best_in_sample_stats = result_in_sample[best_i, best_j, :].copy()
    best_out_sample_stats = result_out_sample[best_i, best_j, :].copy()

    return GridSearchResult(
        result_in_sample=result_in_sample,
        result_out_sample=result_out_sample,
        target_in_sample=target_in_sample,
        chn_len_values=chn_len_values,
        stop_pct_values=stop_pct_values,
        labels=labels,
        target_label=target_label,
        best_i=int(best_i),
        best_j=int(best_j),
        best_chn_len=best_chn_len,
        best_stop_pct=best_stop_pct,
        best_target=best_target,
        best_in_sample_stats=best_in_sample_stats,
        best_out_sample_stats=best_out_sample_stats,
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


def run_grid_search_with_params(
    df: pd.DataFrame,
    in_sample: tuple[datetime, datetime],
    out_sample: tuple[datetime, datetime],
    params: BacktestRunParams,
    *,
    verbose: bool = True,
) -> GridSearchResult:
    if not isinstance(params, BacktestRunParams):
        raise TypeError("params must be a BacktestRunParams instance")
    return run_grid_search(
        df=df,
        in_sample=in_sample,
        out_sample=out_sample,
        bars_back=params.bars_back,
        slpg=params.slpg,
        pv=params.pv,
        e0=params.e0,
        chn_len_values=params.chn_len_values,
        stop_pct_values=params.stop_pct_values,
        verbose=verbose,
    )
