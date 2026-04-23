"""
Run rolling PL optimization windows.

Each optimization window uses `bars_back_years` worth of bars (4 years by default)
for in-sample, then evaluates on the following one quarter out-of-sample, advances
by one quarter, and records the best parameters.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from MarketConfig import MarketConfig
from OptimConfig import OptimConfig

from backtest import (
    calculate_bars_back,
    filter_session,
    load_ohlcv_csv,
    run_grid_search,
)

_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    optim_cfg = OptimConfig.from_yaml(_ROOT / "configs" / "optim.yml")
    cfg = MarketConfig.from_yaml(_ROOT / "configs" / "PL.yml")
    print(optim_cfg)
    print(cfg)

    df = load_ohlcv_csv(str(_ROOT / "data" / "PL-5minHLV.csv"))
    print("Loaded dataframe with shape:", df.shape)
    df = filter_session(df, cfg.time_open, cfg.time_close)
    print("Filtered trading session to shape:", df.shape)

    bars_back = calculate_bars_back(cfg.minutes_per_session, years=optim_cfg.bars_back_years)
    quarter_bars = calculate_bars_back(cfg.minutes_per_session, years=0.25)
    slpg = float(cfg.slippage)
    pv = float(cfg.point_value * cfg.pv_multiplier)
    e0 = 100_000.0
    chn_len_values = optim_cfg.chn_len_values()
    stop_pct_values = optim_cfg.stop_pct_values()
    core_bars_back = int(chn_len_values.max())
    
    print("Backtest variables:")
    print("--------------------------------")
    print(f"In-sample bars (bars_back_years): {bars_back}")
    print(f"Out-of-sample bars (1 quarter): {quarter_bars}")
    print(f"Step bars (1 quarter): {quarter_bars}")
    print(f"Core warmup bars (max chn_len): {core_bars_back}")
    print(f"Slippage: {slpg}")
    print(f"Point value: {pv}")
    print(f"Initial equity: {e0}")

    start = perf_counter()
    rows: list[dict[str, object]] = []
    n = len(df)
    total_window_bars = bars_back + quarter_bars
    for start_idx in range(0, n - total_window_bars + 1, quarter_bars):
        window_df = df.iloc[start_idx : start_idx + total_window_bars].reset_index(drop=True)
        in_start_idx = start_idx
        in_end_idx = start_idx + bars_back - 1
        out_start_idx = in_end_idx + 1
        out_end_idx = start_idx + total_window_bars - 1

        in_sample = (
            pd.Timestamp(window_df["DateTime"].iloc[0]).to_pydatetime(),
            pd.Timestamp(window_df["DateTime"].iloc[bars_back - 1]).to_pydatetime(),
        )
        out_sample = (
            pd.Timestamp(window_df["DateTime"].iloc[bars_back]).to_pydatetime(),
            pd.Timestamp(window_df["DateTime"].iloc[-1]).to_pydatetime(),
        )

        result = run_grid_search(
            df=window_df,
            in_sample=in_sample,
            out_sample=out_sample,
            bars_back=core_bars_back,
            slpg=slpg,
            pv=pv,
            e0=e0,
            chn_len_values=chn_len_values,
            stop_pct_values=stop_pct_values,
            verbose=False,
        )

        rows.append(
            {
                "in_sample_start_idx": in_start_idx,
                "in_sample_end_idx": in_end_idx,
                "start_date": in_sample[0],
                "end_date": in_sample[1],
                "optimal_chn_len": result.best_chn_len,
                "optimal_stop_pct": result.best_stop_pct,
                "out_sample_start_idx": out_start_idx,
                "out_sample_end_idx": out_end_idx,
                "out_sample_start_date": out_sample[0],
                "out_sample_end_date": out_sample[1],
                "best_target": result.best_target,
            }
        )

    end = perf_counter()
    result_df = pd.DataFrame(rows)

    print("--------------------------------")
    print("Completed rolling optimization")
    print(f"Elapsed time: {end - start} seconds")
    print(f"Windows evaluated: {len(result_df)}")

    print("--------------------------------")
    print("Rolling optimization result dataframe:")
    print(result_df)
    results_dir = _ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = results_dir / "pl_rolling_optimization.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"Saved rolling optimization results to: {output_csv}")
