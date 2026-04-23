"""
Run HO futures backtest with the same layout as `run_backtest_pl.py`:
`OptimConfig` + `MarketConfig` (HO.yml), same IS/OOS dates and `optim.yml` grid.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

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
    cfg = MarketConfig.from_yaml(_ROOT / "configs" / "HO.yml")
    print(optim_cfg)
    print(cfg)

    df = load_ohlcv_csv(str(_ROOT / "data" / "HO-5minHLV.csv"))
    print("Loaded dataframe with shape:", df.shape)
    df = filter_session(df, cfg.time_open, cfg.time_close)
    print("Filtered trading session to shape:", df.shape)

    bars_back = 17001 #HARDCODED TO REPLICATE MATLAB CODE TO VERIFY

    in_sample = (datetime(1980, 1, 1), datetime(2000, 1, 1)) #HARDCODED TO REPLICATE MATLAB CODE TO VERIFY
    out_sample = (datetime(2000, 1, 1), datetime(2023, 3, 23)) #HARDCODED TO REPLICATE MATLAB CODE TO VERIFY
    slpg = float(cfg.slippage)
    pv = float(cfg.point_value * cfg.pv_multiplier)
    e0 = 100_000.0
    chn_len_values = optim_cfg.chn_len_values()
    stop_pct_values = optim_cfg.stop_pct_values()

    print("Backtest variables:")
    print("--------------------------------")
    print(f"In-sample: {in_sample}")
    print(f"Out-sample: {out_sample}")
    print(f"Bars back: {bars_back}")
    print(f"Slippage: {slpg}")
    print(f"Point value: {pv}")
    print(f"Initial equity: {e0}")

    

    start = perf_counter()
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
    end = perf_counter()

    print("--------------------------------")
    print("Completed grid search")
    print(f"Elapsed time: {end - start} seconds")
    print("In-sample shape:", result.result_in_sample.shape)
    print("Out-of-sample shape:", result.result_out_sample.shape)
    print("In-sample indices:", result.ind_in_sample)
    print("Out-of-sample indices:", result.ind_out_sample)

    print("--------------------------------")
    print("Optimization summary")
    print("Target:", result.target_label)
    print("Best ChnLen:", result.best_chn_len)
    print(f"Best StopPct: {result.best_stop_pct:.4%}")
    print(f"Best Target: {result.best_target:.6f}")

    print(
        f"Best IN  -> Profit={result.best_in_sample_stats[0]:,.2f}, "
        f"DD={result.best_in_sample_stats[1]:,.2f}, "
        f"Std={result.best_in_sample_stats[2]:,.2f}, "
        f"Trades={result.best_in_sample_stats[3]:.1f}"
    )
    print(
        f"Best OUT -> Profit={result.best_out_sample_stats[0]:,.2f}, "
        f"DD={result.best_out_sample_stats[1]:,.2f}, "
        f"Std={result.best_out_sample_stats[2]:,.2f}, "
        f"Trades={result.best_out_sample_stats[3]:.1f}"
    )
