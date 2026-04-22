from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml


@dataclass
class RangeConfig:
    start: float
    stop: float
    step: float

    def values(self, dtype=np.float64) -> np.ndarray:
        return np.arange(self.start, self.stop, self.step, dtype=dtype)

    def count(self) -> int:
        return len(self.values())

    def __str__(self) -> str:
        return (
            f"start={self.start}, stop={self.stop}, step={self.step}"
        )


@dataclass
class OptimConfig:
    chn_len: RangeConfig
    stop_pct: RangeConfig
    bars_back_years: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OptimConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            chn_len=RangeConfig(
                start=float(data["chn_len"]["start"]),
                stop=float(data["chn_len"]["stop"]),
                step=float(data["chn_len"]["step"]),
            ),
            stop_pct=RangeConfig(
                start=float(data["stop_pct"]["start"]),
                stop=float(data["stop_pct"]["stop"]),
                step=float(data["stop_pct"]["step"]),
            ),
            bars_back_years=float(data["bars_back_years"]),
        )

    def chn_len_values(self) -> np.ndarray:
        return self.chn_len.values(dtype=np.int64)

    def stop_pct_values(self) -> np.ndarray:
        return self.stop_pct.values(dtype=np.float64)

    def __str__(self) -> str:
        total_points = self.chn_len.count() * self.stop_pct.count()
        return (
            "OptimConfig:\n"
            f"  bars_back_years={self.bars_back_years} years,\n"
            f"  chn_len={self.chn_len},\n"
            f"  stop_pct={self.stop_pct},\n"
            f"  total_grid_points={total_points}\n"
        )