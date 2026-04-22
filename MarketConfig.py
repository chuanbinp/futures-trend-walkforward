from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
import yaml


@dataclass
class MarketConfig:
    ticker: str
    bloomberg: str
    description: str
    exchange: str
    currency: str
    point_value: float
    tick_value: float
    margin: float
    slippage: float
    pv_multiplier: float
    time_open: time
    time_close: time
    minutes_per_session: int

    @staticmethod
    def _parse_hhmm(value: str) -> time:
        value = str(value).strip()
        if len(value) != 4 or not value.isdigit():
            raise ValueError(f"Time must be HHMM, got {value}")
        return datetime.strptime(value, "%H%M").time()

    @property
    def time_open_fraction(self) -> float:
        return (self.time_open.hour * 60 + self.time_open.minute) / 1440

    @property
    def time_close_fraction(self) -> float:
        return (self.time_close.hour * 60 + self.time_close.minute) / 1440

    @classmethod
    def from_dict(cls, data: dict) -> "MarketConfig":
        return cls(
            ticker=data["ticker"],
            bloomberg=data["bloomberg"],
            description=data["description"],
            exchange=data["exchange"],
            currency=data["currency"],
            point_value=float(data["point_value"]),
            tick_value=float(data["tick_value"]),
            margin=float(data["margin"]),
            slippage=float(data["slippage"]),
            pv_multiplier=float(data["pv_multiplier"]),
            time_open=cls._parse_hhmm(data["time_open"]),
            time_close=cls._parse_hhmm(data["time_close"]),
            minutes_per_session=int(data["minutes_per_session"]),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MarketConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    
    def __str__(self) -> str:
        return (
            "MarketConfig:\n"
            f"  ticker={self.ticker}, bloomberg={self.bloomberg},\n"
            f"  description={self.description},\n"
            f"  exchange={self.exchange}, currency={self.currency},\n"
            f"  point_value={self.point_value:,.4f}, "
            f"  tick_value={self.tick_value:,.4f}, "
            f"  pv_multiplier={self.pv_multiplier:,.4f},\n"
            f"  slippage={self.slippage:,.4f}, "
            f"  margin={self.margin:,.2f},\n"
            f"  session={self.time_open.strftime('%H:%M')} - "
            f"{self.time_close.strftime('%H:%M')} "
            f"({self.minutes_per_session} min)\n"
        )