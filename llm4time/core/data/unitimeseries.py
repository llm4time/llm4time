import pandas as pd
from ._base import TimeSeries
from ._plots import UniTimeSeriesPlot
from ._imputation import UniTimeSeriesImputation
from ._statistics import UniTimeSeriesStatistics
from ._metrics import UniTimeSeriesMetrics
from typing import Optional, Union, override


class UniTimeSeries(
    pd.Series,
    TimeSeries,
    UniTimeSeriesPlot,
    UniTimeSeriesImputation,
    UniTimeSeriesStatistics,
    UniTimeSeriesMetrics
):
  def __init__(self, data, *args, **kwargs):
    super().__init__(data, *args, **kwargs)

  @property
  def _constructor(self) -> 'UniTimeSeries':
    return UniTimeSeries

  @override
  def agg_duplicates(self, method: str, inplace: bool | None = False) -> Optional['UniTimeSeries']:
    ts = self if inplace else self.copy()

    match method:
      case "first":
        ts = ts[~ts.index.duplicated(keep="first")]
      case "last":
        ts = ts[~ts.index.duplicated(keep="last")]
      case "sum":
        ts = ts.groupby(ts.index).sum()
      case _:
        raise ValueError(
            f"Invalid method: {method}. Choose from 'first', 'last', 'sum'.")

    if inplace:
      self.__dict__.update(ts.__dict__)
    else:
      return ts

  @override
  def mean(self, decimals: int = 4, **kwargs) -> float:
    return round(super().mean(**kwargs), decimals)

  @override
  def median(self, decimals: int = 4, **kwargs) -> float:
    return round(super().median(**kwargs), decimals)

  @override
  def std(self, decimals: int = 4, **kwargs) -> float:
    return round(super().std(**kwargs), decimals)

  @override
  def min(self, decimals: int = 4, **kwargs) -> float:
    return round(super().min(**kwargs), decimals)

  @override
  def max(self, decimals: int = 4, **kwargs) -> float:
    return round(super().max(**kwargs), decimals)

  @override
  def quantile(self, q: float, decimals: int = 4, **kwargs) -> float:
    return round(super().quantile(q, **kwargs), decimals)

  def trend(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> Union['UniTimeSeries', float]:
    res = self.stl(period, freq, decimals)
    if strength:
      return res["t_strength"]
    return res["trend"]

  def seasonal(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> Union['UniTimeSeries', float]:
    res = self.stl(period, freq, decimals)
    if strength:
      return res["s_strength"]
    return res["seasonal"]

  def residual(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> Union['UniTimeSeries', float]:
    res = self.stl(period, freq, decimals)
    if strength:
      return res["r_strength"]
    return res["residual"]
