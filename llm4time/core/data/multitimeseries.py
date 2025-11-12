import pandas as pd
from ._base import TimeSeries
from ._plots import MultiTimeSeriesPlot
from ._imputation import MultiTimeSeriesImputation
from ._statistics import MultiTimeSeriesStatistics
from ._metrics import MultiTimeSeriesMetrics
from .unitimeseries import UniTimeSeries
from typing import Optional, override


class MultiTimeSeries(
    pd.DataFrame,
    TimeSeries,
    MultiTimeSeriesPlot,
    MultiTimeSeriesImputation,
    MultiTimeSeriesStatistics,
    MultiTimeSeriesMetrics
):
  def __init__(self, data, *args, **kwargs):
    super().__init__(data, *args, **kwargs)

  def __getitem__(self, key):
    obj = super().__getitem__(key)
    if isinstance(obj, pd.Series) and not isinstance(obj, UniTimeSeries):
      return UniTimeSeries(obj)
    elif isinstance(obj, pd.DataFrame) and not isinstance(obj, MultiTimeSeries):
      return MultiTimeSeries(obj)
    return obj

  @property
  def _constructor(self) -> 'MultiTimeSeries':
    return MultiTimeSeries

  @property
  def _constructor_sliced(self) -> UniTimeSeries:
    return UniTimeSeries

  @property
  def num_columns(self):
    return self.select_dtypes(include='number').columns

  @property
  def cat_columns(self):
    return self.select_dtypes(exclude='number').columns

  @override
  def agg_duplicates(self, method: str, inplace: bool | None = False) -> Optional['MultiTimeSeries']:
    ts = self if inplace else self.copy()

    match method:
      case "first":
        ts = ts[~ts.index.duplicated(keep="first")]
      case "last":
        ts = ts[~ts.index.duplicated(keep="last")]
      case "sumf":
        ts = pd.concat([
            ts[ts.num_columns].groupby(ts.index).sum(),
            ts[ts.cat_columns].groupby(ts.index).first()
        ], axis=1)
      case "suml":
        ts = pd.concat([
            ts[ts.num_columns].groupby(ts.index).sum(),
            ts[ts.cat_columns].groupby(ts.index).last()
        ], axis=1)
      case _:
        raise ValueError(
            f"Invalid method: {method}. Choose from 'first', 'last', 'sumf', 'suml'.")

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

  def trend(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> pd.DataFrame:
    res = self.stl(period, freq, decimals)
    if strength:
      return res["t_strength"]
    return res["trend"]

  def seasonal(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> pd.DataFrame:
    res = self.stl(period, freq, decimals)
    if strength:
      return res["s_strength"]
    return res["seasonal"]

  def residual(self, strength: bool = False, period: int = None, freq: str = None, decimals: int = 4) -> pd.DataFrame:
    res = self.stl(period, freq, decimals)
    if strength:
      return res["r_strength"]
    return res["residual"]
