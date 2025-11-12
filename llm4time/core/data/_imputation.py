from ._base import TimeSeries, TimeSeriesImputation
from typing import override


class UniTimeSeriesImputation(TimeSeriesImputation):

  @override
  def impute_mean(self, decimals: int = None, inplace: bool = False) -> TimeSeries | None:
    return self.fillna(round(self.mean(), decimals), inplace=inplace)

  @override
  def impute_median(self, decimals: int = None, inplace: bool = False) -> TimeSeries | None:
    return self.fillna(round(self.median(), decimals), inplace=inplace)

  @override
  def impute_ffill(self, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts.ffill(inplace=True)
    ts.bfill(inplace=True)
    if not inplace:
      return ts

  @override
  def impute_bfill(self, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts.bfill(inplace=True)
    ts.ffill(inplace=True)
    if not inplace:
      return ts

  @override
  def impute_sma(self, window: int, min_periods: int = 1, decimals: int = None, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts.fillna(ts.rolling(window=window, min_periods=min_periods)
              .mean().round(decimals), inplace=True)
    ts.impute_ffill(inplace=True)
    if not inplace:
      return ts

  @override
  def impute_ema(self, span: int, adjust: bool = False, decimals: int = None, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts.fillna(ts.ewm(span=span, adjust=adjust)
              .mean().round(decimals), inplace=True)
    ts.impute_ffill(inplace=True)
    if not inplace:
      return ts

  @override
  def impute_interpolate(self, method: str = 'linear', order: int = 2, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    try:
      if method == 'linear':
        ts.interpolate(method='linear', inplace=True)
      elif method == 'spline':
        ts.interpolate(method='spline', order=order, inplace=True)
      else:
        raise ValueError("Supported methods: linear or spline.")
    except Exception:
      ts.interpolate(method='linear', inplace=True)

    ts.impute_ffill(inplace=True)
    if not inplace:
      return ts


class MultiTimeSeriesImputation(TimeSeriesImputation):

  @override
  def impute_mean(self, decimals: int = None, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts[self.num_columns] = ts[self.num_columns].fillna(
        ts[self.num_columns].mean().round(decimals))
    if not inplace:
      return ts

  @override
  def impute_median(self, decimals: int = None, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts[self.num_columns] = ts[self.num_columns].fillna(
        ts[self.num_columns].median().round(decimals))
    if not inplace:
      return ts

  @override
  def impute_ffill(self, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts[self.cat_columns] = ts[self.cat_columns].ffill().bfill()
    ts[self.num_columns] = ts[self.num_columns].ffill().bfill()
    if not inplace:
      return ts

  @override
  def impute_bfill(self, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts[self.cat_columns] = ts[self.cat_columns].bfill().ffill()
    ts[self.num_columns] = ts[self.num_columns].bfill().ffill()
    if not inplace:
      return ts

  @override
  def impute_sma(self, window: int, min_periods: int = 1, decimals: int = None, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts[self.num_columns] = ts[self.num_columns].fillna(
        ts[self.num_columns].rolling(window=window, min_periods=min_periods)
        .mean().round(decimals))
    ts.impute_ffill(inplace=True)
    if not inplace:
      return ts

  @override
  def impute_ema(self, span: int, adjust: bool = False, decimals: int = None, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    ts[self.num_columns] = ts[self.num_columns].fillna(
        ts[self.num_columns].ewm(span=span, adjust=adjust)
        .mean().round(decimals))
    ts.impute_ffill(inplace=True)
    if not inplace:
      return ts

  @override
  def impute_interpolate(self, method: str = 'linear', order: int = 2, inplace: bool = False) -> TimeSeries | None:
    ts = self if inplace else self.copy()
    try:
      if method == 'linear':
        ts[self.num_columns] = ts[self.num_columns].interpolate(method='linear')
      elif method == 'spline':
        ts[self.num_columns] = ts[self.num_columns].interpolate(
            method='spline', order=order)
      else:
        raise ValueError("Supported methods: linear or spline.")
    except Exception:
      ts[self.num_columns] = ts[self.num_columns].interpolate(method='linear')

    ts.impute_ffill(inplace=True)
    if not inplace:
      return ts
