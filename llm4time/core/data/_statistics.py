import numpy as np
import pandas as pd
from ._base import TimeSeriesStatistics
from statsmodels.tsa.seasonal import STL
from llm4time._infra import logger
from typing import override


class UniTimeSeriesStatistics(TimeSeriesStatistics):

  @override
  def stl(self, period: int = None, freq: str = None, decimals: int = 4) -> dict:
    ts = self.copy()
    if freq:
      try:
        ts = ts.asfreq(freq)
      except Exception as e:
        logger.error(f"Failed to set series frequency to '{freq}': {e}")
        return {
            "trend": self.__class__(dtype=float),
            "seasonal": self.__class__(dtype=float),
            "residual": self.__class__(dtype=float),
            "t_strength": pd.NA,
            "s_strength": pd.NA,
            "r_strength": pd.NA,
        }

    try:
      res = STL(ts.dropna(), period=period).fit()
      trend = res.trend.round(decimals or 4)
      seasonal = res.seasonal.round(decimals or 4)
      resid = res.resid.round(decimals or 4)
      var_r = np.var(resid)
      var_t = np.var(trend)
      var_s = np.var(seasonal)
      total_var = var_t + var_s + var_r

      t_strength = round(var_t / total_var, decimals) if total_var > 0 else np.nan
      s_strength = round(var_s / total_var, decimals) if total_var > 0 else np.nan
      r_strength = round(var_r / total_var, decimals) if total_var > 0 else np.nan

      return {
          "trend": self.__class__(trend),
          "seasonal": self.__class__(seasonal),
          "residual": self.__class__(resid),
          "t_strength": t_strength,
          "s_strength": s_strength,
          "r_strength": r_strength,
      }
    except Exception as e:
      logger.error(f"STL decomposition failed: {e}")
      return None


class MultiTimeSeriesStatistics(TimeSeriesStatistics):

  @override
  def stl(self, period: int = None, freq: str = None, decimals: int = 4) -> dict:
    trend_dict, seasonal_dict, resid_dict = {}, {}, {}
    t_strengths, s_strengths, r_strengths = {}, {}, {}

    for col in self.columns:
      ts = self[col].copy()
      if hasattr(ts, "stl"):
        res = ts.stl(period=period, freq=freq, decimals=decimals)
        trend_dict[col] = res["trend"]
        seasonal_dict[col] = res["seasonal"]
        resid_dict[col] = res["residual"]
        t_strengths[col] = res["t_strength"]
        s_strengths[col] = res["s_strength"]
        r_strengths[col] = res["r_strength"]
      else:
        logger.warning(f"Column '{col}' does not support STL decomposition.")

    return {
        "trend": pd.DataFrame(trend_dict),
        "seasonal": pd.DataFrame(seasonal_dict),
        "residual": pd.DataFrame(resid_dict),
        "t_strength": pd.Series(t_strengths),
        "s_strength": pd.Series(s_strengths),
        "r_strength": pd.Series(r_strengths),
    }
