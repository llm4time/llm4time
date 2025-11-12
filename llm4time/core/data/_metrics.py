import numpy as np
import pandas as pd
from ._base import TimeSeriesMetrics
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from typing import override


class UniTimeSeriesMetrics(TimeSeriesMetrics):

  @override
  def smape(self, y_pred: list[float], decimals: int = 2) -> float:
    y_true = self[~np.isnan(self)]
    y_pred = np.array(y_pred)[~np.isnan(y_pred)]
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    epsilon = 1e-10
    smape = np.mean(numerator / (denominator + epsilon)) * 100
    return round(smape, decimals)

  @override
  def mae(self, y_pred: list[float], decimals: int = 2) -> float:
    y_true = self[~np.isnan(self)]
    y_pred = np.array(y_pred)[~np.isnan(y_pred)]
    mae = mean_absolute_error(y_true, y_pred)
    return round(mae, decimals)

  @override
  def rmse(self, y_pred: list[float], decimals: int = 2) -> float:
    y_true = self[~np.isnan(self)]
    y_pred = np.array(y_pred)[~np.isnan(y_pred)]
    rmse = root_mean_squared_error(y_true, y_pred)
    return round(rmse, decimals)

  def metrics(self, y_pred: list[float], decimals: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        "smape": self.smape(y_pred, decimals),
        "mae": self.mae(y_pred, decimals),
        "rmse": self.rmse(y_pred, decimals)
    }, index=[self.name]).T


class MultiTimeSeriesMetrics:

  @override
  def smape(self, y_pred: pd.DataFrame, decimals: int = 2) -> pd.Series:
    return pd.Series({
        col: self[col].smape(y_pred[col], decimals)
        for col in self.columns
    })

  @override
  def mae(self, y_pred: pd.DataFrame, decimals: int = 2) -> pd.Series:
    return pd.Series({
        col: self[col].mae(y_pred[col], decimals)
        for col in self.columns
    })

  @override
  def rmse(self, y_pred: pd.DataFrame, decimals: int = 2) -> pd.Series:
    return pd.Series({
        col: self[col].rmse(y_pred[col], decimals)
        for col in self.columns
    })

  def metrics(self, y_pred: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    return pd.DataFrame({
        "smape": self.smape(y_pred, decimals),
        "mae": self.mae(y_pred, decimals),
        "rmse": self.rmse(y_pred, decimals)
    }).T
