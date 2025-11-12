import plotly.graph_objects as go
from llm4time.core import UniTimeSeries, MultiTimeSeries
from llm4time.core._utils.colors import get_color, get_lightness_map
from typing import Sequence
import pandas as pd
import math


def _get_groups(
    series: Sequence[UniTimeSeries | MultiTimeSeries | pd.Series | pd.DataFrame],
    groups: list[str] = None
) -> list[str | None]:
  if groups is None:
    return [None] * len(series)
  elif len(groups) == 1:
    return groups * len(series)
  elif len(groups) != len(series):
    raise ValueError("The number of groups must match the number of series provided.")
  return groups


def linechart(
    *series: Sequence[UniTimeSeries | MultiTimeSeries | pd.Series | pd.DataFrame],
    groups: list[str] = None,
    showlegend: bool = True,
    lightness: float = 0.7,
    **kwargs
) -> go.Figure:
  groups = _get_groups(series, groups)

  fig = go.Figure()
  for i, s in enumerate(series):
    if isinstance(s, UniTimeSeries) or isinstance(s, pd.Series):
      name = f"{groups[i]} - {s.name}" if groups[i] else s.name
      color = get_color(i, lightness)
      fig.add_trace(go.Scatter(
          x=s.index, y=s.values, mode="lines", name=name, line=dict(color=color)))
    elif isinstance(s, MultiTimeSeries) or isinstance(s, pd.DataFrame):
      lightness_values = get_lightness_map(len(s.columns), lightness) \
          if groups[i] else [lightness] * len(s.columns)
      for j, c in enumerate(s.columns):
        name = f"{groups[i]} - {c}" if groups[i] else c
        color = get_color(i if groups[i] else j, lightness_values[j])
        fig.add_trace(go.Scatter(
            x=s.index, y=s[c], mode="lines", name=name, line=dict(color=color)))
    else:
      raise TypeError(f"Type not supported: {type(s).__name__}.")

  fig.update_layout(showlegend=showlegend, **kwargs)
  return fig


def lineplot(
    *series: Sequence[UniTimeSeries | MultiTimeSeries | pd.Series | pd.DataFrame],
    groups: list[str] = None,
    showlegend: bool = True,
    lightness: float = 0.7,
    **kwargs
) -> go.Figure:
  groups = _get_groups(series, groups)

  fig = go.Figure()
  for i, s in enumerate(series):
    if isinstance(s, UniTimeSeries) or isinstance(s, pd.Series):
      name = f"{groups[i]} - {s.name}" if groups[i] else s.name
      color = get_color(i, lightness)
      fig.add_trace(go.Scatter(
          x=list(range(len(s))), y=s, mode="lines", name=name, line=dict(color=color)))
    elif isinstance(s, MultiTimeSeries) or isinstance(s, pd.DataFrame):
      lightness_values = get_lightness_map(len(s.columns), lightness) \
          if groups[i] else [lightness] * len(s.columns)
      for j, c in enumerate(s.columns):
        name = f"{groups[i]} - {c}" if groups[i] else c
        color = get_color(i if groups[i] else j, lightness_values[j])
        fig.add_trace(go.Scatter(
            x=list(range(len(s))), y=s[c], mode="lines", name=name, line=dict(color=color)))
    else:
      raise TypeError(f"Type not supported: {type(s).__name__}.")

  fig.update_layout(showlegend=showlegend, **kwargs)
  return fig


def barplot(
    *series: Sequence[UniTimeSeries | MultiTimeSeries | pd.Series | pd.DataFrame],
    x: list[str] = None,
    groups: list[str] = None,
    lightness: float = 0.7,
    **kwargs
) -> go.Figure:
  groups = _get_groups(series, groups)

  fig = go.Figure()
  for i, s in enumerate(series):
    if isinstance(s, MultiTimeSeries):
      lightness_values = get_lightness_map(len(s.columns), lightness) \
          if groups[i] else [lightness] * len(s.columns)
      for j, c in enumerate(s.columns):
        stats = ["mean", "std", "max", "min", "median"]
        y = [getattr(s[c], func)() for func in stats]
        name = f"{groups[i]} - {c}" if groups[i] else c
        color = get_color(i, lightness_values[j]) if groups[i] else get_color(
            j, lightness_values[j])
        fig.add_trace(go.Bar(x=x or stats, y=y, name=name, marker_color=color))
    elif isinstance(s, UniTimeSeries):
      stats = ["mean", "std", "max", "min", "median"]
      y = [getattr(s, func)() for func in stats]
      name = groups[i] or s.name
      color = get_color(i, lightness)
      fig.add_trace(go.Bar(x=x or stats, y=y, name=name, marker_color=color))
    elif isinstance(s, pd.Series):
      name = groups[i] or s.name
      color = get_color(i, lightness)
      fig.add_trace(go.Bar(x=x or s.index.astype(str),
                           y=s.values, name=name, marker_color=color))
    elif isinstance(s, pd.DataFrame):
      lightness_values = get_lightness_map(len(s.columns), lightness) \
          if groups[i] else [lightness] * len(s.columns)
      for j, c in enumerate(s.columns):
        name = f"{groups[i]} - {c}" if groups[i] else c
        color = get_color(i if groups[i] else j, lightness_values[j])
        fig.add_trace(go.Bar(x=x or s.index.astype(str),
                             y=s[c].values, name=name, marker_color=color))
    else:
      raise TypeError(f"Type not supported: {type(s).__name__}.")

  fig.update_layout(barmode="group", **kwargs)
  return fig
