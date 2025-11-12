import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .._utils.colors import get_color, adjust_lightness, get_lightness_map
from ._base import TimeSeriesPlot
from typing import override


class UniTimeSeriesPlot(TimeSeriesPlot):

  @override
  def linechart(self, showlegend: bool = True, lightness: float = 0.7, **kwargs) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=self.index, y=self, mode="lines", name=self.name,
        line=dict(color=get_color(0, lightness))
    ))
    fig.update_layout(showlegend=showlegend, **kwargs)
    return fig

  @override
  def lineplot(self, showlegend: bool = True, lightness: float = 0.7, **kwargs) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(self.values))),
        y=self.values, mode="lines", name=self.name,
        line=dict(color=get_color(0, lightness))
    ))
    fig.update_layout(showlegend=showlegend, **kwargs)
    return fig

  @override
  def barplot(self, x: list[str] = None, lightness: float = 0.7, **kwargs) -> go.Figure:
    stats = ["mean", "std", "max", "min", "median"]
    y = [getattr(self, func)() for func in stats]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x or stats, y=y, marker_color=get_color(0, lightness)))
    fig.update_layout(barmode="group", **kwargs)
    return fig

  @override
  def stlplot(self, titles: list[str] = None, showlegend: bool = True, lightness: float = 0.7, **kwargs) -> go.Figure:
    stl = self.stl()
    titles = titles or ["Trend", "Seasonal", "Residual"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=titles)
    for i, component in enumerate(["trend", "seasonal", "residual"]):
      fig.add_trace(
          go.Scatter(
              x=stl[component].index, y=stl[component].values, mode="lines",
              name=titles[i], line=dict(color=adjust_lightness({
                  "trend": "#FFA500",
                  "seasonal": "#008000",
                  "residual": "#FF0000"
              }[component], lightness))), row=i+1, col=1)
    fig.update_layout(showlegend=showlegend, **kwargs)
    return fig


class MultiTimeSeriesPlot(TimeSeriesPlot):

  @override
  def linechart(self, showlegend: bool = True, lightness: float = 0.7, **kwargs) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(self.num_columns):
      fig.add_trace(go.Scatter(
          x=self.index, y=self[col], mode="lines", name=col,
          line=dict(color=get_color(i, lightness))
      ))
    fig.update_layout(showlegend=showlegend, **kwargs)
    return fig

  @override
  def lineplot(self, showlegend: bool = True, lightness: float = 0.7, **kwargs) -> go.Figure:
    fig = go.Figure()
    x = list(range(len(self.values)))
    for i, col in enumerate(self.num_columns):
      fig.add_trace(go.Scatter(
          x=x, y=self[col], mode="lines", name=col,
          line=dict(color=get_color(i, lightness))
      ))
    fig.update_layout(showlegend=showlegend, **kwargs)
    return fig

  @override
  def barplot(self, x: list[str] = None, lightness: float = 0.7, **kwargs) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(self.num_columns):
      stats = ["mean", "std", "max", "min", "median"]
      y = [getattr(self[col], func)() for func in stats]
      fig.add_trace(go.Bar(
          x=x or stats, y=y, name=col,
          marker_color=get_color(i, lightness)
      ))
    fig.update_layout(barmode="group", **kwargs)
    return fig

  @override
  def stlplot(self, titles: list[str] = None, showlegend: bool = True, lightness: float = 0.7, **kwargs) -> go.Figure:
    stl = self.stl()
    titles = titles or ["Trend", "Seasonal", "Residual"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=titles)
    for i, component in enumerate(["trend", "seasonal", "residual"]):
      df = stl[component]
      for j, col_name in enumerate(df.columns):
        lightness_values = get_lightness_map(len(df.columns), lightness)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[col_name], mode="lines", name=col_name,
                line=dict(color=adjust_lightness({
                    "trend": "#FFA500",
                    "seasonal": "#008000",
                    "residual": "#FF0000"
                }[component], lightness_values[j]))), row=i+1, col=1)
    fig.update_layout(showlegend=showlegend, **kwargs)
    return fig
