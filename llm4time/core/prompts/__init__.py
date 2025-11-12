from ._templates import *
import llm4time.core.data as l4t
from enum import Enum


class PromptType(str, Enum):
  ZERO_SHOT = "zero_shot"
  FEW_SHOT = "few_shot"
  COT = "cot"
  COT_FEW = "cot_few"
  CUSTOM = "custom"


def prompt(
    ts: l4t.TimeSeries,
    periods: int,
    type: PromptType,
    tsformat: l4t.TSFormat = l4t.TSFormat.CSV,
    tstype: l4t.TSType = l4t.TSType.NUMERIC,
    examples: int = 0,
    sampling: l4t.Sampling = None,
    template: str = None,
    **kwargs
) -> str:
  if template is None and type == PromptType.CUSTOM:
    raise ValueError("Template must be set for custom prompt.")
  if examples == 0 and type in [PromptType.FEW_SHOT, PromptType.COT_FEW]:
    raise ValueError("Must contain at least 1 example.")

  base_kwargs = {
      "input_len": len(ts),
      "input": ts.to_str(tsformat, tstype),
      "output_example": ts[:periods].to_str(tsformat, tstype),
      "forecast_horizon": periods,
  }
  base_kwargs.update(kwargs)

  if isinstance(ts, l4t.UniTimeSeries):
    base_kwargs.update({
        "statistics": "\n".join([
            f"- Mean: {ts.mean()}\n"
            f"- Median: {ts.median()}\n"
            f"- Standard Deviation: {ts.std()}\n"
            f"- Minimum Value: {ts.min()}\n"
            f"- Maximum Value: {ts.max()}\n"
            f"- First Quartile (Q1): {ts.quantile(0.25)}\n"
            f"- Terceiro Quartil (Q3): {ts.quantile(0.75)}\n"
            f"- Força da Tendência (STL): {ts.trend(strength=True)}\n"
            f"- Força da Sazonalidade (STL): {ts.seasonal(strength=True)}\n"
        ]),
    })
  elif isinstance(ts, l4t.MultiTimeSeries):
    base_kwargs.update({
        "statistics": "\n".join([
            f"{f'Column: {col}\n' if len(ts.num_columns) > 1 else ''}"
            f"- Mean: {ts[col].mean()}\n"
            f"- Median: {ts[col].median()}\n"
            f"- Standard Deviation: {ts[col].std()}\n"
            f"- Minimum Value: {ts[col].min()}\n"
            f"- Maximum Value: {ts[col].max()}\n"
            f"- First Quartile (Q1): {ts[col].quantile(0.25)}\n"
            f"- Third Quartile (Q3): {ts[col].quantile(0.75)}\n"
            f"- Trend Strength (STL): {ts[col].trend(strength=True)}\n"
            f"- Seasonality Strength (STL): {ts[col].seasonal(strength=True)}"
            f"{'' if i == len(ts.num_columns) - 1 else '\n'}"
            for i, col in enumerate(ts.num_columns)
        ]),
    })
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

  min_periods = periods * 2 * examples
  if len(ts) < min_periods:
    raise ValueError(
        f"For {examples} examples there must be {min_periods} periods in the time series.")

  try:
    sampling = l4t.Sampling(sampling or l4t.Sampling.BACKEND.value)
  except ValueError:
    raise ValueError("Supported samplings: frontend, backend, random, uniform.")

  if "forecast_examples" not in kwargs:
    forecast_examples = "\n".join([
        f"- Example {i}:\n"
        f"Input (history):\n{input.to_str(tsformat, tstype)}\n\n"
        f"Output (forecast):\n<out>\n{output.to_str(tsformat, tstype)}\n</out>"
        f"{'' if i == examples else '\n'}"
        for i, (input, output) in enumerate(
            ts.slide(method=sampling, window=periods, samples=examples),
            start=1)
    ])
    base_kwargs.update({"forecast_examples": forecast_examples})

  prompt_map = {
      PromptType.ZERO_SHOT: ZERO_SHOT,
      PromptType.FEW_SHOT: FEW_SHOT,
      PromptType.COT_FEW: COT_FEW,
      PromptType.COT: COT,
      PromptType.CUSTOM: template
  }
  if type not in prompt_map:
    raise ValueError("Supported prompts: zero_shot, few_shot, cot, cot_few, custom.")

  try:
    return prompt_map[type].format(**base_kwargs)
  except KeyError as e:
    raise ValueError(f"Key {e} not defined.")
