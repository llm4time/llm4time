from ...data import TimeSeries, UniTimeSeries, MultiTimeSeries


def to_plain(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries):
    values = [[v] for v in ts.to_list()]
    columns = [ts.name]
  elif isinstance(ts, MultiTimeSeries):
    values = ts.to_numpy().tolist()
    columns = ts.columns
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

  lines = [
      f"{ts.index.name}: {idx}, " +
      ", ".join(f"{col}: {val}" for col, val in zip(columns, row))
      for idx, row in zip(ts.index, values)
  ]
  return "\n".join(lines)
