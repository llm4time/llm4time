from ...data import TimeSeries, UniTimeSeries, MultiTimeSeries


def to_context(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries):
    header = f"{ts.index.name},{ts.name}"
    values = [[v] for v in ts.to_list()]
  elif isinstance(ts, MultiTimeSeries):
    header = f"{ts.index.name},{",".join(ts.columns)}"
    values = ts.to_numpy().tolist()
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

  lines = [f"{idx}," + ",".join(f"[{v}]" for v in row)
           for idx, row in zip(ts.index, values)]
  return header + "\n" + "\n".join(lines)
