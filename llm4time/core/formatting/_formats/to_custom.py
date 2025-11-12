from ...data import TimeSeries, UniTimeSeries, MultiTimeSeries


def to_custom(ts: TimeSeries, sep: str = "|") -> str:
  if isinstance(ts, UniTimeSeries):
    header = f"{ts.index.name}{sep}{ts.name}"
    values = [[v] for v in ts.to_list()]
  elif isinstance(ts, MultiTimeSeries):
    header = f"{ts.index.name}{sep}" + sep.join(ts.columns)
    values = ts.to_numpy().tolist()
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

  lines = [f"{idx}{sep}" + sep.join(str(v) for v in row)
           for idx, row in zip(ts.index, values)]
  return header + "\n" + "\n".join(lines)
