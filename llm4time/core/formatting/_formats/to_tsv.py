from ...data import TimeSeries, UniTimeSeries, MultiTimeSeries


def to_tsv(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries):
    header = f"{ts.index.name}\t{ts.name}"
    values = [[v] for v in ts.to_list()]
  elif isinstance(ts, MultiTimeSeries):
    header = ts.index.name + "\t" + "\t".join(map(str, ts.columns))
    values = ts.to_numpy().tolist()
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

  lines = [
      f"{idx}\t" + "\t".join(str(v) if v is not None else "nan" for v in row)
      for idx, row in zip(ts.index, values)
  ]
  return header + "\n" + "\n".join(lines)
