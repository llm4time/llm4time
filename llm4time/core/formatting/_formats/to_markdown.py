from ...data import TimeSeries, UniTimeSeries, MultiTimeSeries


def to_markdown(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries):
    header = f"|{ts.index.name}|{ts.name}|"
    values = [[v] for v in ts.to_list()]
  elif isinstance(ts, MultiTimeSeries):
    header = "|" + ts.index.name + "|" + "|".join(ts.columns) + "|"
    values = ts.to_numpy().tolist()
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

  sep = "|" + "|".join("---" for _ in header.split("|") if _ != "") + "|"
  rows = [f"|{idx}|" + "|".join(str(v) for v in row) +
          "|" for idx, row in zip(ts.index, values)]
  return header + "\n" + sep + "\n" + "\n".join(rows)
