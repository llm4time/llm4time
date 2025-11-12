from ...data import TimeSeries, UniTimeSeries, MultiTimeSeries


def to_symbol(ts: TimeSeries) -> str:
  def directions(vals):
    prev = None
    for v in vals:
      if prev is None:
        yield "→"
      elif v > prev:
        yield "↑"
      elif v < prev:
        yield "↓"
      else:
        yield "→"
      prev = v

  if isinstance(ts, UniTimeSeries):
    values = ts.to_list()
    header = f"{ts.index.name},Value,DirectionIndicator"
    lines = [f"{idx},{v},{d}" for idx, v, d in zip(
        ts.index, values, directions(values))]

  elif isinstance(ts, MultiTimeSeries):
    values = ts.to_numpy().tolist()
    cols = ts.columns
    header = ts.index.name + "," + ",".join(f"{c},{c}_DirectionIndicator" for c in cols)

    prev_row = None
    lines = []
    for idx, row in zip(ts.index, values):
      row_parts = []
      if prev_row is None:
        row_parts = [str(v) + ",→" for v in row]
      else:
        row_parts = [
            f"{v},{'↑' if v > pv else '↓' if v < pv else '→'}"
            for v, pv in zip(row, prev_row)
        ]
      lines.append(f"{idx}," + ",".join(row_parts))
      prev_row = row
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

  return header + "\n" + "\n".join(lines)
