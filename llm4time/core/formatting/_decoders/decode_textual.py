from ...data import TimeSeries, UniTimeSeries, MultiTimeSeries
import re


def decode_textual(ts: TimeSeries) -> TimeSeries:
  ts = ts.copy()

  def decode(v):
    s = str(v).strip()
    if re.fullmatch(r"[-\d\s.]+", s):
      return float(s.replace(" ", ""))
    return v

  if isinstance(ts, UniTimeSeries):
    ts = ts.apply(decode)

  elif isinstance(ts, MultiTimeSeries):
    for col in ts.columns:
      ts[col] = ts[col].apply(decode)

  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

  return ts
