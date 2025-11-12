from ...data import TimeSeries, UniTimeSeries, MultiTimeSeries


def to_array(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries) or isinstance(ts, MultiTimeSeries):
    return str(ts.values.tolist() if not ts.empty else [])
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")
