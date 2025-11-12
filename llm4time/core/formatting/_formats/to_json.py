from ...data import TimeSeries, UniTimeSeries, MultiTimeSeries
import json


def to_json(ts: TimeSeries) -> str:
  if isinstance(ts, UniTimeSeries):
    columns = [ts.name]
    values = [[v] for v in ts.to_list()]
  elif isinstance(ts, MultiTimeSeries):
    columns = ts.columns
    values = ts.to_numpy().tolist()
  else:
    raise TypeError(f"Expected TimeSeries, got {type(ts).__name__}.")

  data = [
      {ts.index.name: idx, **{col: val for col, val in zip(columns, row)}}
      for idx, row in zip([str(idx) for idx in ts.index], values)
  ]
  return json.dumps(data)
