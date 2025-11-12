import json
import pandas as pd
from ...data import TimeSeries, read_file


def from_json(string: str) -> TimeSeries:
  data = json.loads(string)
  if not data:
    return None
  df = pd.DataFrame(data)
  return read_file(df, index_col=df.columns[0])
