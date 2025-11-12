import re
import pandas as pd
from io import StringIO
from ...data import TimeSeries, read_file


def from_plain(string: str) -> TimeSeries:
  data = [{k.strip(): v.strip() for k, v in (p.split(":", 1) for p in line.split(","))}
          for line in string.strip().splitlines()]
  df = pd.read_csv(StringIO(pd.DataFrame(data).to_csv(index=False)))
  return read_file(df, index_col=df.columns[0])
