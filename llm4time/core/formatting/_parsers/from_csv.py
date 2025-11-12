import pandas as pd
from io import StringIO
from ...data import TimeSeries, read_file


def from_csv(string: str) -> TimeSeries:
  df = pd.read_csv(StringIO(string))
  return read_file(df, index_col=df.columns[0])
