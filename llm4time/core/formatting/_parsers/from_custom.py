import pandas as pd
from io import StringIO
from ...data import TimeSeries, read_file


def from_custom(string: str) -> TimeSeries:
  df = pd.read_csv(StringIO(string), sep="|")
  return read_file(df, index_col=df.columns[0])
