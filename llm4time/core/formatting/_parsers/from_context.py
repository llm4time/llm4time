import pandas as pd
from io import StringIO
from ...data import TimeSeries, read_file
import re


def from_context(string: str) -> TimeSeries:
  string = re.sub(r'\[([^\]]+)\]', r'\1', string)
  df = pd.read_csv(StringIO(string))
  return read_file(df, index_col=df.columns[0])
