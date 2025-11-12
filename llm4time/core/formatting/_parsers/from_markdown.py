import pandas as pd
from io import StringIO
from ...data import TimeSeries, read_file


def from_markdown(string: str) -> TimeSeries:
  lines = string.strip().splitlines()
  data = "\n".join([line.strip().strip("|") for line in [lines[0]] + lines[2:]])
  df = pd.read_csv(StringIO(data), sep="|", engine="python", skipinitialspace=True)
  return read_file(df, index_col=df.columns[0])
