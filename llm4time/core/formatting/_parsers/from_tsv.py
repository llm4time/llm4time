import pandas as pd
from io import StringIO
from ...data import TimeSeries, read_file


def from_tsv(string: str) -> TimeSeries:
  df = pd.read_csv(StringIO(string), sep="\t")
  return read_file(df, index_col=df.columns[0])
