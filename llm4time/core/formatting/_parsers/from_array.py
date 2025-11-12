import ast
import pandas as pd
from ...data import TimeSeries, read_file


def from_array(string: str) -> TimeSeries:
  data = ast.literal_eval(string) or []
  df = pd.DataFrame(data)
  return read_file(df)
