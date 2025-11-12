import os
import pandas as pd
from llm4time._infra import logger
from .unitimeseries import UniTimeSeries
from .multitimeseries import MultiTimeSeries


def read_file(path_or_df: str | pd.DataFrame, index_col: str = None) -> MultiTimeSeries | UniTimeSeries:
  """
  Carrega dados de séries temporais a partir de um arquivo.

  Esta função identifica a extensão do arquivo e utiliza a função de leitura
  apropriada. Formatos suportados: CSV, XLSX, JSON, Parquet.

  Args:
      path_or_df (str | DataFrame): Caminho para o arquivo de dados ou um DataFrame do pandas.
      index_col (str | None): Define a coluna que será usada como DateTimeIndex.

  Returns:
      MultiTimeSeries | UniTimeSeries: Série Temporal contendo os dados carregados.
  """
  try:
    if isinstance(path_or_df, pd.DataFrame):
      df = path_or_df.copy()
    elif isinstance(path_or_df, str):
      _, ext = os.path.splitext(path_or_df)
      ext = ext.lower()
      readers_map = {
          ".csv": pd.read_csv,
          ".xlsx": pd.read_excel,
          ".json": pd.read_json,
          ".parquet": pd.read_parquet,
      }
      if ext not in readers_map:
        raise ValueError("Supported extensions: .csv, .xlsx, .json, .parquet")

      df = readers_map[ext](path_or_df)
    else:
      raise ValueError("Input must be a file path or a pandas DataFrame.")

    if index_col in df.columns:
      idx1 = pd.to_datetime(df[index_col], dayfirst=False, errors="coerce")
      idx2 = pd.to_datetime(df[index_col], dayfirst=True, errors="coerce")
      df[index_col] = idx1 if idx1.notna().sum() == len(df) else idx2
      df = df.set_index(index_col)
      if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    elif index_col is not None:
      raise ValueError(f"Index column '{index_col}' not found in data.")

    try:
      df.index.freq = pd.infer_freq(df.index)
    except:
      pass

    if df.shape[1] == 1:
      ts = UniTimeSeries(df[df.columns[0]])
    else:
      ts = MultiTimeSeries(df)

    return ts

  except FileNotFoundError:
    logger.error("Time series data file not found.")
    raise FileNotFoundError("File not found.")

  except Exception:
    logger.exception("Error reading time series data file.")
    raise Exception("An unexpected error occurred.")
