from enum import Enum
from typing import Self
from abc import ABC, abstractmethod
import llm4time as lt
import pandas as pd
import random
import os


class Sampling(str, Enum):
  FRONTEND = "frontend"
  BACKEND = "backend"
  RANDOM = "random"
  UNIFORM = "uniform"


class TSFormat(str, Enum):
  ARRAY = "array"
  CONTEXT = "context"
  CSV = "csv"
  CUSTOM = "custom"
  JSON = "json"
  MARKDOWN = "markdown"
  PLAIN = "plain"
  SYMBOL = "symbol"
  TSV = "tsv"


class TSType(str, Enum):
  NUMERIC = "numeric"
  TEXTUAL = "textual"


class TimeSeriesStatistics(ABC):

  @abstractmethod
  def stl(self: Self, period: int | None, freq: str | None, decimals: int = 4) -> dict:
    """
    Realiza a decomposição STL (Seasonal-Trend decomposition using LOESS) da série temporal.

    Args:
        period (int, opcional): O período da sazonalidade.
        freq (str, opcional): A frequência da série temporal.
        decimals (int, opcional): O número de casas decimais a serem arredondadas.

    Returns:
        dict: Um dicionário contendo as componentes da série temporal (tendência, sazonalidade, resíduos) e suas forças relativas.
    """
    ...

  @abstractmethod
  def mean(self: Self, decimals: int | None, **kwargs) -> float:
    """
    Calcula a média aritmética da série temporal.

    Args:
        decimals (int | None): Número de casas decimais para arredondamento.
                               Se None, não arredonda.
        **kwargs: Argumentos adicionais específicos da implementação.

    Returns:
        float: Valor da média da série temporal.
    """
    ...

  @abstractmethod
  def median(self: Self, decimals: int | None, **kwargs) -> float:
    """
    Calcula a mediana da série temporal.

    Args:
        decimals (int | None): Número de casas decimais para arredondamento.
                               Se None, não arredonda.
        **kwargs: Argumentos adicionais específicos da implementação.

    Returns:
        float: Valor da mediana da série temporal.
    """
    ...


class TimeSeriesImputation(ABC):

  @abstractmethod
  def impute_mean(self: Self, decimals: int | None, inplace: bool | None) -> Self | None:
    """
    Substitui valores ausentes pela média da série.

    Calcula a média dos valores presentes na série e preenche os valores ausentes
    com este valor arredondado para o número especificado de casas decimais.

    Args:
        decimals (int | None): Número de casas decimais para arredondar a média.
        inplace (bool | None): Se True, modifica a série original. Caso False, retorna
                              uma nova série.

    Returns:
        TimeSeries | None: Série temporal com valores ausentes imputados
                                    ou None se inplace=True.
    """
    ...

  @abstractmethod
  def impute_median(self: Self, decimals: int | None, inplace: bool | None) -> Self | None:
    """
    Substitui valores ausentes pela mediana da série.

    Calcula a mediana dos valores presentes na série e preenche os valores ausentes
    com este valor arredondado para o número especificado de casas decimais.

    Args:
        decimals (int | None): Número de casas decimais para arredondar a mediana.
        inplace (bool | None): Se True, modifica a série original. Caso False, retorna
                              uma nova série.

    Returns:
        TimeSeries | None: Série temporal com valores ausentes imputados
                                    ou None se inplace=True.
    """
    ...

  @abstractmethod
  def impute_ffill(self: Self, inplace: bool | None) -> Self | None:
    """
    Imputa valores ausentes usando forward fill seguido de backward fill.

    Preenche valores ausentes primeiro propagando o último valor válido
    à frente e, em seguida, propagando o próximo valor válido para trás
    caso ainda existam valores ausentes.

    Args:
        inplace (bool | None): Se True, modifica a série original. Caso False, retorna
                              uma nova série.

    Returns:
        TimeSeries | None: Série temporal com valores ausentes imputados
                                    ou None se inplace=True.
    """
    ...

  @abstractmethod
  def impute_bfill(self: Self, inplace: bool | None) -> Self | None:
    """
    Imputa valores ausentes usando backward fill seguido de forward fill.

    Preenche valores ausentes primeiro propagando o próximo valor válido
    para trás e, em seguida, propagando o último valor válido à frente
    caso ainda existam valores ausentes.

    Args:
        inplace (bool | None): Se True, modifica a série original. Caso False, retorna
                        uma nova série.

    Returns:
        TimeSeries | None: Série temporal com valores ausentes imputados
                                    ou None se inplace=True.
    """
    ...

  @abstractmethod
  def impute_sma(self: Self, window: int, min_periods: int | None, decimals: int | None, inplace: bool | None) -> Self | None:
    """
    Imputa valores ausentes usando média móvel simples (SMA).

    Calcula a média móvel simples com a janela especificada e substitui
    valores ausentes. Após a SMA, aplica forward e backward fill para
    preencher valores restantes.

    Args:
        window (int): Tamanho da janela para cálculo da média móvel.
        min_periods (int): Número mínimo de valores não nulos necessários
                           para calcular a média.
        decimals (int | None): Número de casas decimais para arredondamento.
        inplace (bool | None): Se True, modifica a série original. Caso False, retorna
                        uma nova série.

    Returns:
        TimeSeries | None: Série temporal com valores ausentes imputados
                                    ou None se inplace=True.
    """
    ...

  @abstractmethod
  def impute_ema(self: Self, span: int, adjust: bool | None, decimals: int | None, inplace: bool | None) -> Self | None:
    """
    Imputa valores ausentes usando média móvel exponencial (EMA).

    Calcula a EMA com o parâmetro span especificado e preenche valores ausentes.
    Após a EMA, aplica forward e backward fill para preencher valores restantes.

    Args:
        span (int): Período da média móvel exponencial.
        adjust (bool): Se True, ajusta os pesos para considerar toda a série.
        decimals (int | None): Número de casas decimais para arredondamento.
        inplace (bool | None): Se True, modifica a série original. Caso False, retorna
                        uma nova série.

    Returns:
        TimeSeries | None: Série temporal com valores ausentes imputados
                                    ou None se inplace=True.
    """
    ...

  @abstractmethod
  def impute_interpolate(self: Self, method: str, order: int | None, inplace: bool | None) -> Self | None:
    """
    Imputa valores ausentes usando interpolação.

    Suporta interpolação linear ou spline. Após a interpolação, aplica
    forward e backward fill para preencher valores restantes.

    Args:
        method (str): Método de interpolação. Valores aceitos: 'linear', 'spline'.
        order (int | None): Ordem da spline, caso method='spline'.
        inplace (bool | None): Se True, modifica a série original. Caso False, retorna
                        uma nova série.

    Returns:
        TimeSeries | None: Série temporal com valores ausentes imputados
                                    ou None se inplace=True.

    Raises:
        ValueError: Se o método fornecido não for suportado.
    """
    ...

  @abstractmethod
  def std(self: Self, decimals: int | None, **kwargs) -> float:
    """
    Calcula o desvio padrão da série temporal.

    Args:
        decimals (int | None): Número de casas decimais para arredondamento.
                               Se None, não arredonda.
        **kwargs: Argumentos adicionais específicos da implementação.

    Returns:
        float: Valor do desvio padrão da série temporal.
    """
    ...

  @abstractmethod
  def min(self: Self, decimals: int | None, **kwargs) -> float:
    """
    Retorna o menor valor presente na série temporal.

    Args:
        decimals (int | None): Número de casas decimais para arredondamento.
                               Se None, não arredonda.
        **kwargs: Argumentos adicionais específicos da implementação.

    Returns:
        float: Valor mínimo da série temporal.
    """
    ...

  @abstractmethod
  def max(self: Self, decimals: int | None, **kwargs) -> float:
    """
    Retorna o maior valor presente na série temporal.

    Args:
        decimals (int | None): Número de casas decimais para arredondamento.
                               Se None, não arredonda.
        **kwargs: Argumentos adicionais específicos da implementação.

    Returns:
        float: Valor máximo da série temporal.
    """
    ...

  @abstractmethod
  def quantile(self: Self, decimals: int | None, **kwargs) -> float:
    """
    Calcula o quantil da série temporal para um valor de probabilidade especificado.

    Args:
        decimals (int | None): Número de casas decimais para arredondamento.
                               Se None, não arredonda.
        **kwargs: Argumentos adicionais, incluindo parâmetro 'q' para o quantil desejado (0 <= q <= 1).

    Returns:
        float: Valor do quantil correspondente da série temporal.
    """
    ...


class TimeSeriesMetrics(ABC):

  @abstractmethod
  def smape(self, y_pred: list[float], decimals: int | None) -> float:
    """
    sMAPE — Erro Percentual Absoluto Simétrico Médio.

    Mede a média dos erros percentuais absolutos entre valores observados e preditos,
    normalizando pela média dos valores absolutos observados e preditos.

    Args:
        y_pred (list[float]): Valores preditos.
        decimals (int, optional): Número de casas decimais para arredondamento.

    Returns:
        float: Valor do sMAPE.
    """
    ...

  @abstractmethod
  def mae(self, y_pred: list[float], decimals: int | None) -> float:
    """
    MAE — Erro Absoluto Médio.

    Mede a média dos erros absolutos entre valores observados e preditos,
    fornecendo uma medida direta da acurácia das previsões.

    Args:
        y_pred (list[float]): Valores preditos.
        decimals (int, optional): Número de casas decimais para arredondamento.

    Returns:
        float: Valor do MAE.
    """
    ...

  @abstractmethod
  def rmse(y_pred: list[float], decimals: int | None) -> float:
    """
    RMSE — Raiz do Erro Quadrático Médio.

    Mede a média dos erros quadráticos entre valores observados e preditos,
    penalizando erros maiores.

    Args:
        y_pred (list[float]): Valores preditos.
        decimals (int, optional): Número de casas decimais para arredondamento.

    Returns:
        float: Valor do RMSE.
    """
    ...


class TimeSeriesPlot(ABC):

  @abstractmethod
  def linechart(self: Self):
    """
    Cria um gráfico de linha da série temporal.
    """
    ...

  @abstractmethod
  def lineplot(self: Self):
    """
    Cria um gráfico de linha para uma série temporal indexada por períodos.
    """
    ...

  @abstractmethod
  def barplot(self: Self):
    """
    Cria um gráfico de barras das estatísticas.
    """
    ...

  @abstractmethod
  def stlplot(self: Self):
    """
    Cria gráficos das componentes da decomposição STL (tendência, sazonalidade, resíduos).
    """
    ...


class TimeSeries(ABC):
  @abstractmethod
  def agg_duplicates(self: Self, method: str, inplace: bool | None) -> Self | None:
    """
    Remove ou agrega valores duplicados no índice da série temporal.

    Args:
        method (str): Método para tratar duplicatas.
        inplace (bool | None): Se True, modifica o objeto atual e retorna None.
            Se False, retorna uma nova cópia com duplicatas resolvidas.

    Returns:
        Self | None: Série temporal com duplicatas resolvidas ou None se `inplace=True`.
    """
    ...

  def normalize(self: Self, freq: str, start: str = None, end: str = None) -> Self:
    """
    Reindexa a série temporal para uma frequência específica, preenchendo lacunas.

    Cria um índice contínuo entre as datas de início e fim e reindexa a série,
    preenchendo os valores ausentes com NaN.

    Args:
        freq (str): Frequência da série temporal. Se None, tenta inferir da série.
        start (str | None): Data de início. Se None, usa a menor data da série.
        end (str | None): Data de término. Se None, usa a maior data da série.

    Returns:
        TimeSeries | None: Série temporal reindexada e normalizada.

    Raises:
        ValueError: Se não for possível inferir a frequência automaticamente.
    """
    start_date = pd.to_datetime(start) if start else self.index.min()
    end_date = pd.to_datetime(end) if end else self.index.max()

    if (freq := (self.index.freq or freq)) is None:
      raise ValueError("Error trying to infer frequency automatically.")

    self.index.freq = freq
    full_idx = pd.date_range(start=start_date, end=end_date, freq=freq)
    return self.reindex(full_idx, fill_value=pd.NA)

  def split(self: Self, start: str | pd.DatetimeIndex, end: str | pd.DatetimeIndex, periods: int) -> tuple[Self, Self]:
    """
    Divide a série temporal em duas partes com base em datas de início e fim.

    A primeira parte contém os dados entre start e end,
    enquanto a segunda contém os dados após end.

    Args:
        start (str | pd.DatetimeIndex): Data de início do conjunto de treinamento.
        end (str | pd.DatetimeIndex): Data de término do conjunto de treinamento.
        periods (int): Quantidade de valores a considerar no conjunto de validação.

    Returns:
        tuple[TimeSeries, TimeSeries]: Par de séries temporais (treino, validação).
    """
    train = self[(self.index >= str(start)) & (self.index <= str(end))]
    val = self[self.index > str(end)][:periods]
    return train, val

  def slide(self: Self, method: Sampling, window: int, samples: int, step: int = None) -> list[tuple[Self, Self]]:
    """
    Gera amostras sequenciais da série temporal em pares de janelas (entrada, saída).

    Cria subconjuntos consecutivos da série temporal a partir de diferentes estratégias
    de amostragem, onde cada amostra consiste em uma janela de entrada seguida
    imediatamente por uma janela de saída. O método define como as janelas iniciais
    são selecionadas.

    Args:
        method (Sampling): Estratégia de amostragem. Métodos suportados:
            - 'frontend': Gera janelas sequenciais a partir do início da série.
            - 'backend': Gera janelas sequenciais a partir do final da série.
            - 'random': Seleciona aleatoriamente os pontos iniciais das janelas.
            - 'uniform': Gera janelas distribuídas uniformemente ao longo da série.
        window (int): Tamanho de cada janela.
        samples (int): Número total de amostras a serem geradas.
        step (int): Intervalo entre os pontos iniciais das janelas.

    Returns:
        list[tuple[TimeSeries, TimeSeries]]: Lista de tuplas (entrada, saída), onde cada
            elemento representa uma amostra contendo duas janelas consecutivas da série.

    Raises:
        ValueError: Se o método informado não for um dos suportados:
                    'frontend', 'backend', 'random' ou 'uniform'.
    """
    max_start = len(self) - 2 * window

    if method == Sampling.FRONTEND:
      idxs = [i * 2 * window for i in range(samples)]

    elif method == Sampling.BACKEND:
      total = len(self) // window - 1
      samples = min(samples, total)
      idxs = [len(self) - (samples - i) * 2 * window for i in range(samples)]

    elif method == Sampling.RANDOM:
      if max_start < 0:
        return []
      idxs = sorted(random.sample(range(max_start + 1), k=min(samples, max_start + 1)))

    elif method == Sampling.UNIFORM:
      if max_start < 0 or samples <= 0:
        return []
      if step is None:
        step = max_start / (samples - 1) if samples > 1 else 0
        idxs = [int(i * step) for i in range(samples)]
      else:
        idxs = list(range(0, max_start + 1, step))[:samples]

    else:
      raise ValueError('Supported methods: frontend, backend, random, uniform.')

    windows = []
    for idx in idxs:
      end_out = idx + 2 * window
      if end_out > len(self):
        break
      windows.append((
          self._constructor(self.iloc[idx:idx + window].copy()),
          self._constructor(self.iloc[idx + window:end_out].copy())
      ))
    return windows

  def to_str(self: Self, format: TSFormat, type: TSType = TSType.NUMERIC) -> str:
    """
    Converte a série temporal para uma representação em string em diversos formatos.

    Args:
        format (TSFormat): Formato desejado para a conversão. Formatos suportados:
            - TSFormat.ARRAY: Retorna a série como array.
            - TSFormat.CONTEXT: Retorna a série em formato contextual.
            - TSFormat.CSV: Retorna a série como CSV.
            - TSFormat.CUSTOM: Formato customizado.
            - TSFormat.JSON: Retorna a série como JSON.
            - TSFormat.MARKDOWN: Retorna a série como tabela Markdown.
            - TSFormat.PLAIN: Retorna a série como texto simples.
            - TSFormat.SYMBOL: Retorna a série usando notação simbólica.
            - TSFormat.TSV: Retorna a série como TSV.
        type (TSType, optional): Tipo da representação desejada. Pode ser:
            - TSType.NUMERIC (padrão): Mantém os valores numéricos da série.
            - TSType.TEXTUAL: Converte a série para uma forma textual codificada.

    Returns:
        str: Representação em string da série temporal no formato e tipo especificados.

    Raises:
        ValueError: Se o `format` fornecido não for suportado.
    """
    ts = lt.encode_textual(self) if type == TSType.TEXTUAL else self

    formats_map = {
        TSFormat.ARRAY: lt.to_array,
        TSFormat.CONTEXT: lt.to_context,
        TSFormat.CSV: lt.to_csv,
        TSFormat.CUSTOM: lt.to_custom,
        TSFormat.JSON: lt.to_json,
        TSFormat.MARKDOWN: lt.to_markdown,
        TSFormat.PLAIN: lt.to_plain,
        TSFormat.SYMBOL: lt.to_symbol,
        TSFormat.TSV: lt.to_tsv,
    }
    if format not in formats_map:
      raise ValueError(f"Unknown format: {format}.")
    return formats_map[format](ts)

  def to_file(self: Self, path: str) -> None:
    """
    Salva a série temporal em um arquivo no formato especificado pela extensão.

    Exporta a série temporal para disco em um dos formatos suportados:
    CSV, Excel (XLSX), JSON ou Parquet. A extensão do arquivo é usada para
    determinar automaticamente o formato de exportação.

    Args:
        path (str): Caminho completo do arquivo de saída, incluindo o nome
                    e a extensão (ex: 'data/serie.csv').
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext == ".csv":
      self.to_csv(path, index=True)
    elif ext == ".xlsx":
      self.to_excel(path, index=True)
    elif ext == ".json":
      self.to_json(path, orient="records", date_format="iso")
    elif ext == ".parquet":
      self.to_parquet(path, index=True)
    else:
      raise ValueError(f"Supported extensions: .csv, .xlsx, .json, .parquet")
