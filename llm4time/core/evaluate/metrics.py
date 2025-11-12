import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import sem as scipy_sem


class Metrics:
  def __init__(self, y_val: list[float], y_pred: list[float]) -> None:
    self.y_val = np.array(y_val)[~np.isnan(y_val)]
    self.y_pred = np.array(y_pred)[~np.isnan(y_pred)]

  def smape(self, decimals: int = 2) -> float:
    """
    sMAPE — Erro Percentual Absoluto Simétrico Médio.

    Mede a média dos erros percentuais absolutos entre valores observados e preditos,
    normalizando pela média dos valores absolutos observados e preditos.

    Args:
        decimals (int, optional): Número de casas decimais para arredondamento.
                                  Padrão é 2.

    Returns:
        float: Valor do sMAPE (duas casas decimais).
    """
    numerator = np.abs(self.y_val - self.y_pred)
    denominator = (np.abs(self.y_val) + np.abs(self.y_pred)) / 2
    epsilon = 1e-10
    smape = np.mean(numerator / (denominator + epsilon)) * 100
    return round(smape, decimals)

  def mae(self, decimals: int = 2) -> float:
    """
    MAE — Erro Absoluto Médio.
    Mede a média dos erros absolutos entre valores observados e preditos,
    fornecendo uma medida direta da acurácia das previsões.

    Args:
        decimals (int, optional): Número de casas decimais para arredondamento.
                                  Padrão é 2.

    Returns:
        float: Valor do MAE (duas casas decimais).
    """
    mae = mean_absolute_error(self.y_val, self.y_pred)
    return round(mae, decimals)

  def rmse(self, decimals: int = 2) -> float:
    """
    RMSE — Raiz do Erro Quadrático Médio.

    Mede a média dos erros quadráticos entre valores observados e preditos,
    penalizando erros maiores.

    Returns:
        float: Valor do RMSE (duas casas decimais).
    """
    rmse = root_mean_squared_error(self.y_val, self.y_pred)
    return round(rmse, decimals)

  def sem(errors: list[float], decimals: int = 4) -> float:
    """
    SEM — Erro Padrão da Média.
    Mede a precisão da média dos erros, útil para avaliar a confiabilidade
    das previsões.

    Args:
        errors (list[float]): Lista de erros (diferenças entre valores observados e preditos).
        decimals (int, optional): Número de casas decimais para arredondamento. Padrão é 4.

    Returns:
        float: Valor do SEM (arredondado para o número especificado de casas decimais).
    """
    return round(scipy_sem(errors), decimals)
