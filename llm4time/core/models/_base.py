from enum import Enum
from typing import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass
import re


class Provider(str, Enum):
  LM_STUDIO = "lm_studio"
  OPENAI = "openai"
  AZURE = "azure"

  def __str__(self):
    return {
        Provider.LM_STUDIO: "LM Studio",
        Provider.OPENAI: "OpenAI",
        Provider.AZURE: "Azure"
    }[self]

  @classmethod
  def enum(cls, name: str):
    return next((m for m in cls if str(m) == name), None)


@dataclass(kw_only=True)
class ModelResponse:
  raw: str
  predicted: str
  input_tokens: int
  output_tokens: int
  time: float


class Model(ABC):

  @abstractmethod
  def predict(self: Self, content: str, temperature: float | None, **kwargs) -> ModelResponse:
    """
    Envia uma requisição para o modelo e retorna a resposta.

    Args:
        content (str): Conteúdo da mensagem do usuário a ser enviada.
        temperature (float | None): Grau de aleatoriedade da resposta.
        **kwargs: Argumentos adicionais passados para `client.chat.completions.create`.

    Returns:
        ModelResponse: Resposta do modelo com detalhes.
    """
    ...

  def _output(self, response: str) -> str:
    return re.findall(r'<out>(.*?)</out>', response, re.DOTALL)[-1].strip()
