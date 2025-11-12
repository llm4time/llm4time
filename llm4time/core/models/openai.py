from openai import OpenAI as Client
from ._base import Model, ModelResponse
import time


class OpenAI(Model):

  def __init__(self, model: str, api_key: str, base_url: str) -> None:
    """
    Inicializa a classe OpenAI com configurações de conexão.

    Args:
        model (str): Nome do modelo OpenAI.
        api_key (str): Chave de API para autenticação.
        base_url (str): URL base do endpoint OpenAI.
    """
    self.model = model
    self.api_key = api_key
    self.base_url = base_url

  def predict(self, content: str, temperature: float = 0.7, **kwargs) -> ModelResponse:
    client = Client(api_key=self.api_key, base_url=self.base_url)

    params = {
        "model": self.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature
    }
    params.update(kwargs)

    start_time = time.time()
    response = client.chat.completions.create(**params)
    end_time = time.time()

    raw = response.choices[0].message.content
    usage = response.usage

    return ModelResponse(
        raw=raw,
        predicted=self._output(raw),
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        time=end_time - start_time
    )
