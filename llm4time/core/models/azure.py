from openai import AzureOpenAI as Client
from ._base import Model, ModelResponse
import time


class AzureOpenAI(Model):

  def __init__(self, model: str, api_key: str, azure_endpoint: str, api_version: str) -> ModelResponse:
    """
    Inicializa a classe AzureOpenAI com configurações de conexão.

    Args:
        model (str): Nome do modelo Azure OpenAI.
        api_key (str): Chave de API para autenticação.
        azure_endpoint (str): URL do endpoint Azure OpenAI.
        api_version (str): Versão da API Azure OpenAI.
    """
    self.model = model
    self.api_key = api_key
    self.azure_endpoint = azure_endpoint
    self.api_version = api_version
    super().__init__(model)

  def predict(self, content: str, temperature: float = 0.7, **kwargs) -> ModelResponse:
    client = Client(
        api_key=self.api_key,
        azure_endpoint=self.azure_endpoint,
        api_version=self.api_version
    )

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
