import lmstudio as lms
from ._base import Model, ModelResponse
import time


class LMStudio(Model):

  def __init__(self, model: str) -> None:
    """
    Inicializa a classe LMStudio com o modelo especificado.

    Args:
        model (str): Nome ou caminho do modelo LM Studio.
    """
    self.model = model

  def predict(self, content: str, temperature: float = 0.7, **kwargs) -> tuple[str, int, int, float]:
    client = lms.llm(self.model)

    config = {"temperature": temperature}
    config.update(kwargs)

    start_time = time.time()
    response = client.respond(content, config=config)
    end_time = time.time()

    raw = response.text if hasattr(
        response, 'text') else str(response)

    input_tokens = response.stats.prompt_tokens_count if hasattr(
        response, "stats") else 0
    output_tokens = response.stats.predicted_tokens_count if hasattr(
        response, "stats") else 0

    return ModelResponse(
        raw=raw,
        predicted=self._output(raw),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        time=end_time - start_time
    )
