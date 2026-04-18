import llm4time as l4t
from storage import repository as repo
from config import logger
import numpy as np
import random


class API:
  def __init__(self, model: str, provider: str):
    self.model = model
    self.provider = provider
    self._model_data: dict | None = None

  @property
  def model_data(self) -> dict:
    """Lazy load model data from database."""
    if self._model_data is None:
      self._model_data = self._fetch_model_data()
    return self._model_data

  def _fetch_model_data(self) -> dict:
    """Fetch model configuration from database."""
    models = repo.models().select(self.provider)
    for m in models:
      if m["name"] == self.model:
        return m
    logger.warning(f"Model data not found for {self.model} ({self.provider})")
    return {}

  def response(self, context: str, data: str, temperature: float, **kwargs) -> l4t.ModelResponse:
    """Generate response using the configured provider."""
    context = str(context)
    data = str(data)

    providers = {
        str(l4t.Provider.LM_STUDIO): self._lmstudio,
        str(l4t.Provider.OPENAI): self._openai,
        str(l4t.Provider.AZURE): self._azure_openai,
    }

    handler = providers.get(str(self.provider))
    if handler:
      return handler(context, data, temperature, **kwargs)

    logger.error(f"Unknown provider: {self.provider}")
    return l4t.ModelResponse(
        raw=f"Unknown provider: {self.provider}",
        predicted=None,
        input_tokens=None,
        output_tokens=None,
        time=None,
    )

  def _lmstudio(self, context: str, data: str, temperature: float, **kwargs) -> l4t.ModelResponse:
    """Handle LM Studio provider."""
    return self._call_client(
        lambda model: l4t.LMStudio(model),
        context, data, temperature, **kwargs
    )

  def _openai(self, context: str, data: str, temperature: float, **kwargs) -> l4t.ModelResponse:
    """Handle OpenAI provider."""
    model_data = self.model_data
    api_key = model_data.get("api_key")
    base_url = model_data.get("endpoint")

    if not api_key:
      logger.error("OpenAI API key not configured")
      return self._error_response("API key not configured")

    logger.info(f"BASE_URL: {base_url}")

    return self._call_client(
        lambda model: l4t.OpenAI(api_key=api_key, base_url=base_url, model=model),
        context, data, temperature, **kwargs
    )

  def _azure_openai(self, context: str, data: str, temperature: float, **kwargs) -> l4t.ModelResponse:
    """Handle Azure OpenAI provider."""
    model_data = self.model_data
    api_key = model_data.get("api_key")
    endpoint = model_data.get("endpoint")
    api_version = model_data.get("api_version")

    if not all([api_key, endpoint, api_version]):
      logger.error("Azure OpenAI configuration incomplete")
      return self._error_response("Azure configuration incomplete")

    logger.info(f"ENDPOINT: {endpoint}")
    logger.info(f"API_VERSION: {api_version}")

    return self._call_client(
        lambda model: l4t.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            model=model,
        ),
        context, data, temperature, **kwargs
    )

  def _call_client(self, client_factory, context: str, data: str, temperature: float, **kwargs) -> l4t.ModelResponse:
    """Execute prediction using the provided client factory."""
    try:
      context = str(context)
      data = str(data)
      client = client_factory(self.model)
      response = client.predict(context, data, temperature=temperature, **kwargs)

      logger.info(f"Response: {response.predicted}")
      logger.info(f"Input Tokens: {response.input_tokens}")
      logger.info(f"Output Tokens: {response.output_tokens}")
      logger.info(f"Time: {response.time:.2f} seconds")

      return response
    except Exception as e:
      logger.error(f"Error generating response: {e}")
      return self._error_response(str(e))

  @staticmethod
  def _error_response(message: str) -> l4t.ModelResponse:
    """Create an error response."""
    return l4t.ModelResponse(
        raw=message,
        predicted=None,
        input_tokens=None,
        output_tokens=None,
        time=None,
    )

  @staticmethod
  def _mock(ts: l4t.TimeSeries, tsformat: l4t.TSFormat, tstype: l4t.TSType) -> l4t.ModelResponse:
    """Generate mock response for testing."""
    pred = ts.copy()
    for column in pred.columns:
      pred[column] = pred[column] + np.random.normal(0, 0.5, size=len(pred))

    response = pred.to_str(format=tsformat, type=tstype)
    response_time = round(random.uniform(0.5, 2.5), 2)
    input_tokens = random.randint(10, 500)
    output_tokens = random.randint(10, 500)

    logger.info(f"Input Tokens: {input_tokens}")
    logger.info(f"Output Tokens: {output_tokens}")
    logger.info(f"Time: {response_time:.2f} seconds")

    return l4t.ModelResponse(
        raw=response,
        predicted=response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        time=response_time,
    )
