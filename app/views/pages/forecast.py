import streamlit as st
import llm4time as l4t
import pandas as pd
from helpers import API
import storage.repository as repo
from utils import abspath
from config import logger
import io


# =============================================================================
# Styles
# =============================================================================
st.html("""
<style>
.st-key-time_window .stElementToolbar {
  display: none !important;
}
</style>
""")


# =============================================================================
# Sidebar Components
# =============================================================================

def render_dataset_selection() -> tuple[str | None, bytes | None, str | None]:
  """Render dataset selection and return filename, content, and frequency."""
  files = repo.uploads().select_all()
  if not files:
    st.info("No datasets available. Please [upload](/) a file first.")
    return None, None, None

  filenames = [f["filename"] for f in files]
  filename = st.selectbox("Dataset", filenames)

  if not filename:
    return None, None, None

  content, freq = get_file_content(filename)
  if not content:
    st.error("Failed to load file content.")
    return None, None, None

  return filename, content, freq


def render_column_selection(ts: l4t.MultiTimeSeries) -> list[str] | None:
  """Render column selection."""
  columns = st.multiselect("Select one or more columns", ts.columns.tolist())
  return columns if columns else None


def render_window_selection_datetime(ts: l4t.MultiTimeSeries, freq: str) -> tuple[str, str] | None:
  """Render window selection for datetime-based series (D, M, Y, h, min, s, ms)."""
  min_date = ts.index.min()
  max_date = ts.index.max()

  freq_lower = freq.lower()
  if freq_lower in ['y', 'a']:  # Year
    default_end = min(min_date + pd.DateOffset(years=1), max_date)
  elif freq_lower in ['m', 'ms']:  # Month or Millisecond
    if 'ms' in freq_lower:
      default_end = min(min_date + pd.Timedelta(milliseconds=100), max_date)
    else:
      default_end = min(min_date + pd.DateOffset(months=1), max_date)
  elif freq_lower in ['d', 'b']:  # Day or Business day
    default_end = min(min_date + pd.Timedelta(days=7), max_date)
  elif freq_lower in ['h']:  # Hour
    default_end = min(min_date + pd.Timedelta(hours=24), max_date)
  elif freq_lower in ['t', 'min']:  # Minute
    default_end = min(min_date + pd.Timedelta(hours=1), max_date)
  elif freq_lower in ['s']:  # Second
    default_end = min(min_date + pd.Timedelta(minutes=5), max_date)
  else:
    default_end = min(min_date + pd.Timedelta(days=2), max_date)

  if freq_lower in ['y', 'a', 'm']:
    date = st.date_input(
        "Date Range",
        value=(min_date.date(), default_end.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
        help="Select the input window to define the training data.",
        format="YYYY.MM.DD"
    )

    if len(date) < 2:
      st.warning("Please select both start and end dates.")
      st.stop()

    start_datetime = str(pd.Timestamp.combine(date[0], pd.Timestamp("00:00:00").time()))
    end_datetime = str(pd.Timestamp.combine(date[1], pd.Timestamp("23:59:59").time()))

  elif freq_lower in ['d', 'b']:
    st.write("**Window Selection**")
    date = st.date_input(
        "Date Range",
        value=(min_date.date(), default_end.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
        help="Select the input window to define the training data.",
        format="YYYY.MM.DD"
    )

    if len(date) < 2:
      st.warning("Please select both start and end dates.")
      st.stop()

    start_datetime = str(pd.Timestamp.combine(date[0], pd.Timestamp("00:00:00").time()))
    end_datetime = str(pd.Timestamp.combine(date[1], pd.Timestamp("23:59:59").time()))

  elif freq_lower in ['h', 't', 'min', 's']:
    st.write("**Window Selection**")

    date = st.date_input(
        "Date Range",
        value=(min_date.date(), default_end.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
        help="Select the date range for the training data.",
        format="YYYY.MM.DD"
    )

    if len(date) < 2:
      st.warning("Please select both start and end dates.")
      st.stop()

    dt = st.data_editor(
        pd.DataFrame({
            "start": [min_date.time()],
            "end": [default_end.time()]
        }),
        column_config={
            "start": st.column_config.TimeColumn(label="⏱ Start Time", format="HH:mm:ss"),
            "end": st.column_config.TimeColumn(label="⏱ End Time", format="HH:mm:ss")
        },
        hide_index=True,
        key="time_window"
    )

    start_datetime = str(pd.Timestamp.combine(date[0], dt["start"][0]))
    end_datetime = str(pd.Timestamp.combine(date[1], dt["end"][0]))

  elif freq_lower == 'ms':
    st.write("**Window Selection**")

    col1, col2 = st.columns(2)
    with col1:
      start_datetime = st.text_input(
          "Start Datetime",
          value=str(min_date),
          help="Format: YYYY-MM-DD HH:MM:SS.fff"
      )
    with col2:
      end_datetime = st.text_input(
          "End Datetime",
          value=str(default_end),
          help="Format: YYYY-MM-DD HH:MM:SS.fff"
      )

    try:
      pd.Timestamp(start_datetime)
      pd.Timestamp(end_datetime)
    except Exception:
      st.error("Invalid datetime format. Use: YYYY-MM-DD HH:MM:SS or YYYY-MM-DD HH:MM:SS.fff")
      st.stop()
  else:
    st.write("**Window Selection**")
    col1, col2 = st.columns(2)
    with col1:
      start_datetime = st.text_input(
          "Start Datetime",
          value=str(min_date),
          help="Start of the training window"
      )
    with col2:
      end_datetime = st.text_input(
          "End Datetime",
          value=str(default_end),
          help="End of the training window"
      )

  return start_datetime, end_datetime


def render_horizon_selection(freq: str) -> int:
  """Render horizon forecast selection with appropriate range based on frequency."""
  freq_lower = freq.lower()

  if freq_lower in ['y', 'a']:  # Year
    max_value = 10
    default_value = 1
    help_text = "Number of years to forecast."
  elif freq_lower in ['m']:  # Month
    max_value = 24
    default_value = 3
    help_text = "Number of months to forecast."
  elif freq_lower in ['d', 'b']:  # Day
    max_value = 90
    default_value = 7
    help_text = "Number of days to forecast."
  elif freq_lower in ['h']:  # Hour
    max_value = 168  # 7 days
    default_value = 24
    help_text = "Number of hours to forecast."
  elif freq_lower in ['t', 'min']:  # Minute
    max_value = 1440  # 24 hours
    default_value = 60
    help_text = "Number of minutes to forecast."
  elif freq_lower in ['s']:  # Second
    max_value = 3600  # 1 hour
    default_value = 300
    help_text = "Number of seconds to forecast."
  elif freq_lower == 'ms':  # Millisecond
    max_value = 10000
    default_value = 1000
    help_text = "Number of milliseconds to forecast."
  else:
    max_value = 96
    default_value = 12
    help_text = "Number of periods to forecast."

  return st.slider(
      "Horizon Forecast",
      min_value=1,
      max_value=max_value,
      value=default_value,
      help=help_text
  )


def render_prompt_type_selection() -> tuple[l4t.PromptType, str | None]:
  """Render prompt type selection and custom prompt if needed."""
  prompt_type = st.selectbox(
      "Prompt Type",
      options=list(l4t.PromptType),
      index=0,
      format_func=lambda f: f.name,
      help="Choose the type of prompt to be used."
  )

  prompt_name = None
  if prompt_type == l4t.PromptType.CUSTOM:
    prompts = repo.prompts().select_all()
    if prompts:
      prompt_name = st.selectbox(
          "Prompt",
          options=[p["name"] for p in prompts],
          index=0,
          help="Choose the prompt to be used."
      )

  return prompt_type, prompt_name


def render_examples_sampling_selection(prompt_type: l4t.PromptType) -> tuple[int, l4t.Sampling | None]:
  """Render examples and sampling selection based on prompt type."""
  if prompt_type in (l4t.PromptType.FEW_SHOT, l4t.PromptType.COT_FEW):
    examples = st.slider(
        "Examples",
        min_value=1,
        max_value=5,
        value=1,
        help="Number of examples to be used."
    )
  elif prompt_type == l4t.PromptType.CUSTOM:
    examples = st.slider(
        "Examples",
        min_value=0,
        max_value=5,
        value=0,
        help="Number of examples to be used."
    )
  else:
    examples = 0

  sampling = None
  if examples > 0:
    sampling = st.selectbox(
        "Sampling",
        options=list(l4t.Sampling),
        index=0,
        format_func=lambda f: f.name,
        help="Choose the sampling strategy to be used."
    )

  return examples, sampling


def render_format_type_selection() -> tuple[l4t.TSFormat, l4t.TSType]:
  """Render format and type selection."""
  tsformat = st.selectbox(
      "Series Format",
      options=list(l4t.TSFormat),
      index=0,
      format_func=lambda f: f.name,
      help="Presentation format of the data for the model."
  )

  tstype = st.radio(
      "Series Type",
      options=list(l4t.TSType),
      index=0,
      format_func=lambda f: f.name,
      help="Numeric: [3.662, 3.124], Text: [3 . 6 6 2, 3 . 1 2 4]"
  )

  return tsformat, tstype


def render_prompt_settings() -> dict | None:
  """Render prompt settings section and return configuration."""
  st.write("---")
  st.write("#### ⚙️ Prompt Settings")

  # Dataset selection
  filename, content, freq = render_dataset_selection()
  if not filename:
    return None

  # Load timeseries
  ts = load_timeseries(content)

  # Column selection
  columns = render_column_selection(ts)
  if not columns:
    return None

  # Window selection (varies by frequency)
  start_datetime, end_datetime = render_window_selection_datetime(ts, freq)
  if not start_datetime or not end_datetime:
    return None

  # Horizon forecast
  horizon_forecast = render_horizon_selection(freq)

  # Prompt type and custom prompt
  prompt_type, prompt_name = render_prompt_type_selection()

  # Examples and sampling
  examples, sampling = render_examples_sampling_selection(prompt_type)

  # Format and type
  tsformat, tstype = render_format_type_selection()

  # Split data
  ts = ts[columns]
  train, val = ts.split(
      start=start_datetime,
      end=end_datetime,
      periods=horizon_forecast
  )

  return {
      "filename": filename,
      "columns": columns,
      "start_datetime": start_datetime,
      "end_datetime": end_datetime,
      "freq": freq,
      "horizon_forecast": horizon_forecast,
      "prompt_type": prompt_type,
      "prompt_name": prompt_name,
      "examples": examples,
      "sampling": sampling,
      "tsformat": tsformat,
      "tstype": tstype,
      "train": train,
      "val": val,
  }


# =============================================================================
# Helper functions
# =============================================================================

def load_timeseries(content: bytes) -> l4t.MultiTimeSeries:
  """Load bytes content into MultiTimeSeries."""
  df = pd.read_csv(io.BytesIO(content))
  df = l4t.read_file(df, index_col="datetime")
  return l4t.MultiTimeSeries(df)


def get_file_content(filename: str) -> tuple[bytes, str] | None:
  """Get file content by filename."""
  file = repo.uploads().select_with_content(filename)
  return file["content"], file["freq"]


# =============================================================================
# Model Settings
# =============================================================================

def render_model_settings() -> tuple[str | None, str | None, float]:
  """Render model settings section."""
  st.write("#### ⚙️ Model Settings")

  models = repo.models().select_all()
  if not models:
    st.info("No models configured. Please add a model in [Settings](/settings).")
    return None, None, 0.7

  model_options = {f"{m['provider']} / {m['name']}": m for m in models}
  selected = st.selectbox("Model", list(model_options.keys()), index=0,
                          help="Choose the model to be used.")
  model_data = model_options.get(selected, {})
  model_name = model_data.get("name")
  provider = model_data.get("provider")

  temperature = st.slider(
      "Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
      help="Temperature controls the randomness of the model's response."
  )

  return model_name, provider, temperature


def render_sidebar() -> tuple[str | None, str | None, float, dict | None, bool]:
  """Render complete sidebar."""
  with st.sidebar:
    model_name, provider, temperature = render_model_settings()
    settings = render_prompt_settings()
    confirm = st.button("Generate Analysis", type="primary", width="stretch")

  return model_name, provider, temperature, settings, confirm


# =============================================================================
# Validation
# =============================================================================

def validate_inputs(model_name: str | None, settings: dict | None) -> str | None:
  """Validate inputs and return error message if invalid."""
  if not model_name:
    return "Model not selected. Please select one before continuing."
  if not settings:
    return "Dataset or columns not selected. Please configure before continuing."
  if settings["prompt_type"] == l4t.PromptType.CUSTOM and not settings["prompt_name"]:
    return "Prompt not selected. Please select one before continuing."
  if settings["val"].shape[0] < settings["horizon_forecast"]:
    return "Validation set is smaller than the forecast horizon. Please adjust the date range."
  return None


# =============================================================================
# UI Components - Results
# =============================================================================

def render_overview(model_name: str, provider: str, temperature: float, settings: dict) -> None:
  """Render overview section."""
  st.write("#### OVERVIEW")
  cols = st.columns([3, 3, 2, 4, 3, 2])
  metrics = [
      ("Model", model_name.upper()),
      ("API", provider.upper()),
      ("Temperature", temperature),
      ("Prompt Type", settings["prompt_type"].name),
      ("Series Type", settings["tstype"].name),
      ("Series Format", settings["tsformat"].name),
  ]
  for col, (label, value) in zip(cols, metrics):
    with col:
      st.metric(label, value)


def render_training_set(filename: str, train: l4t.MultiTimeSeries, freq: str) -> None:
  """Render training set section."""
  st.write("---")
  st.write("#### TRAINING SET")

  cols = st.columns([3, 3, 3, 2, 1])
  metrics = [
      ("Dataset", filename),
      ("Start", train.index.min().strftime("%Y-%m-%d")),
      ("End", train.index.max().strftime("%Y-%m-%d")),
      ("Rows", train.shape[0]),
      ("Columns", train.shape[1]),
  ]
  for col, (label, value) in zip(cols, metrics):
    with col:
      st.metric(label, value)

  st.plotly_chart(
      l4t.linechart(train, title="Time Series - Training Set", lightness=0.7),
      width="stretch"
  )
  st.dataframe(train, width="stretch")

  st.write("##### DESCRIPTIVE STATISTICS")
  st.plotly_chart(
      l4t.barplot(train, title="Descriptive Statistics - Training Set",
                  yaxis=dict(type="log", dtick=1), lightness=0.7),
      width="stretch"
  )
  st.dataframe(train.describe().T, width="stretch")

  st.write("##### STL DECOMPOSITION")
  try:
    stl = train.stl(freq=freq)
    st.dataframe(pd.DataFrame({
        "Trend Strength": [stl["t_strength"][col] for col in train.num_columns],
        "Seasonality Strength": [stl["s_strength"][col] for col in train.num_columns],
        "Noise Strength": [stl["r_strength"][col] for col in train.num_columns],
    }, index=train.num_columns), width="stretch")
  except Exception:
    st.info("Could not compute STL decomposition.")


def render_model_response(prompt: str, raw_response: str, expanded: bool = True) -> None:
  """Render model response section."""
  with st.expander("MODEL RESPONSE", expanded=expanded):
    with st.chat_message("user"):
      st.write("###### User")
      st.code(prompt, language="json", line_numbers=True)
    with st.chat_message("assistant"):
      st.write("###### Model")
      st.code(raw_response, language="json5", line_numbers=True)


def render_forecast_metrics(response) -> None:
  """Render forecast metrics row."""
  cols = st.columns(3)
  metrics = [
      ("INPUT TOKENS", response.input_tokens),
      ("OUTPUT TOKENS", response.output_tokens),
      ("RESPONSE TIME", f"{response.time:.2f} seconds"),
  ]
  for col, (label, value) in zip(cols, metrics):
    with col:
      st.metric(label, value)


def render_comparison(val, pred, title_prefix: str) -> None:
  """Render real vs predicted comparison."""
  st.plotly_chart(
      l4t.lineplot(val, pred, groups=["Real", "Predicted"],
                   title=f"Time Series - {title_prefix}"),
      width="stretch"
  )

  col1, col2 = st.columns(2)
  with col1:
    st.write("##### REAL")
    st.dataframe(val, width="stretch")
  with col2:
    st.write("##### PREDICTED")
    st.dataframe(pred, width="stretch")

  st.plotly_chart(
      l4t.barplot(val, pred, groups=["Real", "Predicted"],
                  title=f"Descriptive Statistics - {title_prefix}",
                  yaxis=dict(type="log", dtick=1)),
      width="stretch"
  )

  col1, col2 = st.columns(2)
  with col1:
    st.write("##### REAL")
    st.dataframe(val.describe().drop("count"), width="stretch")
  with col2:
    st.write("##### PREDICTED")
    st.dataframe(pred.describe().drop("count"), width="stretch")


def render_forecast_results(prompt: str, response, val, pred) -> pd.DataFrame | None:
  """Render forecast results section."""
  st.write("---")
  st.write("#### FORECAST RESULTS")

  render_forecast_metrics(response)
  render_model_response(prompt, response.raw)
  render_comparison(val, pred, "Real vs Predicted")

  try:
    metrics = val.metrics(pred)
    st.plotly_chart(
        l4t.barplot(metrics, x=["sMAPE", "MAE", "RMSE"],
                    title="Forecast Metrics",
                    yaxis=dict(type="log", dtick=1)),
        width="stretch"
    )
    st.dataframe(metrics, width="stretch")
    return metrics
  except Exception:
    st.info("Could not compute metrics.")
    return None


# =============================================================================
# Forecast Generation
# =============================================================================

def build_prompt(settings: dict) -> str:
  """Build prompt from settings."""
  prompt_content, prompt_variables = None, {}

  if settings["prompt_type"] == l4t.PromptType.CUSTOM:
    prompt_data = repo.prompts().select(settings["prompt_name"])
    if prompt_data:
      prompt_content = prompt_data["content"]
      prompt_variables = prompt_data.get("variables") or {}

  return l4t.prompt(
      ts=settings["train"],
      forecast_horizon=settings["horizon_forecast"],
      type=settings["prompt_type"],
      tsformat=settings["tsformat"],
      tstype=settings["tstype"],
      examples=settings["examples"],
      sampling=settings["sampling"],
      template=prompt_content,
      stl=settings["train"].stl(freq=settings["freq"]),
      **prompt_variables
  )


def save_to_history(model_name: str, provider: str, temperature: float,
                    settings: dict, prompt: str, response, pred, val, metrics) -> None:
  """Save forecast results to history."""
  repo.history().insert(
      model=model_name,
      provider=provider,
      temperature=temperature,
      dataset=settings["filename"],
      columns=settings["columns"],
      start_time=settings["start_datetime"],
      end_time=settings["end_datetime"],
      prompt_type=settings["prompt_type"].name,
      time_series_format=settings["tsformat"].name,
      time_series_type=settings["tstype"].name,
      examples=settings["examples"],
      sampling=settings["sampling"].name if settings["sampling"] else None,
      forecast_horizon=settings["horizon_forecast"],
      input_tokens=response.input_tokens,
      output_tokens=response.output_tokens,
      response_time=response.time,
      response_raw=response.raw,
      response_predicted=pred.to_str(format="csv"),
      validation=val.to_str(format="csv"),
      metrics=metrics.to_dict(orient="records") if metrics is not None else None,
      statistics_predicted=pred.describe().to_dict(orient="records"),
      statistics_validation=val.describe().to_dict(orient="records"),
      training=settings["train"].to_str(format="csv"),
      prompt=prompt
  )


def run_forecast(model_name: str, provider: str, temperature: float, settings: dict) -> None:
  """Run the forecast pipeline."""
  train = settings["train"]
  val = settings["val"]
  response = None

  # Build prompt
  try:
    context = build_prompt(settings)
    data = train.to_str(format=settings["tsformat"], type=settings["tstype"])
    prompt = context + '\n' + data
  except Exception as e:
    st.toast(f"Error building prompt: {e}", icon="⚠️")
    st.stop()

  # Render overview and training set
  render_overview(model_name, provider, temperature, settings)
  render_training_set(settings["filename"], train, settings["freq"])

  # Generate forecast
  try:
    api = API(model_name, provider)

    if not model_name.startswith("mock"):
      response = api.response(context, data, temperature=temperature)
    else:
      response = API._mock(val, settings["tsformat"], settings["tstype"])

    if not response.predicted:
      render_model_response(prompt, response.raw or "No response", expanded=True)
      st.info("The model did not return a forecast, so the result could not be parsed.")
      return

    pred = l4t.from_str(response.predicted, format=settings["tsformat"])
    pred = l4t.MultiTimeSeries(pred)
    pred.columns = val.columns
    pred.index = val.index

    metrics = render_forecast_results(prompt, response, val, pred)

    save_to_history(model_name, provider, temperature, settings,
                    prompt, response, pred, val, metrics)
    st.toast("Forecast generated and saved to history successfully!", icon="✅")

  except Exception as e:
    logger.error(f"Error during forecast generation: {e}")
    render_model_response(prompt, getattr(response, "raw", "No response"), expanded=True)
    st.info("An error occurred while generating the forecast. Check the model response above.")


# =============================================================================
# Main
# =============================================================================

model_name, provider, temperature, settings, confirm = render_sidebar()

if not confirm:
  with st.container():
    st.write("## LLM4Time Pipeline")
    st.write("##### Follow the steps below to upload your dataset, configure the model and generate predictions.")
    st.image(abspath("assets/llm4time.svg"), width=780)
else:
  error = validate_inputs(model_name, settings)
  if error:
    st.toast("An unexpected error occurred. Please, check your inputs and try again.", icon="⚠️")
  else:
    run_forecast(model_name, provider, temperature, settings)
