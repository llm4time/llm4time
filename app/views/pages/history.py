import streamlit as st
import llm4time as l4t
import pandas as pd
import storage.repository as repo


# =============================================================================
# Constants
# =============================================================================

PROMPT_TYPES = [pt.name for pt in l4t.PromptType]


# =============================================================================
# Data Helpers
# =============================================================================

def parse_timeseries(csv_str: str) -> l4t.MultiTimeSeries:
  """Parse CSV string to MultiTimeSeries."""
  df = l4t.from_str(csv_str, format="csv")
  return l4t.MultiTimeSeries(df)


def get_result_field(result: dict, field: str):
  """Safely get field from result dict."""
  return result.get(field)


# =============================================================================
# Pending Actions
# =============================================================================

def handle_pending_actions() -> None:
  """Process pending clear history action."""
  if not st.session_state.pop("clear_history", False):
    return

  filename = st.session_state.pop("filename", None)
  prompt_types = st.session_state.pop("prompt_types", [])

  try:
    repo.history().remove_many(filename, prompt_types)
    st.toast("History cleared successfully.", icon="✅")
  except Exception as e:
    st.toast(f"Error clearing history: {e}", icon="⚠️")
  finally:
    st.rerun()


# =============================================================================
# Dialogs
# =============================================================================

@st.dialog("Confirm deletion")
def delete_dialog(filename: str, prompt_types: list[str]) -> None:
  st.write(
      f"Are you sure you want to clear the history of the dataset "
      f"**{filename}** for the prompts below?"
  )
  st.markdown("\n".join(f"- **{pt}**" for pt in prompt_types))
  st.caption("**⚠️ This action CANNOT be undone.**")

  col1, col2 = st.columns(2)
  with col1:
    if st.button("No", width="stretch"):
      st.rerun()
  with col2:
    if st.button("Yes, delete", type="primary", width="stretch"):
      st.session_state.clear_history = True
      st.session_state.filename = filename
      st.session_state.prompt_types = prompt_types
      st.rerun()


# =============================================================================
# UI Components - Sidebar
# =============================================================================

def render_sidebar() -> tuple[str | None, list[str], bool, bool]:
  """Render sidebar and return selected options."""
  with st.sidebar:
    files = repo.uploads().select_all()

    if not files:
      st.info("No datasets available. Please [upload](/) a file first.")
      return None, [], False, False

    filenames = [f["filename"] for f in files]
    filename = st.selectbox("Dataset", filenames)

    prompt_types = st.multiselect(
        "Prompts Type",
        options=PROMPT_TYPES,
        default=[l4t.PromptType.ZERO_SHOT.name],
        help="Select the prompt types you want to view."
    )

    if filename and prompt_types:
      st.session_state.history = repo.history().select(filename, prompt_types)[::-1]

    view_btn = st.button(
        "View History", type="primary", width="stretch",
        help="Click to view the prediction history of the selected prompts."
    )
    clear_btn = st.button(
        "Clear History", width="stretch",
        help="Click to clear the prediction history of the selected prompts."
    )

  return filename, prompt_types, view_btn, clear_btn


# =============================================================================
# UI Components - History Display
# =============================================================================

def render_metrics_row(result: dict) -> None:
  """Render metrics row for a result."""
  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric("INPUT TOKENS", result.get("input_tokens"))
  with col2:
    st.metric("OUTPUT TOKENS", result.get("output_tokens"))
  with col3:
    response_time = result.get("response_time", 0)
    st.metric("RESPONSE TIME", f"{response_time:.2f} seconds")


def render_training_section(result: dict, idx: int) -> None:
  """Render training set section."""
  header = (
      f"**TRAINING SET - "
      f"`{result.get('prompt_type')}` • "
      f"`{result.get('time_series_format')}` • "
      f"`{result.get('time_series_type')}`**"
  )

  with st.expander(header, expanded=False):
    train_data = result.get("training")
    if not train_data:
      st.warning("No training data available.")
      return

    train = parse_timeseries(train_data)

    col1, col2, col3, col4, col5 = st.columns([3, 3, 3, 2, 1])
    with col1:
      st.metric("Dataset", result.get("dataset"))
    with col2:
      st.metric("Start", train.index.min().strftime("%Y-%m-%d"))
    with col3:
      st.metric("End", train.index.max().strftime("%Y-%m-%d"))
    with col4:
      st.metric("Rows", train.shape[0])
    with col5:
      st.metric("Columns", train.shape[1])

    st.plotly_chart(
        l4t.linechart(train, title="Time Series - Training Set", lightness=0.7),
        width="stretch", key=f"train_linechart_{idx}"
    )
    st.dataframe(train, width="stretch")

    st.write("##### DESCRIPTIVE STATISTICS")
    st.plotly_chart(
        l4t.barplot(train, title="Descriptive Statistics - Training Set",
                    yaxis=dict(type="log", dtick=1), lightness=0.7),
        width="stretch", key=f"train_barplot_{idx}"
    )
    st.dataframe(train.describe().T, width="stretch")

    st.write("##### STL DECOMPOSITION")
    try:
      stl = train.stl()
      st.dataframe(pd.DataFrame({
          "Trend Strength": [stl["t_strength"][col] for col in train.num_columns],
          "Seasonality Strength": [stl["s_strength"][col] for col in train.num_columns],
          "Noise Strength": [stl["r_strength"][col] for col in train.num_columns],
      }, index=train.num_columns), width="stretch")
    except Exception:
      st.info("Could not compute STL decomposition.")


def render_response_section(result: dict) -> None:
  """Render model response section."""
  header = (
      f"**MODEL RESPONSE - "
      f"`{result.get('model')}` • "
      f"`{result.get('provider')}` • "
      f"`{round(result.get('temperature'), 1)}`**"
  )

  with st.expander(header, expanded=False):
    with st.chat_message("user"):
      st.write("###### User")
      st.code(result.get("prompt", ""), language="json", height=600)
    with st.chat_message("assistant"):
      st.write("###### Model")
      st.code(result.get("response_raw", ""), language="json5")


def render_forecast_section(result: dict, idx: int) -> None:
  """Render forecast results section."""
  header = (
      f"**FORECAST RESULTS - "
      f"`{result.get('input_tokens')}` • "
      f"`{result.get('output_tokens')}` • "
      f"`{result.get('response_time'):.2f}s`**"
  )

  with st.expander(header, expanded=False):
    render_metrics_row(result)

    val_data = result.get("validation")
    pred_data = result.get("response_predicted")

    if not val_data or not pred_data:
      st.warning("No forecast data available.")
      return

    val = l4t.from_str(val_data, format="csv")
    pred = l4t.from_str(pred_data, format="csv")

    st.plotly_chart(
        l4t.lineplot(val, pred, groups=["Real", "Predicted"],
                     title="Time Series - Real vs Predicted"),
        width="stretch", key=f"forecast_linechart_{idx}"
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
                    title="Descriptive Statistics - Real vs Predicted",
                    yaxis=dict(type="log", dtick=1)),
        width="stretch", key=f"forecast_barplot_{idx}"
    )

    col1, col2 = st.columns(2)
    with col1:
      st.write("##### REAL")
      st.dataframe(val.describe().drop("count"), width="stretch")
    with col2:
      st.write("##### PREDICTED")
      st.dataframe(pred.describe().drop("count"), width="stretch")

    try:
      metrics = val.metrics(pred)
      st.plotly_chart(
          l4t.barplot(metrics, x=["sMAPE", "MAE", "RMSE"],
                      title="Forecast Metrics",
                      yaxis=dict(type="log", dtick=1)),
          width="stretch", key=f"forecast_metrics_{idx}"
      )
      st.dataframe(metrics, width="stretch")
    except Exception:
      st.info("Could not compute metrics.")


def render_history_entry(result: dict, idx: int) -> None:
  """Render a single history entry."""
  header = (
      f""
  )

  with st.expander(header, expanded=True):
    render_training_section(result, idx)
    render_response_section(result)
    render_forecast_section(result, idx)


def render_history(history: list[dict]) -> None:
  """Render all history entries."""
  for i, result in enumerate(reversed(history)):
    render_history_entry(result, i)


# =============================================================================
# Main
# =============================================================================

handle_pending_actions()

filename, prompt_types, view_clicked, clear_clicked = render_sidebar()

if view_clicked and not prompt_types:
  st.warning("Please select at least one prompt type to view the predictions.")
elif clear_clicked and not prompt_types:
  st.warning("Please select at least one prompt type to clear the history.")
elif clear_clicked and filename:
  delete_dialog(filename, prompt_types)
elif view_clicked:
  history = st.session_state.get("history", [])
  if history:
    render_history(history)
  else:
    st.info("No history found for the selected dataset and prompt types.")
