import streamlit as st
import llm4time as l4t
from utils import freq_to_description
import storage.repository as repo
import pandas as pd
import io


# =============================================================================
# Data Loading
# =============================================================================

def load_timeseries(content: bytes) -> l4t.MultiTimeSeries:
  """Load content bytes into MultiTimeSeries."""
  df = pd.read_csv(io.BytesIO(content))
  df = l4t.read_file(df, index_col="datetime")
  return l4t.MultiTimeSeries(df)


def get_file_content(filename: str) -> tuple[bytes, str] | None:
  """Get file content by filename."""
  file = repo.uploads().select_with_content(filename)
  if not file:
    return None
  return file["content"], file["freq"]


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar() -> tuple[str | None, list[str]]:
  """Render sidebar and return selected filename and columns."""
  with st.sidebar:
    files = repo.uploads().select_all()

    if not files:
      st.info("No datasets available. Please [upload](/) a file first.")
      return None, []

    filenames = [f["filename"] for f in files]
    filename = st.selectbox("Dataset", filenames)

    columns = []
    if filename:
      content, _ = get_file_content(filename)
      if content:
        ts = load_timeseries(content)
        columns = st.multiselect("Select one or more columns", ts.num_columns)

    st.button("Generate Statistics", type="primary",
              width="stretch", key="confirm")

  return filename, columns


def render_metrics(filename: str, ts: l4t.MultiTimeSeries, freq: str) -> None:
  """Render time series metrics."""
  col1, col2, col3, col4, col5, col6 = st.columns([2, 3, 3, 2, 1, 2])
  with col1:
    st.metric("Dataset", filename)
  with col2:
    st.metric("Start", ts.index.min().strftime("%Y-%m-%d"))
  with col3:
    st.metric("End", ts.index.max().strftime("%Y-%m-%d"))
  with col4:
    st.metric("Rows", ts.shape[0])
  with col5:
    st.metric("Columns", ts.shape[1])
  with col6:
    st.metric("Frequency", freq_to_description(freq))


def render_timeseries(ts: l4t.MultiTimeSeries) -> None:
  """Render time series section."""
  st.write("### Time Series")
  st.plotly_chart(
      ts.linechart(title="Time Series"),
      config={"responsive": True},
      width="stretch",
  )
  st.dataframe(ts, width="stretch")


def render_stl_decomposition(ts: l4t.MultiTimeSeries, freq: str) -> None:
  """Render STL decomposition section."""
  st.write("### STL Decomposition")
  try:
    st.plotly_chart(
        ts.stlplot(
            freq=freq,
            title="Time Series Decomposition (STL)",
            lightness=0.7
        ),
        config={"responsive": True},
        width="stretch",
    )
    stl = ts.stl(freq=freq)
    stl_df = pd.DataFrame({
        "Trend Strength": [stl["t_strength"][col] for col in ts.num_columns],
        "Seasonality Strength": [stl["s_strength"][col] for col in ts.num_columns],
        "Noise Strength": [stl["r_strength"][col] for col in ts.num_columns],
    }, index=ts.num_columns)
    st.dataframe(stl_df, width="stretch")
  except Exception as e:
    print(e)
    st.info("Could not compute STL decomposition.")


def render_descriptive_statistics(ts: l4t.MultiTimeSeries) -> None:
  """Render descriptive statistics section."""
  st.write("### Descriptive Statistics")
  st.plotly_chart(
      ts.barplot(
          title="Descriptive Statistics",
          yaxis=dict(type="log", dtick=1), lightness=0.7,
      ),
      config={"responsive": True},
      width="stretch",
  )
  st.dataframe(ts.describe().T, width="stretch")


# =============================================================================
# Main
# =============================================================================

filename, columns = render_sidebar()

if not st.session_state.get("confirm"):
  pass
elif not filename:
  st.toast("Dataset not selected. Please select one before continuing.", icon="⚠️")
elif not columns:
  st.toast("No columns selected. Please select one or more columns before continuing.", icon="⚠️")
else:
  content, freq = get_file_content(filename)
  if not content:
    st.error("Failed to load file content.")
  else:
    ts = load_timeseries(content)
    ts = ts[columns]

    render_metrics(filename, ts, freq)
    render_timeseries(ts)
    render_stl_decomposition(ts, freq)
    render_descriptive_statistics(ts)
