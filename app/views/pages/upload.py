import streamlit as st
import llm4time as l4t
import storage.repository as repo
from utils import freq_to_description
import pandas as pd
import io
import os

VALID_EXTENSIONS = [".csv", ".xlsx", ".json", ".parquet"]
INVALID_FILENAME_CHARS = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']


# =============================================================================
# File Processing
# =============================================================================

def read_dataframe(file) -> pd.DataFrame:
  """Read uploaded file into DataFrame based on extension."""
  _, ext = os.path.splitext(file.name)
  readers = {
      ".csv": pd.read_csv,
      ".xlsx": pd.read_excel,
      ".json": pd.read_json,
      ".parquet": pd.read_parquet,
  }
  reader = readers.get(ext.lower())
  if not reader:
    raise ValueError(f"Unsupported file extension: {ext}")
  return reader(file)


def process_timeseries(df: pd.DataFrame) -> bytes:
  """Process DataFrame and return CSV bytes."""
  df = df[[st.session_state.index] + st.session_state.columns]
  ts = l4t.read_file(df, index_col=st.session_state.index)
  ts = ts.agg_duplicates(method=st.session_state.duplicates)

  if st.session_state.normalize == "Yes":
    ts = ts.normalize(freq=st.session_state.normalize_freq)

  if st.session_state.imputation == "Yes":
    imputation_methods = {
        "ffill": lambda: ts.impute_ffill(inplace=True),
        "bfill": lambda: ts.impute_bfill(inplace=True),
        "linear": lambda: ts.impute_interpolate(method="linear", inplace=True),
        "spline": lambda: ts.impute_interpolate(
            method="spline", order=st.session_state.spline_order, inplace=True),
        "mean": lambda: ts.impute_mean(inplace=True),
        "median": lambda: ts.impute_median(inplace=True),
        "sma": lambda: ts.impute_sma(window=st.session_state.sma_window, inplace=True),
        "ema": lambda: ts.impute_ema(span=st.session_state.ema_span, inplace=True),
    }
    imputation_methods[st.session_state.fill_method]()

  ts.index.name = "datetime"
  return ts.to_csv().encode("utf-8")


# =============================================================================
# Validation
# =============================================================================

def is_valid_extension(filename: str) -> bool:
  _, ext = os.path.splitext(filename)
  return ext.lower() in VALID_EXTENSIONS


def is_valid_filename(filename: str) -> bool:
  if not filename.strip():
    return False
  return not any(char in filename for char in INVALID_FILENAME_CHARS)


def get_extension(filename: str) -> str:
  _, ext = os.path.splitext(filename)
  return ext.lower().lstrip(".")


# =============================================================================
# Upload Dialog
# =============================================================================

def on_file_upload() -> None:
  """Handle file upload event."""
  file = st.session_state.uploaded_file
  if file is None:
    return

  st.session_state.pop("step", None)
  st.session_state.step = 1

  try:
    df = read_dataframe(file)
    upload_dialog(df)
  except Exception as e:
    st.toast(f"Error loading file: {e}", icon="🚨")


@st.dialog("Configure Dataset")
def upload_dialog(df: pd.DataFrame) -> None:
  step = st.session_state.step

  if step == 1:
    render_step_columns(df)
  elif step == 2:
    render_step_duplicates()
  elif step == 3:
    render_step_normalize()
  elif step == 4:
    render_step_imputation()
  elif step == 5:
    render_step_save()

  render_navigation_buttons(df)


def render_step_columns(df: pd.DataFrame) -> None:
  columns = df.columns.tolist()
  st.session_state.index = st.selectbox(
      "Choose the datetime column:", columns, index=0,
      help="Time reference column that indicates the datetime for each observation.")

  st.session_state.freq = st.selectbox(
      "Frequency:",
      ["ms", "s", "min", "h", "D", "M", "Y"],
      format_func=freq_to_description,
      index=3,
      help="Defines the time unit of the time series."
  )

  st.session_state.columns = st.multiselect(
      "Choose the value columns:", df.columns.drop(st.session_state.index),
      help="Columns that contain the values of the time series.")


def render_step_duplicates() -> None:
  if len(st.session_state.columns) > 1:
    options = ["first", "last", "sumf", "suml"]
    labels = {
        "first": "Keep the first occurrence",
        "last": "Keep the last occurrence",
        "sumf": "Sum numeric values and keep first categories",
        "suml": "Sum numeric values and keep last categories",
    }
  else:
    options = ["first", "last", "sum"]
    labels = {
        "first": "Keep the first occurrence",
        "last": "Keep the last occurrence",
        "sum": "Sum the values",
    }

  st.session_state.duplicates = st.radio(
      "How to handle duplicate timestamps?",
      options=options,
      format_func=lambda x: labels[x],
      index=0,
      help="\n".join(f"- {v}" for v in labels.values()))


def render_step_normalize() -> None:
  st.session_state.normalize = st.radio(
      "Do you want to normalize the dataset?", ["Yes", "No"], index=1,
      help="Ensures that all timestamps between the start and end of the dataset are present.")
  freq = st.session_state.freq
  if st.session_state.normalize == "Yes":
    interval = st.number_input(
        "Interval:", min_value=1, max_value=60, value=1, step=1,
        help="Defines how many frequency units apart the data points will be considered.")
    freq = freq if interval == 1 else f"{interval}{freq}"
  st.session_state.normalize_freq = freq


def render_step_imputation() -> None:
  st.session_state.imputation = st.radio(
      "Do you want to impute missing values?", ["Yes", "No"], index=1,
      help="Fills in the missing values in the dataset according to the selected method.")

  if st.session_state.imputation != "Yes":
    return

  methods = {
      "ffill": "Forward Fill",
      "bfill": "Backward Fill",
      "linear": "Linear Interpolation",
      "spline": "Spline Interpolation",
      "mean": "Mean",
      "median": "Median",
      "sma": "Simple Moving Average",
      "ema": "Exponential Moving Average",
  }
  st.session_state.fill_method = st.selectbox(
      "Imputation Method:",
      options=list(methods.keys()),
      format_func=lambda x: methods[x],
      index=0,
      help=(
          "- Forward Fill: fills using the last known value.\n"
          "- Backward Fill: fills using the next known value.\n"
          "- Linear Interpolation: estimates using linear interpolation.\n"
          "- Spline Interpolation: estimates using spline interpolation.\n"
          "- Mean: fills with column mean.\n"
          "- Median: fills with column median.\n"
          "- SMA: fills using rolling mean.\n"
          "- EMA: fills using exponential weighted mean."
      ))

  if st.session_state.fill_method == "spline":
    st.session_state.spline_order = st.number_input(
        "Spline Order:", min_value=1, max_value=5, value=2, step=1,
        help="Specifies the order of the spline interpolation.")
  elif st.session_state.fill_method == "sma":
    st.session_state.sma_window = st.number_input(
        "Window Size:", min_value=1, max_value=30, value=3, step=1,
        help="Specifies the window size for the simple moving average.")
  elif st.session_state.fill_method == "ema":
    st.session_state.ema_span = st.number_input(
        "Span:", min_value=1, max_value=30, value=3, step=1,
        help="Specifies the span for the exponential moving average.")


def render_step_save() -> None:
  file = st.session_state.uploaded_file
  st.session_state.filename = st.text_input(
      "Save file as:", value=file.name).strip()

  if not is_valid_extension(st.session_state.filename):
    st.warning("A valid file extension is required: .csv, .xlsx, .json, or .parquet")
  elif repo.uploads().exists(st.session_state.filename):
    st.warning(f"File **{st.session_state.filename}** already exists.")


def render_navigation_buttons(df: pd.DataFrame) -> None:
  col1, col2 = st.columns(2)

  with col1:
    if st.session_state.step > 1:
      st.button("Back", on_click=lambda: _change_step(-1), width="stretch")

  with col2:
    if st.session_state.step < 5:
      disabled = st.session_state.step == 1 and len(st.session_state.columns) == 0
      st.button("Next", on_click=lambda: _change_step(1),
                width="stretch", disabled=disabled)
    else:
      disabled = (
          not is_valid_extension(st.session_state.filename) or
          repo.uploads().exists(st.session_state.filename)
      )
      if st.button("Confirm", type="primary", width="stretch", disabled=disabled):
        _confirm_upload(df)
        st.rerun()


def _change_step(delta: int) -> None:
  st.session_state.step += delta


def _confirm_upload(df: pd.DataFrame) -> None:
  content = process_timeseries(df)
  st.session_state.upload_data = {
      "filename": st.session_state.filename,
      "content": content,
      "freq": st.session_state.freq,
  }


# =============================================================================
# Delete Dialog
# =============================================================================

@st.dialog("Confirm deletion")
def delete_dialog(filenames: list) -> None:
  n = len(filenames)
  if n == 1:
    st.write(f"Are you sure you want to delete the file **{filenames[0]}**?")
  else:
    st.write(f"Are you sure you want to delete **{n} files**?")

  st.caption("**⚠️ This action CANNOT be undone.**")

  col1, col2 = st.columns(2)
  with col1:
    if st.button("No", width="stretch"):
      st.rerun()
  with col2:
    if st.button("Yes, delete", type="primary", width="stretch"):
      st.session_state.files_to_delete = filenames
      st.rerun()


# =============================================================================
# Rename Validation
# =============================================================================

def validate_rename(old_name: str, new_name: str) -> bool:
  if not is_valid_filename(new_name):
    st.warning("File name cannot be empty or contain invalid characters.")
    return False

  if not is_valid_extension(new_name):
    st.warning("A valid file extension is required: .csv, .xlsx, .json, or .parquet")
    return False

  if not repo.uploads().exists(old_name):
    st.warning(f"File not found: {old_name}")
    return False

  if repo.uploads().exists(new_name) and old_name != new_name:
    st.warning(f"A file with the name '{new_name}' already exists.")
    return False

  return True


# =============================================================================
# File List Display
# =============================================================================

def render_file_list() -> None:
  files = repo.uploads().select_all()
  if not files:
    return

  files_df = pd.DataFrame([
      {
          "File": f"📁 {os.path.splitext(f['filename'])[0]}",
          "Extension": f['extension'],
          "Frequency": freq_to_description(f["freq"]),
          "Rows": f["rows"],
          "Columns": f["columns"],
          "Size (MB)": round(f["size"], 2),
          "Added": f["created_at"][:16] if f.get("created_at") else "N/A",
          "Modified": f["updated_at"][:16] if f.get("updated_at") else "N/A",
          "Delete": False,
      }
      for f in files
  ])

  edited_df = st.data_editor(
      files_df,
      disabled=["Extension", "Frequency", "Rows", "Columns",
                "Size (MB)", "Added", "Modified"],
      hide_index=True,
      width="stretch",
  )

  # Handle renames
  for idx, row in edited_df.iterrows():
    old_name = files[idx]["filename"]
    new_name = row["File"].replace("📁 ", "") + f".{row['Extension']}"
    if old_name != new_name and validate_rename(old_name, new_name):
      st.session_state.rename_data = {"old": old_name, "new": new_name}
      st.rerun()

  # Handle deletes
  files_to_delete = [
      files[idx]["filename"]
      for idx, row in edited_df.iterrows()
      if row["Delete"]
  ]

  if files_to_delete:
    n = len(files_to_delete)
    if st.button(f"🗑️ Delete {n} file{'s' if n > 1 else ''}", type="primary"):
      delete_dialog(files_to_delete)


# =============================================================================
# Action Handlers
# =============================================================================

def handle_pending_actions() -> None:
  # Handle upload
  if "upload_data" in st.session_state:
    data = st.session_state.upload_data
    content = data["content"]
    filename = data["filename"]
    freq = data["freq"]
    df = pd.read_csv(io.BytesIO(content))

    repo.uploads().insert(
        filename=filename,
        extension=get_extension(filename),
        rows=df.shape[0],
        columns=df.shape[1],
        size=len(content) / (1024 * 1024),
        content=content,
        freq=freq,
    )
    st.toast(f"File '{filename}' uploaded successfully!", icon="✅")
    del st.session_state.upload_data
    st.rerun()

  # Handle rename
  if "rename_data" in st.session_state:
    data = st.session_state.rename_data
    try:
      repo.uploads().rename(data["old"], data["new"])
      st.toast(f"File renamed to '{data['new']}' successfully!", icon="✅")
    except Exception as e:
      st.warning(f"Error renaming file: {e}")
    finally:
      del st.session_state.rename_data
      st.rerun()

  # Handle delete
  if "files_to_delete" in st.session_state:
    filenames = st.session_state.files_to_delete
    n = len(filenames)
    try:
      repo.uploads().remove_many(filenames)
      st.toast(f"File{'s' if n > 1 else ''} deleted successfully!", icon="✅")
    except Exception as e:
      st.warning(f"Error deleting files: {e}")
    finally:
      del st.session_state.files_to_delete
      st.rerun()


# =============================================================================
# Main
# =============================================================================

handle_pending_actions()

st.write("### Upload your time series dataset file")
st.file_uploader(
    "Select a file",
    on_change=on_file_upload,
    key="uploaded_file",
    type=["csv", "xlsx", "json", "parquet"],
)

render_file_list()
