import streamlit as st
import llm4time as l4t
import pandas as pd
import storage.repository as repo
from storage.exceptions import \
    ModelAlreadyExistsError, PromptAlreadyExistsError, PromptNotFoundError
from streamlit_theme import st_theme


# =============================================================================
# Theme & Styles
# =============================================================================

theme = st_theme()
if theme is None:
  st.stop()

st.html(f"""
<style>
.st-key-api-tabs,
.st-key-custom-prompts-tabs {{
  padding: 1rem;
  border-radius: 0.5rem;
  background-color: {theme["secondaryBackgroundColor"]};
}}
</style>
""")


# =============================================================================
# Constants
# =============================================================================

GLOBAL_VARIABLES_PREVIEW = {
    "len_input": 30,
    "horizon_forecast": 7,
    "input": "Date,Value\n2016-07-01,38.662\n2016-07-01,37.124\n...",
    "output_example": "Date,Value\n2016-07-01,38.662\n2016-07-01,37.124\n...",
    "forecast_examples": "Example 1:\nPeriod (history):\n...\nPeriod (forecast):\n...",
    "input_len": 30,
    "forecast_horizon": 7,
    "statistics": "- Mean: 29.07\n- Median: 26.80\n- Standard Deviation: 3.45\n...",
    "examples": "Example 1:\nInput (History):\n...\nOutput (Forecast):\n...",
}

GLOBAL_VARIABLES_DOCS = pd.DataFrame([
    {"Value": "Number of periods to be forecasted."},
    {"Value": "Number of periods in the input time series."},
    {"Value": "Statistical summary of the input time series."},
    {"Value": "Example output containing the same number of periods to be forecasted formatted."},
    {"Value": "Examples containing history and forecast, according to the sampling strategy."},
    {"Value": "Input time series formatted."},
], index=pd.Index([
    "`{horizon_forecast}`", "`{len_input}`", "`{statistics}`",
    "`{output_example}`", "`{forecast_examples}`", "`{input}`"
], name="Key"))


# =============================================================================
# Model Operations
# =============================================================================

def save_model(provider: str, name: str, **kwargs) -> bool:
  """Save a new model configuration."""
  try:
    repo.models().insert(provider=provider, name=name, **kwargs)
    st.toast("Settings saved successfully!", icon="✅")
    return True
  except ModelAlreadyExistsError:
    st.toast(f"The model **{name}** for **{provider}** already exists.", icon="❌")
  except Exception:
    st.toast("Error saving settings.", icon="❌")
  return False


def rename_model(old_name: str, new_name: str, provider: str) -> bool:
  """Rename an existing model."""
  try:
    repo.models().rename(old_name=old_name, new_name=new_name, provider=provider)
    st.toast("Model renamed successfully!", icon="✅")
    return True
  except ModelAlreadyExistsError:
    st.warning(f"The model **{new_name}** for **{provider}** already exists.")
  except Exception as e:
    print("Error renaming model:", e)
    st.warning("Error renaming model.")
  return False


def delete_models(models: list[tuple[str, str]]) -> bool:
  """Delete multiple models."""
  try:
    repo.models().remove_many(models)
    st.toast("Models deleted successfully!", icon="✅")
    return True
  except Exception as e:
    print("Error deleting models:", e)
    st.toast("Error deleting models.", icon="❌")
  return False


# =============================================================================
# Prompt Operations
# =============================================================================

def delete_prompts(names: list[str]) -> bool:
  """Delete multiple prompts."""
  try:
    n = len(names)
    repo.prompts().remove_many(names)
    st.toast(f"Prompt{'s' if n > 1 else ''} deleted successfully!", icon="✅")
    return True
  except Exception:
    st.toast("Error deleting prompts.", icon="❌")
  return False


def save_prompt(name: str, content: str, variables: dict, is_edit: bool) -> bool:
  """Save or update a prompt."""
  try:
    if is_edit:
      repo.prompts().update(name, content, variables)
      st.toast(f"Prompt **'{name}'** updated successfully!", icon="✅")
    else:
      repo.prompts().insert(name=name, content=content, variables=variables)
      st.toast(f"Prompt **'{name}'** created successfully!", icon="✅")
    return True
  except PromptAlreadyExistsError:
    st.warning(
        f"A prompt named **'{name}'** already exists. Please choose another name.")
  except PromptNotFoundError:
    st.warning(f"Prompt **'{name}'** not found.")
  except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
  return False


def rename_prompt(old_name: str, new_name: str) -> bool:
  """Rename a prompt."""
  try:
    repo.prompts().rename(old_name, new_name)
    return True
  except PromptAlreadyExistsError:
    st.warning(f"A prompt named **'{new_name}'** already exists.")
  except PromptNotFoundError:
    st.warning(f"Prompt **'{old_name}'** not found.")
  except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
  return False


# =============================================================================
# Pending Actions Handler
# =============================================================================

def handle_pending_actions() -> None:
  """Process any pending actions from session state."""

  # Handle deletions
  if st.session_state.pop("confirm_delete", False):
    if "models_to_delete" in st.session_state:
      delete_models(st.session_state.pop("models_to_delete"))
      st.rerun()
    elif "prompts_to_delete" in st.session_state:
      delete_prompts(st.session_state.pop("prompts_to_delete"))
      st.rerun()


# =============================================================================
# UI Components - API Configuration
# =============================================================================

def render_api_tabs() -> l4t.Provider:
  """Render API provider tabs and return selected provider."""
  if "mode" not in st.session_state:
    st.session_state.mode = l4t.Provider.LM_STUDIO

  providers = [
      (l4t.Provider.LM_STUDIO, "LM Studio"),
      (l4t.Provider.OPENAI, "OpenAI / Ollama"),
      (l4t.Provider.AZURE, "OpenAI Azure"),
  ]

  cols = st.columns(len(providers))
  for col, (provider, label) in zip(cols, providers):
    with col:
      is_selected = st.session_state.mode == provider
      if st.button(label, type="primary" if is_selected else "tertiary",
                   width="stretch"):
        st.session_state.mode = provider
        st.rerun()

  return st.session_state.mode


def render_lm_studio_form() -> dict | None:
  """Render LM Studio configuration form."""
  st.write(
      "You will be redirected to LM Studio. If you do not have LM Studio installed, "
      "you can download it [here](https://lmstudio.ai)."
  )
  model = st.text_input(
      "Model", placeholder="deepseek-r1",
      help="Enter the name of the model you want to use in LM Studio.",
      icon=":material/robot:"
  )
  return {"model": model} if model else None


def render_openai_form() -> dict | None:
  """Render OpenAI configuration form."""
  st.write(
      "You will be redirected to the API. If you do not have an API key, "
      "you can obtain one [here](https://platform.openai.com/signup)."
  )
  api_key = st.text_input(
      "API Key", type="password", placeholder="********************************",
      help="Enter the API key you want to use.", icon=":material/vpn_key:"
  )
  model = st.text_input(
      "Model", placeholder="deepseek-r1",
      help="Enter the name of the model you want to use.", icon=":material/robot:"
  )
  base_url = st.text_input(
      "Base URL", placeholder="https://api.example.com/openai/v1",
      help="Enter the base URL you want to use.", icon=":material/cloud:"
  )
  if all([api_key, model, base_url]):
    return {"api_key": api_key, "model": model, "base_url": base_url}
  return None


def render_azure_form() -> dict | None:
  """Render Azure OpenAI configuration form."""
  st.write(
      "You will be redirected to the API. If you do not have an API key, "
      "you can obtain one [here](https://portal.azure.com)."
  )
  api_key = st.text_input(
      "API Key", type="password", placeholder="********************************",
      help="Enter the API key you want to use.", icon=":material/vpn_key:"
  )
  model = st.text_input(
      "Model", placeholder="deepseek-r1",
      help="Enter the name of the model you want to use.", icon=":material/robot:"
  )
  api_version = st.text_input(
      "API Version", placeholder="2024-05-01-preview",
      help="Enter the API version you want to use.", icon=":material/date_range:"
  )
  endpoint = st.text_input(
      "Endpoint", placeholder="https://<resource-name>.services.ai.azure.com",
      help="Enter the endpoint you want to use.", icon=":material/cloud:"
  )
  if all([api_key, model, api_version, endpoint]):
    return {"api_key": api_key, "model": model, "api_version": api_version, "endpoint": endpoint}
  return None


def render_api_section() -> None:
  """Render the API configuration section."""
  st.write("### API")

  with st.container(key="api-tabs"):
    mode = render_api_tabs()

  form_renderers = {
      l4t.Provider.LM_STUDIO: render_lm_studio_form,
      l4t.Provider.OPENAI: render_openai_form,
      l4t.Provider.AZURE: render_azure_form,
  }
  form_data = form_renderers[mode]()

  if st.button("Save Settings", type="primary", icon="💾"):
    if not form_data:
      st.toast("Please fill in all fields before saving the settings.", icon="⚠️")
    else:
      provider = str(mode)
      model = form_data["model"]

      kwargs = {}
      if provider == str(l4t.Provider.OPENAI):
        kwargs["api_key"] = form_data["api_key"]
        kwargs["endpoint"] = form_data["base_url"]
      elif provider == str(l4t.Provider.AZURE):
        kwargs["api_key"] = form_data["api_key"]
        kwargs["endpoint"] = form_data["endpoint"]
        kwargs["api_version"] = form_data["api_version"]

      if provider and model and save_model(provider, model, **kwargs):
        st.rerun()


# =============================================================================
# UI Components - Models List
# =============================================================================

@st.dialog("Confirm deletion")
def delete_models_dialog(models: list[tuple[str, str]]) -> None:
  n = len(models)
  if n == 1:
    model, provider = models[0]
    st.write(
        f"Are you sure you want to delete the model **{model}** from the API **{provider}**?")
  else:
    st.write(f"Are you sure you want to delete **{n} models**?")

  st.caption("**⚠️ This action CANNOT be undone.**")

  col1, col2 = st.columns(2)
  with col1:
    if st.button("No", width="stretch"):
      st.rerun()
  with col2:
    if st.button("Yes, delete", type="primary", width="stretch"):
      st.session_state.confirm_delete = True
      st.session_state.models_to_delete = models
      st.rerun()


def render_models_section() -> None:
  """Render the models list section."""
  st.write("---")
  st.write("### Models")

  models = repo.models().select_all()
  if not models:
    st.info("No models configured. Please add a model in the [#api](#api) section.")
    return

  st.write("This section displays the models configured in the [#api](#api) section.")
  df = st.data_editor(
      pd.DataFrame([
          {"Model": f"👾 {m['name']}", "API": m["provider"], "Delete": False}
          for m in models
      ]),
      disabled=["API"],
      hide_index=True, width="stretch"
  )

  # Handle renames
  for idx, row in df.iterrows():
    old_name = models[idx]["name"]
    new_name = row["Model"].replace("👾 ", "")
    if old_name != new_name:
      if rename_model(old_name, new_name, models[idx]["provider"]):
        st.rerun()

  # Handle deletes
  to_delete = [
      (models[idx]["name"], models[idx]["provider"])
      for idx, row in df.iterrows() if row["Delete"]
  ]
  if to_delete:
    n = len(to_delete)
    if st.button(f"🗑️ Delete {n} model{'s' if n > 1 else ''}", type="primary"):
      delete_models_dialog(to_delete)


# =============================================================================
# UI Components - Prompts Editor
# =============================================================================

def render_prompts_tabs() -> str:
  """Render prompt action tabs."""
  if "action" not in st.session_state:
    st.session_state.action = "create"

  col1, col2 = st.columns(2)
  for col, action in zip([col1, col2], ["create", "edit"]):
    with col:
      is_selected = st.session_state.action == action
      if st.button(action.capitalize(), type="primary" if is_selected else "tertiary",
                   width="stretch", icon=":material/note_add:" if action == "create" else ":material/edit_note:"):
        st.session_state.action = action
        st.rerun()

  return st.session_state.action


def render_prompt_editor() -> None:
  """Render the prompt editor section."""
  st.write("---")
  st.write("### Custom Prompts")

  with st.container(key="custom-prompts-tabs"):
    action = render_prompts_tabs()

  prompts = repo.prompts().select_all()
  prompt_name, prompt_content, prompt_variables = "", "", {}

  if action == "create":
    prompt_name = st.text_input("Name", placeholder="Price Prediction")
  else:
    prompt_names = [p["name"] for p in prompts]
    if prompt_names:
      prompt_name = st.selectbox("Prompt", options=prompt_names)
      if prompt_name:
        data = repo.prompts().select(prompt_name)
        if data:
          prompt_content = data["content"]
          prompt_variables = data.get("variables") or {}

  # Variables editor
  df_vars = st.data_editor(
      pd.DataFrame([{"Key": k, "Value": v} for k, v in prompt_variables.items()])
      if prompt_variables else pd.DataFrame(columns=["Key", "Value"]),
      hide_index=True, num_rows="dynamic", width="stretch"
  )
  prompt_variables = {row["Key"]: row["Value"]
                      for _, row in df_vars.iterrows() if row["Key"]}

  # Global variables documentation
  with st.expander("Global variables"):
    st.table(
        GLOBAL_VARIABLES_DOCS.style.set_properties(
            subset=["Value"], **{"color": "gray"})
    )

  # Prompt content and preview
  col1, col2 = st.columns(2)
  with col1:
    prompt_content = st.text_area(
        "Prompt", value=prompt_content, placeholder=l4t.FEW_SHOT,
        label_visibility="collapsed", height=400
    )
  with col2:
    preview_vars = {**GLOBAL_VARIABLES_PREVIEW, **prompt_variables}
    try:
      st.code(prompt_content.format(**preview_vars), language="python", height=400)
    except KeyError as e:
      st.code(f"Error: key {e} not found.", language="python", height=400)
    except Exception as e:
      st.code(f"Error: {e}", language="python", height=400)

  # Save button
  if st.button("💾 Save Prompt", type="primary"):
    if save_prompt(prompt_name, prompt_content, prompt_variables, is_edit=(action == "edit")):
      st.rerun()


# =============================================================================
# UI Components - Prompts List
# =============================================================================

@st.dialog("Confirm deletion")
def delete_prompts_dialog(names: list[str]) -> None:
  n = len(names)
  if n == 1:
    st.write(f"Are you sure you want to delete the prompt **'{names[0]}'**?")
  else:
    st.write(f"Are you sure you want to delete **{n} prompts**?")

  st.caption("**⚠️ This action CANNOT be undone.**")

  col1, col2 = st.columns(2)
  with col1:
    if st.button("No", width="stretch"):
      st.rerun()
  with col2:
    if st.button("Yes, delete", type="primary", width="stretch"):
      st.session_state.confirm_delete = True
      st.session_state.prompts_to_delete = names
      st.rerun()


def render_prompts_section() -> None:
  """Render the prompts list section."""
  st.write("---")
  st.write("### Prompts")

  prompts = repo.prompts().select_all()
  if not prompts:
    st.info(
        "No custom prompts created. Please add a prompt in the [#custom-prompts](#custom-prompts) section."
    )
    return

  st.write(
      "This section displays the custom prompts created in the [#custom-prompts](#custom-prompts) section."
  )
  df = st.data_editor(
      pd.DataFrame([{"Name": f"📄 {p['name']}", "Delete": False} for p in prompts]),
      hide_index=True, width="stretch"
  )

  # Handle renames
  for idx, row in df.iterrows():
    old_name = prompts[idx]["name"]
    new_name = row["Name"].replace("📄 ", "")
    if old_name != new_name:
      if rename_prompt(old_name, new_name):
        st.rerun()

  # Handle deletes
  to_delete = [prompts[idx]["name"] for idx, row in df.iterrows() if row["Delete"]]
  if to_delete:
    n = len(to_delete)
    if st.button(f"🗑️ Delete {n} prompt{'s' if n > 1 else ''}", type="primary"):
      delete_prompts_dialog(to_delete)


# =============================================================================
# Main
# =============================================================================

handle_pending_actions()
render_api_section()
render_models_section()
render_prompt_editor()
render_prompts_section()
