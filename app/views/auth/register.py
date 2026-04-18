import streamlit as st
from storage.exceptions import \
    EmailAlreadyExistsError, UsernameAlreadyExistsError
import re


# =============================================================================
# Theme & Styles
# =============================================================================

st.markdown("""
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css');

/* Google Button */
.st-key-google-btn .stButton > button:first-child {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}
.st-key-google-btn .stButton > button:first-child::before {
  content: "\\f1a0";
  font-family: "Font Awesome 6 Brands";
  font-size: 16px;
}

/* GitHub Button */
.st-key-github-btn .stButton > button:first-child {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}
.st-key-github-btn .stButton > button:first-child::before {
  content: "\\f09b";
  font-family: "Font Awesome 6 Brands";
  font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Constants
# =============================================================================

auth = st.session_state.auth


# =============================================================================
# UI Components - Models List
# =============================================================================

def register(email, username, password):
  if any(x is None or x == "" for x in [email, username, password]):
    st.toast("Please fill in all required fields.", icon="⚠️")
    return
  if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
    st.toast("Please enter a valid email address.", icon="⚠️")
    return
  if not re.match(r"^[a-zA-Z0-9_]{3,20}$", username):
    st.toast("Username must be 3-20 characters long and can only contain letters, numbers, and underscores.", icon="⚠️")
    return
  if len(password) < 6:
    st.toast("Password must be at least 6 characters long.", icon="⚠️")
    return
  try:
    if auth.signup(email=email, username=username, password=password):
      st.rerun()
  except EmailAlreadyExistsError:
    st.toast("Email already exists. Please use a different email.", icon="⚠️")
  except UsernameAlreadyExistsError:
    st.toast("Username already exists. Please choose a different username.", icon="⚠️")
  except Exception:
    st.toast(f"Unexpected error occurred during registration.", icon="⚠️")


st.title("Sign Up")
email = st.text_input("Email", icon=":material/mail:")
username = st.text_input("Username", icon=":material/person:")
password = st.text_input("Password", type="password", icon=":material/lock:")
submit_btn = st.button("Continue", type="primary", width="stretch", on_click=register,
                       kwargs={"email": email, "username": username, "password": password})

st.markdown("Already have an account? <a href='/' target='_self'>Sign in</a>",
            unsafe_allow_html=True)

st.write("---")
st.button(label="Continue with Google",
          type="secondary", width="stretch", key="google-btn",
          on_click=st.login, kwargs={"provider": "google"})

st.button(label="Continue with GitHub",
          type="secondary", width="stretch", key="github-btn")
