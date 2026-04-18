import streamlit as st

auth = st.session_state.auth


CLIENT_ID = st.secrets["github"]["client_id"]
CLIENT_SECRET = st.secrets["github"]["client_secret"]
REDIRECT_URI = st.secrets["github"]["redirect_uri"]


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

def login(username, password):
  if any(x is None or x == "" for x in [username, password]):
    st.toast("Username/email or password is incorrect.", icon="⚠️")
    return
  try:
    if not auth.signin(username, password):
      st.toast("Username/email or password is incorrect.", icon="⚠️")
      return
    st.rerun()
  except Exception:
    st.toast("Username/email or password is incorrect.", icon="⚠️")


st.title("Sign In")
username = st.text_input("Username or email address", icon=":material/person:")
password = st.text_input("Password", type="password", icon=":material/lock:")
submit_btn = st.button("Continue", type="primary", width="stretch",
                       on_click=login, kwargs={"username": username, "password": password})

st.markdown("Don't have an account? <a href='/register' target='_self'>Sign up</a>",
            unsafe_allow_html=True)

st.write("---")
st.button(label="Continue with Google",
          type="secondary", width="stretch", key="google-btn",
          on_click=st.login, kwargs={"provider": "google"})

if st.button(label="Continue with GitHub",
             type="secondary", width="stretch", key="github-btn"):
  auth_url = f"https://github.com/login/oauth/authorize?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
  st.write(
      f'<meta http-equiv="refresh" content="0; url={auth_url}">',
      unsafe_allow_html=True
  )
