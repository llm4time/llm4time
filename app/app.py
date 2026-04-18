import streamlit as st
from config.auth import Auth
from st_pages import add_page_title, get_nav_from_toml
from utils import abspath


if "auth" not in st.session_state:
  st.session_state.auth = Auth()

auth = st.session_state.auth

if not auth.user.is_logged_in:
  nav = get_nav_from_toml(abspath(".streamlit/auth.toml"))
  pg = st.navigation(nav, position="hidden")
  pg.run()
else:
  st.set_page_config(layout="wide")
  nav = get_nav_from_toml(abspath(".streamlit/pages.toml"))
  pg = st.navigation(nav)
  add_page_title(pg)
  pg.run()
