import streamlit as st
from .logging import logger
import storage.repository as repo
from utils import CookieManager, Crypto, load_image
import requests


CLIENT_ID = st.secrets["github"]["client_id"]
CLIENT_SECRET = st.secrets["github"]["client_secret"]

COOKIE_NAME = st.secrets["cookies"]["name"]
COOKIE_KEY = st.secrets["cookies"]["key"]
COOKIE_EXPIRY_DAYS = st.secrets["cookies"]["expiry_days"]


crypto = Crypto()

cookies = CookieManager(
    cookie_name=COOKIE_NAME,
    cookie_key=COOKIE_KEY,
    cookie_expiry_days=COOKIE_EXPIRY_DAYS,
)


class User:
  @property
  def is_logged_in(self):
    return st.session_state.get("authentication_status", False)

  @property
  def sub(self):
    return st.session_state.get("id")

  @property
  def username(self):
    return st.session_state.get("username")

  @property
  def email(self):
    return st.session_state.get("email")

  @property
  def picture(self):
    pic = st.session_state.get("picture")
    if isinstance(pic, bytes):
      return load_image(pic)
    return "app/static/avatar.svg"

  @picture.setter
  def picture(self, pic):
    st.session_state["picture"] = pic


class Auth:
  def __init__(self):
    self.token = cookies.get_cookie()
    self._recover_user()

  @property
  def user(self):
    return User()

  # ---------------------------
  # Session helper
  # ---------------------------
  def _set_session(self, user):
    st.session_state["id"] = user["id"]
    st.session_state["email"] = user["email"]
    st.session_state["username"] = user["username"]
    st.session_state["picture"] = user.get("picture")
    st.session_state["authentication_status"] = True
    cookies.set_cookie()

  # ---------------------------
  # Auth actions
  # ---------------------------
  def signin(self, identifier: str, password: str) -> bool:
    user = self._login(identifier, password)
    if not user:
      return False
    self._set_session(user)
    return True

  def signup(
      self,
      email: str,
      username: str,
      password: str | None = None,
      picture: str | None = None,
      oauth_provider: str | None = None,
      oauth_id: str | None = None
  ) -> bool:
    user_id = self._register(
        email=email,
        username=username,
        password=password,
        picture=picture,
        oauth_provider=oauth_provider,
        oauth_id=oauth_id,
    )
    if not user_id:
      return False
    self._set_session({
        "id": user_id,
        "email": email,
        "username": username,
        "picture": picture,
    })
    return True

  def signout(self) -> None:
    for key in ["id", "email", "username", "picture"]:
      st.session_state.pop(key, None)
    st.session_state["authentication_status"] = False
    st.logout()

  def delete_account(self) -> bool:
    try:
      user_id = st.session_state.get("id")
      if not user_id:
        return False
      repo.users().remove(user_id)
      self.signout()
      return True
    except Exception as e:
      print("Error deleting account:", e)
      return False

  # ---------------------------
  # Internal ops
  # ---------------------------
  def _login(self, identifier, password):
    try:
      users = repo.users()
      user = (
          users.select_by_email(identifier)
          if "@" in identifier
          else users.select_by_username(identifier)
      )
      if not user or not crypto.verify_password(password, user["password"]):
        return None
      return user
    except Exception as e:
      logger.error(f"Error logging in user: {e}")
      return None

  def _register(
      self,
      email: str,
      username: str,
      password: str | None = None,
      picture: str | None = None,
      oauth_provider: str | None = None,
      oauth_id: str | None = None
  ):
    try:
      return repo.users().insert(
          email=email,
          username=username,
          password=password,
          picture=picture,
          oauth_provider=oauth_provider,
          oauth_id=oauth_id
      )
    except Exception as e:
      logger.error(f"Error registering user: {e}")
      return None

  # ---------------------------
  # Recover session
  # ---------------------------
  def _recover_user(self):
    # Case 1: User logged in via Streamlit OAuth
    if st.user.is_logged_in:
      try:
        email = st.user.email
        users = repo.users()

        if not users.exists_email(email):
          picture = load_image(st.user.picture, as_bytes=True)
          self.signup(
              email=email,
              username=email.split("@")[0],
              picture=picture,
              oauth_provider="google",
              oauth_id=st.user.sub,
          )
          return

        google_user = users.select_by_oauth("google", st.user.sub)
        if google_user:
          self._set_session(google_user)
        else:
          st.logout()
        return
      except Exception as e:
        self.signout()
        return

    # Case 2: Cookie session
    if self.token:
      try:
        user = repo.users().select_by_id(self.token["sub"])
        if user:
          self._set_session(user)
      except Exception as e:
        self.signout()

    # Case 3: OAuth login via GitHub
    code = st.query_params.get("code")
    st.query_params.clear()
    if code:
      try:
        response = requests.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code": code,
            },
            headers={"Accept": "application/json"}
        )

        token = response.json().get("access_token")
        if token:
          user = requests.get(
              "https://api.github.com/user",
              headers={"Authorization": f"token {token}"}
          ).json()

          users = repo.users()
          if not users.exists_email(user["email"]):
            picture = load_image(user["avatar_url"], as_bytes=True)
            self.signup(
                email=user["email"],
                username=user["login"],
                picture=picture,
                oauth_provider="github",
                oauth_id=user["id"],
            )
            return

          github_user = users.select_by_oauth("github", user["id"])
          if github_user:
            self._set_session(github_user)
          else:
            st.logout()
          return
      except Exception as e:
        self.signout()
        return
