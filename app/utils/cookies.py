from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import jwt
from jwt import DecodeError, InvalidSignatureError
import streamlit as st
import extra_streamlit_components as stx


class CookieManager:
  """Manages JWT-based authentication cookies."""

  def __init__(
      self,
      cookie_name: Optional[str] = None,
      cookie_key: Optional[str] = None,
      cookie_expiry_days: Optional[float] = None,
  ) -> None:
    self.cookie_name = cookie_name
    self.cookie_key = cookie_key
    self.cookie_expiry_days = cookie_expiry_days
    self.cookie_manager = stx.CookieManager()
    self.token = None
    self.exp_date = None

  def get_cookie(self) -> Optional[Dict[str, Any]]:
    """Retrieve and validate authentication cookie."""
    self.token = (
        st.context.cookies[self.cookie_name]
        if self.cookie_name in st.context.cookies
        else None
    )

    if self.token is not None:
      self.token = self._token_decode()

      if (
          self.token is not False
          and 'sub' in self.token
          and self.token['exp_date'] > datetime.now().timestamp()
      ):
        try:
          self.token['sub'] = self.token['sub']
        except (ValueError, TypeError):
          return None

        return self.token

    return None

  def set_cookie(self) -> None:
    """Create and store authentication cookie with JWT token."""
    if self.cookie_expiry_days != 0:
      self.exp_date = self._set_exp_date()
      token = self._token_encode()

      self.cookie_manager.set(
          self.cookie_name,
          token,
          expires_at=datetime.now() + timedelta(days=self.cookie_expiry_days)
      )

  def delete_cookie(self) -> None:
    """Remove authentication cookie."""
    try:
      self.cookie_manager.delete(self.cookie_name)
    except KeyError as e:
      print(f"Cookie deletion error: {e}")

  def _set_exp_date(self) -> float:
    """Calculate token expiration timestamp."""
    return (datetime.now() + timedelta(days=self.cookie_expiry_days)).timestamp()

  def _token_decode(self) -> Optional[Dict[str, Any]]:
    """Decode JWT token from cookie."""
    try:
      return jwt.decode(self.token, self.cookie_key, algorithms=['HS256'])
    except (DecodeError, InvalidSignatureError) as e:
      return False

  def _token_encode(self) -> str:
    """Encode user session data into JWT token."""
    return jwt.encode(
        {
            'sub': str(st.session_state['id']),
            'exp_date': self.exp_date
        },
        self.cookie_key,
        algorithm='HS256'
    )
