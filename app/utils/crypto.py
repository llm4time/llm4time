import bcrypt
import streamlit as st
from cryptography.fernet import Fernet


class Crypto:
  """Handles encryption and password hashing."""

  def __init__(self):
    key = st.secrets["crypto"]["master_key"]
    if not key:
      raise RuntimeError("MASTER_KEY is not set in Streamlit secrets")

    self.fernet = Fernet(key.encode())

  def hash_password(self, password: str) -> str:
    """Generate irreversible password hash using bcrypt."""
    if password is None:
      raise ValueError("Password cannot be None")

    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)

    return hashed.decode('utf-8')

  def verify_password(self, password: str, hashed_password: str) -> bool:
    """Verify if password matches the stored hash."""
    if password is None or hashed_password is None:
      return False

    try:
      return bcrypt.checkpw(
          password.encode('utf-8'),
          hashed_password.encode('utf-8')
      )
    except Exception:
      return False

  def encrypt(self, value: str | None) -> str | None:
    """Encrypt data using Fernet (reversible - do NOT use for passwords)."""
    if value is None:
      return None

    return self.fernet.encrypt(value.encode()).decode()

  def decrypt(self, value: str | None) -> str | None:
    """Decrypt data encrypted with Fernet."""
    if value is None:
      return None

    try:
      return self.fernet.decrypt(value.encode()).decode()
    except Exception as e:
      print(f"Decryption error: {e}")
      return None
