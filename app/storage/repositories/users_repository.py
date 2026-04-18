import uuid
from typing import Any
from datetime import datetime
from sqlalchemy import select, update as sql_update
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from utils.crypto import Crypto
from storage.interfaces import BaseUsersRepository
from storage.models import User

from storage.exceptions import (
    UserNotFoundError,
    EmailAlreadyExistsError,
    UsernameAlreadyExistsError,
    OAuthAlreadyExistsError,
)


class UsersRepository(BaseUsersRepository):
  def __init__(self, session: Session):
    super().__init__()
    self.session = session
    self.crypto = Crypto()

  @staticmethod
  def _datetime_to_string(dt: datetime) -> str | None:
    """Converte datetime em string no formato YYYY-MM-DD HH:MM:SS"""
    return dt.strftime('%Y-%m-%d %H:%M:%S') if dt else None

  # ---------------------------------------------------------
  # INSERT
  # ---------------------------------------------------------
  def insert(
      self,
      email: str,
      username: str,
      password: str | None = None,
      picture: bytes | None = None,
      oauth_provider: str | None = None,
      oauth_id: str | None = None,
  ) -> str:
    user_id = str(uuid.uuid4())
    hashed_password = self.crypto.hash_password(password) if password else None

    user = User(
        id=user_id,
        email=email,
        username=username,
        password=hashed_password,
        picture=picture,
        oauth_provider=oauth_provider,
        oauth_id=oauth_id,
    )

    try:
      self.session.add(user)
      self.session.commit()
      return user_id
    except IntegrityError as e:
      self.session.rollback()
      error_msg = str(e).lower()
      if "email" in error_msg:
        raise EmailAlreadyExistsError(f"Email '{email}' already exists")
      elif "username" in error_msg:
        raise UsernameAlreadyExistsError(f"Username '{username}' already exists")
      elif "oauth" in error_msg:
        raise OAuthAlreadyExistsError(
            f"OAuth account '{oauth_provider}:{oauth_id}' already exists"
        )
      else:
        raise

  # ---------------------------------------------------------
  # SELECTS
  # ---------------------------------------------------------
  def select_by_id(self, user_id: str) -> dict[str, Any] | None:
    stmt = select(User).where(User.id == user_id, User.deleted == False)
    user = self.session.execute(stmt).scalar_one_or_none()

    if not user:
      return None

    return {
        'id': user.id,
        'email': user.email,
        'username': user.username,
        'password': user.password,
        'picture': user.picture,
        'oauth_provider': user.oauth_provider,
        'oauth_id': user.oauth_id,
        'deleted': user.deleted,
        'created_at': self._datetime_to_string(user.created_at),
        'updated_at': self._datetime_to_string(user.updated_at),
    }

  def select_by_email(self, email: str) -> dict[str, Any] | None:
    stmt = select(User).where(User.email == email, User.deleted == False)
    user = self.session.execute(stmt).scalar_one_or_none()

    if not user:
      return None

    return {
        'id': user.id,
        'email': user.email,
        'username': user.username,
        'password': user.password,
        'picture': user.picture,
        'oauth_provider': user.oauth_provider,
        'oauth_id': user.oauth_id,
        'deleted': user.deleted,
        'created_at': self._datetime_to_string(user.created_at),
        'updated_at': self._datetime_to_string(user.updated_at),
    }

  def select_by_username(self, username: str) -> dict[str, Any] | None:
    stmt = select(User).where(User.username == username, User.deleted == False)
    user = self.session.execute(stmt).scalar_one_or_none()

    if not user:
      return None

    return {
        'id': user.id,
        'email': user.email,
        'username': user.username,
        'password': user.password,
        'picture': user.picture,
        'oauth_provider': user.oauth_provider,
        'oauth_id': user.oauth_id,
        'deleted': user.deleted,
        'created_at': self._datetime_to_string(user.created_at),
        'updated_at': self._datetime_to_string(user.updated_at),
    }

  def select_by_oauth(self, provider: str, oauth_id: str) -> dict[str, Any] | None:
    stmt = select(User).where(
        User.oauth_provider == provider,
        User.oauth_id == oauth_id,
        User.deleted == False
    )
    user = self.session.execute(stmt).scalar_one_or_none()

    if not user:
      return None

    return {
        'id': user.id,
        'email': user.email,
        'username': user.username,
        'password': user.password,
        'picture': user.picture,
        'oauth_provider': user.oauth_provider,
        'oauth_id': user.oauth_id,
        'deleted': user.deleted,
        'created_at': self._datetime_to_string(user.created_at),
        'updated_at': self._datetime_to_string(user.updated_at),
    }

  def select_all(self) -> list[dict[str, Any]]:
    stmt = select(User).where(User.deleted == False).order_by(User.created_at.desc())
    users = self.session.execute(stmt).scalars().all()

    return [
        {
            'id': user.id,
            'email': user.email,
            'username': user.username,
            'password': user.password,
            'picture': user.picture,
            'oauth_provider': user.oauth_provider,
            'oauth_id': user.oauth_id,
            'deleted': user.deleted,
            'created_at': self._datetime_to_string(user.created_at),
            'updated_at': self._datetime_to_string(user.updated_at),
        }
        for user in users
    ]

  # ---------------------------------------------------------
  # EXISTS
  # ---------------------------------------------------------
  def exists_email(self, email: str) -> bool:
    stmt = select(User.id).where(User.email == email, User.deleted == False)
    return self.session.execute(stmt).first() is not None

  def exists_username(self, username: str) -> bool:
    stmt = select(User.id).where(User.username == username, User.deleted == False)
    return self.session.execute(stmt).first() is not None

  def exists_oauth(self, provider: str, oauth_id: str) -> bool:
    stmt = select(User.id).where(
        User.oauth_provider == provider,
        User.oauth_id == oauth_id,
        User.deleted == False
    )
    return self.session.execute(stmt).first() is not None

  # ---------------------------------------------------------
  # UPDATE
  # ---------------------------------------------------------
  def update(
      self,
      user_id: str,
      email: str | None = None,
      username: str | None = None,
      password: str | None = None,
      picture: bytes | None = None,
      oauth_provider: str | None = None,
      oauth_id: str | None = None,
  ) -> bool:
    updates = {}

    if email is not None:
      stmt = select(User.id).where(
          User.email == email,
          User.id != user_id,
          User.deleted == False,
      )
      if self.session.execute(stmt).first():
        raise EmailAlreadyExistsError(f"Email '{email}' already exists")
      updates["email"] = email

    if username is not None:
      stmt = select(User.id).where(
          User.username == username,
          User.id != user_id,
          User.deleted == False,
      )
      if self.session.execute(stmt).first():
        raise UsernameAlreadyExistsError(f"Username '{username}' already exists")
      updates["username"] = username

    if password is not None:
      updates["password"] = self.crypto.hash_password(password)

    if picture is not None:
      updates["picture"] = picture

    if oauth_provider is not None or oauth_id is not None:
      stmt = select(User.id).where(
          User.oauth_provider == oauth_provider,
          User.oauth_id == oauth_id,
          User.id != user_id,
          User.deleted == False,
      )
      if self.session.execute(stmt).first():
        raise OAuthAlreadyExistsError(
            f"OAuth account '{oauth_provider}:{oauth_id}' already exists"
        )
      if oauth_provider is not None:
        updates["oauth_provider"] = oauth_provider
      if oauth_id is not None:
        updates["oauth_id"] = oauth_id

    if not updates:
      return False

    stmt = (
        sql_update(User)
        .where(User.id == user_id, User.deleted == False)
        .values(**updates)
    )
    result = self.session.execute(stmt)
    self.session.commit()

    if result.rowcount == 0:
      raise UserNotFoundError(f"User with ID {user_id} not found")

    return True

  def update_picture(self, user_id: str, picture: bytes | None) -> bool:
    stmt = (
        sql_update(User)
        .where(User.id == user_id, User.deleted == False)
        .values(picture=picture)
    )
    result = self.session.execute(stmt)
    self.session.commit()

    if result.rowcount == 0:
      raise UserNotFoundError(f"User with ID {user_id} not found")

    return True

  def update_password(self, user_id: str, password: str) -> bool:
    hashed_password = self.crypto.hash_password(password)

    stmt = (
        sql_update(User)
        .where(User.id == user_id, User.deleted == False)
        .values(password=hashed_password)
    )
    result = self.session.execute(stmt)
    self.session.commit()

    if result.rowcount == 0:
      raise UserNotFoundError(f"User with ID {user_id} not found")

    return True

  # ---------------------------------------------------------
  # REMOVE
  # ---------------------------------------------------------
  def remove(self, user_id: str) -> bool:
    """Soft delete: marks deleted=TRUE."""
    stmt = (
        sql_update(User)
        .where(User.id == user_id, User.deleted == False)
        .values(deleted=True)
    )
    result = self.session.execute(stmt)
    self.session.commit()

    if result.rowcount == 0:
      raise UserNotFoundError(f"User with ID {user_id} not found")

    return True
