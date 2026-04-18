import uuid
from typing import Any
from datetime import datetime
from sqlalchemy import select, update as sql_update, or_, and_
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from utils.crypto import Crypto
from storage.interfaces import BaseModelsRepository
from storage.models import Model

from storage.exceptions import (
    ModelAlreadyExistsError,
    ModelNotFoundError,
)


class ModelsRepository(BaseModelsRepository):
  def __init__(self, user_id: str, session: Session):
    super().__init__(user_id)
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
      name: str,
      provider: str,
      api_key: str | None = None,
      endpoint: str | None = None,
      api_version: str | None = None,
  ) -> str:
    """Insert model using UUID. Soft-delete aware."""
    model_id = str(uuid.uuid4())
    encrypted_key = self.crypto.encrypt(api_key)

    model = Model(
        id=model_id,
        user_id=self.user_id,
        name=name,
        provider=provider,
        api_key=encrypted_key,
        endpoint=endpoint,
        api_version=api_version,
    )

    try:
      self.session.add(model)
      self.session.commit()
      return model_id
    except IntegrityError:
      self.session.rollback()
      raise ModelAlreadyExistsError(
          f"Model '{name}' with provider '{provider}' already exists"
      )

  # ---------------------------------------------------------
  # SELECTS
  # ---------------------------------------------------------
  def select(self, provider: str) -> list[dict[str, Any]]:
    """Select all models by provider (not deleted)."""
    stmt = select(Model).where(
        Model.user_id == self.user_id,
        Model.provider == provider,
        Model.deleted == False
    ).order_by(Model.name)

    models = self.session.execute(stmt).scalars().all()

    results = []
    for model in models:
      result = {
          'id': model.id,
          'user_id': model.user_id,
          'name': model.name,
          'provider': model.provider,
          'api_key': self.crypto.decrypt(model.api_key),
          'endpoint': model.endpoint,
          'api_version': model.api_version,
          'deleted': model.deleted,
          'created_at': self._datetime_to_string(model.created_at),
          'updated_at': self._datetime_to_string(model.updated_at),
      }
      results.append(result)

    return results

  def select_all(self) -> list[dict[str, Any]]:
    """Select all models for the user (not deleted)."""
    stmt = select(Model).where(
        Model.user_id == self.user_id,
        Model.deleted == False
    ).order_by(Model.provider, Model.name)

    models = self.session.execute(stmt).scalars().all()

    results = []
    for model in models:
      result = {
          'id': model.id,
          'user_id': model.user_id,
          'name': model.name,
          'provider': model.provider,
          'api_key': self.crypto.decrypt(model.api_key),
          'endpoint': model.endpoint,
          'api_version': model.api_version,
          'deleted': model.deleted,
          'created_at': self._datetime_to_string(model.created_at),
          'updated_at': self._datetime_to_string(model.updated_at),
      }
      results.append(result)

    return results

  # ---------------------------------------------------------
  # UPDATE
  # ---------------------------------------------------------
  def rename(self, old_name: str, new_name: str, provider: str) -> bool:
    """Rename model (soft-delete aware)."""
    try:
      stmt = (
          sql_update(Model)
          .where(
              Model.user_id == self.user_id,
              Model.name == old_name,
              Model.provider == provider,
              Model.deleted == False
          )
          .values(name=new_name)
      )

      result = self.session.execute(stmt)
      self.session.commit()

      if result.rowcount == 0:
        raise ModelNotFoundError(
            f"Model '{old_name}' with provider '{provider}' not found"
        )

      return True
    except IntegrityError:
      self.session.rollback()
      raise ModelAlreadyExistsError(
          f"Model '{new_name}' with provider '{provider}' already exists"
      )

  def update(
      self,
      name: str,
      provider: str,
      api_key: str | None = None,
      endpoint: str | None = None,
      api_version: str | None = None,
  ) -> bool:
    """Update model fields (soft-delete aware)."""
    encrypted_key = self.crypto.encrypt(api_key)

    stmt = (
        sql_update(Model)
        .where(
            Model.user_id == self.user_id,
            Model.name == name,
            Model.provider == provider,
            Model.deleted == False
        )
        .values(
            api_key=encrypted_key,
            endpoint=endpoint,
            api_version=api_version
        )
    )

    result = self.session.execute(stmt)
    self.session.commit()

    if result.rowcount == 0:
      raise ModelNotFoundError(
          f"Model '{name}' with provider '{provider}' not found"
      )

    return True

  # ---------------------------------------------------------
  # REMOVE (soft delete)
  # ---------------------------------------------------------
  def remove(self, name: str, provider: str) -> bool:
    """Soft delete: marks deleted = TRUE."""
    stmt = (
        sql_update(Model)
        .where(
            Model.user_id == self.user_id,
            Model.name == name,
            Model.provider == provider,
            Model.deleted == False
        )
        .values(deleted=True)
    )

    result = self.session.execute(stmt)
    self.session.commit()

    if result.rowcount == 0:
      raise ModelNotFoundError(
          f"Model '{name}' with provider '{provider}' not found"
      )

    return True

  def remove_many(self, models: list[tuple[str, str]]) -> int:
    """Soft delete multiple models."""
    if not models:
      return 0

    # Build OR conditions for each (name, provider) pair
    conditions = [
        and_(
            Model.name == name,
            Model.provider == provider
        )
        for name, provider in models
    ]

    stmt = (
        sql_update(Model)
        .where(
            Model.user_id == self.user_id,
            or_(*conditions),
            Model.deleted == False
        )
        .values(deleted=True)
    )

    result = self.session.execute(stmt)
    self.session.commit()

    return result.rowcount
