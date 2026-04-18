import uuid
import json
from typing import Any
from datetime import datetime
from sqlalchemy import select, update as sql_update
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from storage.interfaces import BasePromptsRepository
from storage.models import Prompt
from storage.exceptions import (
    PromptAlreadyExistsError,
    PromptNotFoundError,
)


class PromptsRepository(BasePromptsRepository):
  def __init__(self, user_id: str, session: Session):
    super().__init__(user_id)
    self.session = session

  @staticmethod
  def _datetime_to_string(dt: datetime) -> str | None:
    """Converte datetime em string no formato YYYY-MM-DD HH:MM:SS"""
    return dt.strftime('%Y-%m-%d %H:%M:%S') if dt else None

  # ---------------------------------------------------------
  # INSERT
  # ---------------------------------------------------------
  def insert(self, name: str, content: str, variables: dict[str, Any] | None = None) -> str:
    """Insert prompt with UUID. Raises if name already exists."""
    prompt_id = str(uuid.uuid4())

    prompt = Prompt(
        id=prompt_id,
        user_id=self.user_id,
        name=name,
        content=content,
        variables=json.dumps(variables) if variables else None,
    )

    try:
      self.session.add(prompt)
      self.session.commit()
      return prompt_id
    except IntegrityError:
      self.session.rollback()
      raise PromptAlreadyExistsError(f"Prompt '{name}' already exists")

  # ---------------------------------------------------------
  # SELECTS
  # ---------------------------------------------------------
  def select(self, name: str) -> dict[str, Any] | None:
    """Select a prompt by name (not deleted)."""
    stmt = select(Prompt).where(
        Prompt.user_id == self.user_id,
        Prompt.name == name,
        Prompt.deleted == False
    )

    prompt = self.session.execute(stmt).scalar_one_or_none()

    if not prompt:
      return None

    result = {
        'id': prompt.id,
        'user_id': prompt.user_id,
        'name': prompt.name,
        'content': prompt.content,
        'variables': json.loads(prompt.variables) if prompt.variables else None,
        'deleted': prompt.deleted,
        'created_at': self._datetime_to_string(prompt.created_at),
        'updated_at': self._datetime_to_string(prompt.updated_at),
    }

    return result

  def select_all(self) -> list[dict[str, Any]]:
    """Select all non-deleted prompts ordered by name."""
    stmt = select(Prompt).where(
        Prompt.user_id == self.user_id,
        Prompt.deleted == False
    ).order_by(Prompt.name)

    prompts = self.session.execute(stmt).scalars().all()

    results = []
    for prompt in prompts:
      data = {
          'id': prompt.id,
          'user_id': prompt.user_id,
          'name': prompt.name,
          'content': prompt.content,
          'variables': json.loads(prompt.variables) if prompt.variables else None,
          'deleted': prompt.deleted,
          'created_at': self._datetime_to_string(prompt.created_at),
          'updated_at': self._datetime_to_string(prompt.updated_at),
      }
      results.append(data)

    return results

  # ---------------------------------------------------------
  # UPDATE
  # ---------------------------------------------------------
  def update(self, name: str, content: str, variables: dict[str, Any] | None = None) -> bool:
    """Update content and variables of a prompt."""
    stmt = (
        sql_update(Prompt)
        .where(
            Prompt.user_id == self.user_id,
            Prompt.name == name,
            Prompt.deleted == False
        )
        .values(
            content=content,
            variables=json.dumps(variables) if variables else None
        )
    )

    result = self.session.execute(stmt)
    self.session.commit()

    if result.rowcount == 0:
      raise PromptNotFoundError(f"Prompt '{name}' not found")

    return True

  def rename(self, old_name: str, new_name: str) -> bool:
    """Rename a prompt (keeps UUID)."""
    try:
      stmt = (
          sql_update(Prompt)
          .where(
              Prompt.user_id == self.user_id,
              Prompt.name == old_name,
              Prompt.deleted == False
          )
          .values(name=new_name)
      )

      result = self.session.execute(stmt)
      self.session.commit()

      if result.rowcount == 0:
        raise PromptNotFoundError(f"Prompt '{old_name}' not found")

      return True
    except IntegrityError:
      self.session.rollback()
      raise PromptAlreadyExistsError(f"Prompt '{new_name}' already exists")

  # ---------------------------------------------------------
  # REMOVE (SOFT DELETE)
  # ---------------------------------------------------------
  def remove(self, name: str) -> bool:
    """Soft delete: marks deleted=TRUE."""
    stmt = (
        sql_update(Prompt)
        .where(
            Prompt.user_id == self.user_id,
            Prompt.name == name,
            Prompt.deleted == False
        )
        .values(deleted=True)
    )

    result = self.session.execute(stmt)
    self.session.commit()

    if result.rowcount == 0:
      raise PromptNotFoundError(f"Prompt '{name}' not found")

    return True

  def remove_many(self, names: list[str]) -> int:
    """Soft delete multiple prompts."""
    if not names:
      return 0

    stmt = (
        sql_update(Prompt)
        .where(
            Prompt.user_id == self.user_id,
            Prompt.name.in_(names),
            Prompt.deleted == False
        )
        .values(deleted=True)
    )

    result = self.session.execute(stmt)
    self.session.commit()
    return result.rowcount
