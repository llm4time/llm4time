from abc import ABC, abstractmethod
from typing import Any


class BaseUsersRepository(ABC):

  @abstractmethod
  def insert(
      self,
      email: str,
      username: str,
      password: str,
      picture: bytes | None = None,
      oauth_provider: str | None = None,
      oauth_id: str | None = None,
  ) -> str:
    """Insert a user. Returns the inserted user UUID."""
    pass

  @abstractmethod
  def select_by_id(self, user_id: str) -> dict[str, Any] | None:
    """Select a user by UUID."""
    pass

  @abstractmethod
  def select_by_email(self, email: str) -> dict[str, Any] | None:
    """Select a user by email."""
    pass

  @abstractmethod
  def select_by_username(self, username: str) -> dict[str, Any] | None:
    """Select a user by username."""
    pass

  @abstractmethod
  def select_by_oauth(self, provider: str, oauth_id: str) -> dict[str, Any] | None:
    """Select a user by OAuth credentials."""
    pass

  @abstractmethod
  def select_all(self) -> list[dict[str, Any]]:
    """Select all users."""
    pass

  @abstractmethod
  def exists_email(self, email: str) -> bool:
    """Check if email already exists."""
    pass

  @abstractmethod
  def exists_username(self, username: str) -> bool:
    """Check if username already exists."""
    pass

  @abstractmethod
  def exists_oauth(self, provider: str, oauth_id: str) -> bool:
    """Check if OAuth account already exists."""
    pass

  @abstractmethod
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
    """Update a user's information."""
    pass

  @abstractmethod
  def update_picture(self, user_id: str, picture: bytes | None) -> bool:
    """Update only the user's picture."""
    pass

  @abstractmethod
  def update_password(self, user_id: str, password: str) -> bool:
    """Update only the user's password."""
    pass

  @abstractmethod
  def remove(self, user_id: str) -> bool:
    """Soft-delete a user by UUID."""
    pass


class BaseUploadsRepository(ABC):
  def __init__(self, user_id: str):
    self.user_id = user_id

  @abstractmethod
  def insert(
      self,
      filename: str,
      extension: str,
      rows: int,
      columns: int,
      size: float,
      content: bytes,
      freq: str,
  ) -> str:
    """Insert an upload record. Returns the inserted record UUID."""
    pass

  @abstractmethod
  def exists(self, filename: str) -> bool:
    """Check if a file exists."""
    pass

  @abstractmethod
  def select(self, filename: str) -> dict[str, Any] | None:
    """Select an upload by filename."""
    pass

  @abstractmethod
  def select_all(self) -> list[dict[str, Any]]:
    """Select all uploads."""
    pass

  @abstractmethod
  def remove(self, filename: str) -> bool:
    """Soft-delete a single upload by filename."""
    pass

  @abstractmethod
  def remove_many(self, filenames: list[str]) -> int:
    """Soft-delete multiple uploads. Returns count of removed records."""
    pass

  @abstractmethod
  def remove_all(self) -> int:
    """Soft-delete all uploads. Returns count of removed records."""
    pass

  @abstractmethod
  def update(
      self,
      filename: str,
      rows: int | None = None,
      columns: int | None = None,
      size: float | None = None,
  ) -> bool:
    """Update an upload's metadata."""
    pass

  @abstractmethod
  def rename(self, old_filename: str, new_filename: str) -> bool:
    """Rename an upload."""
    pass


class BaseModelsRepository(ABC):
  def __init__(self, user_id: str):
    self.user_id = user_id

  @abstractmethod
  def insert(
      self,
      name: str,
      provider: str,
      api_key: str | None = None,
      endpoint: str | None = None,
      api_version: str | None = None,
  ) -> str:
    """Insert a model. Returns the inserted model UUID."""
    pass

  @abstractmethod
  def select(self, provider: str) -> list[dict[str, Any]]:
    """Select models by provider."""
    pass

  @abstractmethod
  def select_all(self) -> list[dict[str, Any]]:
    """Select all models."""
    pass

  @abstractmethod
  def remove(self, name: str, provider: str) -> bool:
    """Soft-delete a model."""
    pass

  @abstractmethod
  def remove_many(self, models: list[tuple[str, str]]) -> int:
    """Soft-delete multiple models."""
    pass

  @abstractmethod
  def rename(self, old_name: str, new_name: str, provider: str) -> bool:
    """Rename a model."""
    pass

  @abstractmethod
  def update(
      self,
      name: str,
      provider: str,
      api_key: str | None = None,
      endpoint: str | None = None,
      api_version: str | None = None,
  ) -> bool:
    """Update a model's configuration."""
    pass


class BasePromptsRepository(ABC):
  def __init__(self, user_id: str):
    self.user_id = user_id

  @abstractmethod
  def insert(self, name: str, content: str, variables: dict[str, Any] | None = None) -> str:
    """Insert a prompt. Returns the inserted prompt UUID."""
    pass

  @abstractmethod
  def select(self, name: str) -> dict[str, Any] | None:
    """Select a prompt by name."""
    pass

  @abstractmethod
  def select_all(self) -> list[dict[str, Any]]:
    """Select all prompts."""
    pass

  @abstractmethod
  def remove(self, name: str) -> bool:
    """Remove a single prompt by name."""
    pass

  @abstractmethod
  def remove_many(self, names: list[str]) -> int:
    """Remove multiple prompts."""
    pass

  @abstractmethod
  def update(self, name: str, content: str, variables: dict[str, Any] | None = None) -> bool:
    """Update a prompt's content and variables."""
    pass

  @abstractmethod
  def rename(self, old_name: str, new_name: str) -> bool:
    """Rename a prompt."""
    pass


class BaseHistoryRepository(ABC):
  def __init__(self, user_id: str):
    self.user_id = user_id

  @abstractmethod
  def insert(self, **kwargs) -> str:
    """Insert a record. Returns the inserted record UUID."""
    pass

  @abstractmethod
  def select(self, dataset: str, prompt_types: list[str]) -> list[dict[str, Any]]:
    """Select records by dataset and prompt types."""
    pass

  @abstractmethod
  def select_by_id(self, record_id: str) -> dict[str, Any] | None:
    """Select a single record by UUID."""
    pass

  @abstractmethod
  def group_by(self, columns: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Group records by columns."""
    pass

  @abstractmethod
  def remove(self, record_id: str) -> bool:
    """Soft-delete a record by UUID."""
    pass

  @abstractmethod
  def remove_many(self, dataset: str, prompt_types: list[str]) -> int:
    """Soft-delete multiple records."""
    pass

  @abstractmethod
  def remove_all(self) -> int:
    """Soft-delete all records."""
    pass
