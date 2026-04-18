# Database exceptions
class DatabaseError(Exception):
  """Base exception for database operations."""
  pass


class NotFoundError(DatabaseError):
  """Base exception for records not found."""
  pass


class AlreadyExistsError(DatabaseError):
  """Base exception for duplicate records."""
  pass


# User exceptions
class UserNotFoundError(NotFoundError):
  """Raised when a user is not found in the users table."""
  pass


class UserAlreadyExistsError(AlreadyExistsError):
  """Raised when trying to insert a user that already exists."""
  pass


class EmailAlreadyExistsError(UserAlreadyExistsError):
  """Raised when trying to insert a user with an email that already exists."""
  pass


class UsernameAlreadyExistsError(UserAlreadyExistsError):
  """Raised when trying to insert a user with a username that already exists."""
  pass


class OAuthAlreadyExistsError(UserAlreadyExistsError):
  """Raised when OAuth account already exists."""
  pass


# Upload exceptions


class UploadNotFoundError(NotFoundError):
  """Raised when an upload is not found in the uploads table."""
  pass


class UploadAlreadyExistsError(AlreadyExistsError):
  """Raised when trying to insert an upload that already exists."""
  pass


# Model exceptions
class ModelNotFoundError(NotFoundError):
  """Raised when a model is not found in the models table."""
  pass


class ModelAlreadyExistsError(AlreadyExistsError):
  """Raised when trying to insert a model that already exists."""
  pass


# Prompt exceptions
class PromptNotFoundError(NotFoundError):
  """Raised when a prompt is not found in the prompts table."""
  pass


class PromptAlreadyExistsError(AlreadyExistsError):
  """Raised when trying to insert a prompt that already exists."""
  pass


# History exceptions
class HistoryNotFoundError(NotFoundError):
  """Raised when a record is not found in the history table."""
  pass
