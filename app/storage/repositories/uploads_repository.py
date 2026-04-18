import uuid
from typing import Any
from datetime import datetime
from sqlalchemy import select, update as sql_update
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from storage.interfaces import BaseUploadsRepository
from storage.models import Upload

from storage.exceptions import (
    UploadNotFoundError,
    UploadAlreadyExistsError
)


class UploadsRepository(BaseUploadsRepository):
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
    """Insert upload using UUID. Soft delete-safe."""
    upload_id = str(uuid.uuid4())

    upload = Upload(
        id=upload_id,
        user_id=self.user_id,
        filename=filename,
        extension=extension,
        rows=rows,
        columns=columns,
        size=size,
        content=content,
        freq=freq,
    )

    try:
      self.session.add(upload)
      self.session.commit()
      return upload_id
    except IntegrityError:
      self.session.rollback()
      raise UploadAlreadyExistsError(
          f"Upload '{filename}' already exists"
      )

  # ---------------------------------------------------------
  # SELECTS
  # ---------------------------------------------------------
  def exists(self, filename: str) -> bool:
    stmt = select(Upload.id).where(
        Upload.user_id == self.user_id,
        Upload.filename == filename,
        Upload.deleted == False
    )
    return self.session.execute(stmt).first() is not None

  def select(self, filename: str) -> dict[str, Any] | None:
    stmt = select(
        Upload.id,
        Upload.user_id,
        Upload.filename,
        Upload.extension,
        Upload.rows,
        Upload.columns,
        Upload.size,
        Upload.freq,
        Upload.created_at,
        Upload.updated_at
    ).where(
        Upload.user_id == self.user_id,
        Upload.filename == filename,
        Upload.deleted == False
    )

    result = self.session.execute(stmt).first()

    if not result:
      return None

    return {
        'id': result.id,
        'user_id': result.user_id,
        'filename': result.filename,
        'extension': result.extension,
        'rows': result.rows,
        'columns': result.columns,
        'size': result.size,
        'freq': result.freq,
        'created_at': self._datetime_to_string(result.created_at),
        'updated_at': self._datetime_to_string(result.updated_at),
    }

  def select_with_content(self, filename: str) -> dict[str, Any] | None:
    stmt = select(Upload).where(
        Upload.user_id == self.user_id,
        Upload.filename == filename,
        Upload.deleted == False
    )

    upload = self.session.execute(stmt).scalar_one_or_none()

    if not upload:
      return None

    return {
        'id': upload.id,
        'user_id': upload.user_id,
        'filename': upload.filename,
        'extension': upload.extension,
        'rows': upload.rows,
        'columns': upload.columns,
        'size': upload.size,
        'content': upload.content,
        'freq': upload.freq,
        'deleted': upload.deleted,
        'created_at': self._datetime_to_string(upload.created_at),
        'updated_at': self._datetime_to_string(upload.updated_at),
    }

  def select_all(self) -> list[dict[str, Any]]:
    stmt = select(
        Upload.id,
        Upload.user_id,
        Upload.filename,
        Upload.extension,
        Upload.rows,
        Upload.columns,
        Upload.size,
        Upload.freq,
        Upload.created_at,
        Upload.updated_at
    ).where(
        Upload.user_id == self.user_id,
        Upload.deleted == False
    ).order_by(Upload.created_at.desc())

    results = self.session.execute(stmt).all()

    return [
        {
            'id': row.id,
            'user_id': row.user_id,
            'filename': row.filename,
            'extension': row.extension,
            'rows': row.rows,
            'columns': row.columns,
            'size': row.size,
            'freq': row.freq,
            'created_at': self._datetime_to_string(row.created_at),
            'updated_at': self._datetime_to_string(row.updated_at),
        }
        for row in results
    ]

  # ---------------------------------------------------------
  # UPDATE
  # ---------------------------------------------------------
  def update(
      self,
      filename: str,
      rows: int | None = None,
      columns: int | None = None,
      size: float | None = None,
  ) -> bool:
    updates = {}

    if rows is not None:
      updates['rows'] = rows
    if columns is not None:
      updates['columns'] = columns
    if size is not None:
      updates['size'] = size

    if not updates:
      return False

    stmt = (
        sql_update(Upload)
        .where(
            Upload.user_id == self.user_id,
            Upload.filename == filename,
            Upload.deleted == False
        )
        .values(**updates)
    )

    result = self.session.execute(stmt)
    self.session.commit()

    if result.rowcount == 0:
      raise UploadNotFoundError(f"Upload '{filename}' not found")

    return True

  def rename(self, old_filename: str, new_filename: str) -> bool:
    stmt = (
        sql_update(Upload)
        .where(
            Upload.user_id == self.user_id,
            Upload.filename == old_filename,
            Upload.deleted == False
        )
        .values(filename=new_filename)
    )

    result = self.session.execute(stmt)
    self.session.commit()

    if result.rowcount == 0:
      raise UploadNotFoundError(f"Upload '{old_filename}' not found")

    return True

  # ---------------------------------------------------------
  # REMOVE (Soft Delete)
  # ---------------------------------------------------------
  def remove(self, filename: str) -> bool:
    """Soft delete: deleted = TRUE."""
    stmt = (
        sql_update(Upload)
        .where(
            Upload.user_id == self.user_id,
            Upload.filename == filename,
            Upload.deleted == False
        )
        .values(deleted=True)
    )

    result = self.session.execute(stmt)
    self.session.commit()

    if result.rowcount == 0:
      raise UploadNotFoundError(f"Upload '{filename}' not found")

    return True

  def remove_many(self, filenames: list[str]) -> int:
    if not filenames:
      return 0

    stmt = (
        sql_update(Upload)
        .where(
            Upload.user_id == self.user_id,
            Upload.filename.in_(filenames),
            Upload.deleted == False
        )
        .values(deleted=True)
    )

    result = self.session.execute(stmt)
    self.session.commit()
    return result.rowcount

  def remove_all(self) -> int:
    stmt = (
        sql_update(Upload)
        .where(
            Upload.user_id == self.user_id,
            Upload.deleted == False
        )
        .values(deleted=True)
    )

    result = self.session.execute(stmt)
    self.session.commit()
    return result.rowcount
