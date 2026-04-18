import json
import uuid
from typing import Any
from datetime import datetime
from sqlalchemy import select, update as sql_update, func
from sqlalchemy.orm import Session

from storage.interfaces import BaseHistoryRepository
from storage.models import History
from storage.exceptions import HistoryNotFoundError


class HistoryRepository(BaseHistoryRepository):
  def __init__(self, user_id: str, session: Session):
    super().__init__(user_id)
    self.session = session

  @staticmethod
  def _datetime_to_string(dt: datetime) -> str | None:
    """Converte datetime em string no formato YYYY-MM-DD HH:MM:SS"""
    return dt.strftime('%Y-%m-%d %H:%M:%S') if dt else None

  # --------------------------------------------
  # INSERT
  # --------------------------------------------

  def insert(self, **kwargs) -> str:
    record_id = str(uuid.uuid4())

    # Convert dict/list values to JSON strings
    processed_kwargs = {
        key: json.dumps(value) if isinstance(value, (dict, list)) else value
        for key, value in kwargs.items()
    }

    history = History(
        id=record_id,
        user_id=self.user_id,
        **processed_kwargs
    )

    self.session.add(history)
    self.session.commit()

    return record_id

  # --------------------------------------------
  # SELECT
  # --------------------------------------------

  def select(self, dataset: str, prompt_types: list[str]) -> list[dict[str, Any]]:
    stmt = select(History).where(
        History.user_id == self.user_id,
        History.dataset == dataset,
        History.prompt_type.in_(prompt_types),
        History.deleted == False
    ).order_by(History.created_at.desc())

    results = self.session.execute(stmt).scalars().all()

    return [
        {
            'id': record.id,
            'user_id': record.user_id,
            'model': record.model,
            'provider': record.provider,
            'temperature': record.temperature,
            'dataset': record.dataset,
            'columns': record.columns,
            'start_time': record.start_time,
            'end_time': record.end_time,
            'prompt_type': record.prompt_type,
            'time_series_format': record.time_series_format,
            'time_series_type': record.time_series_type,
            'examples': record.examples,
            'sampling': record.sampling,
            'forecast_horizon': record.forecast_horizon,
            'input_tokens': record.input_tokens,
            'output_tokens': record.output_tokens,
            'response_time': record.response_time,
            'response_raw': record.response_raw,
            'response_predicted': record.response_predicted,
            'validation': record.validation,
            'metrics': record.metrics,
            'statistics_predicted': record.statistics_predicted,
            'statistics_validation': record.statistics_validation,
            'training': record.training,
            'prompt': record.prompt,
            'deleted': record.deleted,
            'created_at': self._datetime_to_string(record.created_at),
            'updated_at': self._datetime_to_string(record.updated_at),
        }
        for record in results
    ]

  def select_by_id(self, record_id: str) -> dict[str, Any] | None:
    stmt = select(History).where(
        History.id == record_id,
        History.user_id == self.user_id,
        History.deleted == False
    )

    record = self.session.execute(stmt).scalar_one_or_none()

    if not record:
      return None

    return {
        'id': record.id,
        'user_id': record.user_id,
        'model': record.model,
        'provider': record.provider,
        'temperature': record.temperature,
        'dataset': record.dataset,
        'columns': record.columns,
        'start_time': record.start_time,
        'end_time': record.end_time,
        'prompt_type': record.prompt_type,
        'time_series_format': record.time_series_format,
        'time_series_type': record.time_series_type,
        'examples': record.examples,
        'sampling': record.sampling,
        'forecast_horizon': record.forecast_horizon,
        'input_tokens': record.input_tokens,
        'output_tokens': record.output_tokens,
        'response_time': record.response_time,
        'response_raw': record.response_raw,
        'response_predicted': record.response_predicted,
        'validation': record.validation,
        'metrics': record.metrics,
        'statistics_predicted': record.statistics_predicted,
        'statistics_validation': record.statistics_validation,
        'training': record.training,
        'prompt': record.prompt,
        'deleted': record.deleted,
        'created_at': self._datetime_to_string(record.created_at),
        'updated_at': self._datetime_to_string(record.updated_at),
    }

  # --------------------------------------------
  # GROUP BY
  # --------------------------------------------

  def group_by(self, columns: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    # Build list of column attributes from History model
    column_attrs = [getattr(History, col) for col in columns]

    stmt = (
        select(*column_attrs, func.count().label('count'))
        .where(
            History.user_id == self.user_id,
            History.deleted == False
        )
        .group_by(*column_attrs)
        .order_by(*column_attrs)
    )

    results = self.session.execute(stmt).all()

    # Convert results to list of dicts
    rows = []
    for row in results:
      row_dict = {}
      for i, col in enumerate(columns):
        row_dict[col] = row[i]
      row_dict['count'] = row[-1]  # Last element is the count
      rows.append(row_dict)

    return rows, columns + ["count"]

  # --------------------------------------------
  # SOFT DELETE
  # --------------------------------------------

  def remove(self, record_id: str) -> bool:
    stmt = (
        sql_update(History)
        .where(
            History.id == record_id,
            History.user_id == self.user_id,
            History.deleted == False
        )
        .values(deleted=True)
    )

    result = self.session.execute(stmt)
    self.session.commit()

    if result.rowcount == 0:
      raise HistoryNotFoundError(f"Record with id {record_id} not found")

    return True

  def remove_many(self, dataset: str, prompt_types: list[str]) -> int:
    stmt = (
        sql_update(History)
        .where(
            History.user_id == self.user_id,
            History.dataset == dataset,
            History.prompt_type.in_(prompt_types),
            History.deleted == False
        )
        .values(deleted=True)
    )

    result = self.session.execute(stmt)
    self.session.commit()
    return result.rowcount

  def remove_all(self) -> int:
    stmt = (
        sql_update(History)
        .where(
            History.user_id == self.user_id,
            History.deleted == False
        )
        .values(deleted=True)
    )

    result = self.session.execute(stmt)
    self.session.commit()
    return result.rowcount
