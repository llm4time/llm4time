from sqlalchemy import (
    Column, String, Boolean, Integer, Float, Text,
    ForeignKey, CheckConstraint, Index, DateTime, func
)
from sqlalchemy.orm import relationship
from .base import Base


class History(Base):
  __tablename__ = "history"

  id = Column(String, primary_key=True)
  user_id = Column(String, ForeignKey("users.id"), nullable=False)
  model = Column(String)
  provider = Column(String)
  temperature = Column(Float, CheckConstraint("temperature >= 0 AND temperature <= 2"))
  dataset = Column(String)
  columns = Column(Text)
  start_time = Column(String)
  end_time = Column(String)
  prompt_type = Column(String, CheckConstraint(
      "prompt_type IN ('ZERO_SHOT', 'FEW_SHOT', 'COT', 'COT_FEW', 'CUSTOM')"
  ))
  time_series_format = Column(String)
  time_series_type = Column(String, CheckConstraint(
      "time_series_type IN ('NUMERIC', 'TEXTUAL')"
  ))
  examples = Column(Integer)
  sampling = Column(String, CheckConstraint(
      "sampling IN ('FRONTEND', 'BACKEND', 'RANDOM', 'UNIFORM')"
  ))
  forecast_horizon = Column(Integer)
  input_tokens = Column(Integer)
  output_tokens = Column(Integer)
  response_time = Column(Float)
  response_raw = Column(Text)
  response_predicted = Column(Text)
  validation = Column(Text)
  metrics = Column(Text)
  statistics_predicted = Column(Text)
  statistics_validation = Column(Text)
  training = Column(Text)
  prompt = Column(Text)
  deleted = Column(Boolean, default=False)
  created_at = Column(DateTime, default=func.now())
  updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

  user = relationship("User", back_populates="history")

  __table_args__ = (
      Index("idx_history_user_id", "user_id"),
      Index("idx_history_dataset", "dataset"),
      Index("idx_history_created_at", "created_at"),
  )
