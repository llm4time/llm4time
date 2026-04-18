from sqlalchemy import (
    Column, String, Boolean, ForeignKey, Index, DateTime, func
)
from sqlalchemy.orm import relationship
from .base import Base


class Model(Base):
  __tablename__ = "models"

  id = Column(String, primary_key=True)
  user_id = Column(String, ForeignKey("users.id"), nullable=False)
  name = Column(String, nullable=False)
  provider = Column(String, nullable=False)
  api_key = Column(String)
  endpoint = Column(String)
  api_version = Column(String)
  deleted = Column(Boolean, default=False)
  created_at = Column(DateTime, default=func.now())
  updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

  user = relationship("User", back_populates="models")

  __table_args__ = (
      Index("idx_models_user_name_provider_not_deleted", "user_id", "name", "provider",
            unique=True, sqlite_where=Column("deleted") == False),
      Index("idx_models_user_id", "user_id"),
      Index("idx_models_provider", "provider"),
  )
