from sqlalchemy import (
    Column, String, Boolean, Text, ForeignKey, Index, DateTime, func
)
from sqlalchemy.orm import relationship
from .base import Base


class Prompt(Base):
  __tablename__ = "prompts"

  id = Column(String, primary_key=True)
  user_id = Column(String, ForeignKey("users.id"), nullable=False)
  name = Column(String, nullable=False)
  content = Column(Text, nullable=False)
  variables = Column(Text)
  deleted = Column(Boolean, default=False)
  created_at = Column(DateTime, default=func.now())
  updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

  user = relationship("User", back_populates="prompts")

  __table_args__ = (
      Index("idx_prompts_user_name_not_deleted", "user_id", "name",
            unique=True, sqlite_where=Column("deleted") == False),
      Index("idx_prompts_user_id", "user_id"),
      Index("idx_prompts_name", "name"),
  )
