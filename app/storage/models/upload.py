from sqlalchemy import (
    Column, String, Boolean, Integer, Float, LargeBinary,
    ForeignKey, CheckConstraint, Index, DateTime, func
)
from sqlalchemy.orm import relationship
from .base import Base


class Upload(Base):
  __tablename__ = "uploads"

  id = Column(String, primary_key=True)
  user_id = Column(String, ForeignKey("users.id"), nullable=False)
  filename = Column(String, nullable=False)
  extension = Column(String, nullable=False)
  rows = Column(Integer, CheckConstraint("rows >= 0"))
  columns = Column(Integer, CheckConstraint("columns >= 0"))
  size = Column(Float, CheckConstraint("size >= 0"))
  content = Column(LargeBinary, nullable=False)
  freq = Column(String, nullable=False)
  deleted = Column(Boolean, default=False)
  created_at = Column(DateTime, default=func.now())
  updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

  user = relationship("User", back_populates="uploads")

  __table_args__ = (
      Index("idx_uploads_user_filename_not_deleted", "user_id", "filename",
            unique=True, sqlite_where=Column("deleted") == False),
      Index("idx_uploads_user_id", "user_id"),
      Index("idx_uploads_filename", "filename"),
  )
