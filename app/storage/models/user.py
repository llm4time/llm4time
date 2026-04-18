from sqlalchemy import (
    Column, String, Boolean, LargeBinary, Index, DateTime, func
)
from sqlalchemy.orm import relationship
from .base import Base


class User(Base):
  __tablename__ = "users"

  id = Column(String, primary_key=True)
  email = Column(String, nullable=False, unique=True)
  username = Column(String, nullable=False, unique=True)
  password = Column(String)
  picture = Column(LargeBinary)
  oauth_provider = Column(String)
  oauth_id = Column(String)
  deleted = Column(Boolean, default=False)
  created_at = Column(DateTime, default=func.now())
  updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

  uploads = relationship("Upload", back_populates="user", cascade="all, delete-orphan")
  models = relationship("Model", back_populates="user", cascade="all, delete-orphan")
  prompts = relationship("Prompt", back_populates="user", cascade="all, delete-orphan")
  history = relationship("History", back_populates="user", cascade="all, delete-orphan")

  __table_args__ = (
      Index("idx_users_email", "email"),
      Index("idx_users_username", "username"),
  )
