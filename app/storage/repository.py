import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .repositories import (
    UsersRepository,
    UploadsRepository,
    ModelsRepository,
    PromptsRepository,
    HistoryRepository,
)

from config.database import DB_URL


# Create engine and session factory
engine = create_engine(
    url=DB_URL,
    # connect_args={"check_same_thread": False},  # Needed for SQLite with Streamlit
    echo=False  # Set to True for SQL query logging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _get_session() -> Session:
  """Get a new database session."""
  return SessionLocal()


def _get_user_id() -> str:
  """Get the current authenticated user's ID."""
  auth = st.session_state.auth
  return auth.user.sub


# Repository factory functions
def users() -> UsersRepository:
  return UsersRepository(session=_get_session())


def uploads() -> UploadsRepository:
  return UploadsRepository(user_id=_get_user_id(), session=_get_session())


def models() -> ModelsRepository:
  return ModelsRepository(user_id=_get_user_id(), session=_get_session())


def prompts() -> PromptsRepository:
  return PromptsRepository(user_id=_get_user_id(), session=_get_session())


def history() -> HistoryRepository:
  return HistoryRepository(user_id=_get_user_id(), session=_get_session())
