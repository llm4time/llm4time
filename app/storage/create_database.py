from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from .models.base import Base


def create_database(path: str, engine: str = "sqlite"):
  try:
    engine_obj = create_engine(f"{engine}:///{path}")
    Base.metadata.create_all(engine_obj)
    print(f"[OK] Database successfully created at: {path} using {engine}.")
  except SQLAlchemyError as e:
    print("[ERROR] Failed to create database:", str(e))
    raise
