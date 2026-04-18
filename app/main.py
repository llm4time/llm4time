import subprocess
import glob
import os

from utils import abspath
from storage import create_database
from config.database import DB_ENGINE, DB_PATH


def run():
  # Create database directory
  os.makedirs(abspath("database"), exist_ok=True)

  # Initialize database if it doesn't exist
  if not glob.glob(abspath("database/*.db")):
    create_database(DB_PATH, engine=DB_ENGINE)

  # Run Streamlit app
  subprocess.run(["streamlit", "run", abspath("app.py")])


if __name__ == "__main__":
  run()
