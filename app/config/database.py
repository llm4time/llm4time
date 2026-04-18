from utils import abspath

DB_ENGINE = "sqlite"
DB_NAME = "database.db"
DB_PATH = abspath(f"database/{DB_NAME}")
DB_URL = f"{DB_ENGINE}:///{DB_PATH}"
