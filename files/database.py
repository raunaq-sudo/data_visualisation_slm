"""database.py — shared connection helper.

Imported by app.py and all routers so every module uses
the same context manager and path constants.
"""
import sqlite3
from contextlib import contextmanager

DB_FILE_PATH  = "db_setup/data.db"
ADMIN_DB_PATH = "db_setup/dashboard_system.db"


@contextmanager
def get_db(db_path: str = ADMIN_DB_PATH):
    """Yield a SQLite connection that always commits or rolls back."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
