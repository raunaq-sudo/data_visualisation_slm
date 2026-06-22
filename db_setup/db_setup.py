import sqlite3
import os

from contextlib import contextmanager

# Define the database file name
DB_NAME = "dashboard_system.db"
DATA_DB_NAME = "data.db"
# SQL script to create tables
create_tables_script = """
-- Enable foreign key support
PRAGMA foreign_keys = ON;

-- 1. Table: user_auth
CREATE TABLE IF NOT EXISTS user_auth (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    role_name TEXT NOT NULL,
    data_source TEXT
);

-- 2. Table: user_dashboard
CREATE TABLE IF NOT EXISTS user_dashboard (
    dashboard_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    status TEXT NOT NULL,
    description TEXT,
    FOREIGN KEY (user_id) REFERENCES user_auth(user_id) ON DELETE CASCADE
);

-- 3. Table: dashboard_widget_mapping
CREATE TABLE IF NOT EXISTS dashboard_widget_mapping (
    dashboard_id INTEGER,
    widget_id INTEGER,
    row_location INTEGER NOT NULL,
    column_location INTEGER NOT NULL,
    PRIMARY KEY (dashboard_id, widget_id),
    FOREIGN KEY (dashboard_id) REFERENCES user_dashboard(dashboard_id) ON DELETE CASCADE
);

-- 4. Table: widget_query_mapping
CREATE TABLE IF NOT EXISTS widget_query_mapping (
    widget_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    status TEXT NOT NULL,
    user_agent_conversation TEXT NOT NULL,
    widget_type TEXT NOT NULL,
    FOREIGN KEY (widget_id) REFERENCES dashboard_widget_mapping(widget_id) ON DELETE CASCADE
);

-- 5. Table: chat_message_history
CREATE TABLE IF NOT EXISTS chat_message_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    message_history TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chat_id)
);

-- 6. Table: metadata_data_table
CREATE TABLE IF NOT EXISTS metadata_data_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    column_type TEXT NOT NULL,
    column_description,
    UNIQUE(table_name, column_name)
);

-- 7. Table: metadata_data_table_description
CREATE TABLE IF NOT EXISTS metadata_data_table_description (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    table_description,
    UNIQUE(table_name)
);
"""

@contextmanager
def get_db_connection(db_path: str = DB_NAME):
    """Context manager ensuring DB connections are always closed, even on error."""
    connection = sqlite3.connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()



def initialize_database():
    with get_db_connection() as connection:
    
        # Connect to SQLite (It will create the file if it doesn't exist)
        cursor = connection.cursor()
        
        # Execute the script containing multiple SQL statements
        cursor.executescript(create_tables_script)
        
        # Commit changes
        connection.commit()
        print(f"Database '{DB_NAME}' initialized and tables created successfully!")
        

def update_metadata():
    with get_db_connection() as connection:
        with get_db_connection(DATA_DB_NAME) as connection_data:
        
            cursor = connection.cursor()

            # Connection to data.db
            cursor_data = connection_data.cursor()

            # fetch schema from data db
            cursor_data.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            # Fetch trusted table names directly from sqlite_master
            trusted_table_names = [row[0] for row in cursor_data.fetchall()]

            table_schemas = []
            for table_name in trusted_table_names:
                # PRAGMA table_info accepts parameterized identifiers via cursor description;
                # since SQLite doesn't support ? placeholders in PRAGMA, we use the trusted
                # name fetched directly from sqlite_master (not from user input).
                cursor_data.execute(f"PRAGMA table_info(\"{table_name.replace('\"', '')}\");")
                columns_info = cursor_data.fetchall()
                rows = [
                        (
                            table_name,
                            col[1],  # column_name
                            col[2],  # column_type
                            None
                        )
                        for col in columns_info
                    ]

                cursor.executemany(
                        """
                        INSERT INTO metadata_data_table
                        (
                            table_name,
                            column_name,
                            column_type,
                            column_description
                        )
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(table_name, column_name)
                        DO UPDATE SET
                            column_type = excluded.column_type,
                            column_description = excluded.column_description
                        """,
                        rows
                    )
                cursor.execute(
                    """
                    INSERT INTO metadata_data_table_description(
                        table_name,
                        table_description
                    )
                    VALUES (?, ?)
                    ON CONFLICT(table_name)
                    DO NOTHING
                    
                    
                    """,
                    [table_name, None]
                )

def check_schema(data_db_name, metadata_db_name):
    """
    Compare actual database schema with metadata_data_table.
    Returns a list of discrepancies.
    """

    issues = []

    with sqlite3.connect(data_db_name) as data_conn, \
         sqlite3.connect(metadata_db_name) as meta_conn:

        data_cursor = data_conn.cursor()
        meta_cursor = meta_conn.cursor()

        # Get all user tables
        data_cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name NOT LIKE 'sqlite_%'
        """)

        tables = [row[0] for row in data_cursor.fetchall()]

        for table_name in tables:

            # Actual schema
            data_cursor.execute(f'PRAGMA table_info("{table_name}")')
            actual_columns = {
                row[1]: row[2]
                for row in data_cursor.fetchall()
            }

            # Metadata schema
            meta_cursor.execute(
                """
                SELECT column_name, column_type
                FROM metadata_data_table
                WHERE table_name = ?
                """,
                (table_name,)
            )

            metadata_columns = {
                row[0]: row[1]
                for row in meta_cursor.fetchall()
            }

            # Missing in metadata
            for column_name, column_type in actual_columns.items():
                if column_name not in metadata_columns:
                    issues.append({
                        "table": table_name,
                        "column": column_name,
                        "issue": "Missing from metadata"
                    })

            # Missing in actual DB
            for column_name in metadata_columns:
                if column_name not in actual_columns:
                    issues.append({
                        "table": table_name,
                        "column": column_name,
                        "issue": "Missing from database"
                    })

            # Type mismatch
            for column_name in (
                set(actual_columns.keys())
                & set(metadata_columns.keys())
            ):
                if actual_columns[column_name].upper() != \
                   metadata_columns[column_name].upper():

                    issues.append({
                        "table": table_name,
                        "column": column_name,
                        "issue": "Type mismatch",
                        "database_type": actual_columns[column_name],
                        "metadata_type": metadata_columns[column_name]
                    })

    return issues

if __name__ == "__main__":
    initialize_database()
    update_metadata()
    issues = check_schema(DATA_DB_NAME, DB_NAME)
    print(issues)

